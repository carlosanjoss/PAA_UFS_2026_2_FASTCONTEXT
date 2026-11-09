"""Audita e compara as 180 tentativas controladas do FastContext.

Leitura dos resultados individuais, sem executar modelos ou recuperacao.
Uso: python -m scripts.analyze_rag_model_comparison
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from experiments import continue_rag_model_comparison as continuation
from experiments import run_rag_model_comparison as core
from src.utils.config import PROJECT_ROOT


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_bytes(rows: list[dict], columns: list[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=columns,
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def save_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        require(
            path.read_bytes() == data,
            f"Analise ja existe com outro conteudo: {path}",
        )
        print(f"VERIFICADO (sem sobrescrever): {path.name}")
        return

    with path.open("xb") as stream:
        stream.write(data)

    print(f"CRIADO: {path.name}")


def mean_or_none(values: list[float]) -> float | None:
    return round(statistics.mean(values), 6) if values else None


def median_or_none(values: list[float]) -> float | None:
    return round(statistics.median(values), 6) if values else None


def main() -> None:
    cfg, manifest, frozen, _ = core.load_inputs()

    roles = {
        "baseline": cfg["baseline"]["model"],
        "candidate": cfg["candidate"]["model"],
    }

    conditions = list(cfg["conditions"])
    root = PROJECT_ROOT / cfg["output"]["directory"]

    status_path = root / "generation_status.csv"
    status_summary_path = root / "generation_status_summary.json"

    require(
        status_path.is_file(),
        "Execute antes o modulo de continuacao: generation_status.csv ausente.",
    )

    require(
        status_summary_path.is_file(),
        "Resumo de status ausente.",
    )

    with status_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        declared_rows = list(csv.DictReader(stream))

    declared = {row["run_key"]: row for row in declared_rows}

    require(
        len(declared) == len(declared_rows),
        "Run keys duplicados no status CSV.",
    )

    summary = core.read_json(status_summary_path)

    require(
        summary["frozen_prompts_sha256"] == manifest["frozen_prompts_sha256"],
        "Hash dos prompts no resumo diverge.",
    )

    require(
        summary["source_signature"] == manifest["current_source_signature"],
        "Assinatura de fonte diverge.",
    )

    attempts: dict[tuple[str, str], dict] = {}
    expected_keys: set[str] = set()
    counts = Counter()

    group_records: dict[tuple[str, str], list[dict]] = defaultdict(list)

    paired_rows = []

    for item in frozen:
        for role, model in roles.items():
            key = f"{role}|{item['run_id']}"
            expected_keys.add(key)

            good_path = continuation.success_path(root, role, item)

            fail_path = continuation.fail_path(root, role, item)

            require(
                not (good_path.exists() and fail_path.exists()),
                f"Sucesso e falha simultaneos: {key}",
            )

            require(
                good_path.exists() or fail_path.exists(),
                f"Tentativa pendente: {key}",
            )

            if good_path.exists():
                record = core.read_json(good_path)

                core.validate_record(
                    record,
                    item,
                    role,
                    model,
                    cfg,
                    manifest,
                )

                require(
                    record["run_key"] == key,
                    f"Run key divergente: {key}",
                )

                result = {
                    "status": "success",
                    "record": record,
                }

                group_records[(role, item["condition"])].append(record)

            else:
                error = core.read_json(fail_path)

                continuation.validate_failure(
                    error,
                    item,
                    role,
                    model,
                    cfg,
                    manifest,
                )

                result = {
                    "status": "truncated",
                    "failure": error,
                }

            attempts[(role, item["run_id"])] = result

            counts[result["status"]] += 1

            declared_row = declared.get(key)

            require(
                declared_row is not None,
                f"Run nao consta no status CSV: {key}",
            )

            for field, expected in (
                ("status", result["status"]),
                ("model", model),
                ("condition", item["condition"]),
                ("query_id", item["query_id"]),
                (
                    "prompt_sha256",
                    item["prompt_sha256"],
                ),
            ):
                require(
                    declared_row[field] == expected,
                    f"Status CSV incoerente ({field}): {key}",
                )

    require(
        len(frozen) == 90 and len(expected_keys) == 180,
        "A matriz nao tem 90 prompts x 2 modelos.",
    )

    require(
        set(declared) == expected_keys,
        "O status CSV contem runs extras ou faltantes.",
    )

    require(
        summary["expected"] == 180,
        "Quantidade esperada divergente no resumo.",
    )

    require(
        summary["counts"].get("pending") == 0,
        "Existem tentativas pendentes.",
    )

    for kind in ("success", "truncated"):
        require(
            summary["counts"].get(kind) == counts[kind],
            f"Contagem divergente: {kind}",
        )

    # Comparacao pareada: somente respostas completas
    # dos dois modelos recebem delta de latencia.
    for item in frozen:
        base = attempts[("baseline", item["run_id"])]

        cand = attempts[("candidate", item["run_id"])]

        both = base["status"] == cand["status"] == "success"

        a = base.get("record")
        b = cand.get("record")

        paired_rows.append(
            {
                "run_id": item["run_id"],
                "query_id": item["query_id"],
                "condition": item["condition"],
                "prompt_sha256": item["prompt_sha256"],
                "baseline_status": base["status"],
                "candidate_status": cand["status"],
                "baseline_generation_ms": (a["generation_time_ms"] if a else ""),
                "candidate_generation_ms": (b["generation_time_ms"] if b else ""),
                "candidate_minus_baseline_ms": (
                    round(
                        b["generation_time_ns"] / 1e6 - a["generation_time_ns"] / 1e6,
                        6,
                    )
                    if both
                    else ""
                ),
                "baseline_citation_valid": (
                    a["citation_valid"] if a and item["condition"] != "no_rag" else ""
                ),
                "candidate_citation_valid": (
                    b["citation_valid"] if b and item["condition"] != "no_rag" else ""
                ),
            }
        )

    by_condition = []
    pairs_summary = []

    for condition in conditions:
        relevant = [item for item in frozen if item["condition"] == condition]

        for role, model in roles.items():
            records = group_records[(role, condition)]

            success_n = len(records)
            truncated_n = len(relevant) - success_n

            times = [r["generation_time_ns"] / 1e6 for r in records]

            rag = condition != "no_rag"

            by_condition.append(
                {
                    "role": role,
                    "model": model,
                    "condition": condition,
                    "planned": len(relevant),
                    "success": success_n,
                    "truncated": truncated_n,
                    "success_rate": round(success_n / len(relevant), 6),
                    "generation_mean_ms_success_only": mean_or_none(times),
                    "generation_median_ms_success_only": median_or_none(times),
                    "citation_accepted_n": (
                        sum(r["citation_valid"] is True for r in records) if rag else ""
                    ),
                    "with_valid_citation_n": (
                        sum(bool(r["valid_citations"]) for r in records) if rag else ""
                    ),
                    "citation_retry_mean_success_only": (
                        mean_or_none(
                            [float(r["citation_retry_count"]) for r in records]
                        )
                        if rag
                        else ""
                    ),
                }
            )

        subset = [r for r in paired_rows if r["condition"] == condition]

        comparable = [r for r in subset if r["candidate_minus_baseline_ms"] != ""]

        pairs_summary.append(
            {
                "condition": condition,
                "planned_pairs": len(subset),
                "both_success": len(comparable),
                "only_baseline_success": sum(
                    r["baseline_status"] == "success"
                    and r["candidate_status"] != "success"
                    for r in subset
                ),
                "only_candidate_success": sum(
                    r["candidate_status"] == "success"
                    and r["baseline_status"] != "success"
                    for r in subset
                ),
                "both_truncated": sum(
                    r["baseline_status"] == "truncated"
                    and r["candidate_status"] == "truncated"
                    for r in subset
                ),
                "paired_mean_candidate_minus_baseline_ms": mean_or_none(
                    [float(r["candidate_minus_baseline_ms"]) for r in comparable]
                ),
            }
        )

    analysis_dir = root / "analysis"

    save_new(
        analysis_dir / "resumo_por_modelo_condicao.csv",
        csv_bytes(
            by_condition,
            list(by_condition[0]),
        ),
    )

    save_new(
        analysis_dir / "comparacao_pareada.csv",
        csv_bytes(
            paired_rows,
            list(paired_rows[0]),
        ),
    )

    save_new(
        analysis_dir / "resumo_pareado.csv",
        csv_bytes(
            pairs_summary,
            list(pairs_summary[0]),
        ),
    )

    audit = {
        "expected_attempts": len(expected_keys),
        "successful_answers": counts["success"],
        "truncation_failures": counts["truncated"],
        "pending": 0,
        "frozen_prompts_sha256": manifest["frozen_prompts_sha256"],
        "source_signature": manifest["current_source_signature"],
        "generation_status_sha256": sha256_file(status_path),
        "generation_status_summary_sha256": sha256_file(status_summary_path),
        "analysis_note": (
            "Tempos somente de geracao; medias usam "
            "apenas sucessos. Diferencas pareadas usam "
            "somente pares com duas respostas completas. "
            "Aceitacao de citacao nao comprova suporte "
            "factual. Nao foram calculadas metricas de "
            "qualidade humana nem testes de significancia."
        ),
    }

    audit_bytes = (
        json.dumps(
            audit,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")

    save_new(
        analysis_dir / "auditoria_180.json",
        audit_bytes,
    )

    print("\nAUDITORIA DAS 180 TENTATIVAS CONCLUIDA")
    print(f"Respostas completas: {counts['success']}/180")
    print(f"Truncamentos: {counts['truncated']}/180")
    print("Pendentes: 0")

    print("\nPor modelo e condicao:")

    for row in by_condition:
        print(
            f"  {row['model']:12s} "
            f"{row['condition']:9s}  "
            f"completas={row['success']:2d}/30  "
            f"truncadas={row['truncated']:2d}  "
            "media_geracao="
            f"{row['generation_mean_ms_success_only']} ms"
        )

    print(f"\nArquivos em: {analysis_dir}")
    print("Os registros de geracao e o baseline historico nao foram alterados.")


if __name__ == "__main__":
    main()
