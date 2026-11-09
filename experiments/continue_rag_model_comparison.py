"""Continua a comparacao RAG sem alterar o protocolo ou os runs salvos.

Falhas definitivas por truncamento sao registradas separadamente,
nunca como respostas completas.

Uso:
    python -m experiments.continue_rag_model_comparison
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
from time import perf_counter_ns

from experiments import run_rag_model_comparison as core
from src.rag.providers.base import (
    LLMMessage,
    LLMTruncatedResponseError,
)
from src.rag.providers.ollama import OllamaProvider
from src.utils.config import PROJECT_ROOT


def fail_path(output: Path, role: str, item: dict) -> Path:
    return output / "failures" / role / item["condition"] / f"{item['query_id']}.json"


def success_path(output: Path, role: str, item: dict) -> Path:
    return output / "runs" / role / item["condition"] / f"{item['query_id']}.json"


def validate_failure(
    failure: dict,
    item: dict,
    role: str,
    model: str,
    cfg: dict,
    manifest: dict,
) -> None:
    content = {k: v for k, v in failure.items() if k != "record_sha256"}

    core.require(
        failure.get("record_sha256") == core.sha256(core.stable_json(content)),
        f"SHA invalido: {role}|{item['run_id']}",
    )

    checks = {
        "run_key": f"{role}|{item['run_id']}",
        "run_id": item["run_id"],
        "condition": item["condition"],
        "query_id": item["query_id"],
        "model_role": role,
        "model": model,
        "provider": "ollama",
        "prompt_sha256": item["prompt_sha256"],
        "frozen_prompts_sha256": manifest["frozen_prompts_sha256"],
        "source_signature": manifest["current_source_signature"],
        "corpus_fingerprint": manifest["corpus_fingerprint"],
        "generation": cfg["generation"],
        "error_type": "LLMTruncatedResponseError",
        "status": "truncated",
    }

    for key, value in checks.items():
        core.require(
            failure.get(key) == value,
            f"Falha incoerente ({key}): {role}|{item['run_id']}",
        )

    core.require(
        isinstance(failure.get("elapsed_ns"), int) and failure["elapsed_ns"] >= 0,
        "Tempo da falha invalido.",
    )


def save_failure(
    path: Path,
    item: dict,
    role: str,
    model: str,
    cfg: dict,
    manifest: dict,
    error: LLMTruncatedResponseError,
    elapsed_ns: int,
) -> None:
    entry = {
        "status": "truncated",
        "error_type": "LLMTruncatedResponseError",
        "error_message": str(error),
        "run_key": f"{role}|{item['run_id']}",
        "run_id": item["run_id"],
        "condition": item["condition"],
        "query_id": item["query_id"],
        "model_role": role,
        "model": model,
        "provider": "ollama",
        "prompt_sha256": item["prompt_sha256"],
        "frozen_prompts_sha256": manifest["frozen_prompts_sha256"],
        "source_signature": manifest["current_source_signature"],
        "corpus_fingerprint": manifest["corpus_fingerprint"],
        "generation": dict(cfg["generation"]),
        "elapsed_ns": elapsed_ns,
        "created_at_utc": core.now_utc(),
        "answer": None,
    }

    entry["record_sha256"] = core.sha256(core.stable_json(entry))

    validate_failure(entry, item, role, model, cfg, manifest)

    path.parent.mkdir(parents=True, exist_ok=True)

    # A abertura exclusiva impede sobrescrita.
    with path.open("xb") as stream:
        data = json.dumps(
            entry,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        stream.write(data.encode("utf-8") + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_status(
    output: Path,
    selected: list[dict],
    roles: dict[str, str],
    cfg: dict,
    manifest: dict,
) -> tuple[int, int, int]:
    """Gera a matriz completa de status experimental."""

    rows = []
    failures = []

    for item in selected:
        for role, model in roles.items():
            success = success_path(output, role, item)
            failure = fail_path(output, role, item)

            core.require(
                not (success.exists() and failure.exists()),
                "Uma execucao possui sucesso e falha "
                f"simultaneamente: {role}|{item['run_id']}",
            )

            status = "pending"
            error = ""

            if success.exists():
                record = core.read_json(success)

                core.validate_record(
                    record,
                    item,
                    role,
                    model,
                    cfg,
                    manifest,
                )
                status = "success"

            elif failure.exists():
                entry = core.read_json(failure)

                validate_failure(
                    entry,
                    item,
                    role,
                    model,
                    cfg,
                    manifest,
                )

                failures.append(entry)
                error = entry["error_message"]
                status = "truncated"

            rows.append(
                {
                    "run_key": f"{role}|{item['run_id']}",
                    "model_role": role,
                    "model": model,
                    "condition": item["condition"],
                    "query_id": item["query_id"],
                    "prompt_sha256": item["prompt_sha256"],
                    "status": status,
                    "error_message": error,
                }
            )

    # Verifica se existem falhas fora da matriz congelada.
    expected = {
        fail_path(output, role, item).resolve() for item in selected for role in roles
    }

    failures_root = output / "failures"

    if failures_root.exists():
        actual = {path.resolve() for path in failures_root.rglob("*.json")}
        core.require(
            actual.issubset(expected),
            "Foram encontrados registros de falha inesperados.",
        )

    stream = io.StringIO(newline="")

    writer = csv.DictWriter(
        stream,
        fieldnames=[
            "run_key",
            "model_role",
            "model",
            "condition",
            "query_id",
            "prompt_sha256",
            "status",
            "error_message",
        ],
        lineterminator="\n",
    )

    writer.writeheader()
    writer.writerows(rows)

    core.atomic_write(
        output / "generation_status.csv",
        stream.getvalue().encode("utf-8"),
    )

    core.atomic_write(
        output / "generation_failures.jsonl",
        b"".join(core.stable_json(failure) + b"\n" for failure in failures),
    )

    counts = {
        status: sum(row["status"] == status for row in rows)
        for status in (
            "success",
            "truncated",
            "pending",
        )
    }

    summary = {
        "expected": len(rows),
        "counts": counts,
        "frozen_prompts_sha256": manifest["frozen_prompts_sha256"],
        "source_signature": manifest["current_source_signature"],
        "note": ("Uma geracao truncada nao constitui uma resposta completa."),
    }

    core.atomic_write(
        output / "generation_status_summary.json",
        (
            json.dumps(
                summary,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8"),
    )

    return (
        counts["success"],
        counts["truncated"],
        counts["pending"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
    )
    args = parser.parse_args()

    # Reutiliza todas as validacoes originais.
    cfg, manifest, frozen, by_id = core.load_inputs()

    selected = frozen

    if args.smoke:
        first = frozen[0]["query_id"]
        selected = [item for item in frozen if item["query_id"] == first]
        core.require(
            len(selected) == 3,
            "Smoke exige tres condicoes.",
        )

    output_root = PROJECT_ROOT / cfg["output"]["directory"]

    output = output_root / "smoke" if args.smoke else output_root

    core.require(
        output_root.resolve()
        != (PROJECT_ROOT / "experiments" / "rag_results").resolve(),
        "Diretorio de saida nao pode ser o baseline historico.",
    )

    roles = {
        "baseline": cfg["baseline"]["model"],
        "candidate": cfg["candidate"]["model"],
    }

    total = len(selected) * len(roles)

    print("=" * 65)
    print("FASTCONTEXT - CONTINUACAO CONTROLADA")
    print("=" * 65)
    print("Modelos:", roles)
    print(
        "SHA-256 dos prompts:",
        manifest["frozen_prompts_sha256"],
    )

    completed, expected = core.write_outputs(
        output,
        cfg,
        manifest,
        frozen,
        args.smoke,
    )

    successes, truncated, pending = write_status(
        output,
        selected,
        roles,
        cfg,
        manifest,
    )

    core.require(
        completed == successes and total == expected,
        "Inconsistencia entre os registros e o status.",
    )

    print(
        f"Situacao inicial: {successes} respostas, "
        f"{truncated} truncamentos, "
        f"{pending} pendentes."
    )

    if pending == 0:
        print("Todas as entradas possuem resultados ou falhas registradas.")
        return 0

    providers = {}

    for role, model in roles.items():
        provider = OllamaProvider(
            model=model,
            base_url=cfg["candidate"]["base_url"],
        )

        core.require(
            provider.is_available(),
            f"Modelo indisponivel: {model}",
        )

        providers[role] = provider

    core.verify_model_provenance(
        output,
        cfg["candidate"]["base_url"],
        roles,
    )

    if cfg["warmup"]["enabled"]:
        print("Aquecimento fora das medicoes...")

        for role, provider in providers.items():
            response = provider.generate(
                messages=[
                    LLMMessage(
                        role="system",
                        content="Reply directly and concisely.",
                    ),
                    LLMMessage(
                        role="user",
                        content="Reply only with OK.",
                    ),
                ],
                config=core.make_generation_config(cfg),
            )

            core.check_response(
                roles[role],
                "ollama",
                response.model,
                response.provider,
            )

        print("Aquecimento concluido.")

    checkpoint_every = int(cfg["checkpoint"]["every_completed_runs"])

    new_attempts = 0

    for index, item in enumerate(selected, 1):
        for role, model in roles.items():
            success = success_path(output, role, item)
            failure = fail_path(output, role, item)

            if success.exists() or failure.exists():
                continue

            print(
                f"[{index}/{len(selected)}] {role} ({model}) | {item['run_id']}",
                flush=True,
            )

            start = perf_counter_ns()

            try:
                record = core.execute_one(
                    item,
                    role,
                    model,
                    providers[role],
                    cfg,
                    manifest,
                    by_id,
                )

            except LLMTruncatedResponseError as exc:
                elapsed_ns = perf_counter_ns() - start

                save_failure(
                    failure,
                    item,
                    role,
                    model,
                    cfg,
                    manifest,
                    exc,
                    elapsed_ns,
                )

                print(
                    f"  TRUNCAMENTO REGISTRADO: {exc}",
                    flush=True,
                )

            else:
                core.validate_record(
                    record,
                    item,
                    role,
                    model,
                    cfg,
                    manifest,
                )

                core.save_run(
                    success,
                    record,
                )

                print(
                    "  Resposta completa salva.",
                    flush=True,
                )

            new_attempts += 1

            if new_attempts % checkpoint_every == 0:
                core.write_outputs(
                    output,
                    cfg,
                    manifest,
                    frozen,
                    args.smoke,
                )

                ok, failed, remaining = write_status(
                    output,
                    selected,
                    roles,
                    cfg,
                    manifest,
                )

                print(
                    f"  Checkpoint: {ok} respostas, "
                    f"{failed} truncadas, "
                    f"{remaining} pendentes.",
                    flush=True,
                )

    core.write_outputs(
        output,
        cfg,
        manifest,
        frozen,
        args.smoke,
    )

    ok, failed, remaining = write_status(
        output,
        selected,
        roles,
        cfg,
        manifest,
    )

    print("=" * 65)
    print(
        f"FINAL: {ok} respostas completas; "
        f"{failed} truncamentos; "
        f"{remaining} pendentes; "
        f"{total} execucoes previstas."
    )

    print("Relatorios: generation_status.csv e generation_failures.jsonl")

    print("Nenhum limite foi alterado. Nenhum resultado anterior foi sobrescrito.")

    core.require(
        remaining == 0,
        "Ainda existem execucoes pendentes.",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
