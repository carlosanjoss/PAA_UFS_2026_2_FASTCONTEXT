"""Prepara planilhas cegas para avaliacao humana da comparacao FastContext.

Nao executa modelos, nao altera runs, nao atribui notas e nao distribui
itens entre avaliadores. Reutiliza campos/escala da avaliacao AV2 ja documentada.

Execucao na raiz do projeto:
    python -m scripts.prepare_rag_model_comparison_evaluation
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
from collections import Counter
from pathlib import Path

from experiments import continue_rag_model_comparison as continuation
from experiments import run_rag_model_comparison as core
from src.utils.config import PROJECT_ROOT

QUALITY_COLUMNS = (
    "blind_id",
    "question",
    "answer",
    "correctness",
    "completeness",
    "clarity",
    "hallucination_present",
    "notes",
)
GROUNDING_COLUMNS = (
    "blind_id",
    "question",
    "answer",
    "retrieved_context",
    "groundedness",
    "notes",
)
KEY_COLUMNS = (
    "blind_id",
    "dimension",
    "run_key",
    "run_id",
    "condition",
    "query_id",
    "model_role",
    "model",
    "prompt_sha256",
    "record_sha256",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def csv_payload(rows: list[dict], columns: tuple[str, ...]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8-sig")


def json_payload(obj: dict) -> bytes:
    return (
        json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def write_immutable(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(
            path.read_bytes() == data,
            f"Arquivo existente difere; nao sobrescrevi: {path}",
        )
        print(f"VALIDADO: {path}")
        return
    try:
        with path.open("xb") as stream:
            stream.write(data)
            stream.flush()
    except FileExistsError as exc:
        raise RuntimeError(f"Criacao concorrente: {path}") from exc
    print(f"CRIADO: {path}")


def shuffled_rows(items: list[dict], frozen_sha: str, dimension: str) -> list[dict]:
    """Ordem cega reprodutivel, sem modificar os registros de geracao."""
    ordered = sorted(items, key=lambda row: row["record"]["run_key"])
    seed_hex = digest(f"{frozen_sha}|{dimension}|avaliacao".encode())
    rng = random.Random(int(seed_hex, 16))
    rng.shuffle(ordered)
    return ordered


def context_text(item: dict, by_id: dict[str, dict]) -> str:
    sections = []
    for chunk_id in item["context_chunk_ids"]:
        require(chunk_id in by_id, f"Trecho do contexto ausente: {chunk_id}")
        chunk = by_id[chunk_id]
        sections.append(
            f"[chunk_id: {chunk_id}]\n"
            f"source_path: {chunk['source_path']}\n"
            f"section_title: {chunk['section_title']}\n"
            f"content:\n{chunk['content']}"
        )
    return "\n\n---\n\n".join(sections)


def collect(cfg: dict, manifest: dict, frozen: list[dict], by_id: dict):
    output = PROJECT_ROOT / cfg["output"]["directory"]
    roles = {
        "baseline": cfg["baseline"]["model"],
        "candidate": cfg["candidate"]["model"],
    }
    quality_items = []
    grounding_items = []
    counts = Counter()
    failures = []

    require(len(frozen) == 90, "Esperados exatamente 90 prompts congelados.")
    for item in frozen:
        for role, model in roles.items():
            run_key = f"{role}|{item['run_id']}"
            success = continuation.success_path(output, role, item)
            failure = continuation.fail_path(output, role, item)
            require(
                success.exists() != failure.exists(),
                f"Registro pendente ou duplicado: {run_key}",
            )
            if failure.exists():
                failed = core.read_json(failure)
                continuation.validate_failure(failed, item, role, model, cfg, manifest)
                failures.append(run_key)
                counts["truncated"] += 1
                continue

            record = core.read_json(success)
            core.validate_record(record, item, role, model, cfg, manifest)
            require(record["run_key"] == run_key, f"Chave divergente: {run_key}")
            require(
                isinstance(record["answer"], str) and record["answer"].strip(),
                f"Resposta vazia: {run_key}",
            )
            entry = {"item": item, "record": record}
            quality_items.append(entry)
            counts["quality"] += 1
            if item["condition"] != "no_rag":
                require(
                    len(item["context_chunk_ids"]) == int(cfg["retrieval"]["top_k"]),
                    f"Top-k divergente: {run_key}",
                )
                grounding_items.append(entry)
                counts["grounding"] += 1

    require(
        sum(counts[k] for k in ("quality", "truncated")) == 180,
        "A matriz de 180 tentativas nao foi completada.",
    )
    require(
        counts["quality"] == 179 and counts["truncated"] == 1,
        "Contagens diferentes da auditoria fornecida; revisar antes de seguir.",
    )
    require(
        counts["grounding"] == 119,
        "Esperadas 119 respostas RAG completas para fundamentacao.",
    )
    return quality_items, grounding_items, failures, counts


def build_forms(quality_items, grounding_items, by_id, frozen_sha):
    quality_rows = []
    grounding_rows = []
    admin_rows = []
    ids = set()

    def register(blind_id: str, dimension: str, entry: dict) -> None:
        require(blind_id not in ids, f"ID cego repetido: {blind_id}")
        ids.add(blind_id)
        item = entry["item"]
        record = entry["record"]
        admin_rows.append(
            {
                "blind_id": blind_id,
                "dimension": dimension,
                "run_key": record["run_key"],
                "run_id": item["run_id"],
                "condition": item["condition"],
                "query_id": item["query_id"],
                "model_role": record["model_role"],
                "model": record["model"],
                "prompt_sha256": item["prompt_sha256"],
                "record_sha256": record["record_sha256"],
            }
        )

    for index, entry in enumerate(
        shuffled_rows(quality_items, frozen_sha, "qualidade"), 1
    ):
        blind_id = f"Q{index:04d}"
        register(blind_id, "qualidade", entry)
        quality_rows.append(
            {
                "blind_id": blind_id,
                "question": entry["item"]["question"],
                "answer": entry["record"]["answer"],
                "correctness": "",
                "completeness": "",
                "clarity": "",
                "hallucination_present": "",
                "notes": "",
            }
        )

    for index, entry in enumerate(
        shuffled_rows(grounding_items, frozen_sha, "fundamentacao"), 1
    ):
        blind_id = f"G{index:04d}"
        register(blind_id, "fundamentacao", entry)
        grounding_rows.append(
            {
                "blind_id": blind_id,
                "question": entry["item"]["question"],
                "answer": entry["record"]["answer"],
                "retrieved_context": context_text(entry["item"], by_id),
                "groundedness": "",
                "notes": "",
            }
        )

    return quality_rows, grounding_rows, admin_rows


def main() -> None:
    cfg, manifest, frozen, by_id = core.load_inputs()
    output = PROJECT_ROOT / cfg["output"]["directory"]
    audit_path = output / "analysis" / "auditoria_180.json"
    require(audit_path.is_file(), "Auditoria final ausente. Execute a Etapa 5.")
    audit = core.read_json(audit_path)
    require(
        audit.get("expected_attempts") == 180
        and audit.get("successful_answers") == 179
        and audit.get("truncation_failures") == 1
        and audit.get("pending") == 0,
        "Auditoria final diverge das 180 tentativas verificadas.",
    )
    frozen_sha = manifest["frozen_prompts_sha256"]
    require(audit["frozen_prompts_sha256"] == frozen_sha, "SHA congelado divergente.")
    require(
        audit["source_signature"] == manifest["current_source_signature"],
        "Assinatura de codigo divergente.",
    )

    quality, grounding, failures, counts = collect(cfg, manifest, frozen, by_id)
    quality_rows, grounding_rows, admin_rows = build_forms(
        quality,
        grounding,
        by_id,
        frozen_sha,
    )
    require(
        len(quality_rows) == 179
        and len(grounding_rows) == 119
        and len(admin_rows) == 298,
        "Quantidade de linhas cegas invalida.",
    )

    out = output / "avaliacao_humana_cega"
    files = {
        "avaliadores/qualidade_cega.csv": csv_payload(quality_rows, QUALITY_COLUMNS),
        "avaliadores/fundamentacao_cega.csv": csv_payload(
            grounding_rows, GROUNDING_COLUMNS
        ),
        "administrativo/chave_cego_admin.csv": csv_payload(admin_rows, KEY_COLUMNS),
    }

    manual = (
        "# Instrumentos cegos — comparacao Qwen × Llama\n\n"
        "As duas planilhas foram produzidas das 179 respostas completas e "
        "preservam integralmente os textos das respostas. O modelo e a condicao "
        "nao aparecem nas planilhas dos avaliadores.\n\n"
        "## Rubrica ja documentada na avaliacao AV2\n\n"
        "- correctness: 1 (incorreto) a 5 (tecnicamente correto).\n"
        "- completeness: 1 (nao responde) a 5 (atende de forma completa).\n"
        "- clarity: 1 (muito confuso) a 5 (claro e organizado).\n"
        "- hallucination_present: 0 (nao ha afirmacao inventada evidente) "
        "ou 1 (afirmacao tecnica nao fundamentada ou API inventada).\n"
        "- groundedness: 1 (sem suporte no contexto) a 5 (totalmente "
        "alinhada aos excertos recuperados).\n"
        "- notes: justificativa curta e especifica.\n\n"
        "A escala acima reproduz os campos da avaliacao AV2 existente. "
        "Nenhuma nota foi atribuida automaticamente.\n\n"
        "## Escopo\n\n"
        "- Qualidade: 179 itens (inclui No-RAG).\n"
        "- Fundamentacao: 119 itens RAG completos; No-RAG fica fora "
        "desta dimensao por nao possuir contexto recuperado.\n"
        "- Uma tentativa truncada nao integra as planilhas de respostas completas; "
        "consta na auditoria experimental.\n"
        "- Os textos dos chunks seguem os IDs congelados e a ordem original.\n"
        "- As colunas de notas e justificativas estao intencionalmente vazias.\n\n"
        "## Regras de integridade\n\n"
        "Mantenha `administrativo/chave_cego_admin.csv` fora do alcance dos "
        "avaliadores. Nao compartilhe o manifesto administrativo com eles. "
        "As planilhas sao masters cegos: este programa nao define o numero "
        "de avaliadores, nao aloca itens, nao calcula concordancia e nao "
        "simula julgamentos. Defina o plano de distribuicao antes da coleta.\n"
    ).encode()
    files["avaliadores/LEIA-ME.md"] = manual

    report = {
        "source_frozen_prompts_sha256": frozen_sha,
        "source_signature": manifest["current_source_signature"],
        "source_final_audit_sha256": digest(audit_path.read_bytes()),
        "expected_attempts": 180,
        "complete_answers": 179,
        "truncation_failures": failures,
        "quality_form_rows": len(quality_rows),
        "grounding_form_rows": len(grounding_rows),
        "blind_key_rows": len(admin_rows),
        "notes": "Sem avaliacao automatica; nenhuma distribuicao entre avaliadores.",
        "output_sha256": {name: digest(data) for name, data in files.items()},
    }
    files["administrativo/auditoria_preparacao.json"] = json_payload(report)

    for relative, data in files.items():
        write_immutable(out / relative, data)

    print("=" * 68)
    print("PREPARACAO CEGA CONCLUIDA")
    print("Qualidade:", len(quality_rows))
    print("Fundamentacao RAG:", len(grounding_rows))
    print("Falhas definitivas:", failures)
    print("Chave administrativa:", len(admin_rows))
    print("Saidas:", out)
    print("Nenhuma resposta anterior foi modificada; nenhuma nota foi criada.")


if __name__ == "__main__":
    main()
