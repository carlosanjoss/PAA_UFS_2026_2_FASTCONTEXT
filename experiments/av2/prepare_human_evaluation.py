
"""Prepare a double-blind human evaluation of frozen AV2 x RAG generations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import random
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from src.utils.config import PROJECT_ROOT

CONDITIONS = ("semantic", "literal", "hybrid")
ANNOTATORS = ("carlos", "yann", "wilson", "matheus", "gabriela")
SEED = 42
EXPECTED_SHA256 = "d8bad807ba4bc5a72ceb4a1f9d0b5fd6b12045b7e11aa405cf0917fa8e85d4c5"
INPUT = PROJECT_ROOT / "experiments/av2/rag_comparison/final/generations.csv"
OUTPUT = PROJECT_ROOT / "experiments/av2/human_evaluation"
CODEBOOK = PROJECT_ROOT / "data/rag_evaluation/evaluation_codebook.md"
QUALITY_FIELDS = (
    "blind_id", "question", "answer", "correctness", "completeness",
    "clarity", "hallucination_present", "notes",
)
GROUNDED_FIELDS = (
    "blind_id", "question", "answer", "retrieved_context", "groundedness", "notes",
)
MANIFEST_FIELDS = (
    "task_type", "blind_id", "query_id", "condition", "annotator_1",
    "annotator_2", "signature", "context_sha256", "citation_valid",
    "citation_count", "citation_retry_count", "valid_citations",
    "invalid_citations", "generation_time_ms",
)
REQUIRED_FIELDS = {
    "signature", "query_id", "category", "condition", "question", "model",
    "provider", "top_k", "retrieved_chunk_ids", "context_sha256",
    "generation_time_ms", "citation_valid", "citation_count",
    "citation_retry_count", "valid_citations", "invalid_citations",
    "answer", "retrieved_context", "generation_metadata",
}


def sha256(path: Path) -> str:
    """Fingerprint exact input bytes, not normalized CSV data."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_frozen(path: Path, expected_sha256: str) -> list[dict[str, str]]:
    """Reject changed, incomplete or otherwise incompatible generations."""
    actual_sha256 = sha256(path)
    if actual_sha256.lower() != expected_sha256.lower():
        raise ValueError(f"SHA-256 divergente: {actual_sha256}")

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not REQUIRED_FIELDS.issubset(reader.fieldnames):
            raise ValueError("Colunas obrigatórias ausentes no generations.csv")
        rows = list(reader)

    if len(rows) != 90:
        raise ValueError(f"Esperadas 90 respostas; encontradas {len(rows)}")
    if len({row["signature"] for row in rows}) != 1:
        raise ValueError("As assinaturas de geração não são idênticas")
    if Counter(row["condition"] for row in rows) != dict.fromkeys(CONDITIONS, 30):
        raise ValueError("As condições devem ter 30 respostas cada")

    expected_ids = {f"q{i:02d}" for i in range(1, 31)}
    if {row["query_id"] for row in rows} != expected_ids:
        raise ValueError("IDs das 30 perguntas divergentes")
    if len({(row["query_id"], row["condition"]) for row in rows}) != 90:
        raise ValueError("Pares pergunta-condição duplicados")

    questions: dict[str, str] = {}
    for row in rows:
        query_id = row["query_id"]
        if query_id in questions and questions[query_id] != row["question"]:
            raise ValueError(f"Pergunta mudou entre condições: {query_id}")
        questions[query_id] = row["question"]
        if row["model"] != "qwen2.5:3b" or row["provider"] != "ollama":
            raise ValueError("Modelo/provedor divergente")
        if row["top_k"] != "5":
            raise ValueError("Top-k diferente de cinco")
        if not row["question"].strip() or not row["answer"].strip():
            raise ValueError("Pergunta ou resposta vazia")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", row["context_sha256"]):
            raise ValueError("Hash do contexto inválido")
        if not row["retrieved_context"].strip():
            raise ValueError("Contexto de auditoria não registrado")

        chunk_ids = json.loads(row["retrieved_chunk_ids"])
        if (not isinstance(chunk_ids, list) or len(chunk_ids) > 5
                or len(chunk_ids) != len(set(chunk_ids))
                or any(not isinstance(chunk, str) or not chunk for chunk in chunk_ids)):
            raise ValueError("IDs de chunks inválidos")
        if (not chunk_ids and
                row["retrieved_context"].strip() != "No context was retrieved."):
            raise ValueError("Representação inesperada de contexto vazio")
        for field in ("valid_citations", "invalid_citations"):
            if not isinstance(json.loads(row[field]), list):
                raise ValueError(f"Campo JSON inválido: {field}")
        meta = json.loads(row["generation_metadata"])
        if (str(meta.get("requested_seed")) != "42"
                or float(meta.get("requested_temperature", -1)) != 0.0
                or meta.get("done_reason") != "stop"):
            raise ValueError("Parâmetros ou finalização incompatíveis")
    return rows


def without_chunk_citations(answer: str, chunk_ids: list[str]) -> str:
    """Use the AV1 rule: hide exact [chunk_id] citations in quality."""
    for chunk_id in sorted(set(chunk_ids), key=len, reverse=True):
        answer = re.sub(re.escape(f"[{chunk_id}]"), "", answer)
    answer = re.sub(r"[ \t]+([.,;:!?])", r"\1", answer)
    answer = re.sub(r"[ \t]{2,}", " ", answer)
    answer = re.sub(r"\n[ \t]+", "\n", answer)
    return answer.strip()


def opaque_id(task_type: str, row: dict[str, str]) -> str:
    """Deterministic, opaque ID (same construction as AV1)."""
    run_id = f"{row['condition']}:{row['query_id']}"
    digest = hashlib.sha256(
        f"{SEED}|{task_type}|{run_id}".encode()
    ).hexdigest()[:12].upper()
    return f"{'Q' if task_type == 'quality' else 'G'}-{digest}"


def balanced_pairs(ids: list[str], seed: int) -> dict[str, tuple[str, str]]:
    """Nine items per evaluator pair, as in the AV1 pairing algorithm."""
    pairs = list(itertools.combinations(ANNOTATORS, 2))
    if len(ids) % len(pairs):
        raise ValueError("Quantidade incompatível com balanceamento exato")
    randomized_ids = sorted(ids)
    random.Random(seed).shuffle(randomized_ids)
    schedule = pairs * (len(ids) // len(pairs))
    random.Random(seed + 1).shuffle(schedule)
    return dict(zip(randomized_ids, schedule, strict=True))


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def generate(rows: list[dict[str, str]], destination: Path, codebook: Path,
             input_sha256: str) -> None:
    """Create new derived files; never replace existing human judgments."""
    if destination.exists():
        raise FileExistsError(f"Diretório existente, não sobrescrito: {destination}")
    if not codebook.is_file():
        raise FileNotFoundError(f"Rubrica oficial da AV1 não encontrada: {codebook}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".av2_human_eval_", dir=destination.parent))
    try:
        tasks: dict[str, list[dict[str, str]]] = {"quality": [], "groundedness": []}
        mapping: dict[str, dict[str, str]] = {}
        for row in rows:
            chunk_ids: list[str] = json.loads(row["retrieved_chunk_ids"])
            for task in ("quality", "groundedness"):
                blind_id = opaque_id(task, row)
                if blind_id in mapping:
                    raise ValueError("Colisão de identificadores cegos")
                mapping[blind_id] = row
                base = {"blind_id": blind_id, "question": row["question"],
                        "answer": row["answer"], "notes": ""}
                if task == "quality":
                    base["answer"] = without_chunk_citations(row["answer"], chunk_ids)
                    for field in (
                        "correctness", "completeness", "clarity",
                        "hallucination_present",
                    ):
                        base[field] = ""
                else:
                    base["retrieved_context"] = row["retrieved_context"]
                    base["groundedness"] = ""
                tasks[task].append(base)

        assignments = {
            "quality": balanced_pairs(
                [r["blind_id"] for r in tasks["quality"]], SEED + 101
            ),
            "groundedness": balanced_pairs(
                [r["blind_id"] for r in tasks["groundedness"]], SEED + 202
            ),
        }
        manifests: list[dict[str, str]] = []
        count = Counter()
        task_fields = (("quality", QUALITY_FIELDS),
                       ("groundedness", GROUNDED_FIELDS))
        for task, fields in task_fields:
            items = tasks[task]
            random.Random(SEED + (11 if task == "quality" else 22)).shuffle(items)
            write_csv(stage / f"{task}_blinded.csv", fields, items)
            for item in items:
                blind_id = item["blind_id"]
                pair = assignments[task][blind_id]
                original = mapping[blind_id]
                manifests.append({
                    "task_type": task, "blind_id": blind_id,
                    "query_id": original["query_id"],
                    "condition": original["condition"],
                    "annotator_1": pair[0], "annotator_2": pair[1],
                    **{field: original[field] for field in MANIFEST_FIELDS
                       if field in original},
                })
                for name in pair:
                    count[(name, task)] += 1

            for idx, name in enumerate(ANNOTATORS):
                personal = [
                    item for item in items
                    if name in assignments[task][item["blind_id"]]
                ]
                shuffle_seed = SEED + (1000 if task == "quality" else 2000) + idx
                random.Random(shuffle_seed).shuffle(personal)
                write_csv(
                    stage / "assignments" / f"{name}_{task}.csv", fields, personal
                )

        for name in ANNOTATORS:
            if (count[(name, "quality")], count[(name, "groundedness")]) != (36, 36):
                raise ValueError(f"Balanceamento inválido para {name}")
        write_csv(stage / "hidden_manifest.csv", MANIFEST_FIELDS, manifests)
        summary = [{"annotator": name,
                    "quality_items": str(count[(name, "quality")]),
                    "groundedness_items": str(count[(name, "groundedness")]),
                    "total_items": "72"} for name in ANNOTATORS]
        summary_fields = (
            "annotator", "quality_items", "groundedness_items", "total_items"
        )
        write_csv(stage / "assignment_summary.csv", summary_fields, summary)

        rubric = codebook.read_text(encoding="utf-8")
        if "No-RAG, Optimized ou Semantic" not in rubric:
            raise ValueError(
                "A rubrica AV1 esperada mudou; verificar antes de prosseguir"
            )
        rubric = rubric.replace(
            "No-RAG, Optimized ou Semantic", "Semantic, Literal ou Hybrid"
        )
        (stage / "evaluation_codebook.md").write_text(rubric, encoding="utf-8")
        (stage / "README_ADMIN.md").write_text(
            "# AV2 × RAG — Avaliação humana cega\n\n"
            f"SHA-256 de generations.csv: `{input_sha256}`\n\n"
            f"SHA-256 da rubrica AV1 original: `{sha256(codebook)}`\n\n"
            "Seed: 42; duas avaliações independentes por item.\n"
            "Qualidade: 90 × 2; groundedness: 90 × 2; total: 360.\n"
            "Cada avaliador: 36 de qualidade + 36 de groundedness.\n\n"
            "**Compartilhar apenas** rubrica e seus próprios dois CSVs.\n"
            "**Não compartilhar** manifesto, resumo ou gerações.\n"
            "Não revelar condições até a adjudicação final.\n"
            "Não modificar dados além das colunas de avaliação e notes.\n",
            encoding="utf-8",
        )
        stage.rename(destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--codebook", type=Path, default=CODEBOOK)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    rows = read_frozen(args.input, EXPECTED_SHA256)
    print("AV2 × RAG — 90 gerações verificadas; SHA-256 confirmado")
    if not args.check_only:
        generate(rows, args.output, args.codebook, sha256(args.input))
        print(f"Avaliação cega preparada em: {args.output}")
        print("90 qualidade + 90 groundedness; 360 julgamentos distribuídos")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
