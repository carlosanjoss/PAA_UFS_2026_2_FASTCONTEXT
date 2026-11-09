"""Audita, em modo somente leitura, o baseline RAG do FastContext."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import yaml

from src.ingestion.chunk_loader import load_chunks_jsonl
from src.semantic.persistence import calculate_corpus_fingerprint

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "rag_model_comparison.yaml"


def checksum(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    with CONFIG_PATH.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)["model_comparison"]

    baseline = config["baseline"]
    corpus = config["corpus"]
    generation = config["generation"]
    conditions = tuple(config["conditions"])

    baseline_path = ROOT / baseline["generations_csv"]
    corpus_path = ROOT / corpus["chunks_file"]

    for path in (baseline_path, corpus_path):
        require(
            path.is_file(),
            f"Arquivo obrigatorio ausente: {path}",
        )

    # Verifica o baseline contra a versao do Git HEAD.
    # Preserva o SHA historico, sem tratá-lo como equivalente.
    import subprocess

    local_bytes = baseline_path.read_bytes()
    normalized_bytes = local_bytes.replace(b"\r\n", b"\n")

    actual_hash = hashlib.sha256(local_bytes).hexdigest()
    normalized_hash = hashlib.sha256(normalized_bytes).hexdigest()

    documented_hash = baseline["expected_sha256"].lower()
    pinned_git_hash = baseline["git_head_sha256_lf"].lower()

    require(
        normalized_hash == pinned_git_hash,
        "SHA-256 normalizado nao corresponde ao "
        "baseline versionado aprovado: "
        f"{normalized_hash}",
    )

    git_result = subprocess.run(
        [
            "git",
            "show",
            "HEAD:experiments/rag_results/generations.csv",
        ],
        capture_output=True,
        check=True,
    )

    git_bytes = git_result.stdout.replace(b"\r\n", b"\n")
    git_hash = hashlib.sha256(git_bytes).hexdigest()

    require(
        git_hash == pinned_git_hash,
        f"Git HEAD diferente do hash fixado: {git_hash}",
    )

    require(
        normalized_bytes == git_bytes,
        "Conteudo local diferente do Git HEAD.",
    )

    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    print("INTEGRIDADE DO BASELINE")
    print("-" * 60)
    print(f"SHA historico:     {documented_hash}")
    print(f"SHA Git validado:  {git_hash}")
    print(f"SHA local original:{actual_hash}")
    print(f"Commit Git HEAD:   {git_commit}")

    if git_hash != documented_hash:
        print(
            "AVISO: o hash historico nao corresponde "
            "ao conteudo do Git HEAD. "
            "A divergencia foi preservada no registro."
        )

    print("Local e Git HEAD equivalentes apos normalizacao.")
    print()

    with baseline_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        rows = list(csv.DictReader(stream))

    require(
        len(rows) == int(baseline["expected_rows"]),
        "Quantidade de linhas inesperada.",
    )

    chunks = load_chunks_jsonl(corpus_path)
    fingerprint = calculate_corpus_fingerprint(chunks)

    require(
        len(chunks) == int(corpus["expected_chunks"]),
        "Numero de chunks divergente.",
    )

    require(
        fingerprint == corpus["expected_fingerprint"],
        f"Fingerprint do corpus divergente: {fingerprint}",
    )

    chunk_ids = {str(chunk["chunk_id"]) for chunk in chunks}

    counts: Counter[str] = Counter()
    run_ids: set[str] = set()
    questions: dict[str, str] = {}

    for row in rows:
        condition = row["condition"]
        query_id = row["query_id"]
        run_id = row["run_id"]

        require(
            condition in conditions,
            f"Condicao inesperada: {condition}",
        )

        require(
            run_id == f"{condition}|{query_id}",
            f"Run ID invalido: {run_id}",
        )

        require(
            run_id not in run_ids,
            f"Run ID repetido: {run_id}",
        )

        run_ids.add(run_id)
        counts[condition] += 1

        require(
            row["expected_model"] == baseline["model"],
            f"Modelo esperado divergente: {run_id}",
        )

        require(
            row["actual_model"] == baseline["model"],
            f"Modelo real divergente: {run_id}",
        )

        require(
            row["actual_provider"] == baseline["provider"],
            f"Provider divergente: {run_id}",
        )

        require(
            float(row["temperature"]) == float(generation["temperature"]),
            f"Temperatura divergente: {run_id}",
        )

        require(
            int(row["max_tokens"]) == int(generation["max_tokens"]),
            f"Max tokens divergente: {run_id}",
        )

        require(
            row["think"].lower() == str(generation["think"]).lower(),
            f"Think divergente: {run_id}",
        )

        require(
            int(row["generation_seed"]) == int(generation["seed"]),
            f"Seed divergente: {run_id}",
        )

        require(
            row["corpus_fingerprint"] == fingerprint,
            f"Fingerprint divergente: {run_id}",
        )

        require(
            int(row["corpus_chunks"]) == len(chunks),
            f"Corpus chunks divergente: {run_id}",
        )

        if query_id in questions:
            require(
                questions[query_id] == row["question"],
                f"Pergunta alterada: {query_id}",
            )
        else:
            questions[query_id] = row["question"]

        retrieved = json.loads(row["retrieved_chunk_ids"])

        expected_count = (
            0 if condition == "no_rag" else int(config["retrieval"]["top_k"])
        )

        require(
            len(retrieved) == expected_count,
            f"Top-k divergente: {run_id}",
        )

        require(
            len(set(retrieved)) == len(retrieved),
            f"Chunk duplicado: {run_id}",
        )

        require(
            all(item in chunk_ids for item in retrieved),
            f"Chunk ausente do corpus: {run_id}",
        )

    require(
        len(questions) == int(config["queries"]["expected_count"]),
        "Quantidade de perguntas divergente.",
    )

    for condition in conditions:
        require(
            counts[condition] == len(questions),
            f"Condicao incompleta: {condition}",
        )

    print("AUDITORIA APROVADA - baseline somente leitura")
    print(f"CSV SHA-256: {actual_hash}")
    print(f"Corpus: {len(chunks)} chunks")
    print(f"Fingerprint: {fingerprint}")
    print(f"Perguntas unicas: {len(questions)}")
    print(f"Resultados: {len(rows)}")

    print("Condicoes: " + ", ".join(f"{key}={counts[key]}" for key in conditions))

    print("Nenhum arquivo original foi alterado.")


if __name__ == "__main__":
    main()
