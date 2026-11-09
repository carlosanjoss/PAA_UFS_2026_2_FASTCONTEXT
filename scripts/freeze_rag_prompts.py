"""Congela os 90 prompts do experimento RAG sem alterar o baseline."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import yaml

from experiments.run_rag_experiments import (
    CONFIG_PATH as ORIGINAL_CONFIG_PATH,
)
from experiments.run_rag_experiments import (
    NO_RAG_SYSTEM_PROMPT,
    PIPELINE_PATH,
    PROMPT_PATH,
    RUNNER_PATH,
    _calculate_signature,
    _load_ground_truth,
)
from src.ingestion.chunk_loader import load_chunks_jsonl
from src.rag.prompt import ContextChunk, build_rag_prompt
from src.semantic.persistence import calculate_corpus_fingerprint
from src.utils.config import PROJECT_ROOT

CONFIG_PATH = PROJECT_ROOT / "config" / "rag_model_comparison.yaml"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalized_sha256(data: bytes) -> str:
    return sha256(data.replace(b"\r\n", b"\n"))


def load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    require(isinstance(data, dict), f"YAML invalido: {path}")
    return data


def json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def write_verified(path: Path, content: bytes) -> str:
    if path.exists():
        require(
            path.read_bytes() == content,
            f"Arquivo existente diferente: {path}",
        )
        return "ja existe e foi validado"

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("xb") as stream:
        stream.write(content)

    return "criado"


def main() -> None:
    comparison = load_yaml(CONFIG_PATH)["model_comparison"]
    original = load_yaml(ORIGINAL_CONFIG_PATH)["rag_experiment"]

    baseline_cfg = comparison["baseline"]
    corpus_cfg = comparison["corpus"]
    generation_cfg = comparison["generation"]
    conditions = tuple(comparison["conditions"])

    require(
        conditions == tuple(original["conditions"]),
        "As condicoes nao coincidem com o experimento original.",
    )

    require(
        comparison["retrieval"]["top_k"] == original["retrieval"]["top_k"],
        "Top-k foi alterado.",
    )

    require(
        baseline_cfg["model"] == original["generation"]["model"],
        "Modelo baseline divergente da configuracao original.",
    )

    require(
        baseline_cfg["provider"] == original["generation"]["provider"],
        "Provider baseline divergente da configuracao original.",
    )

    for field in (
        "temperature",
        "max_tokens",
        "think",
        "seed",
        "citation_retries",
        "max_truncation_retries",
        "max_retry_tokens",
    ):
        require(
            generation_cfg[field] == original["generation"][field],
            f"Parametro experimental alterado: {field}",
        )

    baseline_path = PROJECT_ROOT / baseline_cfg["generations_csv"]

    ground_truth_path = PROJECT_ROOT / comparison["queries"]["ground_truth_file"]

    chunks_path = PROJECT_ROOT / corpus_cfg["chunks_file"]

    for path in (baseline_path, ground_truth_path, chunks_path):
        require(path.is_file(), f"Arquivo ausente: {path}")

    baseline_bytes = baseline_path.read_bytes()
    baseline_hash = normalized_sha256(baseline_bytes)

    require(
        baseline_hash == baseline_cfg["git_head_sha256_lf"].lower(),
        f"Baseline diferente da versao fixada: {baseline_hash}",
    )

    chunks = load_chunks_jsonl(chunks_path)
    fingerprint = calculate_corpus_fingerprint(chunks)

    require(
        len(chunks) == corpus_cfg["expected_chunks"],
        "Contagem de chunks divergente.",
    )

    require(
        fingerprint == corpus_cfg["expected_fingerprint"],
        "Fingerprint do corpus divergente.",
    )

    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}

    require(
        len(chunks_by_id) == len(chunks),
        "Chunk IDs duplicados.",
    )

    queries = _load_ground_truth(
        ground_truth_path,
        expected_count=comparison["queries"]["expected_count"],
        valid_chunk_ids=set(chunks_by_id),
    )

    queries_by_id = {query.query_id: query for query in queries}

    signature = _calculate_signature(
        config_path=ORIGINAL_CONFIG_PATH,
        ground_truth_path=ground_truth_path,
        prompt_path=PROMPT_PATH,
        pipeline_path=PIPELINE_PATH,
        runner_path=RUNNER_PATH,
        corpus_fingerprint=fingerprint,
        query_ids=tuple(query.query_id for query in queries),
        conditions=conditions,
        top_k=int(comparison["retrieval"]["top_k"]),
        expected_provider=baseline_cfg["provider"],
        expected_model=baseline_cfg["model"],
        temperature=generation_cfg["temperature"],
        max_tokens=generation_cfg["max_tokens"],
        think=generation_cfg["think"],
        generation_seed=generation_cfg["seed"],
        smoke=False,
    )

    with baseline_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        rows = list(csv.DictReader(stream))

    require(
        len(rows) == baseline_cfg["expected_rows"],
        "Total do baseline divergente.",
    )

    signatures = {row["experiment_signature"] for row in rows}

    require(
        len(signatures) == 1,
        "Baseline com assinaturas experimentais distintas.",
    )

    baseline_signature = signatures.pop()

    require(
        signature == baseline_signature,
        "A assinatura do codigo/configuracao atual difere do baseline.\n"
        f"Assinatura original: {baseline_signature}\n"
        f"Assinatura atual:    {signature}\n"
        "Nao congele prompts sem auditar essa divergencia.",
    )

    seen: set[str] = set()
    counts: Counter[str] = Counter()
    frozen: list[dict] = []

    for row in rows:
        run_id = row["run_id"]
        condition = row["condition"]
        query_id = row["query_id"]

        require(
            condition in conditions,
            f"Condicao invalida: {run_id}",
        )

        require(
            run_id == f"{condition}|{query_id}",
            f"Run ID invalido: {run_id}",
        )

        require(
            run_id not in seen,
            f"Run ID duplicado: {run_id}",
        )

        require(
            query_id in queries_by_id,
            f"Pergunta desconhecida: {query_id}",
        )

        seen.add(run_id)
        counts[condition] += 1

        query = queries_by_id[query_id]

        require(
            row["question"] == query.question,
            f"Texto da pergunta divergente: {run_id}",
        )

        require(
            row["category"] == query.category,
            f"Categoria divergente: {run_id}",
        )

        require(
            json.loads(row["relevant_chunk_ids"]) == list(query.relevant_chunks),
            f"Ground truth divergente: {run_id}",
        )

        require(
            row["corpus_fingerprint"] == fingerprint,
            f"Corpus divergente: {run_id}",
        )

        require(
            row["actual_model"] == baseline_cfg["model"],
            f"Modelo diferente: {run_id}",
        )

        require(
            row["actual_provider"] == baseline_cfg["provider"],
            f"Provedor diferente: {run_id}",
        )

        require(
            float(row["temperature"]) == generation_cfg["temperature"],
            f"Temperatura: {run_id}",
        )

        require(
            int(row["max_tokens"]) == generation_cfg["max_tokens"],
            f"Max tokens: {run_id}",
        )

        require(
            row["think"].lower() == str(generation_cfg["think"]).lower(),
            f"Think: {run_id}",
        )

        require(
            int(row["generation_seed"]) == generation_cfg["seed"],
            f"Seed: {run_id}",
        )

        chunk_ids = json.loads(row["retrieved_chunk_ids"])

        require(
            isinstance(chunk_ids, list),
            f"Chunks invalidos: {run_id}",
        )

        require(
            len(set(chunk_ids)) == len(chunk_ids),
            f"Chunk repetido: {run_id}",
        )

        require(
            all(chunk_id in chunks_by_id for chunk_id in chunk_ids),
            f"Chunk desconhecido: {run_id}",
        )

        if condition == "no_rag":
            require(
                not chunk_ids,
                f"No-RAG com chunks: {run_id}",
            )

            messages = [
                {
                    "role": "system",
                    "content": NO_RAG_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": query.question,
                },
            ]

        else:
            require(
                len(chunk_ids) == comparison["retrieval"]["top_k"],
                f"Top-k: {run_id}",
            )

            context = [
                ContextChunk(
                    chunk_id=chunk_id,
                    content=chunks_by_id[chunk_id]["content"],
                    source_path=chunks_by_id[chunk_id]["source_path"],
                    section_title=chunks_by_id[chunk_id]["section_title"],
                )
                for chunk_id in chunk_ids
            ]

            prompt = build_rag_prompt(
                query=query.question,
                chunks=context,
            )

            messages = [
                {
                    "role": "system",
                    "content": prompt.system,
                },
                {
                    "role": "user",
                    "content": prompt.user,
                },
            ]

        frozen.append(
            {
                "run_id": run_id,
                "condition": condition,
                "query_id": query_id,
                "question": query.question,
                "context_chunk_ids": chunk_ids,
                "messages": messages,
                "prompt_sha256": sha256(json_bytes(messages)),
            }
        )

    for condition in conditions:
        require(
            counts[condition] == len(queries),
            f"Condicao incompleta: {condition}",
        )

    require(
        len(seen) == len(queries) * len(conditions),
        "Matriz incompleta.",
    )

    prompts_bytes = b"".join(json_bytes(record) + b"\n" for record in frozen)

    output_dir = PROJECT_ROOT / comparison["output"]["directory"]

    prompts_path = output_dir / comparison["output"]["frozen_prompts"]

    manifest_path = output_dir / "frozen_prompts.manifest.json"

    manifest = {
        "baseline_csv_sha256_lf": baseline_hash,
        "baseline_historical_sha256": (baseline_cfg["expected_sha256"].lower()),
        "baseline_experiment_signature": baseline_signature,
        "corpus_fingerprint": fingerprint,
        "rows": len(frozen),
        "condition_counts": dict(sorted(counts.items())),
        "frozen_prompts_sha256": sha256(prompts_bytes),
        "generation": generation_cfg,
    }

    manifest_bytes = (
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )

    # Proibe sobrescrever um congelamento divergente.
    for path, content in (
        (prompts_path, prompts_bytes),
        (manifest_path, manifest_bytes),
    ):
        if path.exists():
            require(
                path.read_bytes() == content,
                f"Artefato existente divergente: {path}",
            )

    status_prompts = write_verified(
        prompts_path,
        prompts_bytes,
    )

    status_manifest = write_verified(
        manifest_path,
        manifest_bytes,
    )

    print("CONGELAMENTO DOS PROMPTS APROVADO")
    print(f"Assinatura experimental: {baseline_signature}")
    print(f"Fingerprint do corpus: {fingerprint}")
    print(f"Prompts: {len(frozen)}")

    print("Distribuicao: " + ", ".join(f"{key}={counts[key]}" for key in conditions))

    print(f"SHA-256 dos prompts: {sha256(prompts_bytes)}")

    print(f"Prompts ({status_prompts}): {prompts_path}")

    print(f"Manifesto ({status_manifest}): {manifest_path}")

    print("Baseline preservado; nenhuma nova resposta foi gerada.")


if __name__ == "__main__":
    main()
