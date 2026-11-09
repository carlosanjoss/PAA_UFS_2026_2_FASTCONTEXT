"""Congela 90 entradas para comparar dois LLMs em um NOVO experimento.

Nao altera o baseline historico, o corpus ou o experimento original.
Nao afirma que os prompts reconstruidos equivalem aos usados historicamente.
Requer execucao a partir do projeto FastContext:
python -m scripts.freeze_model_comparison_prompts
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

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


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise RuntimeError(reason)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_csv_sha256(path: Path) -> str:
    """Aceita CRLF ou LF; nao modifica o arquivo de origem."""
    return sha256(path.read_bytes().replace(b"\r\n", b"\n"))


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes())


def stable_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def read_config(path: Path, root_key: str) -> dict[str, Any]:
    require(path.is_file(), f"Arquivo ausente: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"YAML invalido: {path}")
    result = value.get(root_key)
    require(isinstance(result, dict), f"Secao ausente: {root_key}")
    return result


def git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def write_immutable(path: Path, data: bytes) -> str:
    """Arquivo existente so e aceito se os bytes forem identicos."""
    if path.exists():
        require(
            path.is_file() and path.read_bytes() == data,
            f"Arquivo existente divergente (nao sobrescrito): {path}",
        )
        return "ja existia, conteudo identico"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
    return "criado"


def main() -> None:
    cfg = read_config(CONFIG_PATH, "model_comparison")
    original = read_config(ORIGINAL_CONFIG_PATH, "rag_experiment")

    baseline = cfg["baseline"]
    candidate = cfg["candidate"]
    generation = cfg["generation"]
    conditions = tuple(cfg["conditions"])
    original_generation = original["generation"]
    top_k = int(cfg["retrieval"]["top_k"])

    require(
        conditions == tuple(original["conditions"]),
        "Condicoes diferentes do protocolo original.",
    )
    require(
        top_k == int(original["retrieval"]["top_k"]),
        "Top-k diferente do protocolo original.",
    )
    require(
        baseline["model"] == original_generation["model"]
        and baseline["provider"] == original_generation["provider"],
        "Identidade do baseline diferente do protocolo original.",
    )
    require(
        candidate["model"] != baseline["model"]
        and candidate["provider"] == baseline["provider"] == "ollama",
        "Modelos devem ser distintos e utilizar Ollama.",
    )
    for key in (
        "temperature",
        "max_tokens",
        "think",
        "seed",
        "citation_retries",
        "max_truncation_retries",
        "max_retry_tokens",
    ):
        require(
            generation[key] == original_generation[key],
            f"Parametro divergente da configuracao original LOCAL: {key} "
            f"({generation[key]!r} != {original_generation[key]!r})",
        )

    baseline_path = PROJECT_ROOT / baseline["generations_csv"]
    corpus_path = PROJECT_ROOT / cfg["corpus"]["chunks_file"]
    ground_truth_path = PROJECT_ROOT / cfg["queries"]["ground_truth_file"]

    for path in (baseline_path, corpus_path, ground_truth_path):
        require(path.is_file(), f"Arquivo necessario ausente: {path}")

    baseline_hash = canonical_csv_sha256(baseline_path)
    require(
        baseline_hash == baseline["git_head_sha256_lf"].lower(),
        "O CSV local nao corresponde ao hash normalizado aprovado: " + baseline_hash,
    )

    corpus = load_chunks_jsonl(corpus_path)
    corpus_fingerprint = calculate_corpus_fingerprint(corpus)
    require(
        len(corpus) == int(cfg["corpus"]["expected_chunks"]),
        "Quantidade de chunks diferente da original.",
    )
    require(
        corpus_fingerprint == cfg["corpus"]["expected_fingerprint"],
        "Fingerprint do corpus divergente.",
    )
    chunks_by_id = {item["chunk_id"]: item for item in corpus}
    require(len(chunks_by_id) == len(corpus), "Chunk IDs duplicados.")

    queries = _load_ground_truth(
        ground_truth_path,
        expected_count=int(cfg["queries"]["expected_count"]),
        valid_chunk_ids=set(chunks_by_id),
    )
    query_by_id = {query.query_id: query for query in queries}
    require(len(query_by_id) == len(queries), "Query IDs duplicados.")

    with baseline_path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(
        len(rows) == int(baseline["expected_rows"]),
        "Baseline com quantidade de linhas inesperada.",
    )
    historical_signatures = {row["experiment_signature"] for row in rows}
    require(
        len(historical_signatures) == 1,
        "Baseline contem varias assinaturas experimentais.",
    )
    historical_signature = historical_signatures.pop()

    # A assinatura atual e apenas uma PROVENIENCIA do novo experimento;
    # nao exigimos que coincida com a assinatura historica nao recuperada.
    current_signature = _calculate_signature(
        config_path=ORIGINAL_CONFIG_PATH,
        ground_truth_path=ground_truth_path,
        prompt_path=PROMPT_PATH,
        pipeline_path=PIPELINE_PATH,
        runner_path=RUNNER_PATH,
        corpus_fingerprint=corpus_fingerprint,
        query_ids=tuple(query.query_id for query in queries),
        conditions=conditions,
        top_k=top_k,
        expected_provider=baseline["provider"],
        expected_model=baseline["model"],
        temperature=float(generation["temperature"]),
        max_tokens=int(generation["max_tokens"]),
        think=bool(generation["think"]),
        generation_seed=int(generation["seed"]),
        smoke=False,
    )

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    counts: Counter[str] = Counter()

    for row in rows:
        run_id = row["run_id"]
        condition = row["condition"]
        query_id = row["query_id"]
        require(condition in conditions, f"Condicao inesperada: {run_id}")
        require(run_id == f"{condition}|{query_id}", f"Run ID invalido: {run_id}")
        require(run_id not in seen, f"Run ID duplicado: {run_id}")
        require(query_id in query_by_id, f"Query ID desconhecido: {run_id}")
        seen.add(run_id)
        counts[condition] += 1
        query = query_by_id[query_id]

        require(row["question"] == query.question, f"Pergunta mudou: {run_id}")
        require(row["category"] == query.category, f"Categoria mudou: {run_id}")
        require(
            json.loads(row["relevant_chunk_ids"]) == list(query.relevant_chunks),
            f"Relevancias divergentes: {run_id}",
        )
        require(
            row["corpus_fingerprint"] == corpus_fingerprint
            and int(row["corpus_chunks"]) == len(corpus),
            f"Corpus divergente: {run_id}",
        )
        require(
            row["expected_model"] == row["actual_model"] == baseline["model"],
            f"Modelo baseline divergente: {run_id}",
        )
        require(
            row["expected_provider"] == row["actual_provider"] == baseline["provider"],
            f"Provider baseline divergente: {run_id}",
        )
        require(
            float(row["temperature"]) == float(generation["temperature"])
            and int(row["max_tokens"]) == int(generation["max_tokens"])
            and row["think"].lower() == str(generation["think"]).lower()
            and int(row["generation_seed"]) == int(generation["seed"]),
            f"Parametros base divergentes: {run_id}",
        )

        chunk_ids = json.loads(row["retrieved_chunk_ids"])
        require(isinstance(chunk_ids, list), f"Chunk IDs invalidos: {run_id}")
        require(len(chunk_ids) == len(set(chunk_ids)), f"Chunks repetidos: {run_id}")
        require(
            all(isinstance(cid, str) and cid in chunks_by_id for cid in chunk_ids),
            f"Chunk ID nao encontrado no corpus: {run_id}",
        )
        if condition == "no_rag":
            require(len(chunk_ids) == 0, f"No-RAG tem contexto: {run_id}")
            messages = [
                {"role": "system", "content": NO_RAG_SYSTEM_PROMPT},
                {"role": "user", "content": query.question},
            ]
        else:
            require(len(chunk_ids) == top_k, f"Top-k inesperado: {run_id}")
            context_chunks = [
                ContextChunk(
                    chunk_id=cid,
                    content=chunks_by_id[cid]["content"],
                    source_path=chunks_by_id[cid]["source_path"],
                    section_title=chunks_by_id[cid]["section_title"],
                )
                for cid in chunk_ids
            ]
            prompt = build_rag_prompt(query=query.question, chunks=context_chunks)
            messages = [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ]

        records.append(
            {
                "run_id": run_id,
                "condition": condition,
                "query_id": query_id,
                "category": query.category,
                "question": query.question,
                "context_chunk_ids": chunk_ids,
                "messages": messages,
                "prompt_sha256": sha256(stable_json(messages)),
            }
        )

    expected_runs = {
        f"{condition}|{query.query_id}" for condition in conditions for query in queries
    }
    require(seen == expected_runs, "Matriz de 90 execucoes incompleta.")
    for condition in conditions:
        require(counts[condition] == len(queries), f"Condicao incompleta: {condition}")

    prompts_bytes = b"".join(stable_json(item) + b"\n" for item in records)
    prompts_sha = sha256(prompts_bytes)

    output_dir = PROJECT_ROOT / cfg["output"]["directory"]
    prompts_path = output_dir / cfg["output"]["frozen_prompts"]
    manifest_path = output_dir / "frozen_prompts.manifest.json"

    manifest = {
        "experiment_kind": "new_controlled_model_comparison",
        "historical_prompts_verified_identical": False,
        "historical_signature": historical_signature,
        "current_source_signature": current_signature,
        "git_head_at_freeze": git_head(),
        "baseline_csv_sha256_lf": baseline_hash,
        "historical_documented_sha256": baseline["expected_sha256"].lower(),
        "corpus_fingerprint": corpus_fingerprint,
        "corpus_chunks": len(corpus),
        "ground_truth_sha256": file_sha256(ground_truth_path),
        "original_config_sha256": file_sha256(ORIGINAL_CONFIG_PATH),
        "comparison_config_sha256": file_sha256(CONFIG_PATH),
        "prompt_source_sha256": file_sha256(PROMPT_PATH),
        "runner_source_sha256": file_sha256(RUNNER_PATH),
        "no_rag_prompt_source_sha256": file_sha256(RUNNER_PATH),
        "conditions": list(conditions),
        "condition_counts": dict(sorted(counts.items())),
        "rows": len(records),
        "baseline_model": baseline["model"],
        "candidate_model": candidate["model"],
        "provider": "ollama",
        "generation": dict(generation),
        "frozen_prompts_sha256": prompts_sha,
    }
    manifest_bytes = (
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        ).encode("utf-8")
        + b"\n"
    )

    # Verifica TODOS os caminhos antes da primeira escrita.
    for path, data in (
        (prompts_path, prompts_bytes),
        (manifest_path, manifest_bytes),
    ):
        if path.exists():
            require(
                path.is_file() and path.read_bytes() == data,
                f"Artefato existente divergente, sem sobrescrita: {path}",
            )

    status_prompts = write_immutable(prompts_path, prompts_bytes)
    status_manifest = write_immutable(manifest_path, manifest_bytes)

    print("CONGELAMENTO DO NOVO EXPERIMENTO APROVADO")
    print("Historico preservado e nao assumido identico.")
    print(f"Assinatura historica: {historical_signature}")
    print(f"Assinatura fonte atual: {current_signature}")
    print(f"Corpus: {len(corpus)} chunks; fingerprint: {corpus_fingerprint}")
    print(f"Prompts congelados: {len(records)}")
    print("Condicoes: " + ", ".join(f"{c}={counts[c]}" for c in conditions))
    print(f"SHA-256 prompts: {prompts_sha}")
    print(f"Prompts ({status_prompts}): {prompts_path}")
    print(f"Manifesto ({status_manifest}): {manifest_path}")
    print("Nenhuma geracao foi executada; nenhum resultado antigo foi alterado.")


if __name__ == "__main__":
    main()
