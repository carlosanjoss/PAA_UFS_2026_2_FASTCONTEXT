"""Executa uma comparacao controlada entre dois LLMs no FastContext.

Requer o congelamento aprovado em scripts.freeze_model_comparison_prompts.
Nao reexecuta retrieval, nao altera arquivos historicos e nao usa fallback de modelo.

Teste:    python -m experiments.run_rag_model_comparison --smoke
Completo: python -m experiments.run_rag_model_comparison
Retomada automatica a partir de arquivos individuais imutaveis em runs/.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import tempfile
from collections import Counter
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import requests
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
from src.rag.pipeline import RAGPipeline
from src.rag.prompt import ContextChunk, build_rag_prompt
from src.rag.providers.base import (
    GenerationConfig,
    LLMMessage,
    LLMResponse,
    LLMTruncatedResponseError,
)
from src.rag.providers.ollama import OllamaProvider
from src.semantic.persistence import calculate_corpus_fingerprint
from src.utils.config import PROJECT_ROOT

CONFIG_PATH = PROJECT_ROOT / "config" / "rag_model_comparison.yaml"
EXPECTED_FROZEN_PROMPTS_SHA256 = (
    "a8137120a9210faece92145b75bfc74415afd8fd0ab5ef711fd8b619ee22d2bd"
)
EXPECTED_SOURCE_SIGNATURE = (
    "87392a4cfc4185dc98af6c80b110b9c825d848bc2b2933b612f102be34efdde0"
)

# Nomes explicitamente separados. Os nomes do candidato seguem o YAML congelado.
BASELINE_OUTPUTS = {
    "csv": "generations_qwen25_3b.csv",
    "jsonl": "generations_qwen25_3b.jsonl",
    "checkpoint": "checkpoint_qwen25_3b.json",
}
COMBINED_OUTPUTS = {
    "csv": "generations_both_models.csv",
    "jsonl": "generations_both_models.jsonl",
    "checkpoint": "checkpoint_both_models.json",
}

CSV_COLUMNS = (
    "run_key",
    "model_role",
    "model",
    "provider",
    "run_id",
    "condition",
    "query_id",
    "category",
    "question",
    "prompt_sha256",
    "frozen_prompts_sha256",
    "source_signature",
    "corpus_fingerprint",
    "context_chunk_ids",
    "temperature",
    "max_tokens",
    "think",
    "seed",
    "max_truncation_retries",
    "max_retry_tokens",
    "citation_retries",
    "generation_time_ns",
    "generation_time_ms",
    "citation_valid",
    "citation_retry_count",
    "citation_count",
    "valid_citations",
    "invalid_citations",
    "answer",
    "generation_metadata",
    "created_at_utc",
    "record_sha256",
)


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(data, dict), f"Objeto JSON invalido: {path}")
    return data


def read_yaml(path: Path, root_key: str) -> dict[str, Any]:
    require(path.is_file(), f"YAML nao encontrado: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    require(isinstance(data, dict), f"YAML invalido: {path}")
    section = data.get(root_key)
    require(isinstance(section, dict), f"Secao YAML ausente: {root_key}")
    return section


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes())


def now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def atomic_write(path: Path, data: bytes) -> None:
    """Escreve apenas artefatos NOVOS desta extensao, via arquivo temporario."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=".partial_",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary_name = stream.name
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def load_inputs() -> tuple[
    dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]
]:
    """Valida a proveniencia e reconstrui contextos somente para checagem."""
    cfg = read_yaml(CONFIG_PATH, "model_comparison")
    original = read_yaml(ORIGINAL_CONFIG_PATH, "rag_experiment")
    output_dir = PROJECT_ROOT / cfg["output"]["directory"]
    prompts_path = output_dir / cfg["output"]["frozen_prompts"]
    manifest_path = output_dir / "frozen_prompts.manifest.json"
    require(prompts_path.is_file(), "Prompts nao congelados. Execute a Etapa 2.")
    require(manifest_path.is_file(), "Manifesto do congelamento ausente.")
    manifest = read_json(manifest_path)
    require(
        manifest.get("frozen_prompts_sha256") == EXPECTED_FROZEN_PROMPTS_SHA256,
        "SHA dos prompts nao corresponde ao congelamento aprovado pelo usuario.",
    )
    require(
        manifest.get("current_source_signature") == EXPECTED_SOURCE_SIGNATURE,
        "Assinatura nao corresponde a fonte validada do novo experimento.",
    )

    require(
        file_sha256(prompts_path) == manifest["frozen_prompts_sha256"],
        "Os prompts congelados foram modificados.",
    )
    require(
        file_sha256(CONFIG_PATH) == manifest["comparison_config_sha256"],
        "O YAML de comparacao mudou depois do congelamento.",
    )
    require(
        file_sha256(ORIGINAL_CONFIG_PATH) == manifest["original_config_sha256"],
        "A configuracao do experimento original mudou apos o congelamento.",
    )
    require(
        file_sha256(PROMPT_PATH) == manifest["prompt_source_sha256"],
        "O construtor de prompts mudou apos o congelamento.",
    )
    require(
        file_sha256(RUNNER_PATH) == manifest["runner_source_sha256"],
        "O executor original mudou apos o congelamento.",
    )

    conditions = tuple(cfg["conditions"])
    require(conditions == tuple(original["conditions"]), "Condicoes modificadas.")
    require(cfg["generation"] == manifest["generation"], "Geracao modificada.")
    require(
        cfg["candidate"]["model"] == manifest["candidate_model"], "Candidato mudou."
    )
    require(cfg["baseline"]["model"] == manifest["baseline_model"], "Baseline mudou.")
    require(
        cfg["baseline"]["provider"] == cfg["candidate"]["provider"] == "ollama",
        "O protocolo requer provider Ollama em ambas as condicoes.",
    )
    require(
        cfg["baseline"]["model"] != cfg["candidate"]["model"],
        "Os dois modelos devem ser distintos.",
    )
    for field in cfg["generation"]:
        require(
            cfg["generation"][field] == original["generation"][field],
            f"Parametro original divergente: {field}",
        )
    require(
        int(cfg["retrieval"]["top_k"]) == int(original["retrieval"]["top_k"]),
        "Top-k modificado.",
    )
    require(
        int(cfg["checkpoint"]["every_completed_runs"])
        == int(original["checkpoint"]["every_completed_runs"])
        and int(cfg["checkpoint"]["every_completed_runs"]) > 0,
        "Periodicidade do checkpoint divergente.",
    )
    require(
        bool(cfg["warmup"]["enabled"]) == bool(original["warmup"]["enabled"]),
        "Habilitacao de aquecimento divergente.",
    )

    corpus_path = PROJECT_ROOT / cfg["corpus"]["chunks_file"]
    ground_truth_path = PROJECT_ROOT / cfg["queries"]["ground_truth_file"]
    corpus = load_chunks_jsonl(corpus_path)
    fingerprint = calculate_corpus_fingerprint(corpus)
    require(
        fingerprint == manifest["corpus_fingerprint"], "Corpus diferente do congelado."
    )
    require(
        fingerprint == cfg["corpus"]["expected_fingerprint"]
        and len(corpus)
        == manifest["corpus_chunks"]
        == cfg["corpus"]["expected_chunks"],
        "Corpus nao corresponde ao protocolo.",
    )
    require(
        file_sha256(ground_truth_path) == manifest["ground_truth_sha256"],
        "O ground truth mudou apos o congelamento.",
    )
    by_id = {chunk["chunk_id"]: chunk for chunk in corpus}
    require(len(by_id) == len(corpus), "IDs duplicados no corpus.")
    queries = _load_ground_truth(
        ground_truth_path,
        expected_count=int(cfg["queries"]["expected_count"]),
        valid_chunk_ids=set(by_id),
    )
    query_by_id = {q.query_id: q for q in queries}

    generation = cfg["generation"]
    source_signature = _calculate_signature(
        config_path=ORIGINAL_CONFIG_PATH,
        ground_truth_path=ground_truth_path,
        prompt_path=PROMPT_PATH,
        pipeline_path=PIPELINE_PATH,
        runner_path=RUNNER_PATH,
        corpus_fingerprint=fingerprint,
        query_ids=tuple(q.query_id for q in queries),
        conditions=conditions,
        top_k=int(cfg["retrieval"]["top_k"]),
        expected_provider=cfg["baseline"]["provider"],
        expected_model=cfg["baseline"]["model"],
        temperature=float(generation["temperature"]),
        max_tokens=int(generation["max_tokens"]),
        think=bool(generation["think"]),
        generation_seed=int(generation["seed"]),
        smoke=False,
    )
    require(
        source_signature == manifest["current_source_signature"],
        "O codigo participante da assinatura mudou depois do congelamento.",
    )

    frozen: list[dict[str, Any]] = []
    for line_num, line in enumerate(
        prompts_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        require(bool(line.strip()), f"Linha vazia no congelamento: {line_num}")
        value = json.loads(line)
        require(isinstance(value, dict), f"Prompt invalido: linha {line_num}")
        frozen.append(value)
    require(len(frozen) == manifest["rows"] == 90, "Congelamento incompleto.")
    require(
        manifest["conditions"] == list(conditions),
        "Manifesto das condicoes divergente.",
    )
    counts: Counter[str] = Counter()
    seen: set[str] = set()
    for item in frozen:
        condition, query_id = item["condition"], item["query_id"]
        run_id = f"{condition}|{query_id}"
        require(
            condition in conditions and item["run_id"] == run_id and run_id not in seen,
            f"Run ID invalido ou repetido: {run_id}",
        )
        seen.add(run_id)
        counts[condition] += 1
        require(query_id in query_by_id, f"Pergunta desconhecida: {query_id}")
        query = query_by_id[query_id]
        require(
            item["question"] == query.question and item["category"] == query.category,
            f"Pergunta ou categoria mudou: {run_id}",
        )
        chunks = item["context_chunk_ids"]
        require(isinstance(chunks, list), f"Contexto invalido: {run_id}")
        require(
            len(chunks) == len(set(chunks)) and all(cid in by_id for cid in chunks),
            f"Chunks invalidos: {run_id}",
        )
        if condition == "no_rag":
            require(not chunks, f"No-RAG com chunks: {run_id}")
            expected = [
                {"role": "system", "content": NO_RAG_SYSTEM_PROMPT},
                {"role": "user", "content": query.question},
            ]
        else:
            require(len(chunks) == cfg["retrieval"]["top_k"], f"Top-k: {run_id}")
            context = [
                ContextChunk(
                    chunk_id=cid,
                    content=by_id[cid]["content"],
                    source_path=by_id[cid]["source_path"],
                    section_title=by_id[cid]["section_title"],
                )
                for cid in chunks
            ]
            prompt = build_rag_prompt(query=query.question, chunks=context)
            expected = [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ]
        require(
            expected == item["messages"], f"Prompt nao reproduz congelamento: {run_id}"
        )
        require(
            sha256(stable_json(expected)) == item["prompt_sha256"],
            f"Hash do prompt divergente: {run_id}",
        )
    require(
        len(seen) == len(queries) * len(conditions)
        and all(counts[c] == len(queries) for c in conditions),
        "A matriz congelada de 90 entradas nao esta completa.",
    )
    return cfg, manifest, frozen, by_id


def make_generation_config(cfg: dict[str, Any]) -> GenerationConfig:
    g = cfg["generation"]
    return GenerationConfig(
        temperature=float(g["temperature"]),
        max_tokens=int(g["max_tokens"]),
        think=bool(g["think"]),
        seed=int(g["seed"]),
    )


def check_response(
    model: str, provider: str, actual_model: str, actual_provider: str
) -> None:
    require(
        actual_model == model and actual_provider == provider,
        f"Modelo ou provedor inesperado: {actual_model}/{actual_provider} "
        f"(esperado {model}/{provider}).",
    )


def generate_no_rag_with_retry(
    provider: OllamaProvider,
    messages: list[LLMMessage],
    generation_config: GenerationConfig,
    *,
    max_truncation_retries: int,
    max_retry_tokens: int,
) -> LLMResponse:
    """Reaplica ao No-RAG a politica de truncamento do RAGPipeline original.

    Preserva o prompt e a seed. Somente max_tokens aumenta quando
    a resposta anterior termina com done_reason='length'.
    """
    require(max_truncation_retries >= 0, "Retentativas negativas.")
    require(max_retry_tokens > 0, "Limite de tokens invalido.")

    current_config = generation_config
    retries = 0

    while True:
        response = provider.generate(messages=messages, config=current_config)
        metadata = dict(response.metadata or {})

        if metadata.get("done_reason") != "length":
            # Para respostas sem retentativa, preserva o formato dos
            # metadados No-RAG ja persistidos antes desta correcao.
            if retries:
                metadata["truncation_retries"] = retries
                metadata["effective_max_tokens"] = current_config.max_tokens
                return LLMResponse(
                    text=response.text,
                    provider=response.provider,
                    model=response.model,
                    metadata=metadata,
                )
            return response

        if retries >= max_truncation_retries:
            raise LLMTruncatedResponseError(
                "Resposta No-RAG permaneceu truncada apos "
                f"{retries} retentativas (limite {current_config.max_tokens} tokens)."
            )

        next_tokens = min(current_config.max_tokens * 2, max_retry_tokens)
        if next_tokens <= current_config.max_tokens:
            raise LLMTruncatedResponseError(
                "Resposta No-RAG atingiu o limite de tokens para retentativas."
            )

        retries += 1
        current_config = replace(current_config, max_tokens=next_tokens)


def execute_one(
    item: dict[str, Any],
    role: str,
    model: str,
    provider: OllamaProvider,
    cfg: dict[str, Any],
    manifest: dict[str, Any],
    by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    g = cfg["generation"]
    generation_config = make_generation_config(cfg)
    condition = item["condition"]
    metadata: dict[str, Any]
    if condition == "no_rag":
        messages = [LLMMessage(**message) for message in item["messages"]]
        start = perf_counter_ns()
        response = generate_no_rag_with_retry(
            provider=provider,
            messages=messages,
            generation_config=generation_config,
            max_truncation_retries=int(g["max_truncation_retries"]),
            max_retry_tokens=int(g["max_retry_tokens"]),
        )
        duration_ns = perf_counter_ns() - start
        check_response(model, "ollama", response.model, response.provider)
        metadata = dict(response.metadata or {})
        answer = response.text
        citation_valid = None
        citation_retry_count = None
        citation_count = None
        valid_citations: list[str] = []
        invalid_citations: list[str] = []
    else:
        context = [
            ContextChunk(
                chunk_id=cid,
                content=by_id[cid]["content"],
                source_path=by_id[cid]["source_path"],
                section_title=by_id[cid]["section_title"],
            )
            for cid in item["context_chunk_ids"]
        ]
        # answer() reutiliza o prompt, a validacao e a politica de retentativas
        # do FastContext original. A igualdade do prompt foi conferida em load_inputs().
        pipeline = RAGPipeline(
            provider=provider,
            max_truncation_retries=int(g["max_truncation_retries"]),
            max_retry_tokens=int(g["max_retry_tokens"]),
            max_citation_retries=int(g["citation_retries"]),
        )
        result = pipeline.answer(
            query=item["question"],
            chunks=context,
            generation_config=generation_config,
        )
        check_response(model, "ollama", result.model, result.provider)
        require(
            [chunk.chunk_id for chunk in result.context_chunks]
            == item["context_chunk_ids"],
            f"Contexto alterado: {item['run_id']}",
        )
        duration_ns = result.generation_time_ns
        answer = result.answer
        metadata = dict(result.metadata or {})
        citation_valid = result.citation_valid
        citation_retry_count = result.citation_retry_count
        citation_count = result.citation_count
        valid_citations = list(result.valid_citations)
        invalid_citations = list(result.invalid_citations)

    record: dict[str, Any] = {
        "run_key": f"{role}|{item['run_id']}",
        "model_role": role,
        "model": model,
        "provider": "ollama",
        "run_id": item["run_id"],
        "condition": condition,
        "query_id": item["query_id"],
        "category": item["category"],
        "question": item["question"],
        "prompt_sha256": item["prompt_sha256"],
        "frozen_prompts_sha256": manifest["frozen_prompts_sha256"],
        "source_signature": manifest["current_source_signature"],
        "corpus_fingerprint": manifest["corpus_fingerprint"],
        "context_chunk_ids": list(item["context_chunk_ids"]),
        "temperature": generation_config.temperature,
        "max_tokens": generation_config.max_tokens,
        "think": generation_config.think,
        "seed": generation_config.seed,
        "max_truncation_retries": int(g["max_truncation_retries"]),
        "max_retry_tokens": int(g["max_retry_tokens"]),
        "citation_retries": int(g["citation_retries"]),
        "generation_time_ns": duration_ns,
        "generation_time_ms": round(duration_ns / 1_000_000, 6),
        "citation_valid": citation_valid,
        "citation_retry_count": citation_retry_count,
        "citation_count": citation_count,
        "valid_citations": valid_citations,
        "invalid_citations": invalid_citations,
        "answer": answer,
        "generation_metadata": metadata,
        "created_at_utc": now_utc(),
    }
    record["record_sha256"] = sha256(stable_json(record))
    return record


def validate_record(
    record: dict[str, Any],
    item: dict[str, Any],
    role: str,
    model: str,
    cfg: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    integrity = record.get("record_sha256")
    without_hash = {
        key: value for key, value in record.items() if key != "record_sha256"
    }
    require(
        integrity == sha256(stable_json(without_hash)),
        f"Checksum invalido: {role}|{item['run_id']}",
    )
    expected = {
        "run_key": f"{role}|{item['run_id']}",
        "model_role": role,
        "model": model,
        "provider": "ollama",
        "run_id": item["run_id"],
        "condition": item["condition"],
        "query_id": item["query_id"],
        "category": item["category"],
        "question": item["question"],
        "prompt_sha256": item["prompt_sha256"],
        "frozen_prompts_sha256": manifest["frozen_prompts_sha256"],
        "source_signature": manifest["current_source_signature"],
        "corpus_fingerprint": manifest["corpus_fingerprint"],
        "context_chunk_ids": item["context_chunk_ids"],
        "temperature": float(cfg["generation"]["temperature"]),
        "max_tokens": int(cfg["generation"]["max_tokens"]),
        "think": bool(cfg["generation"]["think"]),
        "seed": int(cfg["generation"]["seed"]),
        "max_truncation_retries": int(cfg["generation"]["max_truncation_retries"]),
        "max_retry_tokens": int(cfg["generation"]["max_retry_tokens"]),
        "citation_retries": int(cfg["generation"]["citation_retries"]),
    }
    for field, value in expected.items():
        require(
            record.get(field) == value,
            f"Registro incoerente {field}: {record['run_key']}",
        )
    require(
        isinstance(record.get("answer"), str)
        and isinstance(record.get("generation_time_ns"), int)
        and record["generation_time_ns"] >= 0,
        f"Resposta ou tempo invalidos: {record['run_key']}",
    )
    require(
        record.get("citation_valid") is None
        if item["condition"] == "no_rag"
        else isinstance(record.get("citation_valid"), bool),
        f"Estado de citacao invalido: {record['run_key']}",
    )


def save_run(path: Path, record: dict[str, Any]) -> None:
    require(not path.exists(), f"Registro ja existe, recusei sobrescrever: {path}")
    atomic_write(
        path,
        json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        ).encode("utf-8")
        + b"\n",
    )


def write_outputs(
    out: Path,
    cfg: dict[str, Any],
    manifest: dict[str, Any],
    frozen: list[dict[str, Any]],
    smoke: bool,
) -> tuple[int, int]:
    """Reconstroi planilhas/checkpoints somente a partir dos runs validados."""
    roles = {
        "baseline": cfg["baseline"]["model"],
        "candidate": cfg["candidate"]["model"],
    }
    selected = frozen
    if smoke:
        first_id = next(iter(dict.fromkeys(x["query_id"] for x in frozen)))
        selected = [x for x in frozen if x["query_id"] == first_id]
    expected_files = {
        (out / "runs" / role / item["condition"] / f"{item['query_id']}.json").resolve()
        for item in selected
        for role in roles
    }
    runs_root = out / "runs"
    if runs_root.exists():
        actual_files = {p.resolve() for p in runs_root.rglob("*.json")}
        require(
            actual_files.issubset(expected_files),
            "Ha registros inesperados no diretorio runs/. Interrompendo.",
        )

    collected: dict[str, list[dict[str, Any]]] = {role: [] for role in roles}
    for item in selected:
        for role, model in roles.items():
            path = out / "runs" / role / item["condition"] / f"{item['query_id']}.json"
            if path.is_file():
                record = read_json(path)
                validate_record(record, item, role, model, cfg, manifest)
                collected[role].append(record)

    candidate_names = {
        "csv": cfg["output"]["generations_csv"],
        "jsonl": cfg["output"]["generations_jsonl"],
        "checkpoint": cfg["output"]["checkpoint"],
    }

    def emit(names: dict[str, str], rows: list[dict[str, Any]], expected: int) -> None:
        serialized = b"".join(stable_json(row) + b"\n" for row in rows)
        atomic_write(out / names["jsonl"], serialized)
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            converted = {
                field: json.dumps(row[field], ensure_ascii=False, sort_keys=True)
                if isinstance(row.get(field), (list, dict))
                else row.get(field)
                for field in CSV_COLUMNS
            }
            writer.writerow(converted)
        atomic_write(out / names["csv"], stream.getvalue().encode("utf-8"))
        checkpoint = {
            "mode": "smoke" if smoke else "full",
            "frozen_prompts_sha256": manifest["frozen_prompts_sha256"],
            "source_signature": manifest["current_source_signature"],
            "expected_rows": expected,
            "completed_rows": len(rows),
            "completed_run_keys": [r["run_key"] for r in rows],
            "status": "complete" if len(rows) == expected else "in_progress",
            "last_updated_utc": now_utc(),
        }
        atomic_write(
            out / names["checkpoint"],
            json.dumps(checkpoint, ensure_ascii=False, indent=2).encode("utf-8")
            + b"\n",
        )

    emit(BASELINE_OUTPUTS, collected["baseline"], len(selected))
    emit(candidate_names, collected["candidate"], len(selected))
    combined = []
    lookup = {r["run_key"]: r for records in collected.values() for r in records}
    for item in selected:
        for role in roles:
            key = f"{role}|{item['run_id']}"
            if key in lookup:
                combined.append(lookup[key])
    emit(COMBINED_OUTPUTS, combined, len(selected) * len(roles))
    return len(combined), len(selected) * len(roles)


def verify_model_provenance(out: Path, base_url: str, roles: dict[str, str]) -> None:
    """Registra os digests reais para evitar mudanca entre retomadas."""
    endpoint = base_url.rstrip("/")
    try:
        tags_response = requests.get(f"{endpoint}/api/tags", timeout=(5, 15))
        tags_response.raise_for_status()
        version_response = requests.get(f"{endpoint}/api/version", timeout=(5, 15))
        version_response.raise_for_status()
        tags = tags_response.json()
        version = version_response.json()
    except (requests.RequestException, ValueError) as exc:
        raise RuntimeError(
            "Nao foi possivel registrar a versao/modelos do Ollama."
        ) from exc

    installed = tags.get("models")
    require(isinstance(installed, list), "Resposta /api/tags invalida.")
    models: dict[str, dict[str, Any]] = {}
    for role, model in roles.items():
        matches = [
            entry
            for entry in installed
            if isinstance(entry, dict)
            and (entry.get("name") == model or entry.get("model") == model)
        ]
        require(len(matches) == 1, f"Modelo ausente ou ambiguo no Ollama: {model}")
        entry = matches[0]
        digest = entry.get("digest")
        require(isinstance(digest, str) and digest, f"Digest nao informado: {model}")
        models[role] = {
            "name": model,
            "digest": digest,
            "size_bytes": entry.get("size"),
            "details": entry.get("details"),
        }
    data = {
        "models": models,
        "ollama_version": version.get("version"),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "base_url": endpoint,
    }
    path = out / "model_provenance.json"
    raw = (
        json.dumps(data, sort_keys=True, ensure_ascii=False, indent=2).encode("utf-8")
        + b"\n"
    )
    if path.exists():
        require(
            path.read_bytes() == raw,
            "Versao do Ollama, ambiente ou digest de modelo mudou desde a primeira "
            "execucao deste experimento. Confira model_provenance.json antes de "
            "continuar.",
        )
    else:
        atomic_write(path, raw)
    print("Modelos e ambiente registrados; digests fixados para retomadas.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Executa uma pergunta nas tres condicoes para ambos os modelos "
            "(6 geracoes)."
        ),
    )
    args = parser.parse_args()

    cfg, manifest, frozen, by_id = load_inputs()
    selected = frozen
    if args.smoke:
        first_id = next(iter(dict.fromkeys(x["query_id"] for x in frozen)))
        selected = [x for x in frozen if x["query_id"] == first_id]
        require(len(selected) == 3, "Smoke requer tres condicoes da mesma pergunta.")

    output_root = PROJECT_ROOT / cfg["output"]["directory"]
    output_dir = output_root / "smoke" if args.smoke else output_root
    require(
        output_root.resolve()
        != (PROJECT_ROOT / "experiments" / "rag_results").resolve(),
        "Saida aponta para o diretorio do baseline historico.",
    )
    for key in ("generations_csv", "generations_jsonl", "checkpoint"):
        name = cfg["output"][key]
        require(
            isinstance(name, str)
            and Path(name).name == name
            and name not in (".", ".."),
            f"Nome de arquivo de saida invalido: {key}",
        )
    roles = {
        "baseline": cfg["baseline"]["model"],
        "candidate": cfg["candidate"]["model"],
    }

    print("=" * 68)
    print("FASTCONTEXT - COMPARACAO CONTROLADA DE MODELOS")
    print("=" * 68)
    print("Modo:", "smoke (6 geracoes)" if args.smoke else "final (180 geracoes)")
    print("Modelos:", roles)
    print("Prompts congelados (SHA-256):", manifest["frozen_prompts_sha256"])
    print("Assinatura do novo experimento:", manifest["current_source_signature"])
    print("Saidas:", output_dir)
    print("OBS: nao executa recuperacao; tempo = somente geracao.")

    # Checa integridade de TODOS os runs existentes antes de se conectar aos modelos.
    completed, total = write_outputs(output_dir, cfg, manifest, frozen, args.smoke)
    print(f"Retomada validada: {completed}/{total} respostas ja salvas.")
    if completed == total:
        print(
            "EXPERIMENTO JA COMPLETO - nada foi sobrescrito nos registros individuais."
        )
        return 0

    providers: dict[str, OllamaProvider] = {}
    for role, model in roles.items():
        provider = OllamaProvider(model=model, base_url=cfg["candidate"]["base_url"])
        require(
            provider.is_available(),
            f"Modelo {model!r} nao instalado ou Ollama inacessivel. "
            f"Execute: ollama pull {model}",
        )
        providers[role] = provider

    verify_model_provenance(output_dir, cfg["candidate"]["base_url"], roles)

    if bool(cfg["warmup"]["enabled"]):
        print("Aquecimento fora das medicoes, um por modelo...")
        gen_cfg = make_generation_config(cfg)
        for role, provider in providers.items():
            response = provider.generate(
                messages=[
                    LLMMessage(role="system", content="Reply directly and concisely."),
                    LLMMessage(role="user", content="Reply only with OK."),
                ],
                config=gen_cfg,
            )
            check_response(roles[role], "ollama", response.model, response.provider)
        print("Aquecimento concluido.")

    checkpoint_every = int(cfg["checkpoint"]["every_completed_runs"])
    saved_after_resume = completed
    for index, item in enumerate(selected, 1):
        for role, model in roles.items():
            path = (
                output_dir
                / "runs"
                / role
                / item["condition"]
                / f"{item['query_id']}.json"
            )
            if path.exists():
                continue
            print(
                f"[{index}/{len(selected)}] {role} ({model}) | {item['run_id']}",
                flush=True,
            )
            record = execute_one(
                item, role, model, providers[role], cfg, manifest, by_id
            )
            validate_record(record, item, role, model, cfg, manifest)
            save_run(path, record)
            saved_after_resume += 1
            if (
                saved_after_resume % checkpoint_every == 0
                or saved_after_resume == total
            ):
                completed, total = write_outputs(
                    output_dir,
                    cfg,
                    manifest,
                    frozen,
                    args.smoke,
                )
                print(f"  Checkpoint: {completed}/{total}", flush=True)
            else:
                print(
                    f"  Registro individual salvo: {saved_after_resume}/{total} "
                    f"(checkpoint a cada {checkpoint_every})",
                    flush=True,
                )

    completed, total = write_outputs(output_dir, cfg, manifest, frozen, args.smoke)
    require(completed == total, f"Experimento incompleto: {completed}/{total}.")
    print("=" * 68)
    print(f"EXPERIMENTO CONCLUIDO: {completed}/{total}")
    print("Resultados individuais, CSVs, JSONLs e checkpoints verificados.")
    print("Baseline historico nao foi alterado.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
