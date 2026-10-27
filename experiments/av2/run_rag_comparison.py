
"""Generate paired AV2 RAG answers from frozen retrieval Top-5 results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

from src.ingestion.chunk_loader import load_chunks_jsonl
from src.rag.pipeline import RAGPipeline
from src.rag.prompt import ContextChunk, build_context
from src.rag.providers.base import GenerationConfig
from src.rag.providers.ollama import OllamaProvider
from src.rag.settings import load_rag_settings
from src.semantic.persistence import calculate_corpus_fingerprint
from src.utils.config import PROJECT_ROOT

CONDITIONS = ("semantic", "literal", "hybrid")

FINGERPRINT = (
    "e08b47b55d7237f38bcaba48c79a468a40b92e4f3120e968ae9ece999f5e8c2d"
)

INPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "av2"
    / "retrieval_comparison"
    / "final"
)

OUTPUT_ROOT = (
    PROJECT_ROOT / "experiments" / "av2" / "rag_comparison"
)

GROUND_TRUTH = (
    PROJECT_ROOT / "data" / "ground_truth" / "ground_truth.final.json"
)

CORPUS = PROJECT_ROOT / "data" / "chunks" / "chunks.jsonl"

RAG_CONFIG = PROJECT_ROOT / "config" / "rag_experiments.yaml"

PROMPT_SOURCE = PROJECT_ROOT / "src" / "rag" / "prompt.py"

PIPELINE_SOURCE = PROJECT_ROOT / "src" / "rag" / "pipeline.py"

COLUMNS = (
    "signature",
    "query_id",
    "category",
    "condition",
    "question",
    "model",
    "provider",
    "top_k",
    "retrieved_chunk_ids",
    "context_sha256",
    "retrieval_mean_time_ms_reference",
    "generation_time_ms",
    "citation_valid",
    "citation_count",
    "citation_retry_count",
    "valid_citations",
    "invalid_citations",
    "known_relevant_citations",
    "answer",
    "retrieved_context",
    "generation_metadata",
)


def hash_file(path: Path) -> str:
    """Hash one experiment input without loading it into memory."""

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def experiment_signature(
    files: tuple[Path, ...],
) -> str:
    """Bind resumed rows to the exact same source artifacts."""

    digest = hashlib.sha256()

    for path in files:
        digest.update(
            path.relative_to(PROJECT_ROOT).as_posix().encode()
        )
        digest.update(hash_file(path).encode())

    return digest.hexdigest()


def load_inputs() -> tuple[
    list[dict[str, str]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, Any],
    str,
]:
    """Load frozen rankings, questions, corpus and AV1 RAG settings."""

    input_manifest = json.loads(
        (INPUT_DIR / "manifest.json").read_text(
            encoding="utf-8"
        )
    )

    if (
        input_manifest.get("corpus_fingerprint") != FINGERPRINT
        or input_manifest.get("top_k") != 5
        or input_manifest.get("query_count") != 30
    ):
        raise ValueError(
            "Incompatible frozen retrieval comparison manifest."
        )

    with (INPUT_DIR / "per_query.csv").open(
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        required = {
            "query_id",
            "condition",
            "retrieved_chunk_ids",
            "repetitions",
            "mean_time_ms",
        }

        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                "Missing required retrieval comparison columns."
            )

        rankings = list(reader)

    if len(rankings) != 90:
        raise ValueError(
            "Expected exactly 90 frozen retrieval rankings."
        )

    questions_doc = json.loads(
        GROUND_TRUTH.read_text(encoding="utf-8")
    )

    if (
        questions_doc["metadata"]["corpus_fingerprint"]
        != FINGERPRINT
    ):
        raise ValueError("Incompatible ground truth.")

    questions = {
        q["query_id"]: q
        for q in questions_doc["queries"]
    }

    if len(questions) != 30:
        raise ValueError(
            "Expected exactly 30 ground-truth questions."
        )

    chunks = load_chunks_jsonl(CORPUS)

    if (
        len(chunks) != 1305
        or calculate_corpus_fingerprint(chunks) != FINGERPRINT
    ):
        raise ValueError(
            "The frozen AV1 corpus has changed."
        )

    corpus_by_id = {
        str(c["chunk_id"]): c
        for c in chunks
    }

    if len(corpus_by_id) != len(chunks):
        raise ValueError(
            "Duplicate corpus chunk IDs."
        )

    keys: set[tuple[str, str]] = set()

    for row in rankings:
        query_id = row["query_id"]
        condition = row["condition"]

        if (
            query_id not in questions
            or condition not in CONDITIONS
        ):
            raise ValueError(
                "Invalid condition or question in frozen rankings."
            )

        key = (query_id, condition)

        if key in keys:
            raise ValueError(
                "Duplicate query/condition ranking."
            )

        keys.add(key)

        if int(row["repetitions"]) != 5:
            raise ValueError(
                "Frozen rankings must have five repetitions."
            )

        ids = json.loads(row["retrieved_chunk_ids"])

        if (
            not isinstance(ids, list)
            or len(ids) > 5
            or len(ids) != len(set(ids))
            or any(
                not isinstance(x, str)
                or x not in corpus_by_id
                for x in ids
            )
        ):
            raise ValueError(
                f"Invalid Top-5 IDs for {key}."
            )

    if len(keys) != 90:
        raise ValueError(
            "Missing query/condition rankings."
        )

    config = yaml.safe_load(
        RAG_CONFIG.read_text(encoding="utf-8")
    )["rag_experiment"]

    generation = config["generation"]

    if (
        config["retrieval"]["top_k"] != 5
        or generation["provider"] != "ollama"
        or generation["model"] != "qwen2.5:3b"
        or generation["temperature"] != 0.0
        or generation["seed"] != 42
    ):
        raise ValueError(
            "AV1 RAG generation configuration differs "
            "from the frozen protocol."
        )

    if (
        int(generation["max_truncation_retries"]) > 0
        and int(generation["max_retry_tokens"])
        <= int(generation["max_tokens"])
    ):
        raise ValueError(
            "max_retry_tokens must exceed max_tokens; "
            "otherwise a truncated generation cannot be "
            "retried with a larger token budget."
        )

    signature = experiment_signature(
        (
            INPUT_DIR / "per_query.csv",
            INPUT_DIR / "manifest.json",
            GROUND_TRUTH,
            CORPUS,
            RAG_CONFIG,
            PROMPT_SOURCE,
            PIPELINE_SOURCE,
            Path(__file__).resolve(),
        )
    )

    return (
        rankings,
        questions,
        corpus_by_id,
        generation,
        signature,
    )


def context_for(
    ids: list[str],
    corpus_by_id: dict[str, dict[str, Any]],
) -> tuple[ContextChunk, ...]:
    """Build RAG context in exactly the frozen ranking order."""

    return tuple(
        ContextChunk(
            chunk_id=chunk_id,
            content=str(
                corpus_by_id[chunk_id]["content"]
            ),
            source_path=str(
                corpus_by_id[chunk_id]["source_path"]
            ),
            section_title=str(
                corpus_by_id[chunk_id]["section_title"]
            ),
        )
        for chunk_id in ids
    )


def read_existing(
    path: Path,
    signature: str,
) -> dict[tuple[str, str], dict[str, str]]:
    """Validate an interrupted CSV before continuing generation."""

    if not path.is_file():
        raise FileNotFoundError(
            "No generations.csv exists for --resume."
        )

    with path.open(
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError(
                "Existing generations.csv has a different schema."
            )

        rows = list(reader)

    existing: dict[
        tuple[str, str],
        dict[str, str],
    ] = {}

    for row in rows:
        if row["signature"] != signature:
            raise ValueError(
                "Experiment inputs changed; refusing to resume."
            )

        key = (
            row["query_id"],
            row["condition"],
        )

        if key in existing:
            raise ValueError(
                "Duplicate completed generation row."
            )

        existing[key] = row

    return existing


def save_rows(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    """Atomically preserve progress after each completed generation."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(".pending")

    with temporary.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=COLUMNS,
        )

        writer.writeheader()
        writer.writerows(rows)

        handle.flush()
        os.fsync(handle.fileno())

    temporary.replace(path)


def main() -> int:
    """Generate the complete paired AV2 RAG dataset."""

    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    parser.add_argument(
        "--no-warmup",
        action="store_true",
    )

    args = parser.parse_args()

    output = OUTPUT_ROOT / (
        "smoke" if args.smoke else "final"
    )

    csv_path = output / "generations.csv"
    manifest_path = output / "manifest.json"

    (
        rankings,
        questions,
        corpus_by_id,
        generation,
        signature,
    ) = load_inputs()

    by_key = {
        (r["query_id"], r["condition"]): r
        for r in rankings
    }

    query_ids = sorted(questions)

    if args.smoke:
        query_ids = query_ids[:1]

    if args.resume:
        completed = read_existing(
            csv_path,
            signature,
        )

        allowed_keys = {
            (q, c)
            for q in query_ids
            for c in CONDITIONS
        }

        if not set(completed) <= allowed_keys:
            raise ValueError(
                "Existing rows do not belong to this experiment."
            )

    else:
        if csv_path.exists() or manifest_path.exists():
            raise FileExistsError(
                f"Output already exists at {output}. "
                "Use --resume to continue."
            )

        completed = {}

    settings = load_rag_settings()

    if settings.ollama.model != generation["model"]:
        raise ValueError(
            f"OLLAMA_MODEL must be {generation['model']!r}; "
            f"configured {settings.ollama.model!r}."
        )

    provider = OllamaProvider(
        model=str(generation["model"]),
        base_url=settings.ollama.base_url,
    )

    if not provider.is_available():
        raise RuntimeError(
            "Ollama is unavailable or qwen2.5:3b "
            "is not installed."
        )

    pipeline = RAGPipeline(
        provider=provider,
        max_truncation_retries=int(
            generation["max_truncation_retries"]
        ),
        max_retry_tokens=int(
            generation["max_retry_tokens"]
        ),
        max_citation_retries=int(
            generation["citation_retries"]
        ),
    )

    config = GenerationConfig(
        temperature=float(
            generation["temperature"]
        ),
        max_tokens=int(
            generation["max_tokens"]
        ),
        think=bool(
            generation["think"]
        ),
        seed=int(
            generation["seed"]
        ),
    )

    if (
        not args.smoke
        and not args.resume
        and not args.no_warmup
    ):
        first = by_key[
            (query_ids[0], "semantic")
        ]

        ids = json.loads(
            first["retrieved_chunk_ids"]
        )

        print(
            "Warming up Ollama (not recorded)...",
            flush=True,
        )

        pipeline.answer(
            questions[query_ids[0]]["question"],
            context_for(
                ids,
                corpus_by_id,
            ),
            config,
        )

    target = (
        len(query_ids) * len(CONDITIONS)
    )

    print(
        f"AV2 RAG generation: "
        f"{len(completed)}/{target} completed",
        flush=True,
    )

    for index, query_id in enumerate(query_ids):
        question = questions[query_id]

        order = (
            CONDITIONS[index % 3:]
            + CONDITIONS[:index % 3]
        )

        for condition in order:
            key = (query_id, condition)

            if key in completed:
                continue

            row = by_key[key]

            ids: list[str] = json.loads(
                row["retrieved_chunk_ids"]
            )

            context = context_for(
                ids,
                corpus_by_id,
            )

            context_text = build_context(context)

            print(
                f"Generating {query_id}/{condition}...",
                flush=True,
            )

            rag_result = pipeline.answer(
                question["question"],
                context,
                config,
            )

            relevant = set(
                question["relevant_chunks"]
            )

            record: dict[str, Any] = {
                "signature": signature,
                "query_id": query_id,
                "category": question["category"],
                "condition": condition,
                "question": question["question"],
                "model": rag_result.model,
                "provider": rag_result.provider,
                "top_k": 5,
                "retrieved_chunk_ids": json.dumps(
                    ids,
                    ensure_ascii=False,
                ),
                "context_sha256": hashlib.sha256(
                    context_text.encode()
                ).hexdigest(),
                "retrieval_mean_time_ms_reference": (
                    row["mean_time_ms"]
                ),
                "generation_time_ms": (
                    rag_result.generation_time_ns
                    / 1_000_000
                ),
                "citation_valid": (
                    rag_result.citation_valid
                ),
                "citation_count": (
                    rag_result.citation_count
                ),
                "citation_retry_count": (
                    rag_result.citation_retry_count
                ),
                "valid_citations": json.dumps(
                    rag_result.valid_citations
                ),
                "invalid_citations": json.dumps(
                    rag_result.invalid_citations
                ),
                "known_relevant_citations": sum(
                    citation in relevant
                    for citation in rag_result.valid_citations
                ),
                "answer": rag_result.answer,
                "retrieved_context": context_text,
                "generation_metadata": json.dumps(
                    rag_result.metadata or {}
                ),
            }

            completed[key] = {
                k: str(v)
                for k, v in record.items()
            }

            ordered = [
                completed[k]
                for k in sorted(completed)
            ]

            save_rows(
                csv_path,
                ordered,
            )

            print(
                f"Saved {len(completed)}/{target}",
                flush=True,
            )

    manifest = {
        "signature": signature,
        "conditions": list(CONDITIONS),
        "queries": len(query_ids),
        "generations": len(completed),
        "corpus_fingerprint": FINGERPRINT,
        "model": generation["model"],
        "top_k": 5,
        "temperature": generation["temperature"],
        "seed": generation["seed"],
        "ranking_source": (
            "experiments/av2/"
            "retrieval_comparison/final/per_query.csv"
        ),
        "retrieval_latencies": (
            "Reference means from an earlier, "
            "separate experiment"
        ),
        "evaluation_note": (
            "Citation validity is not factual "
            "accuracy or groundedness."
        ),
    }

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    print(
        "AV2 RAG COMPARISON COMPLETED",
        flush=True,
    )

    print(
        f"Generations: {len(completed)}",
        flush=True,
    )

    print(
        f"Output: {csv_path}",
        flush=True,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
