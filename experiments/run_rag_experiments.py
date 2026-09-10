"""Run the reproducible FastContext RAG generation experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import yaml

from src.app.bootstrap import create_application
from src.ingestion.chunk_loader import load_chunks_jsonl
from src.rag.pipeline import RAGPipeline
from src.rag.providers.base import (
    GenerationConfig,
    LLMMessage,
    LLMResponse,
)
from src.rag.providers.ollama import OllamaProvider
from src.semantic.persistence import calculate_corpus_fingerprint
from src.services.factory import create_fastcontext_service
from src.utils.config import PROJECT_ROOT

CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "rag_experiments.yaml"
)

PROMPT_PATH = (
    PROJECT_ROOT
    / "src"
    / "rag"
    / "prompt.py"
)

PIPELINE_PATH = (
    PROJECT_ROOT
    / "src"
    / "rag"
    / "pipeline.py"
)

RUNNER_PATH = Path(__file__).resolve()

CONDITIONS = (
    "no_rag",
    "linear",
    "indexed",
    "optimized",
    "semantic",
)

RAG_CONDITIONS = (
    "linear",
    "indexed",
    "optimized",
    "semantic",
)

LEXICAL_CONDITIONS = (
    "linear",
    "indexed",
    "optimized",
)

CHECKPOINT_SCHEMA_VERSION = 1

NO_RAG_SYSTEM_PROMPT = """You are a FastAPI technical assistant.

Answer the question directly using your internal knowledge.

Requirements:
- Use at most three short sentences.
- Do not explain your reasoning.
- Do not repeat the question.
- Do not use citations or pretend that external context was provided.
"""

CSV_FIELDS = (
    "experiment_signature",
    "run_id",
    "condition",
    "query_id",
    "category",
    "question",
    "corpus_chunks",
    "corpus_fingerprint",
    "top_k",
    "expected_provider",
    "expected_model",
    "actual_provider",
    "actual_model",
    "temperature",
    "max_tokens",
    "think",
    "retrieval_time_ns",
    "generation_time_ns",
    "end_to_end_time_ns",
    "retrieved_chunk_ids",
    "relevant_chunk_ids",
    "relevant_retrieved_chunk_ids",
    "retrieval_hit_count",
    "precision_at_5",
    "recall_at_5",
    "citation_valid",
    "citation_retry_count",
    "citation_count",
    "valid_citations",
    "invalid_citations",
    "citation_relevant_count",
    "citation_relevance",
    "answer_word_count",
    "answer",
    "generation_metadata",
)


@dataclass(frozen=True, slots=True)
class EvaluationQuery:
    """One ground-truth query used by the RAG experiment."""

    query_id: str
    category: str
    question: str
    relevant_chunks: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OutputPaths:
    """Paths used by one normal or smoke experiment."""

    csv_path: Path
    jsonl_path: Path
    checkpoint_path: Path


def build_parser() -> argparse.ArgumentParser:
    """Build the RAG experiment command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Run FastContext RAG generation experiments."
        )
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run all five conditions for the first query only."
        ),
    )

    output_group = (
        parser.add_mutually_exclusive_group()
    )

    output_group.add_argument(
        "--force",
        action="store_true",
        help=(
            "Delete only this RAG experiment output "
            "and start again."
        ),
    )

    output_group.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from compatible persisted RAG results."
        ),
    )

    return parser


def main() -> int:
    """Run the configured RAG experiment."""

    args = (
        build_parser()
        .parse_args()
    )

    config = _load_config()
    experiment = config[
        "rag_experiment"
    ]

    conditions = tuple(
        str(value)
        for value in experiment[
            "conditions"
        ]
    )

    _validate_conditions(
        conditions
    )

    top_k = int(
        experiment[
            "retrieval"
        ][
            "top_k"
        ]
    )

    generation = experiment[
        "generation"
    ]

    expected_provider = str(
        generation[
            "provider"
        ]
    )

    expected_model = str(
        generation[
            "model"
        ]
    )

    temperature = float(
        generation[
            "temperature"
        ]
    )

    max_tokens = int(
        generation[
            "max_tokens"
        ]
    )

    think = bool(
        generation[
            "think"
        ]
    )

    expected_chunks = int(
        experiment[
            "corpus"
        ][
            "expected_chunks"
        ]
    )

    expected_fingerprint = str(
        experiment[
            "corpus"
        ][
            "expected_fingerprint"
        ]
    )

    chunks_path = _project_path(
        experiment[
            "corpus"
        ][
            "chunks_file"
        ]
    )

    ground_truth_path = (
        _project_path(
            experiment[
                "queries"
            ][
                "ground_truth_file"
            ]
        )
    )

    expected_query_count = int(
        experiment[
            "queries"
        ][
            "expected_count"
        ]
    )

    outputs = _resolve_output_paths(
        experiment,
        smoke=args.smoke,
    )

    chunks = load_chunks_jsonl(
        chunks_path
    )

    corpus_fingerprint = (
        calculate_corpus_fingerprint(
            chunks
        )
    )

    _validate_corpus(
        chunks=chunks,
        fingerprint=corpus_fingerprint,
        expected_count=expected_chunks,
        expected_fingerprint=(
            expected_fingerprint
        ),
    )

    queries = _load_ground_truth(
        ground_truth_path,
        expected_count=expected_query_count,
        valid_chunk_ids={
            str(
                chunk[
                    "chunk_id"
                ]
            )
            for chunk in chunks
        },
    )

    if args.smoke:
        queries = queries[
            :1
        ]

    experiment_signature = (
        _calculate_signature(
            config_path=CONFIG_PATH,
            ground_truth_path=(
                ground_truth_path
            ),
            prompt_path=PROMPT_PATH,
            pipeline_path=PIPELINE_PATH,
            runner_path=RUNNER_PATH,
            corpus_fingerprint=(
                corpus_fingerprint
            ),
            query_ids=tuple(
                query.query_id
                for query in queries
            ),
            conditions=conditions,
            top_k=top_k,
            expected_provider=(
                expected_provider
            ),
            expected_model=(
                expected_model
            ),
            temperature=temperature,
            max_tokens=max_tokens,
            think=think,
            smoke=args.smoke,
        )
    )

    expected_run_ids = (
        _expected_run_ids(
            conditions=conditions,
            queries=queries,
        )
    )

    completed_run_ids = (
        _prepare_outputs(
            outputs=outputs,
            force=args.force,
            resume=args.resume,
            expected_run_ids=(
                expected_run_ids
            ),
            expected_signature=(
                experiment_signature
            ),
        )
    )

    print(
        "FastContext RAG Experiment Runner"
    )
    print(
        "=" * 60
    )
    print(
        "Mode: "
        + (
            "smoke"
            if args.smoke
            else "final"
        )
    )
    print(
        "Conditions: "
        + ", ".join(
            conditions
        )
    )
    print(
        f"Queries: {len(queries)}"
    )
    print(
        "Expected generations: "
        f"{len(expected_run_ids)}"
    )
    print(
        "Already completed: "
        f"{len(completed_run_ids)}"
    )
    print(
        f"Top-k: {top_k}"
    )
    print(
        f"Model: {expected_model}"
    )
    print(
        f"Temperature: {temperature}"
    )
    print(
        f"Corpus chunks: {len(chunks)}"
    )
    print(
        "Corpus fingerprint: "
        f"{corpus_fingerprint}"
    )
    print()

    application = create_application()
    settings = application.settings

    if expected_provider != "ollama":
        raise ValueError(
            "This experiment runner currently "
            "requires provider 'ollama'."
        )

    if (
        settings.ollama.model
        != expected_model
    ):
        raise ValueError(
            "Configured Ollama primary model "
            "does not match the experiment model."
        )

    provider = OllamaProvider(
        model=expected_model,
        base_url=(
            settings.ollama.base_url
        ),
    )

    if not provider.is_available():
        raise RuntimeError(
            "The required primary Ollama model "
            f"'{expected_model}' is unavailable."
        )

    rag_pipeline = RAGPipeline(
        provider=provider,
        max_truncation_retries=int(
            generation[
                "max_truncation_retries"
            ]
        ),
        max_retry_tokens=int(
            generation[
                "max_retry_tokens"
            ]
        ),
        max_citation_retries=int(
            generation[
                "citation_retries"
            ]
        ),
    )

    registry = application.registry

    services = {
        condition: (
            create_fastcontext_service(
                algorithm=condition,
                registry=registry,
                rag_pipeline=rag_pipeline,
            )
        )
        for condition in RAG_CONDITIONS
    }

    generation_config = (
        GenerationConfig(
            temperature=temperature,
            max_tokens=max_tokens,
            think=think,
        )
    )

    warmup = experiment[
        "warmup"
    ]

    if bool(
        warmup[
            "enabled"
        ]
    ):
        _run_warmup(
            provider=provider,
            services=services,
            query=str(
                warmup[
                    "query"
                ]
            ),
            top_k=top_k,
            expected_model=(
                expected_model
            ),
            generation_config=(
                generation_config
            ),
        )

    checkpoint_every = int(
        experiment[
            "checkpoint"
        ][
            "every_completed_runs"
        ]
    )

    completed_count = len(
        completed_run_ids
    )

    for query_index, query in enumerate(
        queries
    ):
        condition_order = (
            _condition_order_for_query(
                conditions,
                query_index,
            )
        )

        print(
            f"[{query.query_id}] "
            + " -> ".join(
                condition_order
            )
        )

        for condition in condition_order:
            run_id = _build_run_id(
                condition=condition,
                query_id=query.query_id,
            )

            if run_id in completed_run_ids:
                continue

            print(
                "  Running "
                f"{run_id}..."
            )

            if condition == "no_rag":
                row = _run_no_rag(
                    query=query,
                    provider=provider,
                    experiment_signature=(
                        experiment_signature
                    ),
                    corpus_chunks=len(
                        chunks
                    ),
                    corpus_fingerprint=(
                        corpus_fingerprint
                    ),
                    expected_provider=(
                        expected_provider
                    ),
                    expected_model=(
                        expected_model
                    ),
                    generation_config=(
                        generation_config
                    ),
                )

            else:
                row = _run_rag(
                    condition=condition,
                    query=query,
                    service=services[
                        condition
                    ],
                    experiment_signature=(
                        experiment_signature
                    ),
                    corpus_chunks=len(
                        chunks
                    ),
                    corpus_fingerprint=(
                        corpus_fingerprint
                    ),
                    top_k=top_k,
                    expected_provider=(
                        expected_provider
                    ),
                    expected_model=(
                        expected_model
                    ),
                    generation_config=(
                        generation_config
                    ),
                )

            _append_result(
                outputs=outputs,
                row=row,
            )

            completed_run_ids.add(
                run_id
            )

            completed_count += 1

            if (
                checkpoint_every > 0
                and (
                    completed_count
                    % checkpoint_every
                    == 0
                )
            ):
                _write_checkpoint(
                    path=(
                        outputs
                        .checkpoint_path
                    ),
                    experiment_signature=(
                        experiment_signature
                    ),
                    completed_run_ids=(
                        completed_run_ids
                    ),
                    expected_run_ids=(
                        expected_run_ids
                    ),
                )

            print(
                "  Completed "
                f"{completed_count}/"
                f"{len(expected_run_ids)}"
            )

    _write_checkpoint(
        path=outputs.checkpoint_path,
        experiment_signature=(
            experiment_signature
        ),
        completed_run_ids=(
            completed_run_ids
        ),
        expected_run_ids=(
            expected_run_ids
        ),
    )

    _verify_final_results(
        csv_path=outputs.csv_path,
        expected_run_ids=(
            expected_run_ids
        ),
        expected_signature=(
            experiment_signature
        ),
    )

    print()
    print(
        "FastContext RAG experiment complete."
    )
    print(
        f"CSV: {outputs.csv_path}"
    )
    print(
        f"JSONL: {outputs.jsonl_path}"
    )
    print(
        "Checkpoint: "
        f"{outputs.checkpoint_path}"
    )

    return 0


def _run_warmup(
    *,
    provider: OllamaProvider,
    services: Mapping[str, Any],
    query: str,
    top_k: int,
    expected_model: str,
    generation_config: GenerationConfig,
) -> None:
    """Warm the LLM and every retriever outside measurements."""

    print(
        "Running unmeasured warm-up..."
    )

    response = provider.generate(
        messages=[
            LLMMessage(
                role="system",
                content=(
                    "Reply directly and concisely."
                ),
            ),
            LLMMessage(
                role="user",
                content=(
                    "Reply only with OK."
                ),
            ),
        ],
        config=generation_config,
    )

    _assert_primary_model(
        response=response,
        expected_model=expected_model,
    )

    for condition in RAG_CONDITIONS:
        services[
            condition
        ].retrieve(
            query=query,
            top_k=top_k,
        )

    print(
        "Warm-up complete."
    )
    print()


def _run_no_rag(
    *,
    query: EvaluationQuery,
    provider: OllamaProvider,
    experiment_signature: str,
    corpus_chunks: int,
    corpus_fingerprint: str,
    expected_provider: str,
    expected_model: str,
    generation_config: GenerationConfig,
) -> dict[str, Any]:
    """Generate the closed-book no-RAG baseline."""

    start_time = perf_counter_ns()

    response = provider.generate(
        messages=[
            LLMMessage(
                role="system",
                content=(
                    NO_RAG_SYSTEM_PROMPT
                ),
            ),
            LLMMessage(
                role="user",
                content=query.question,
            ),
        ],
        config=generation_config,
    )

    end_to_end_time_ns = (
        perf_counter_ns()
        - start_time
    )

    _assert_primary_model(
        response=response,
        expected_model=expected_model,
    )

    if (
        response.provider
        != expected_provider
    ):
        raise RuntimeError(
            "No-RAG provider mismatch: "
            f"{response.provider!r}."
        )

    return {
        "experiment_signature": (
            experiment_signature
        ),
        "run_id": _build_run_id(
            condition="no_rag",
            query_id=query.query_id,
        ),
        "condition": "no_rag",
        "query_id": query.query_id,
        "category": query.category,
        "question": query.question,
        "corpus_chunks": corpus_chunks,
        "corpus_fingerprint": (
            corpus_fingerprint
        ),
        "top_k": None,
        "expected_provider": (
            expected_provider
        ),
        "expected_model": expected_model,
        "actual_provider": (
            response.provider
        ),
        "actual_model": response.model,
        "temperature": (
            generation_config.temperature
        ),
        "max_tokens": (
            generation_config.max_tokens
        ),
        "think": generation_config.think,
        "retrieval_time_ns": None,
        "generation_time_ns": (
            end_to_end_time_ns
        ),
        "end_to_end_time_ns": (
            end_to_end_time_ns
        ),
        "retrieved_chunk_ids": [],
        "relevant_chunk_ids": list(
            query.relevant_chunks
        ),
        "relevant_retrieved_chunk_ids": [],
        "retrieval_hit_count": None,
        "precision_at_5": None,
        "recall_at_5": None,
        "citation_valid": None,
        "citation_retry_count": None,
        "citation_count": None,
        "valid_citations": [],
        "invalid_citations": [],
        "citation_relevant_count": None,
        "citation_relevance": None,
        "answer_word_count": (
            _word_count(
                response.text
            )
        ),
        "answer": response.text,
        "generation_metadata": (
            response.metadata
            or {}
        ),
    }


def _run_rag(
    *,
    condition: str,
    query: EvaluationQuery,
    service: Any,
    experiment_signature: str,
    corpus_chunks: int,
    corpus_fingerprint: str,
    top_k: int,
    expected_provider: str,
    expected_model: str,
    generation_config: GenerationConfig,
) -> dict[str, Any]:
    """Run one measured RAG condition."""

    result = service.ask(
        query=query.question,
        top_k=top_k,
        generation_config=(
            generation_config
        ),
    )

    if (
        result.rag.model
        != expected_model
    ):
        raise RuntimeError(
            "RAG model mismatch. "
            f"Expected {expected_model!r}, "
            f"received {result.rag.model!r}."
        )

    if (
        result.rag.provider
        != expected_provider
    ):
        raise RuntimeError(
            "RAG provider mismatch. "
            f"Expected {expected_provider!r}, "
            f"received "
            f"{result.rag.provider!r}."
        )

    retrieved_chunk_ids = [
        chunk.chunk_id
        for chunk in (
            result.retrieval.chunks
        )
    ]

    (
        precision_at_5,
        recall_at_5,
        relevant_retrieved,
    ) = _retrieval_quality(
        retrieved_chunk_ids=(
            retrieved_chunk_ids
        ),
        relevant_chunk_ids=(
            query.relevant_chunks
        ),
        k=top_k,
    )

    (
        citation_relevant_count,
        citation_relevance,
    ) = _citation_relevance(
        valid_citations=(
            result.rag.valid_citations
        ),
        relevant_chunk_ids=(
            query.relevant_chunks
        ),
    )

    return {
        "experiment_signature": (
            experiment_signature
        ),
        "run_id": _build_run_id(
            condition=condition,
            query_id=query.query_id,
        ),
        "condition": condition,
        "query_id": query.query_id,
        "category": query.category,
        "question": query.question,
        "corpus_chunks": corpus_chunks,
        "corpus_fingerprint": (
            corpus_fingerprint
        ),
        "top_k": top_k,
        "expected_provider": (
            expected_provider
        ),
        "expected_model": expected_model,
        "actual_provider": (
            result.rag.provider
        ),
        "actual_model": result.rag.model,
        "temperature": (
            generation_config.temperature
        ),
        "max_tokens": (
            generation_config.max_tokens
        ),
        "think": generation_config.think,
        "retrieval_time_ns": (
            result.retrieval_time_ns
        ),
        "generation_time_ns": (
            result.generation_time_ns
        ),
        "end_to_end_time_ns": (
            result.end_to_end_time_ns
        ),
        "retrieved_chunk_ids": (
            retrieved_chunk_ids
        ),
        "relevant_chunk_ids": list(
            query.relevant_chunks
        ),
        "relevant_retrieved_chunk_ids": (
            relevant_retrieved
        ),
        "retrieval_hit_count": len(
            relevant_retrieved
        ),
        "precision_at_5": (
            precision_at_5
        ),
        "recall_at_5": recall_at_5,
        "citation_valid": (
            result.rag.citation_valid
        ),
        "citation_retry_count": (
            result.rag
            .citation_retry_count
        ),
        "citation_count": (
            result.rag.citation_count
        ),
        "valid_citations": list(
            result.rag.valid_citations
        ),
        "invalid_citations": list(
            result.rag.invalid_citations
        ),
        "citation_relevant_count": (
            citation_relevant_count
        ),
        "citation_relevance": (
            citation_relevance
        ),
        "answer_word_count": (
            _word_count(
                result.rag.answer
            )
        ),
        "answer": result.rag.answer,
        "generation_metadata": (
            result.rag.metadata
            or {}
        ),
    }


def _retrieval_quality(
    *,
    retrieved_chunk_ids: Sequence[str],
    relevant_chunk_ids: Sequence[str],
    k: int,
) -> tuple[
    float,
    float,
    list[str],
]:
    """Calculate Precision@k and Recall@k."""

    if k <= 0:
        raise ValueError(
            "k must be greater than zero."
        )

    relevant = set(
        relevant_chunk_ids
    )

    retrieved_at_k = list(
        retrieved_chunk_ids[
            :k
        ]
    )

    relevant_retrieved = [
        chunk_id
        for chunk_id
        in retrieved_at_k
        if chunk_id in relevant
    ]

    precision = (
        len(
            relevant_retrieved
        )
        / k
    )

    recall = (
        len(
            relevant_retrieved
        )
        / len(
            relevant
        )
        if relevant
        else 0.0
    )

    return (
        precision,
        recall,
        relevant_retrieved,
    )


def _citation_relevance(
    *,
    valid_citations: Sequence[str],
    relevant_chunk_ids: Sequence[str],
) -> tuple[
    int,
    float | None,
]:
    """Measure how many valid citations are human-relevant."""

    if not valid_citations:
        return (
            0,
            None,
        )

    relevant = set(
        relevant_chunk_ids
    )

    count = sum(
        citation
        in relevant
        for citation in valid_citations
    )

    return (
        count,
        count
        / len(
            valid_citations
        ),
    )


def _condition_order_for_query(
    conditions: Sequence[str],
    query_index: int,
) -> tuple[str, ...]:
    """Rotate conditions to balance execution order."""

    if not conditions:
        raise ValueError(
            "At least one condition is required."
        )

    offset = (
        query_index
        % len(
            conditions
        )
    )

    return tuple(
        conditions[
            offset:
        ]
    ) + tuple(
        conditions[
            :offset
        ]
    )


def _build_run_id(
    *,
    condition: str,
    query_id: str,
) -> str:
    """Build the stable identifier of one generation."""

    return (
        f"{condition}|{query_id}"
    )


def _expected_run_ids(
    *,
    conditions: Sequence[str],
    queries: Sequence[EvaluationQuery],
) -> set[str]:
    """Build the complete expected run matrix."""

    return {
        _build_run_id(
            condition=condition,
            query_id=query.query_id,
        )
        for query in queries
        for condition in conditions
    }


def _load_ground_truth(
    path: Path,
    *,
    expected_count: int,
    valid_chunk_ids: set[str],
) -> tuple[EvaluationQuery, ...]:
    """Load and validate completed ground truth."""

    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        data,
        dict,
    ):
        raise TypeError(
            "Ground-truth root must be an object."
        )

    metadata = data.get(
        "metadata"
    )

    raw_queries = data.get(
        "queries"
    )

    if not isinstance(
        metadata,
        dict,
    ):
        raise TypeError(
            "Ground-truth metadata must be an object."
        )

    if (
        metadata.get(
            "annotation_status"
        )
        != "complete"
    ):
        raise ValueError(
            "RAG experiments require completed "
            "ground truth."
        )

    if not isinstance(
        raw_queries,
        list,
    ):
        raise TypeError(
            "Ground-truth queries must be a list."
        )

    if (
        len(
            raw_queries
        )
        != expected_count
    ):
        raise ValueError(
            "Unexpected ground-truth query count."
        )

    queries: list[
        EvaluationQuery
    ] = []

    seen_ids: set[str] = set()

    for item in raw_queries:
        if not isinstance(
            item,
            dict,
        ):
            raise TypeError(
                "Every ground-truth query "
                "must be an object."
            )

        query_id = _required_string(
            item.get(
                "query_id"
            ),
            "query_id",
        )

        if query_id in seen_ids:
            raise ValueError(
                "Duplicate query_id in "
                "ground truth."
            )

        seen_ids.add(
            query_id
        )

        relevant_raw = item.get(
            "relevant_chunks"
        )

        if not isinstance(
            relevant_raw,
            list,
        ):
            raise TypeError(
                "relevant_chunks must be a list."
            )

        relevant = tuple(
            _required_string(
                value,
                "relevant chunk ID",
            )
            for value in relevant_raw
        )

        if not relevant:
            raise ValueError(
                "Completed ground truth contains "
                f"no relevant chunks for {query_id}."
            )

        unknown = (
            set(
                relevant
            )
            - valid_chunk_ids
        )

        if unknown:
            raise ValueError(
                "Ground truth references unknown "
                f"chunks for {query_id}: "
                f"{sorted(unknown)}"
            )

        queries.append(
            EvaluationQuery(
                query_id=query_id,
                category=_required_string(
                    item.get(
                        "category"
                    ),
                    "category",
                ),
                question=_required_string(
                    item.get(
                        "question"
                    ),
                    "question",
                ),
                relevant_chunks=relevant,
            )
        )

    return tuple(
        queries
    )


def _validate_corpus(
    *,
    chunks: Sequence[Mapping[str, Any]],
    fingerprint: str,
    expected_count: int,
    expected_fingerprint: str,
) -> None:
    """Reject a corpus different from the frozen corpus."""

    if (
        len(
            chunks
        )
        != expected_count
    ):
        raise ValueError(
            "Frozen corpus chunk count mismatch."
        )

    if (
        fingerprint
        != expected_fingerprint
    ):
        raise ValueError(
            "Frozen corpus fingerprint mismatch."
        )


def _validate_conditions(
    conditions: Sequence[str],
) -> None:
    """Validate the configured RAG conditions."""

    if (
        tuple(
            conditions
        )
        != CONDITIONS
    ):
        raise ValueError(
            "RAG conditions must be exactly: "
            + ", ".join(
                CONDITIONS
            )
        )


def _load_config() -> dict[str, Any]:
    """Load the RAG experiment YAML."""

    data = yaml.safe_load(
        CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        data,
        dict,
    ):
        raise TypeError(
            "RAG experiment config "
            "must be a mapping."
        )

    return data


def _project_path(
    value: object,
) -> Path:
    """Resolve a project-relative path."""

    path = Path(
        _required_string(
            value,
            "project path",
        )
    )

    if path.is_absolute():
        return path

    return (
        PROJECT_ROOT
        / path
    )


def _resolve_output_paths(
    experiment: Mapping[str, Any],
    *,
    smoke: bool,
) -> OutputPaths:
    """Resolve final or smoke output paths."""

    output = experiment[
        "output"
    ]

    if not isinstance(
        output,
        Mapping,
    ):
        raise TypeError(
            "output configuration "
            "must be a mapping."
        )

    directory = _project_path(
        output[
            "directory"
        ]
    )

    if smoke:
        return OutputPaths(
            csv_path=(
                directory
                / "generations.smoke.csv"
            ),
            jsonl_path=(
                directory
                / "generations.smoke.jsonl"
            ),
            checkpoint_path=(
                directory
                / "checkpoint.smoke.json"
            ),
        )

    return OutputPaths(
        csv_path=(
            directory
            / _required_string(
                output[
                    "csv"
                ],
                "CSV output name",
            )
        ),
        jsonl_path=(
            directory
            / _required_string(
                output[
                    "jsonl"
                ],
                "JSONL output name",
            )
        ),
        checkpoint_path=(
            directory
            / _required_string(
                output[
                    "checkpoint"
                ],
                "checkpoint output name",
            )
        ),
    )


def _prepare_outputs(
    *,
    outputs: OutputPaths,
    force: bool,
    resume: bool,
    expected_run_ids: set[str],
    expected_signature: str,
) -> set[str]:
    """Prepare output files and recover completed runs."""

    for path in (
        outputs.csv_path,
        outputs.jsonl_path,
        outputs.checkpoint_path,
    ):
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    if force:
        for path in (
            outputs.csv_path,
            outputs.jsonl_path,
            outputs.checkpoint_path,
        ):
            path.unlink(
                missing_ok=True
            )

    existing_any = any(
        path.exists()
        for path in (
            outputs.csv_path,
            outputs.jsonl_path,
            outputs.checkpoint_path,
        )
    )

    if (
        existing_any
        and not resume
        and not force
    ):
        raise FileExistsError(
            "RAG experiment output already exists. "
            "Use --resume to continue or --force "
            "to replace only these RAG outputs."
        )

    if not outputs.csv_path.exists():
        _initialize_csv(
            outputs.csv_path
        )

    completed = _read_completed_runs(
        path=outputs.csv_path,
        expected_run_ids=(
            expected_run_ids
        ),
        expected_signature=(
            expected_signature
        ),
    )

    _synchronize_jsonl(
        csv_path=outputs.csv_path,
        jsonl_path=outputs.jsonl_path,
    )

    return completed


def _initialize_csv(
    path: Path,
) -> None:
    """Create the raw generation CSV with its header."""

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=CSV_FIELDS,
        )

        writer.writeheader()

        file.flush()
        os.fsync(
            file.fileno()
        )


def _append_result(
    *,
    outputs: OutputPaths,
    row: Mapping[str, Any],
) -> None:
    """Persist one completed generation immediately."""

    csv_row = _to_csv_row(
        row
    )

    with outputs.csv_path.open(
        "a",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=CSV_FIELDS,
        )

        writer.writerow(
            csv_row
        )

        file.flush()
        os.fsync(
            file.fileno()
        )

    with outputs.jsonl_path.open(
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            json.dumps(
                dict(
                    row
                ),
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )

        file.flush()
        os.fsync(
            file.fileno()
        )


def _to_csv_row(
    row: Mapping[str, Any],
) -> dict[str, str]:
    """Serialize one typed result for CSV storage."""

    result: dict[
        str,
        str,
    ] = {}

    json_fields = {
        "retrieved_chunk_ids",
        "relevant_chunk_ids",
        "relevant_retrieved_chunk_ids",
        "valid_citations",
        "invalid_citations",
        "generation_metadata",
    }

    for field in CSV_FIELDS:
        value = row.get(
            field
        )

        if value is None:
            result[
                field
            ] = ""

        elif field in json_fields:
            result[
                field
            ] = json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
            )

        elif isinstance(
            value,
            bool,
        ):
            result[
                field
            ] = (
                "true"
                if value
                else "false"
            )

        else:
            result[
                field
            ] = str(
                value
            )

    return result


def _read_completed_runs(
    *,
    path: Path,
    expected_run_ids: set[str],
    expected_signature: str,
) -> set[str]:
    """Read and validate persisted raw generations."""

    completed: set[str] = set()

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(
            file
        )

        if (
            tuple(
                reader.fieldnames
                or ()
            )
            != CSV_FIELDS
        ):
            raise ValueError(
                "RAG results CSV schema mismatch."
            )

        for line_number, row in enumerate(
            reader,
            start=2,
        ):
            if None in row:
                raise ValueError(
                    "Unexpected extra CSV fields at "
                    f"line {line_number}."
                )

            if any(
                value is None
                for value in row.values()
            ):
                raise ValueError(
                    "Truncated CSV row at "
                    f"line {line_number}."
                )

            signature = row[
                "experiment_signature"
            ]

            if (
                signature
                != expected_signature
            ):
                raise ValueError(
                    "Existing RAG results use a "
                    "different experiment signature."
                )

            run_id = row[
                "run_id"
            ]

            if (
                run_id
                not in expected_run_ids
            ):
                raise ValueError(
                    "Unexpected run_id in RAG results: "
                    f"{run_id!r}."
                )

            if run_id in completed:
                raise ValueError(
                    "Duplicate run_id in RAG results: "
                    f"{run_id!r}."
                )

            completed.add(
                run_id
            )

    return completed


def _synchronize_jsonl(
    *,
    csv_path: Path,
    jsonl_path: Path,
) -> None:
    """Regenerate JSONL from the source-of-truth CSV."""

    rows: list[
        dict[
            str,
            Any,
        ]
    ] = []

    with csv_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(
            file
        )

        for row in reader:
            rows.append(
                _from_csv_row(
                    row
                )
            )

    with jsonl_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for row in rows:
            file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

        file.flush()
        os.fsync(
            file.fileno()
        )


def _from_csv_row(
    row: Mapping[str, str],
) -> dict[str, Any]:
    """Recover useful JSON types from one CSV row."""

    result: dict[
        str,
        Any,
    ] = dict(
        row
    )

    json_fields = (
        "retrieved_chunk_ids",
        "relevant_chunk_ids",
        "relevant_retrieved_chunk_ids",
        "valid_citations",
        "invalid_citations",
        "generation_metadata",
    )

    for field in json_fields:
        raw = row.get(
            field,
            "",
        )

        result[
            field
        ] = (
            json.loads(
                raw
            )
            if raw
            else (
                {}
                if field
                == "generation_metadata"
                else []
            )
        )

    return result


def _write_checkpoint(
    *,
    path: Path,
    experiment_signature: str,
    completed_run_ids: set[str],
    expected_run_ids: set[str],
) -> None:
    """Write a resumable checkpoint with OneDrive-safe retries."""

    status = (
        "complete"
        if (
            completed_run_ids
            == expected_run_ids
        )
        else "running"
    )

    payload = {
        "schema_version": (
            CHECKPOINT_SCHEMA_VERSION
        ),
        "experiment_signature": (
            experiment_signature
        ),
        "status": status,
        "completed_runs": len(
            completed_run_ids
        ),
        "expected_runs": len(
            expected_run_ids
        ),
        "completed_run_ids": sorted(
            completed_run_ids
        ),
    }

    temp_path = path.with_suffix(
        path.suffix
        + ".tmp"
    )

    temp_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    last_error: OSError | None = None

    for attempt in range(
        8
    ):
        try:
            os.replace(
                temp_path,
                path,
            )
            return

        except OSError as exc:
            last_error = exc
            time.sleep(
                0.15
                * (
                    attempt
                    + 1
                )
            )

    temp_path.unlink(
        missing_ok=True
    )

    if path.exists():
        print(
            "Warning: checkpoint file is locked. "
            "The CSV remains the source of truth."
        )
        return

    if last_error is not None:
        raise last_error


def _verify_final_results(
    *,
    csv_path: Path,
    expected_run_ids: set[str],
    expected_signature: str,
) -> None:
    """Require the raw CSV to contain the complete matrix."""

    completed = _read_completed_runs(
        path=csv_path,
        expected_run_ids=(
            expected_run_ids
        ),
        expected_signature=(
            expected_signature
        ),
    )

    if (
        completed
        != expected_run_ids
    ):
        missing = sorted(
            expected_run_ids
            - completed
        )

        raise RuntimeError(
            "RAG experiment ended incomplete. "
            f"Missing: {missing[:10]}"
        )


def _calculate_signature(
    *,
    config_path: Path,
    ground_truth_path: Path,
    prompt_path: Path,
    pipeline_path: Path,
    runner_path: Path,
    corpus_fingerprint: str,
    query_ids: Sequence[str],
    conditions: Sequence[str],
    top_k: int,
    expected_provider: str,
    expected_model: str,
    temperature: float,
    max_tokens: int,
    think: bool,
    smoke: bool,
) -> str:
    """Calculate the immutable experiment signature."""

    payload = {
        "config_sha256": (
            _sha256_file(
                config_path
            )
        ),
        "ground_truth_sha256": (
            _sha256_file(
                ground_truth_path
            )
        ),
        "prompt_sha256": (
            _sha256_file(
                prompt_path
            )
        ),
        "pipeline_sha256": (
            _sha256_file(
                pipeline_path
            )
        ),
        "runner_sha256": (
            _sha256_file(
                runner_path
            )
        ),
        "corpus_fingerprint": (
            corpus_fingerprint
        ),
        "query_ids": list(
            query_ids
        ),
        "conditions": list(
            conditions
        ),
        "top_k": top_k,
        "provider": expected_provider,
        "model": expected_model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "think": think,
        "smoke": smoke,
    }

    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        serialized
    ).hexdigest()


def _sha256_file(
    path: Path,
) -> str:
    """Calculate SHA-256 of one file."""

    hasher = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:
        for block in iter(
            lambda: file.read(
                1024
                * 1024
            ),
            b"",
        ):
            hasher.update(
                block
            )

    return hasher.hexdigest()


def _assert_primary_model(
    *,
    response: LLMResponse,
    expected_model: str,
) -> None:
    """Abort if a generation used another model."""

    if (
        response.model
        != expected_model
    ):
        raise RuntimeError(
            "Unexpected generation model. "
            f"Expected {expected_model!r}, "
            f"received {response.model!r}."
        )


def _word_count(
    text: str,
) -> int:
    """Count whitespace-separated answer tokens."""

    return len(
        text.split()
    )


def _required_string(
    value: object,
    field_name: str,
) -> str:
    """Return one required non-empty string."""

    if not isinstance(
        value,
        str,
    ):
        raise TypeError(
            f"{field_name} must be a string."
        )

    normalized = value.strip()

    if not normalized:
        raise ValueError(
            f"{field_name} cannot be empty."
        )

    return normalized


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
