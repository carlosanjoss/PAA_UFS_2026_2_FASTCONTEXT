"""Run reproducible FastContext retrieval experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.evaluation.performance import measure_retrieval
from src.evaluation.retrieval_metrics import evaluate_retrieval
from src.evaluation.statistics import calculate_statistics
from src.ingestion.chunk_loader import load_chunks_jsonl
from src.retrieval.base import Retriever
from src.retrieval.indexed_retriever import IndexedRetriever
from src.retrieval.linear_retriever import LinearRetriever
from src.retrieval.optimized_retriever import OptimizedRetriever
from src.retrieval.semantic_retriever import SemanticRetriever
from src.semantic.persistence import calculate_corpus_fingerprint
from src.utils.config import PROJECT_ROOT

CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "experiments.yaml"
)

CHUNKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "chunks"
    / "chunks.jsonl"
)

SUPPORTED_ALGORITHMS = (
    "linear",
    "indexed",
    "optimized",
    "semantic",
)

PERFORMANCE_METRICS = (
    "total_time_ns",
    "retrieval_time_ns",
    "sorting_time_ns",
    "peak_memory_bytes",
    "peak_memory_mb",
    "comparisons",
    "chunks_scored",
    "candidates_found",
    "result_count",
    "query_embedding_time_ns",
    "faiss_search_time_ns",
)

SCENARIO_METRICS = (
    "retriever_setup_time_ns",
    "index_build_time_ns",
    "index_load_time_ns",
)

CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(
    frozen=True,
    slots=True,
)
class EvaluationQuery:
    """Represent one evaluation query."""

    query_id: str
    category: str
    question: str
    relevant_chunks: tuple[
        str,
        ...,
    ]


def build_parser() -> argparse.ArgumentParser:
    """Create the experiment CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Run FastContext retrieval experiments."
        )
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run one query, one repetition, "
            "and the full corpus."
        ),
    )

    parser.add_argument(
        "--skip-memory",
        action="store_true",
        help=(
            "Skip the separate "
            "memory-profiling pass."
        ),
    )

    output_group = parser.add_mutually_exclusive_group()

    output_group.add_argument(
        "--force",
        action="store_true",
        help=(
            "Start a new experiment and overwrite "
            "existing raw results, summary, and checkpoint."
        ),
    )

    output_group.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume a compatible interrupted experiment "
            "without repeating completed measured runs."
        ),
    )

    return parser


def main() -> int:
    """Run the configured experiment matrix."""
    args = (
        build_parser()
        .parse_args()
    )

    config = _load_config()

    experiments = config[
        "experiments"
    ]

    seed = int(
        experiments[
            "reproducibility"
        ][
            "random_seed"
        ]
    )

    algorithms = tuple(
        str(value)
        for value in experiments[
            "algorithms"
        ]
    )

    fractions = tuple(
        float(value)
        for value in experiments[
            "corpus_sizes"
        ]
    )

    k_values = tuple(
        int(value)
        for value in experiments[
            "k_values"
        ]
    )

    repetitions = int(
        experiments[
            "repetitions"
        ]
    )

    warmup = experiments[
        "warmup"
    ]

    warmup_repetitions = (
        int(
            warmup[
                "repetitions"
            ]
        )
        if bool(
            warmup[
                "enabled"
            ]
        )
        else 0
    )

    raw_output = _project_path(
        experiments[
            "output"
        ][
            "raw_results"
        ]
    )

    summary_output = _project_path(
        experiments[
            "output"
        ][
            "processed_results"
        ]
    )

    checkpoint_output = (
        _checkpoint_path(
            raw_output
        )
    )

    ground_truth_path = (
        _project_path(
            experiments[
                "queries"
            ][
                "ground_truth_file"
            ]
        )
    )

    expected_query_count = int(
        experiments[
            "queries"
        ][
            "expected_count"
        ]
    )

    _validate_experiment_config(
        algorithms=algorithms,
        fractions=fractions,
        k_values=k_values,
        repetitions=repetitions,
    )

    if args.smoke:
        fractions = (
            1.0,
        )

        repetitions = 1

    if args.force:
        raw_output.unlink(
            missing_ok=True
        )

        summary_output.unlink(
            missing_ok=True
        )

        checkpoint_output.unlink(
            missing_ok=True
        )

        _checkpoint_temp_path(
            checkpoint_output
        ).unlink(
            missing_ok=True
        )

    elif not args.resume and (
        raw_output.exists()
        or summary_output.exists()
        or checkpoint_output.exists()
    ):
        raise FileExistsError(
            "Experiment output already exists. "
            "Use --resume to continue a compatible "
            "interrupted experiment or --force to "
            "start a new one."
        )

    elif args.resume and not raw_output.exists():
        raise FileNotFoundError(
            "Cannot resume because raw results do not exist: "
            f"{raw_output}"
        )

    elif args.resume and not checkpoint_output.exists():
        raise FileNotFoundError(
            "Cannot resume safely because the experiment "
            "checkpoint does not exist: "
            f"{checkpoint_output}. Use --force to start "
            "a new experiment instead of mixing unverifiable "
            "raw results."
        )

    chunks = load_chunks_jsonl(
        CHUNKS_PATH
    )

    full_ids = {
        str(
            chunk[
                "chunk_id"
            ]
        )
        for chunk in chunks
    }

    full_fingerprint = (
        calculate_corpus_fingerprint(
            chunks
        )
    )

    (
        annotation_status,
        queries,
    ) = _load_ground_truth(
        ground_truth_path,
        expected_count=(
            expected_query_count
        ),
        valid_chunk_ids=(
            full_ids
        ),
    )

    if args.smoke:
        queries = queries[
            :1
        ]

    subsets = (
        _build_nested_subsets(
            chunks,
            fractions,
            seed=seed,
        )
    )

    subset_metadata = {
        fraction: (
            len(
                subsets[
                    fraction
                ]
            ),
            calculate_corpus_fingerprint(
                subsets[
                    fraction
                ]
            ),
        )
        for fraction in fractions
    }

    top_k = max(
        k_values
    )

    expected_run_ids = (
        _expected_run_ids(
            algorithms=algorithms,
            fractions=fractions,
            queries=queries,
            repetitions=repetitions,
        )
    )

    expected_runs = len(
        expected_run_ids
    )

    fieldnames = (
        _raw_fieldnames(
            k_values
        )
    )

    signature_payload = (
        _build_signature_payload(
            algorithms=algorithms,
            fractions=fractions,
            queries=queries,
            repetitions=repetitions,
            warmup_repetitions=(
                warmup_repetitions
            ),
            k_values=k_values,
            top_k=top_k,
            seed=seed,
            full_corpus_chunks=(
                len(
                    chunks
                )
            ),
            full_corpus_fingerprint=(
                full_fingerprint
            ),
            ground_truth_path=(
                ground_truth_path
            ),
            annotation_status=(
                annotation_status
            ),
            smoke=bool(
                args.smoke
            ),
            memory_profiling=(
                not args.skip_memory
            ),
            raw_fieldnames=(
                fieldnames
            ),
        )
    )

    signature = (
        _calculate_signature(
            signature_payload
        )
    )

    if args.resume:
        rows = _read_raw_results(
            raw_output,
            expected_fieldnames=(
                fieldnames
            ),
        )

        completed_run_ids = (
            _validate_existing_rows(
                rows,
                expected_run_ids=(
                    expected_run_ids
                ),
                algorithms=algorithms,
                fractions=fractions,
                queries=queries,
                repetitions=repetitions,
                subset_metadata=(
                    subset_metadata
                ),
                full_corpus_chunks=(
                    len(
                        chunks
                    )
                ),
                top_k=top_k,
                warmup_repetitions=(
                    warmup_repetitions
                ),
                seed=seed,
            )
        )

        checkpoint = (
            _load_checkpoint(
                checkpoint_output
            )
        )

        _validate_checkpoint(
            checkpoint,
            signature=signature,
            expected_runs=(
                expected_runs
            ),
            raw_completed_run_ids=(
                completed_run_ids
            ),
        )

        summary_output.unlink(
            missing_ok=True
        )

        _ensure_trailing_newline(
            raw_output
        )

        _write_checkpoint(
            checkpoint_output,
            signature=signature,
            signature_payload=(
                signature_payload
            ),
            expected_runs=(
                expected_runs
            ),
            completed_run_ids=(
                completed_run_ids
            ),
            status=(
                "complete"
                if len(
                    completed_run_ids
                )
                == expected_runs
                else "in_progress"
            ),
            raw_output=raw_output,
            summary_output=(
                summary_output
            ),
        )

    else:
        raw_output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with raw_output.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            _flush_file(file)

        completed_run_ids: set[
            str
        ] = set()

        _write_checkpoint(
            checkpoint_output,
            signature=signature,
            signature_payload=(
                signature_payload
            ),
            expected_runs=(
                expected_runs
            ),
            completed_run_ids=(
                completed_run_ids
            ),
            status="in_progress",
            raw_output=raw_output,
            summary_output=(
                summary_output
            ),
        )

    print(
        "FastContext Experiment Runner"
    )

    print(
        "=" * 60
    )

    print(
        "Algorithms: "
        + ", ".join(
            algorithms
        )
    )

    print(
        "Corpus fractions: "
        f"{fractions}"
    )

    print(
        f"Queries: {len(queries)}"
    )

    print(
        f"Repetitions: {repetitions}"
    )

    print(
        "Measured retrieval runs: "
        f"{expected_runs}"
    )

    print(
        f"Retrieval top-k: {top_k}"
    )

    print(
        f"Quality cutoffs: {k_values}"
    )

    print(
        "Ground truth: "
        f"{annotation_status}"
    )

    print(
        "Memory profiling: "
        + (
            "off"
            if args.skip_memory
            else "on"
        )
    )

    print(
        "Resume mode: "
        + (
            "on"
            if args.resume
            else "off"
        )
    )

    print(
        "Completed measured runs: "
        f"{len(completed_run_ids)}/"
        f"{expected_runs}"
    )

    print(
        "Checkpoint: "
        f"{checkpoint_output}"
    )

    print()

    with raw_output.open(
        "a",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        for fraction in fractions:
            subset = subsets[
                fraction
            ]

            subset_ids = {
                str(
                    chunk[
                        "chunk_id"
                    ]
                )
                for chunk in subset
            }

            (
                subset_chunk_count,
                fingerprint,
            ) = subset_metadata[
                fraction
            ]

            is_full_corpus = (
                subset_chunk_count
                == len(
                    chunks
                )
                and fingerprint
                == full_fingerprint
            )

            fraction_run_ids = {
                run_id
                for run_id in expected_run_ids
                if _run_id_fraction(
                    run_id
                )
                == f"{fraction:.6f}"
            }

            missing_fraction_runs = (
                fraction_run_ids
                - completed_run_ids
            )

            print(
                f"Corpus {fraction:.2f}: "
                f"{len(subset)} chunks, "
                f"{fingerprint[:12]}..."
            )

            if not missing_fraction_runs:
                print(
                    "  already complete; "
                    "skipping retriever setup"
                )
                continue

            (
                retrievers,
                setup_times,
            ) = _create_retrievers(
                algorithms,
                subset,
                allow_persisted_semantic_index=(
                    is_full_corpus
                ),
            )

            for query in queries:
                missing_algorithms = [
                    algorithm
                    for algorithm in algorithms
                    if any(
                        _build_run_id(
                            algorithm,
                            fraction,
                            query.query_id,
                            repetition,
                        )
                        not in completed_run_ids
                        for repetition in range(
                            1,
                            repetitions + 1,
                        )
                    )
                ]

                if not missing_algorithms:
                    continue

                for algorithm in (
                    missing_algorithms
                ):
                    for _ in range(
                        warmup_repetitions
                    ):
                        retrievers[
                            algorithm
                        ].retrieve(
                            query.question,
                            top_k=top_k,
                        )

                for repetition in range(
                    1,
                    repetitions + 1,
                ):
                    for algorithm in algorithms:
                        run_id = (
                            _build_run_id(
                                algorithm,
                                fraction,
                                query.query_id,
                                repetition,
                            )
                        )

                        if run_id in completed_run_ids:
                            continue

                        performance = (
                            measure_retrieval(
                                retrievers[
                                    algorithm
                                ],
                                query.question,
                                top_k=top_k,
                                profile_memory=(
                                    not args.skip_memory
                                ),
                            )
                        )

                        result = (
                            performance.result
                        )

                        metadata = (
                            result.metadata
                            if isinstance(
                                result.metadata,
                                Mapping,
                            )
                            else {}
                        )

                        retrieved_ids = tuple(
                            chunk.chunk_id
                            for chunk
                            in result.chunks
                        )

                        row: dict[
                            str,
                            object,
                        ] = {
                            "run_id": run_id,
                            "algorithm": (
                                algorithm
                            ),
                            "corpus_fraction": (
                                fraction
                            ),
                            "corpus_chunks": (
                                len(
                                    subset
                                )
                            ),
                            "full_corpus_chunks": (
                                len(
                                    chunks
                                )
                            ),
                            "corpus_fingerprint": (
                                fingerprint
                            ),
                            "query_id": (
                                query.query_id
                            ),
                            "category": (
                                query.category
                            ),
                            "question": (
                                query.question
                            ),
                            "repetition": (
                                repetition
                            ),
                            "retrieval_top_k": (
                                top_k
                            ),
                            "warmup_repetitions": (
                                warmup_repetitions
                            ),
                            "random_seed": (
                                seed
                            ),
                            "retriever_setup_time_ns": (
                                setup_times[
                                    algorithm
                                ]
                            ),
                            "total_time_ns": (
                                performance
                                .total_time_ns
                            ),
                            "retrieval_time_ns": (
                                performance
                                .retrieval_time_ns
                            ),
                            "sorting_time_ns": (
                                performance
                                .sorting_time_ns
                            ),
                            "index_build_time_ns": (
                                performance
                                .index_build_time_ns
                            ),
                            "peak_memory_bytes": (
                                performance
                                .peak_memory_bytes
                            ),
                            "peak_memory_mb": (
                                performance
                                .peak_memory_mb
                            ),
                            "memory_method": (
                                performance
                                .memory_method
                            ),
                            "comparisons": (
                                performance
                                .comparisons
                            ),
                            "chunks_scored": (
                                performance
                                .chunks_scored
                            ),
                            "candidates_found": (
                                performance
                                .candidates_found
                            ),
                            "result_count": (
                                len(
                                    result.chunks
                                )
                            ),
                            "retrieved_chunk_ids": (
                                json.dumps(
                                    retrieved_ids
                                )
                            ),
                            "index_source": (
                                _metadata_value(
                                    metadata,
                                    "index_source",
                                )
                            ),
                            "persistence_status": (
                                _metadata_value(
                                    metadata,
                                    "persistence_status",
                                )
                            ),
                            "index_load_time_ns": (
                                _metadata_value(
                                    metadata,
                                    "index_load_time_ns",
                                )
                            ),
                            "query_embedding_time_ns": (
                                _metadata_value(
                                    metadata,
                                    "query_embedding_time_ns",
                                )
                            ),
                            "faiss_search_time_ns": (
                                _metadata_value(
                                    metadata,
                                    "faiss_search_time_ns",
                                )
                            ),
                        }

                        row.update(
                            _quality_fields(
                                retrieved_ids=(
                                    retrieved_ids
                                ),
                                relevant_ids=(
                                    query
                                    .relevant_chunks
                                ),
                                available_ids=(
                                    subset_ids
                                ),
                                k_values=(
                                    k_values
                                ),
                                ground_truth_complete=(
                                    annotation_status
                                    == "complete"
                                ),
                            )
                        )

                        writer.writerow(
                            row
                        )

                        _flush_file(file)

                        completed_run_ids.add(
                            run_id
                        )

                        _write_checkpoint(
                            checkpoint_output,
                            signature=signature,
                            signature_payload=(
                                signature_payload
                            ),
                            expected_runs=(
                                expected_runs
                            ),
                            completed_run_ids=(
                                completed_run_ids
                            ),
                            status="in_progress",
                            raw_output=(
                                raw_output
                            ),
                            summary_output=(
                                summary_output
                            ),
                        )

                        completed = len(
                            completed_run_ids
                        )

                        if (
                            completed % 25
                            == 0
                            or completed
                            == expected_runs
                        ):
                            print(
                                "Progress: "
                                f"{completed}/"
                                f"{expected_runs}"
                            )

    if len(
        completed_run_ids
    ) != expected_runs:
        raise RuntimeError(
            "Experiment ended before all measured "
            "runs were completed. Resume with --resume."
        )

    rows = _read_raw_results(
        raw_output,
        expected_fieldnames=(
            fieldnames
        ),
    )

    final_run_ids = (
        _validate_existing_rows(
            rows,
            expected_run_ids=(
                expected_run_ids
            ),
            algorithms=algorithms,
            fractions=fractions,
            queries=queries,
            repetitions=repetitions,
            subset_metadata=(
                subset_metadata
            ),
            full_corpus_chunks=(
                len(
                    chunks
                )
            ),
            top_k=top_k,
            warmup_repetitions=(
                warmup_repetitions
            ),
            seed=seed,
        )
    )

    if final_run_ids != expected_run_ids:
        raise RuntimeError(
            "Final raw-results validation failed: "
            "the measured run matrix is incomplete."
        )

    summary = _summarize(
        rows,
        k_values,
    )

    _write_summary(
        summary_output,
        summary,
    )

    _write_checkpoint(
        checkpoint_output,
        signature=signature,
        signature_payload=(
            signature_payload
        ),
        expected_runs=(
            expected_runs
        ),
        completed_run_ids=(
            final_run_ids
        ),
        status="complete",
        raw_output=raw_output,
        summary_output=(
            summary_output
        ),
    )

    print()

    print(
        f"Raw rows: {len(rows)}"
    )

    print(
        "Summary rows: "
        f"{len(summary)}"
    )

    print(
        f"Raw results: {raw_output}"
    )

    print(
        f"Summary: {summary_output}"
    )

    print(
        f"Checkpoint: {checkpoint_output}"
    )

    if (
        annotation_status
        != "complete"
    ):
        print(
            "Quality fields are empty "
            "because human ground truth "
            "is pending."
        )

    return 0


def _create_retrievers(
    algorithms: Sequence[str],
    chunks: list[
        dict[
            str,
            Any,
        ]
    ],
    *,
    allow_persisted_semantic_index: bool,
) -> tuple[
    dict[
        str,
        Retriever,
    ],
    dict[
        str,
        int,
    ],
]:
    """Create one retriever instance per algorithm."""
    retrievers: dict[
        str,
        Retriever,
    ] = {}

    setup_times: dict[
        str,
        int,
    ] = {}

    for algorithm in algorithms:
        started = (
            time.perf_counter_ns()
        )

        if algorithm == "linear":
            retriever: Retriever = (
                LinearRetriever(
                    chunks
                )
            )

        elif algorithm == "indexed":
            retriever = (
                IndexedRetriever(
                    chunks
                )
            )

        elif algorithm == "optimized":
            retriever = (
                OptimizedRetriever(
                    chunks
                )
            )

        elif algorithm == "semantic":
            retriever = (
                SemanticRetriever(
                    chunks,
                    use_persisted_index=(
                        allow_persisted_semantic_index
                    ),
                )
            )

        else:
            raise ValueError(
                "Unsupported algorithm: "
                f"{algorithm}"
            )

        retrievers[
            algorithm
        ] = retriever

        setup_times[
            algorithm
        ] = (
            time.perf_counter_ns()
            - started
        )

        print(
            f"  {algorithm}: "
            f"{setup_times[algorithm] / 1_000_000:.2f} "
            "ms setup"
        )

    return (
        retrievers,
        setup_times,
    )


def _quality_fields(
    *,
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
    available_ids: set[str],
    k_values: Sequence[int],
    ground_truth_complete: bool,
) -> dict[
    str,
    object,
]:
    """Calculate quality at all k values from one Top-k ranking."""
    relevant_full = set(
        relevant_ids
    )

    relevant_available = (
        relevant_full
        & available_ids
    )

    fields: dict[
        str,
        object,
    ] = {
        "quality_available": False,
        "relevant_total_full": (
            len(
                relevant_full
            )
        ),
        "relevant_total_available": (
            len(
                relevant_available
            )
        ),
    }

    for k in k_values:
        for name in (
            "precision",
            "recall",
            "mrr",
            "hit_rate",
        ):
            fields[
                f"{name}_at_{k}"
            ] = None

    if (
        not ground_truth_complete
        or not relevant_available
    ):
        return fields

    fields[
        "quality_available"
    ] = True

    relevant = tuple(
        sorted(
            relevant_available
        )
    )

    for k in k_values:
        metrics = (
            evaluate_retrieval(
                retrieved_ids,
                relevant,
                k,
            )
        )

        fields[
            f"precision_at_{k}"
        ] = metrics.precision

        fields[
            f"recall_at_{k}"
        ] = metrics.recall

        fields[
            f"mrr_at_{k}"
        ] = (
            metrics.reciprocal_rank
        )

        fields[
            f"hit_rate_at_{k}"
        ] = metrics.hit_rate

    return fields


def _build_nested_subsets(
    chunks: list[
        dict[
            str,
            Any,
        ]
    ],
    fractions: Sequence[float],
    *,
    seed: int,
) -> dict[
    float,
    list[
        dict[
            str,
            Any,
        ]
    ],
]:
    """Build deterministic nested subsets while preserving original order."""
    indices = list(
        range(
            len(
                chunks
            )
        )
    )

    random.Random(
        seed
    ).shuffle(
        indices
    )

    subsets: dict[
        float,
        list[
            dict[
                str,
                Any,
            ]
        ],
    ] = {}

    for fraction in fractions:
        size = (
            len(
                chunks
            )
            if math.isclose(
                fraction,
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            else max(
                1,
                int(
                    len(
                        chunks
                    )
                    * fraction
                ),
            )
        )

        selected = set(
            indices[
                :size
            ]
        )

        subsets[
            fraction
        ] = [
            chunk
            for index, chunk
            in enumerate(
                chunks
            )
            if index
            in selected
        ]

    return subsets


def _load_ground_truth(
    path: Path,
    *,
    expected_count: int,
    valid_chunk_ids: set[str],
) -> tuple[
    str,
    tuple[
        EvaluationQuery,
        ...,
    ],
]:
    """Load pending or completed ground truth."""
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
            "Ground-truth root "
            "must be a mapping."
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
            "Ground-truth metadata "
            "must be a mapping."
        )

    if not isinstance(
        raw_queries,
        list,
    ):
        raise TypeError(
            "Ground-truth queries "
            "must be a list."
        )

    if (
        len(
            raw_queries
        )
        != expected_count
    ):
        raise ValueError(
            "Ground-truth query count "
            "does not match configuration."
        )

    status = metadata.get(
        "annotation_status"
    )

    if not isinstance(
        status,
        str,
    ):
        raise TypeError(
            "annotation_status "
            "must be a string."
        )

    if status not in {
        "pending",
        "complete",
    }:
        raise ValueError(
            "annotation_status must be "
            "'pending' or 'complete'."
        )

    queries: list[
        EvaluationQuery
    ] = []

    seen_ids: set[
        str
    ] = set()

    for item in raw_queries:
        if not isinstance(
            item,
            dict,
        ):
            raise TypeError(
                "Each ground-truth query "
                "must be a mapping."
            )

        query_id = (
            _required_string(
                item.get(
                    "query_id"
                ),
                "query_id",
            )
        )

        if query_id in seen_ids:
            raise ValueError(
                "Duplicate query_id: "
                f"{query_id}"
            )

        seen_ids.add(
            query_id
        )

        raw_relevant = item.get(
            "relevant_chunks"
        )

        if not isinstance(
            raw_relevant,
            list,
        ):
            raise TypeError(
                "relevant_chunks "
                "must be a list."
            )

        relevant = tuple(
            _required_string(
                value,
                "relevant chunk ID",
            )
            for value
            in raw_relevant
        )

        if (
            len(
                relevant
            )
            != len(
                set(
                    relevant
                )
            )
        ):
            raise ValueError(
                "Duplicate relevant chunks "
                f"for {query_id}."
            )

        if (
            set(
                relevant
            )
            - valid_chunk_ids
        ):
            raise ValueError(
                "Unknown relevant chunks "
                f"for {query_id}."
            )

        if (
            status == "complete"
            and not relevant
        ):
            raise ValueError(
                "Completed ground truth "
                "has no relevant chunks "
                f"for {query_id}."
            )

        queries.append(
            EvaluationQuery(
                query_id=(
                    query_id
                ),
                category=(
                    _required_string(
                        item.get(
                            "category"
                        ),
                        "category",
                    )
                ),
                question=(
                    _required_string(
                        item.get(
                            "question"
                        ),
                        "question",
                    )
                ),
                relevant_chunks=(
                    relevant
                ),
            )
        )

    return (
        status,
        tuple(
            queries
        ),
    )


def _summarize(
    rows: Sequence[
        Mapping[
            str,
            str,
        ]
    ],
    k_values: Sequence[int],
) -> list[
    dict[
        str,
        object,
    ]
]:
    """Create long-form summary statistics."""
    groups: dict[
        tuple[
            str,
            str,
            str,
            str,
        ],
        list[
            Mapping[
                str,
                str,
            ]
        ],
    ] = defaultdict(
        list
    )

    for row in rows:
        key = (
            row[
                "algorithm"
            ],
            row[
                "corpus_fraction"
            ],
            row[
                "corpus_chunks"
            ],
            row[
                "corpus_fingerprint"
            ],
        )

        groups[
            key
        ].append(
            row
        )

    quality_metrics = tuple(
        f"{name}_at_{k}"
        for k in k_values
        for name in (
            "precision",
            "recall",
            "mrr",
            "hit_rate",
        )
    )

    output: list[
        dict[
            str,
            object,
        ]
    ] = []

    for key, group in sorted(
        groups.items()
    ):
        for metric in (
            PERFORMANCE_METRICS
        ):
            _append_statistics(
                output,
                key,
                metric,
                _values(
                    group,
                    metric,
                ),
            )

        for metric in (
            SCENARIO_METRICS
        ):
            _append_statistics(
                output,
                key,
                metric,
                _values(
                    group[
                        :1
                    ],
                    metric,
                ),
            )

        first_repetition = [
            row
            for row in group
            if (
                row[
                    "repetition"
                ]
                == "1"
                and row[
                    "quality_available"
                ].lower()
                == "true"
            )
        ]

        for metric in (
            quality_metrics
        ):
            _append_statistics(
                output,
                key,
                metric,
                _values(
                    first_repetition,
                    metric,
                ),
            )

    return output


def _append_statistics(
    output: list[
        dict[
            str,
            object,
        ]
    ],
    key: tuple[
        str,
        str,
        str,
        str,
    ],
    metric: str,
    values: Sequence[float],
) -> None:
    """Append one descriptive-statistics row."""
    if not values:
        return

    stats = (
        calculate_statistics(
            values
        )
    )

    (
        algorithm,
        fraction,
        chunks,
        fingerprint,
    ) = key

    output.append(
        {
            "algorithm": (
                algorithm
            ),
            "corpus_fraction": (
                fraction
            ),
            "corpus_chunks": (
                chunks
            ),
            "corpus_fingerprint": (
                fingerprint
            ),
            "metric": metric,
            "count": stats.count,
            "mean": stats.mean,
            "median": stats.median,
            "std": stats.std,
            "minimum": stats.minimum,
            "maximum": stats.maximum,
        }
    )


def _raw_fieldnames(
    k_values: Sequence[int],
) -> list[str]:
    """Return the raw-results schema."""
    fields = [
        "run_id",
        "algorithm",
        "corpus_fraction",
        "corpus_chunks",
        "full_corpus_chunks",
        "corpus_fingerprint",
        "query_id",
        "category",
        "question",
        "repetition",
        "retrieval_top_k",
        "warmup_repetitions",
        "random_seed",
        "retriever_setup_time_ns",
        "total_time_ns",
        "retrieval_time_ns",
        "sorting_time_ns",
        "index_build_time_ns",
        "peak_memory_bytes",
        "peak_memory_mb",
        "memory_method",
        "comparisons",
        "chunks_scored",
        "candidates_found",
        "result_count",
        "retrieved_chunk_ids",
        "quality_available",
        "relevant_total_full",
        "relevant_total_available",
        "index_source",
        "persistence_status",
        "index_load_time_ns",
        "query_embedding_time_ns",
        "faiss_search_time_ns",
    ]

    for k in k_values:
        for name in (
            "precision",
            "recall",
            "mrr",
            "hit_rate",
        ):
            fields.append(
                f"{name}_at_{k}"
            )

    return fields


def _write_summary(
    path: Path,
    rows: Sequence[
        Mapping[
            str,
            object,
        ]
    ],
) -> None:
    """Write summary.csv."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = (
        "algorithm",
        "corpus_fraction",
        "corpus_chunks",
        "corpus_fingerprint",
        "metric",
        "count",
        "mean",
        "median",
        "std",
        "minimum",
        "maximum",
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


def _load_config() -> dict[
    str,
    Any,
]:
    """Load experiments.yaml."""
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
            "Experiment configuration "
            "root must be a mapping."
        )

    experiments = data.get(
        "experiments"
    )

    if not isinstance(
        experiments,
        dict,
    ):
        raise TypeError(
            "experiments must "
            "be a mapping."
        )

    return data


def _read_csv(
    path: Path,
) -> list[
    dict[
        str,
        str,
    ]
]:
    """Read a UTF-8 CSV file."""
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        return [
            dict(
                row
            )
            for row
            in csv.DictReader(
                file
            )
        ]


def _read_raw_results(
    path: Path,
    *,
    expected_fieldnames: Sequence[str],
) -> list[
    dict[
        str,
        str,
    ]
]:
    """Read raw results and reject malformed or truncated CSV rows."""
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(
            file
        )

        fieldnames = reader.fieldnames

        if fieldnames != list(
            expected_fieldnames
        ):
            raise ValueError(
                "Raw-results schema does not match the current runner. "
                f"Expected {list(expected_fieldnames)}, "
                f"found {fieldnames}."
            )

        rows: list[
            dict[
                str,
                str,
            ]
        ] = []

        for line_number, row in enumerate(
            reader,
            start=2,
        ):
            if None in row:
                raise ValueError(
                    "Raw-results row contains unexpected extra fields "
                    f"at CSV line {line_number}."
                )

            if any(
                value is None
                for value in row.values()
            ):
                raise ValueError(
                    "Raw-results row is truncated or incomplete "
                    f"at CSV line {line_number}."
                )

            rows.append(
                {
                    str(key): str(value)
                    for key, value
                    in row.items()
                }
            )

    return rows


def _build_run_id(
    algorithm: str,
    fraction: float,
    query_id: str,
    repetition: int,
) -> str:
    """Build the canonical identity of one measured retrieval run."""
    return (
        f"{algorithm}|"
        f"{fraction:.6f}|"
        f"{query_id}|"
        f"{repetition}"
    )


def _run_id_fraction(
    run_id: str,
) -> str:
    """Return the canonical fraction component from a run ID."""
    parts = run_id.split(
        "|"
    )

    if len(parts) != 4:
        raise ValueError(
            f"Invalid run_id: {run_id!r}"
        )

    return parts[1]


def _expected_run_ids(
    *,
    algorithms: Sequence[str],
    fractions: Sequence[float],
    queries: Sequence[EvaluationQuery],
    repetitions: int,
) -> set[str]:
    """Build the complete expected measured-run matrix."""
    return {
        _build_run_id(
            algorithm,
            fraction,
            query.query_id,
            repetition,
        )
        for fraction in fractions
        for query in queries
        for repetition in range(
            1,
            repetitions + 1,
        )
        for algorithm in algorithms
    }


def _validate_existing_rows(
    rows: Sequence[
        Mapping[
            str,
            str,
        ]
    ],
    *,
    expected_run_ids: set[str],
    algorithms: Sequence[str],
    fractions: Sequence[float],
    queries: Sequence[EvaluationQuery],
    repetitions: int,
    subset_metadata: Mapping[
        float,
        tuple[
            int,
            str,
        ],
    ],
    full_corpus_chunks: int,
    top_k: int,
    warmup_repetitions: int,
    seed: int,
) -> set[str]:
    """Validate existing raw rows before resuming or finalizing."""
    query_map = {
        query.query_id: query
        for query in queries
    }

    algorithm_set = set(
        algorithms
    )

    completed: set[
        str
    ] = set()

    for row_number, row in enumerate(
        rows,
        start=2,
    ):
        run_id = row.get(
            "run_id",
            "",
        ).strip()

        if not run_id:
            raise ValueError(
                f"Raw-results row {row_number} has an empty run_id."
            )

        if run_id in completed:
            raise ValueError(
                f"Duplicate run_id in raw results: {run_id}"
            )

        if run_id not in expected_run_ids:
            raise ValueError(
                "Raw results contain a run that is not part of "
                f"the current experiment matrix: {run_id}"
            )

        algorithm = row[
            "algorithm"
        ].strip()

        if algorithm not in algorithm_set:
            raise ValueError(
                f"Unknown algorithm in raw row {row_number}: "
                f"{algorithm!r}"
            )

        query_id = row[
            "query_id"
        ].strip()

        query = query_map.get(
            query_id
        )

        if query is None:
            raise ValueError(
                f"Unknown query_id in raw row {row_number}: "
                f"{query_id!r}"
            )

        try:
            fraction = float(
                row[
                    "corpus_fraction"
                ]
            )

            repetition = int(
                row[
                    "repetition"
                ]
            )
        except ValueError as exc:
            raise ValueError(
                f"Invalid numeric run identity at row {row_number}."
            ) from exc

        matched_fraction = next(
            (
                value
                for value in fractions
                if math.isclose(
                    fraction,
                    value,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ),
            None,
        )

        if matched_fraction is None:
            raise ValueError(
                f"Unexpected corpus fraction at row {row_number}: "
                f"{fraction}"
            )

        if not 1 <= repetition <= repetitions:
            raise ValueError(
                f"Unexpected repetition at row {row_number}: "
                f"{repetition}"
            )

        canonical_run_id = (
            _build_run_id(
                algorithm,
                matched_fraction,
                query_id,
                repetition,
            )
        )

        if run_id != canonical_run_id:
            raise ValueError(
                f"run_id metadata mismatch at row {row_number}: "
                f"{run_id!r} != {canonical_run_id!r}"
            )

        expected_chunks, expected_fingerprint = (
            subset_metadata[
                matched_fraction
            ]
        )

        _require_int_field(
            row,
            "corpus_chunks",
            expected_chunks,
            row_number,
        )

        _require_int_field(
            row,
            "full_corpus_chunks",
            full_corpus_chunks,
            row_number,
        )

        _require_int_field(
            row,
            "retrieval_top_k",
            top_k,
            row_number,
        )

        _require_int_field(
            row,
            "warmup_repetitions",
            warmup_repetitions,
            row_number,
        )

        _require_int_field(
            row,
            "random_seed",
            seed,
            row_number,
        )

        if row[
            "corpus_fingerprint"
        ].strip() != expected_fingerprint:
            raise ValueError(
                "Corpus fingerprint mismatch in raw results "
                f"at row {row_number}."
            )

        if row[
            "category"
        ] != query.category:
            raise ValueError(
                f"Query category mismatch at row {row_number}."
            )

        if row[
            "question"
        ] != query.question:
            raise ValueError(
                f"Query text mismatch at row {row_number}."
            )

        try:
            retrieved_ids = json.loads(
                row[
                    "retrieved_chunk_ids"
                ]
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Invalid retrieved_chunk_ids JSON "
                f"at row {row_number}."
            ) from exc

        if not isinstance(
            retrieved_ids,
            list,
        ):
            raise ValueError(
                "retrieved_chunk_ids must decode to a list "
                f"at row {row_number}."
            )

        _require_int_field(
            row,
            "result_count",
            len(
                retrieved_ids
            ),
            row_number,
        )

        completed.add(
            run_id
        )

    return completed


def _require_int_field(
    row: Mapping[
        str,
        str,
    ],
    field: str,
    expected: int,
    row_number: int,
) -> None:
    """Require one integer CSV field to match an expected value."""
    try:
        actual = int(
            row[
                field
            ]
        )
    except ValueError as exc:
        raise ValueError(
            f"Invalid integer field {field!r} at row {row_number}."
        ) from exc

    if actual != expected:
        raise ValueError(
            f"Field {field!r} mismatch at row {row_number}: "
            f"expected {expected}, found {actual}."
        )


def _checkpoint_path(
    raw_output: Path,
) -> Path:
    """Return the checkpoint path associated with raw results."""
    return raw_output.with_name(
        raw_output.name
        + ".checkpoint.json"
    )


def _checkpoint_temp_path(
    checkpoint_path: Path,
) -> Path:
    """Return the temporary path used for atomic checkpoint writes."""
    return checkpoint_path.with_name(
        checkpoint_path.name
        + ".tmp"
    )


def _build_signature_payload(
    *,
    algorithms: Sequence[str],
    fractions: Sequence[float],
    queries: Sequence[EvaluationQuery],
    repetitions: int,
    warmup_repetitions: int,
    k_values: Sequence[int],
    top_k: int,
    seed: int,
    full_corpus_chunks: int,
    full_corpus_fingerprint: str,
    ground_truth_path: Path,
    annotation_status: str,
    smoke: bool,
    memory_profiling: bool,
    raw_fieldnames: Sequence[str],
) -> dict[
    str,
    object,
]:
    """Build the scientific compatibility signature payload."""
    return {
        "checkpoint_schema_version": (
            CHECKPOINT_SCHEMA_VERSION
        ),
        "algorithms": list(
            algorithms
        ),
        "corpus_fractions": [
            f"{value:.12g}"
            for value in fractions
        ],
        "query_ids": [
            query.query_id
            for query in queries
        ],
        "repetitions": repetitions,
        "warmup_repetitions": (
            warmup_repetitions
        ),
        "k_values": list(
            k_values
        ),
        "retrieval_top_k": top_k,
        "random_seed": seed,
        "full_corpus_chunks": (
            full_corpus_chunks
        ),
        "full_corpus_fingerprint": (
            full_corpus_fingerprint
        ),
        "ground_truth_sha256": (
            _sha256_file(
                ground_truth_path
            )
        ),
        "annotation_status": (
            annotation_status
        ),
        "smoke": smoke,
        "memory_profiling": (
            memory_profiling
        ),
        "raw_results_schema": list(
            raw_fieldnames
        ),
        "environment": {
            "python": (
                platform.python_version()
            ),
            "system": (
                platform.system()
            ),
            "release": (
                platform.release()
            ),
            "machine": (
                platform.machine()
            ),
            "processor": (
                platform.processor()
            ),
        },
    }


def _calculate_signature(
    payload: Mapping[
        str,
        object,
    ],
) -> str:
    """Return a deterministic SHA-256 signature for experiment settings."""
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
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
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:
        for block in iter(
            lambda: file.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                block
            )

    return digest.hexdigest()


def _write_checkpoint(
    path: Path,
    *,
    signature: str,
    signature_payload: Mapping[
        str,
        object,
    ],
    expected_runs: int,
    completed_run_ids: set[str],
    status: str,
    raw_output: Path,
    summary_output: Path,
) -> None:
    """Atomically persist resume metadata after durable raw-result writes."""
    if status not in {
        "in_progress",
        "complete",
    }:
        raise ValueError(
            f"Invalid checkpoint status: {status!r}"
        )

    if status == "complete" and (
        len(
            completed_run_ids
        )
        != expected_runs
    ):
        raise ValueError(
            "A complete checkpoint must contain every expected run."
        )

    data = {
        "schema_version": (
            CHECKPOINT_SCHEMA_VERSION
        ),
        "signature": signature,
        "signature_payload": dict(
            signature_payload
        ),
        "status": status,
        "expected_runs": (
            expected_runs
        ),
        "completed_runs": (
            len(
                completed_run_ids
            )
        ),
        "completed_run_ids": sorted(
            completed_run_ids
        ),
        "raw_results": str(
            raw_output
        ),
        "summary_results": str(
            summary_output
        ),
    }

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        _checkpoint_temp_path(
            path
        )
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )

        file.write(
            "\n"
        )

        _flush_file(
            file
        )

    temporary_path.replace(
        path
    )


def _load_checkpoint(
    path: Path,
) -> dict[
    str,
    Any,
]:
    """Load a checkpoint JSON object."""
    try:
        data = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Checkpoint JSON is invalid: {path}"
        ) from exc

    if not isinstance(
        data,
        dict,
    ):
        raise TypeError(
            "Checkpoint root must be an object."
        )

    return data


def _validate_checkpoint(
    checkpoint: Mapping[
        str,
        Any,
    ],
    *,
    signature: str,
    expected_runs: int,
    raw_completed_run_ids: set[str],
) -> None:
    """Validate checkpoint compatibility with raw results and current settings."""
    if checkpoint.get(
        "schema_version"
    ) != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            "Checkpoint schema version is incompatible."
        )

    if checkpoint.get(
        "signature"
    ) != signature:
        raise ValueError(
            "Checkpoint signature does not match the current "
            "experiment configuration, corpus, ground truth, "
            "memory mode, smoke mode, or execution environment. "
            "Use --force for a new experiment."
        )

    if checkpoint.get(
        "expected_runs"
    ) != expected_runs:
        raise ValueError(
            "Checkpoint expected-run count is incompatible."
        )

    raw_checkpoint_ids = checkpoint.get(
        "completed_run_ids"
    )

    if not isinstance(
        raw_checkpoint_ids,
        list,
    ) or not all(
        isinstance(
            value,
            str,
        )
        for value in raw_checkpoint_ids
    ):
        raise TypeError(
            "Checkpoint completed_run_ids must be a list of strings."
        )

    checkpoint_ids = set(
        raw_checkpoint_ids
    )

    if len(
        checkpoint_ids
    ) != len(
        raw_checkpoint_ids
    ):
        raise ValueError(
            "Checkpoint contains duplicate completed run IDs."
        )

    completed_runs = checkpoint.get(
        "completed_runs"
    )

    if completed_runs != len(
        checkpoint_ids
    ):
        raise ValueError(
            "Checkpoint completed_runs does not match "
            "completed_run_ids."
        )

    if not checkpoint_ids.issubset(
        raw_completed_run_ids
    ):
        raise ValueError(
            "Checkpoint claims completed runs that are missing "
            "from raw results. Raw results are the source of truth."
        )

    status = checkpoint.get(
        "status"
    )

    if status not in {
        "in_progress",
        "complete",
    }:
        raise ValueError(
            "Checkpoint status is invalid."
        )

    if status == "complete" and (
        len(
            raw_completed_run_ids
        )
        != expected_runs
    ):
        raise ValueError(
            "Checkpoint is marked complete but raw results "
            "do not contain every expected run."
        )


def _ensure_trailing_newline(
    path: Path,
) -> None:
    """Ensure appending a CSV row cannot concatenate with the previous row."""
    if path.stat().st_size == 0:
        return

    with path.open(
        "rb+"
    ) as file:
        file.seek(
            -1,
            os.SEEK_END,
        )

        last_byte = file.read(
            1
        )

        if last_byte not in {
            b"\n",
            b"\r",
        }:
            file.seek(
                0,
                os.SEEK_END,
            )

            file.write(
                b"\n"
            )

            file.flush()
            os.fsync(
                file.fileno()
            )


def _flush_file(
    file: Any,
) -> None:
    """Flush Python and OS buffers for a durable checkpoint boundary."""
    file.flush()
    os.fsync(
        file.fileno()
    )


def _values(
    rows: Sequence[
        Mapping[
            str,
            str,
        ]
    ],
    field: str,
) -> list[float]:
    return [
        float(
            row[
                field
            ]
        )
        for row in rows
        if row.get(
            field,
            "",
        ).strip()
    ]


def _metadata_value(
    metadata: Mapping[
        str,
        Any,
    ],
    key: str,
) -> object | None:
    value = metadata.get(
        key
    )

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        return value

    return None


def _required_string(
    value: object,
    field: str,
) -> str:
    if not isinstance(
        value,
        str,
    ):
        raise TypeError(
            f"{field} must be a string."
        )

    normalized = value.strip()

    if not normalized:
        raise ValueError(
            f"{field} cannot be empty."
        )

    return normalized


def _project_path(
    value: object,
) -> Path:
    if not isinstance(
        value,
        str,
    ):
        raise TypeError(
            "Configured path "
            "must be a string."
        )

    path = Path(
        value
    )

    if path.is_absolute():
        return path

    return (
        PROJECT_ROOT
        / path
    ).resolve()


def _validate_experiment_config(
    *,
    algorithms: Sequence[str],
    fractions: Sequence[float],
    k_values: Sequence[int],
    repetitions: int,
) -> None:
    if (
        not algorithms
        or len(
            algorithms
        )
        != len(
            set(
                algorithms
            )
        )
        or (
            set(
                algorithms
            )
            - set(
                SUPPORTED_ALGORITHMS
            )
        )
    ):
        raise ValueError(
            "Invalid experiment algorithms."
        )

    if (
        not fractions
        or len(
            fractions
        )
        != len(
            set(
                fractions
            )
        )
        or any(
            not 0.0
            < value
            <= 1.0
            for value
            in fractions
        )
    ):
        raise ValueError(
            "Invalid corpus fractions."
        )

    if (
        not k_values
        or len(
            k_values
        )
        != len(
            set(
                k_values
            )
        )
        or any(
            value <= 0
            for value
            in k_values
        )
    ):
        raise ValueError(
            "Invalid k values."
        )

    if repetitions <= 0:
        raise ValueError(
            "repetitions must "
            "be greater than zero."
        )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
