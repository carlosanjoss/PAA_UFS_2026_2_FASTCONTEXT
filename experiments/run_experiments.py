"""Run reproducible FastContext retrieval experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
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

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Overwrite existing result files."
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

    elif (
        raw_output.exists()
        or summary_output.exists()
    ):
        raise FileExistsError(
            "Experiment output already exists. "
            "Use --force to overwrite it."
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

    top_k = max(
        k_values
    )

    expected_runs = (
        len(
            algorithms
        )
        * len(
            fractions
        )
        * len(
            queries
        )
        * repetitions
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

    print()

    raw_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = (
        _raw_fieldnames(
            k_values
        )
    )

    completed = 0

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

            fingerprint = (
                calculate_corpus_fingerprint(
                    subset
                )
            )

            is_full_corpus = (
                len(
                    subset
                )
                == len(
                    chunks
                )
                and fingerprint
                == full_fingerprint
            )

            print(
                f"Corpus {fraction:.2f}: "
                f"{len(subset)} chunks, "
                f"{fingerprint[:12]}..."
            )

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
                for algorithm in algorithms:
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
                            "run_id": (
                                f"{algorithm}|"
                                f"{fraction:.6f}|"
                                f"{query.query_id}|"
                                f"{repetition}"
                            ),
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

                        file.flush()

                        completed += 1

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

    rows = _read_csv(
        raw_output
    )

    summary = _summarize(
        rows,
        k_values,
    )

    _write_summary(
        summary_output,
        summary,
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
        for row
        in rows
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