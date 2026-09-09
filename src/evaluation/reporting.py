"""Reporting utilities for FastContext experimental results."""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.evaluation.statistics import calculate_statistics

CsvRow = dict[str, str]
ReportRow = dict[str, object]

PERFORMANCE_TABLE_FIELDS = (
    "algorithm",
    "corpus_fraction",
    "corpus_chunks",
    "sample_count",
    "retrieval_time_ms_mean",
    "retrieval_time_ms_median",
    "retrieval_time_ms_std",
    "retrieval_time_ms_min",
    "retrieval_time_ms_max",
    "total_time_ms_mean",
    "total_time_ms_median",
    "total_time_ms_std",
    "total_time_ms_min",
    "total_time_ms_max",
    "sorting_time_ms_mean",
    "sorting_time_ms_median",
    "sorting_time_ms_std",
    "sorting_time_ms_min",
    "sorting_time_ms_max",
    "peak_memory_mb_mean",
    "peak_memory_mb_median",
    "peak_memory_mb_std",
    "peak_memory_mb_min",
    "peak_memory_mb_max",
    "comparisons_mean",
    "comparisons_median",
    "comparisons_std",
    "comparisons_min",
    "comparisons_max",
    "chunks_scored_mean",
    "chunks_scored_median",
    "chunks_scored_std",
    "chunks_scored_min",
    "chunks_scored_max",
    "candidates_found_mean",
    "candidates_found_median",
    "candidates_found_std",
    "candidates_found_min",
    "candidates_found_max",
)

QUALITY_TABLE_FIELDS = (
    "algorithm",
    "corpus_fraction",
    "corpus_chunks",
    "k",
    "query_count",
    "precision_mean",
    "precision_median",
    "precision_std",
    "precision_min",
    "precision_max",
    "recall_mean",
    "recall_median",
    "recall_std",
    "recall_min",
    "recall_max",
    "mrr_mean",
    "mrr_median",
    "mrr_std",
    "mrr_min",
    "mrr_max",
    "hit_rate_mean",
    "hit_rate_median",
    "hit_rate_std",
    "hit_rate_min",
    "hit_rate_max",
)

PERFORMANCE_METRICS = (
    (
        "retrieval_time_ns",
        "retrieval_time_ms",
        1 / 1_000_000,
    ),
    (
        "total_time_ns",
        "total_time_ms",
        1 / 1_000_000,
    ),
    (
        "sorting_time_ns",
        "sorting_time_ms",
        1 / 1_000_000,
    ),
    (
        "peak_memory_mb",
        "peak_memory_mb",
        1.0,
    ),
    (
        "comparisons",
        "comparisons",
        1.0,
    ),
    (
        "chunks_scored",
        "chunks_scored",
        1.0,
    ),
    (
        "candidates_found",
        "candidates_found",
        1.0,
    ),
)


def load_results_csv(
    path: Path,
) -> list[CsvRow]:
    """Load raw experiment results."""
    if not path.is_file():
        raise FileNotFoundError(
            "Experiment results file was not found."
        )

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(
            file
        )

        if reader.fieldnames is None:
            raise ValueError(
                "Experiment results CSV has no header."
            )

        rows = [
            dict(row)
            for row in reader
        ]

    if not rows:
        raise ValueError(
            "Experiment results CSV is empty."
        )

    return rows


def aggregate_performance(
    rows: Sequence[
        Mapping[str, str]
    ],
) -> list[ReportRow]:
    """Aggregate performance results by algorithm and corpus size."""
    groups: dict[
        tuple[str, float, int],
        list[
            Mapping[str, str]
        ],
    ] = defaultdict(
        list
    )

    for row in rows:
        key = (
            _required_string(
                row,
                "algorithm",
            ),
            _required_float(
                row,
                "corpus_fraction",
            ),
            _required_int(
                row,
                "corpus_chunks",
            ),
        )

        groups[
            key
        ].append(
            row
        )

    output: list[
        ReportRow
    ] = []

    for key, group in sorted(
        groups.items(),
        key=lambda item: (
            item[0][1],
            item[0][0],
        ),
    ):
        (
            algorithm,
            fraction,
            corpus_chunks,
        ) = key

        report_row: ReportRow = {
            "algorithm": algorithm,
            "corpus_fraction": fraction,
            "corpus_chunks": corpus_chunks,
            "sample_count": len(
                group
            ),
        }

        for (
            source_field,
            output_prefix,
            scale,
        ) in PERFORMANCE_METRICS:
            values = [
                value * scale
                for value in _optional_values(
                    group,
                    source_field,
                )
            ]

            _add_statistics(
                report_row,
                output_prefix,
                values,
            )

        output.append(
            report_row
        )

    return output


def aggregate_quality(
    rows: Sequence[
        Mapping[str, str]
    ],
    *,
    k_values: Sequence[int] = (
        1,
        3,
        5,
        10,
    ),
) -> list[ReportRow]:
    """Aggregate query-level quality metrics without counting repetitions twice."""
    first_repetition_rows = [
        row
        for row in rows
        if (
            row.get(
                "repetition",
                "",
            ).strip()
            == "1"
            and row.get(
                "quality_available",
                "",
            ).strip().lower()
            == "true"
        )
    ]

    if not first_repetition_rows:
        return []

    groups: dict[
        tuple[str, float, int],
        list[
            Mapping[str, str]
        ],
    ] = defaultdict(
        list
    )

    for row in first_repetition_rows:
        key = (
            _required_string(
                row,
                "algorithm",
            ),
            _required_float(
                row,
                "corpus_fraction",
            ),
            _required_int(
                row,
                "corpus_chunks",
            ),
        )

        groups[
            key
        ].append(
            row
        )

    output: list[
        ReportRow
    ] = []

    for key, group in sorted(
        groups.items(),
        key=lambda item: (
            item[0][1],
            item[0][0],
        ),
    ):
        (
            algorithm,
            fraction,
            corpus_chunks,
        ) = key

        query_ids = {
            _required_string(
                row,
                "query_id",
            )
            for row in group
        }

        for k in k_values:
            report_row: ReportRow = {
                "algorithm": algorithm,
                "corpus_fraction": fraction,
                "corpus_chunks": corpus_chunks,
                "k": k,
                "query_count": len(
                    query_ids
                ),
            }

            for metric in (
                "precision",
                "recall",
                "mrr",
                "hit_rate",
            ):
                values = (
                    _optional_values(
                        group,
                        f"{metric}_at_{k}",
                    )
                )

                _add_statistics(
                    report_row,
                    metric,
                    values,
                )

            output.append(
                report_row
            )

    return output


def write_csv_table(
    path: Path,
    rows: Sequence[
        Mapping[str, object]
    ],
    *,
    fieldnames: Sequence[str],
) -> None:
    """Write a reproducible report table."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                fieldnames
            ),
            extrasaction="ignore",
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


def full_corpus_rows(
    rows: Sequence[
        Mapping[str, object]
    ],
) -> list[
    Mapping[str, object]
]:
    """Return rows corresponding to the largest corpus size."""
    if not rows:
        return []

    corpus_sizes = [
        _object_int(
            row.get(
                "corpus_chunks"
            )
        )
        for row in rows
    ]

    maximum = max(
        corpus_sizes
    )

    return [
        row
        for row in rows
        if (
            _object_int(
                row.get(
                    "corpus_chunks"
                )
            )
            == maximum
        )
    ]


def numeric_report_value(
    row: Mapping[
        str,
        object,
    ],
    field: str,
) -> float | None:
    """Read one optional numeric report value."""
    value = row.get(
        field
    )

    if value is None:
        return None

    if isinstance(
        value,
        bool,
    ):
        return None

    if isinstance(
        value,
        (int, float),
    ):
        return float(
            value
        )

    return None


def _add_statistics(
    row: ReportRow,
    prefix: str,
    values: Sequence[float],
) -> None:
    """Add descriptive statistics to one report row."""
    suffixes = (
        "mean",
        "median",
        "std",
        "min",
        "max",
    )

    if not values:
        for suffix in suffixes:
            row[
                f"{prefix}_{suffix}"
            ] = None

        return

    stats = (
        calculate_statistics(
            values
        )
    )

    row[
        f"{prefix}_mean"
    ] = stats.mean

    row[
        f"{prefix}_median"
    ] = stats.median

    row[
        f"{prefix}_std"
    ] = stats.std

    row[
        f"{prefix}_min"
    ] = stats.minimum

    row[
        f"{prefix}_max"
    ] = stats.maximum


def _optional_values(
    rows: Sequence[
        Mapping[str, str]
    ],
    field: str,
) -> list[float]:
    """Read available numeric values from CSV rows."""
    values: list[
        float
    ] = []

    for row in rows:
        raw = row.get(
            field,
            "",
        ).strip()

        if not raw:
            continue

        values.append(
            float(
                raw
            )
        )

    return values


def _required_string(
    row: Mapping[
        str,
        str,
    ],
    field: str,
) -> str:
    """Read one required CSV string."""
    value = row.get(
        field,
        "",
    ).strip()

    if not value:
        raise ValueError(
            "Required experiment field is empty."
        )

    return value


def _required_float(
    row: Mapping[
        str,
        str,
    ],
    field: str,
) -> float:
    """Read one required CSV float."""
    return float(
        _required_string(
            row,
            field,
        )
    )


def _required_int(
    row: Mapping[
        str,
        str,
    ],
    field: str,
) -> int:
    """Read one required CSV integer."""
    return int(
        _required_string(
            row,
            field,
        )
    )


def _object_int(
    value: Any,
) -> int:
    """Normalize one report value as an integer."""
    if isinstance(
        value,
        bool,
    ) or not isinstance(
        value,
        (int, float),
    ):
        raise TypeError(
            "Expected a numeric corpus size."
        )

    return int(
        value
    )