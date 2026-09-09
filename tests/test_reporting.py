"""Tests for FastContext experiment reporting."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.evaluation.reporting import (
    PERFORMANCE_TABLE_FIELDS,
    aggregate_performance,
    aggregate_quality,
    full_corpus_rows,
    load_results_csv,
    write_csv_table,
)


def test_aggregate_performance() -> None:
    rows = [
        {
            "algorithm": "linear",
            "corpus_fraction": "1.0",
            "corpus_chunks": "100",
            "retrieval_time_ns": "1000000",
            "total_time_ns": "2000000",
            "sorting_time_ns": "500000",
            "peak_memory_mb": "1.0",
            "comparisons": "10",
            "chunks_scored": "100",
            "candidates_found": "50",
        },
        {
            "algorithm": "linear",
            "corpus_fraction": "1.0",
            "corpus_chunks": "100",
            "retrieval_time_ns": "3000000",
            "total_time_ns": "4000000",
            "sorting_time_ns": "1500000",
            "peak_memory_mb": "3.0",
            "comparisons": "30",
            "chunks_scored": "100",
            "candidates_found": "70",
        },
    ]

    result = (
        aggregate_performance(
            rows
        )
    )

    assert len(
        result
    ) == 1

    row = result[
        0
    ]

    assert (
        row[
            "sample_count"
        ]
        == 2
    )

    assert (
        row[
            "retrieval_time_ms_mean"
        ]
        == pytest.approx(
            2.0
        )
    )

    assert (
        row[
            "total_time_ms_mean"
        ]
        == pytest.approx(
            3.0
        )
    )

    assert (
        row[
            "peak_memory_mb_mean"
        ]
        == pytest.approx(
            2.0
        )
    )

    assert (
        row[
            "comparisons_mean"
        ]
        == pytest.approx(
            20.0
        )
    )


def test_aggregate_performance_handles_missing_values() -> None:
    rows = [
        {
            "algorithm": "semantic",
            "corpus_fraction": "1.0",
            "corpus_chunks": "100",
            "retrieval_time_ns": "1000000",
            "total_time_ns": "2000000",
            "sorting_time_ns": "",
            "peak_memory_mb": "",
            "comparisons": "",
            "chunks_scored": "100",
            "candidates_found": "10",
        }
    ]

    result = (
        aggregate_performance(
            rows
        )
    )

    assert (
        result[
            0
        ][
            "comparisons_mean"
        ]
        is None
    )

    assert (
        result[
            0
        ][
            "peak_memory_mb_mean"
        ]
        is None
    )


def test_quality_is_empty_when_ground_truth_is_pending() -> None:
    rows = [
        {
            "algorithm": "linear",
            "corpus_fraction": "1.0",
            "corpus_chunks": "100",
            "query_id": "q01",
            "repetition": "1",
            "quality_available": "False",
        }
    ]

    assert (
        aggregate_quality(
            rows
        )
        == []
    )


def test_quality_uses_only_first_repetition() -> None:
    rows = [
        {
            "algorithm": "semantic",
            "corpus_fraction": "1.0",
            "corpus_chunks": "100",
            "query_id": "q01",
            "repetition": "1",
            "quality_available": "True",
            "precision_at_1": "1.0",
            "recall_at_1": "0.5",
            "mrr_at_1": "1.0",
            "hit_rate_at_1": "1.0",
        },
        {
            "algorithm": "semantic",
            "corpus_fraction": "1.0",
            "corpus_chunks": "100",
            "query_id": "q01",
            "repetition": "2",
            "quality_available": "True",
            "precision_at_1": "0.0",
            "recall_at_1": "0.0",
            "mrr_at_1": "0.0",
            "hit_rate_at_1": "0.0",
        },
    ]

    result = (
        aggregate_quality(
            rows,
            k_values=(
                1,
            ),
        )
    )

    assert len(
        result
    ) == 1

    assert (
        result[
            0
        ][
            "query_count"
        ]
        == 1
    )

    assert (
        result[
            0
        ][
            "precision_mean"
        ]
        == 1.0
    )


def test_full_corpus_rows_returns_largest_corpus() -> None:
    rows = [
        {
            "algorithm": "linear",
            "corpus_chunks": 25,
        },
        {
            "algorithm": "linear",
            "corpus_chunks": 100,
        },
        {
            "algorithm": "semantic",
            "corpus_chunks": 100,
        },
    ]

    result = (
        full_corpus_rows(
            rows
        )
    )

    assert len(
        result
    ) == 2

    assert all(
        row[
            "corpus_chunks"
        ]
        == 100
        for row in result
    )


def test_write_and_load_report_csv(
    tmp_path: Path,
) -> None:
    path = (
        tmp_path
        / "table.csv"
    )

    rows = [
        {
            "algorithm": "linear",
            "corpus_fraction": 1.0,
            "corpus_chunks": 100,
            "sample_count": 5,
        }
    ]

    write_csv_table(
        path,
        rows,
        fieldnames=(
            PERFORMANCE_TABLE_FIELDS
        ),
    )

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        saved = list(
            csv.DictReader(
                file
            )
        )

    assert len(
        saved
    ) == 1

    assert (
        saved[
            0
        ][
            "algorithm"
        ]
        == "linear"
    )


def test_load_results_csv_rejects_empty_file(
    tmp_path: Path,
) -> None:
    path = (
        tmp_path
        / "results.csv"
    )

    path.write_text(
        "algorithm,corpus_fraction\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="is empty",
    ):
        load_results_csv(
            path
        )