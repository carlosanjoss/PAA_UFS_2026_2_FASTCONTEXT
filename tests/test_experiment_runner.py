"""Tests for the FastContext experiment runner."""

from __future__ import annotations

import pytest

from experiments.run_experiments import (
    _build_nested_subsets,
    _quality_fields,
    _summarize,
)


def _chunks(
    count: int,
) -> list[
    dict[
        str,
        object,
    ]
]:
    return [
        {
            "chunk_id": (
                f"chunk-{index}"
            ),
            "content": (
                f"content {index}"
            ),
        }
        for index in range(
            count
        )
    ]


def test_nested_subsets_have_expected_sizes() -> None:
    chunks = _chunks(
        20
    )

    subsets = (
        _build_nested_subsets(
            chunks,
            (
                0.25,
                0.50,
                0.75,
                1.0,
            ),
            seed=42,
        )
    )

    assert len(
        subsets[
            0.25
        ]
    ) == 5

    assert len(
        subsets[
            0.50
        ]
    ) == 10

    assert len(
        subsets[
            0.75
        ]
    ) == 15

    assert len(
        subsets[
            1.0
        ]
    ) == 20


def test_nested_subsets_are_nested() -> None:
    chunks = _chunks(
        20
    )

    subsets = (
        _build_nested_subsets(
            chunks,
            (
                0.25,
                0.50,
                0.75,
                1.0,
            ),
            seed=42,
        )
    )

    ids_25 = {
        chunk[
            "chunk_id"
        ]
        for chunk
        in subsets[
            0.25
        ]
    }

    ids_50 = {
        chunk[
            "chunk_id"
        ]
        for chunk
        in subsets[
            0.50
        ]
    }

    ids_75 = {
        chunk[
            "chunk_id"
        ]
        for chunk
        in subsets[
            0.75
        ]
    }

    ids_100 = {
        chunk[
            "chunk_id"
        ]
        for chunk
        in subsets[
            1.0
        ]
    }

    assert ids_25 < ids_50
    assert ids_50 < ids_75
    assert ids_75 < ids_100


def test_full_subset_preserves_original_order() -> None:
    chunks = _chunks(
        20
    )

    subsets = (
        _build_nested_subsets(
            chunks,
            (
                1.0,
            ),
            seed=42,
        )
    )

    assert (
        subsets[
            1.0
        ]
        == chunks
    )


def test_quality_is_empty_while_ground_truth_is_pending() -> None:
    fields = (
        _quality_fields(
            retrieved_ids=(
                "chunk-a",
                "chunk-b",
            ),
            relevant_ids=(),
            available_ids={
                "chunk-a",
                "chunk-b",
            },
            k_values=(
                1,
                2,
            ),
            ground_truth_complete=False,
        )
    )

    assert (
        fields[
            "quality_available"
        ]
        is False
    )

    assert (
        fields[
            "precision_at_1"
        ]
        is None
    )

    assert (
        fields[
            "recall_at_2"
        ]
        is None
    )


def test_quality_uses_relevant_chunks_available_in_subset() -> None:
    fields = (
        _quality_fields(
            retrieved_ids=(
                "chunk-a",
                "chunk-b",
                "chunk-c",
            ),
            relevant_ids=(
                "chunk-b",
                "chunk-x",
            ),
            available_ids={
                "chunk-a",
                "chunk-b",
                "chunk-c",
            },
            k_values=(
                1,
                3,
            ),
            ground_truth_complete=True,
        )
    )

    assert (
        fields[
            "quality_available"
        ]
        is True
    )

    assert (
        fields[
            "relevant_total_full"
        ]
        == 2
    )

    assert (
        fields[
            "relevant_total_available"
        ]
        == 1
    )

    assert (
        fields[
            "precision_at_1"
        ]
        == 0.0
    )

    assert (
        fields[
            "recall_at_1"
        ]
        == 0.0
    )

    assert (
        fields[
            "precision_at_3"
        ]
        == pytest.approx(
            1 / 3
        )
    )

    assert (
        fields[
            "recall_at_3"
        ]
        == 1.0
    )

    assert (
        fields[
            "mrr_at_3"
        ]
        == pytest.approx(
            0.5
        )
    )

    assert (
        fields[
            "hit_rate_at_3"
        ]
        == 1.0
    )


def test_summary_uses_all_repetitions_for_performance() -> None:
    rows = [
        {
            "algorithm": "linear",
            "corpus_fraction": "1.0",
            "corpus_chunks": "1305",
            "corpus_fingerprint": "abc",
            "repetition": "1",
            "quality_available": "False",
            "total_time_ns": "10",
        },
        {
            "algorithm": "linear",
            "corpus_fraction": "1.0",
            "corpus_chunks": "1305",
            "corpus_fingerprint": "abc",
            "repetition": "2",
            "quality_available": "False",
            "total_time_ns": "20",
        },
    ]

    summary = _summarize(
        rows,
        (
            1,
        ),
    )

    total_time = next(
        row
        for row in summary
        if row[
            "metric"
        ]
        == "total_time_ns"
    )

    assert (
        total_time[
            "count"
        ]
        == 2
    )

    assert (
        total_time[
            "mean"
        ]
        == pytest.approx(
            15.0
        )
    )


def test_summary_does_not_duplicate_quality_across_repetitions() -> None:
    rows = [
        {
            "algorithm": "semantic",
            "corpus_fraction": "1.0",
            "corpus_chunks": "1305",
            "corpus_fingerprint": "abc",
            "repetition": "1",
            "quality_available": "True",
            "precision_at_1": "1.0",
        },
        {
            "algorithm": "semantic",
            "corpus_fraction": "1.0",
            "corpus_chunks": "1305",
            "corpus_fingerprint": "abc",
            "repetition": "2",
            "quality_available": "True",
            "precision_at_1": "1.0",
        },
    ]

    summary = _summarize(
        rows,
        (
            1,
        ),
    )

    precision = next(
        row
        for row in summary
        if row[
            "metric"
        ]
        == "precision_at_1"
    )

    assert (
        precision[
            "count"
        ]
        == 1
    )

    assert (
        precision[
            "mean"
        ]
        == 1.0
    )