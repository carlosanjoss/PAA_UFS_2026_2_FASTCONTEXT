"""Tests for FastContext retrieval-quality metrics."""

from __future__ import annotations

import pytest

from src.evaluation.retrieval_metrics import (
    evaluate_retrieval,
    hit_rate_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_precision_at_k() -> None:
    retrieved = [
        "chunk-a",
        "chunk-b",
        "chunk-c",
        "chunk-d",
        "chunk-e",
    ]

    relevant = [
        "chunk-b",
        "chunk-d",
        "chunk-x",
    ]

    assert precision_at_k(
        retrieved,
        relevant,
        5,
    ) == pytest.approx(
        0.4
    )


def test_precision_at_k_uses_k_as_denominator() -> None:
    retrieved = [
        "chunk-a",
    ]

    relevant = [
        "chunk-a",
    ]

    assert precision_at_k(
        retrieved,
        relevant,
        3,
    ) == pytest.approx(
        1 / 3
    )


def test_recall_at_k() -> None:
    retrieved = [
        "chunk-a",
        "chunk-b",
        "chunk-c",
        "chunk-d",
    ]

    relevant = [
        "chunk-b",
        "chunk-d",
        "chunk-x",
        "chunk-y",
    ]

    assert recall_at_k(
        retrieved,
        relevant,
        4,
    ) == pytest.approx(
        0.5
    )


def test_recall_is_zero_without_relevant_chunks() -> None:
    assert recall_at_k(
        [
            "chunk-a",
            "chunk-b",
        ],
        [],
        2,
    ) == 0.0


def test_reciprocal_rank_first_position() -> None:
    assert reciprocal_rank(
        [
            "chunk-a",
            "chunk-b",
        ],
        [
            "chunk-a",
        ],
    ) == 1.0


def test_reciprocal_rank_later_position() -> None:
    assert reciprocal_rank(
        [
            "chunk-a",
            "chunk-b",
            "chunk-c",
        ],
        [
            "chunk-c",
        ],
    ) == pytest.approx(
        1 / 3
    )


def test_reciprocal_rank_respects_k() -> None:
    assert reciprocal_rank(
        [
            "chunk-a",
            "chunk-b",
            "chunk-c",
        ],
        [
            "chunk-c",
        ],
        k=2,
    ) == 0.0


def test_reciprocal_rank_is_zero_without_hit() -> None:
    assert reciprocal_rank(
        [
            "chunk-a",
            "chunk-b",
        ],
        [
            "chunk-x",
        ],
    ) == 0.0


def test_hit_rate_is_one_when_relevant_chunk_is_found() -> None:
    assert hit_rate_at_k(
        [
            "chunk-a",
            "chunk-b",
        ],
        [
            "chunk-b",
        ],
        2,
    ) == 1.0


def test_hit_rate_is_zero_when_no_relevant_chunk_is_found() -> None:
    assert hit_rate_at_k(
        [
            "chunk-a",
            "chunk-b",
        ],
        [
            "chunk-x",
        ],
        2,
    ) == 0.0


def test_evaluate_retrieval_calculates_all_metrics() -> None:
    metrics = evaluate_retrieval(
        [
            "chunk-a",
            "chunk-b",
            "chunk-c",
            "chunk-d",
            "chunk-e",
        ],
        [
            "chunk-b",
            "chunk-d",
            "chunk-x",
        ],
        5,
    )

    assert metrics.k == 5

    assert metrics.precision == pytest.approx(
        0.4
    )

    assert metrics.recall == pytest.approx(
        2 / 3
    )

    assert metrics.reciprocal_rank == pytest.approx(
        0.5
    )

    assert metrics.hit_rate == 1.0

    assert (
        metrics.relevant_retrieved
        == 2
    )

    assert (
        metrics.relevant_total
        == 3
    )

    assert (
        metrics.retrieved_count
        == 5
    )


def test_evaluate_retrieval_handles_empty_ranking() -> None:
    metrics = evaluate_retrieval(
        [],
        [
            "chunk-a",
        ],
        5,
    )

    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.reciprocal_rank == 0.0
    assert metrics.hit_rate == 0.0
    assert metrics.relevant_retrieved == 0
    assert metrics.relevant_total == 1
    assert metrics.retrieved_count == 0


@pytest.mark.parametrize(
    "k",
    [
        0,
        -1,
        -10,
    ],
)
def test_metrics_reject_non_positive_k(
    k: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        precision_at_k(
            [],
            [],
            k,
        )


@pytest.mark.parametrize(
    "k",
    [
        1.5,
        "5",
        True,
    ],
)
def test_metrics_reject_non_integer_k(
    k: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="must be an integer",
    ):
        precision_at_k(
            [],
            [],
            k,  # type: ignore[arg-type]
        )


def test_metrics_reject_duplicate_retrieved_ids() -> None:
    with pytest.raises(
        ValueError,
        match="Retrieved chunk IDs must be unique",
    ):
        evaluate_retrieval(
            [
                "chunk-a",
                "chunk-a",
            ],
            [
                "chunk-a",
            ],
            2,
        )


def test_metrics_reject_duplicate_relevant_ids() -> None:
    with pytest.raises(
        ValueError,
        match="Relevant chunk IDs must be unique",
    ):
        evaluate_retrieval(
            [
                "chunk-a",
            ],
            [
                "chunk-a",
                "chunk-a",
            ],
            1,
        )


def test_metrics_reject_empty_chunk_id() -> None:
    with pytest.raises(
        ValueError,
        match="Chunk IDs cannot be empty",
    ):
        evaluate_retrieval(
            [
                "",
            ],
            [
                "chunk-a",
            ],
            1,
        )