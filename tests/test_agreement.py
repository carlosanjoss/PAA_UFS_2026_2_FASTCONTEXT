"""Tests for FastContext inter-annotator agreement."""

from __future__ import annotations

import pytest

from src.evaluation.agreement import (
    calculate_agreement,
)


def test_perfect_agreement() -> None:
    metrics = calculate_agreement(
        [
            1,
            0,
            1,
            0,
        ],
        [
            1,
            0,
            1,
            0,
        ],
    )

    assert metrics.count == 4
    assert metrics.agreements == 4
    assert metrics.disagreements == 0
    assert metrics.agreement_rate == 1.0
    assert metrics.cohen_kappa == 1.0


def test_partial_agreement() -> None:
    metrics = calculate_agreement(
        [
            1,
            1,
            0,
            0,
        ],
        [
            1,
            0,
            1,
            0,
        ],
    )

    assert metrics.count == 4
    assert metrics.agreements == 2
    assert metrics.disagreements == 2
    assert metrics.agreement_rate == 0.5
    assert metrics.cohen_kappa == pytest.approx(
        0.0
    )


def test_complete_disagreement() -> None:
    metrics = calculate_agreement(
        [
            1,
            1,
            0,
            0,
        ],
        [
            0,
            0,
            1,
            1,
        ],
    )

    assert metrics.agreements == 0
    assert metrics.disagreements == 4
    assert metrics.agreement_rate == 0.0
    assert metrics.cohen_kappa == pytest.approx(
        -1.0
    )


def test_constant_identical_zero_labels_have_undefined_kappa() -> None:
    metrics = calculate_agreement(
        [
            0,
            0,
            0,
        ],
        [
            0,
            0,
            0,
        ],
    )

    assert metrics.agreement_rate == 1.0
    assert metrics.cohen_kappa is None


def test_constant_identical_one_labels_have_undefined_kappa() -> None:
    metrics = calculate_agreement(
        [
            1,
            1,
            1,
        ],
        [
            1,
            1,
            1,
        ],
    )

    assert metrics.agreement_rate == 1.0
    assert metrics.cohen_kappa is None


def test_sequences_must_have_same_length() -> None:
    with pytest.raises(
        ValueError,
        match="same length",
    ):
        calculate_agreement(
            [
                1,
                0,
            ],
            [
                1,
            ],
        )


def test_empty_sequences_are_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="At least one paired annotation",
    ):
        calculate_agreement(
            [],
            [],
        )


@pytest.mark.parametrize(
    "invalid_label",
    [
        -1,
        2,
        10,
    ],
)
def test_non_binary_labels_are_rejected(
    invalid_label: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="only 0 or 1",
    ):
        calculate_agreement(
            [
                invalid_label,
            ],
            [
                0,
            ],
        )


@pytest.mark.parametrize(
    "invalid_label",
    [
        True,
        False,
        1.0,
        "1",
        None,
    ],
)
def test_invalid_label_types_are_rejected(
    invalid_label: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="must contain integers",
    ):
        calculate_agreement(
            [
                invalid_label,
            ],  # type: ignore[list-item]
            [
                0,
            ],
        )
