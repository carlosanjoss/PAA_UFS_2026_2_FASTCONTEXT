"""Tests for FastContext experimental statistics."""

from __future__ import annotations

import pytest

from src.evaluation.statistics import (
    calculate_statistics,
)


def test_calculate_statistics_for_multiple_values() -> None:
    result = calculate_statistics(
        [
            10,
            20,
            30,
            40,
            50,
        ]
    )

    assert result.count == 5

    assert result.mean == pytest.approx(
        30.0
    )

    assert result.median == pytest.approx(
        30.0
    )

    assert result.std == pytest.approx(
        15.811388300841896
    )

    assert result.minimum == pytest.approx(
        10.0
    )

    assert result.maximum == pytest.approx(
        50.0
    )


def test_calculate_statistics_for_even_number_of_values() -> None:
    result = calculate_statistics(
        [
            1,
            2,
            3,
            4,
        ]
    )

    assert result.count == 4

    assert result.mean == pytest.approx(
        2.5
    )

    assert result.median == pytest.approx(
        2.5
    )

    assert result.minimum == pytest.approx(
        1.0
    )

    assert result.maximum == pytest.approx(
        4.0
    )


def test_calculate_statistics_for_single_value() -> None:
    result = calculate_statistics(
        [
            42,
        ]
    )

    assert result.count == 1

    assert result.mean == pytest.approx(
        42.0
    )

    assert result.median == pytest.approx(
        42.0
    )

    assert result.std == 0.0

    assert result.minimum == pytest.approx(
        42.0
    )

    assert result.maximum == pytest.approx(
        42.0
    )


def test_calculate_statistics_accepts_float_values() -> None:
    result = calculate_statistics(
        [
            1.5,
            2.5,
            3.5,
        ]
    )

    assert result.count == 3

    assert result.mean == pytest.approx(
        2.5
    )

    assert result.median == pytest.approx(
        2.5
    )


def test_calculate_statistics_accepts_mixed_numeric_values() -> None:
    result = calculate_statistics(
        [
            1,
            2.5,
            4,
        ]
    )

    assert result.count == 3

    assert result.mean == pytest.approx(
        2.5
    )


def test_calculate_statistics_preserves_negative_values() -> None:
    result = calculate_statistics(
        [
            -10,
            0,
            10,
        ]
    )

    assert result.mean == pytest.approx(
        0.0
    )

    assert result.median == pytest.approx(
        0.0
    )

    assert result.minimum == pytest.approx(
        -10.0
    )

    assert result.maximum == pytest.approx(
        10.0
    )


def test_calculate_statistics_rejects_empty_sequence() -> None:
    with pytest.raises(
        ValueError,
        match="At least one observation is required",
    ):
        calculate_statistics(
            []
        )


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        "10",
        None,
        object(),
    ],
)
def test_calculate_statistics_rejects_non_numeric_values(
    value: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="observations must be numeric",
    ):
        calculate_statistics(
            [
                value,
            ]  # type: ignore[list-item]
        )


@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_calculate_statistics_rejects_non_finite_values(
    value: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="observations must be finite",
    ):
        calculate_statistics(
            [
                value,
            ]
        )