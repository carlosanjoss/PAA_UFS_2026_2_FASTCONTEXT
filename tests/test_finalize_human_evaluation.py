"""Tests for final human-evaluation statistics."""

from __future__ import annotations

from scripts.finalize_human_evaluation import (
    _apply_holm,
    _cochran_q,
    _mcnemar_exact,
    _paired_rank_biserial,
    _wilcoxon_test,
)


def test_wilcoxon_fallback_for_identical_vectors() -> None:
    statistic, p_value = _wilcoxon_test(
        [2, 1, 0],
        [2, 1, 0],
    )

    assert statistic == 0.0
    assert p_value == 1.0


def test_rank_biserial_is_zero_for_identical_vectors() -> None:
    result = _paired_rank_biserial(
        [2, 1, 0],
        [2, 1, 0],
    )

    assert result == 0.0


def test_rank_biserial_is_positive_when_left_is_always_higher() -> None:
    result = _paired_rank_biserial(
        [2, 2, 2],
        [1, 1, 1],
    )

    assert result == 1.0


def test_cochran_q_fallback_when_all_binary_vectors_are_identical() -> None:
    statistic, p_value = _cochran_q(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
    )

    assert statistic == 0.0
    assert p_value == 1.0


def test_exact_mcnemar_is_one_without_discordant_pairs() -> None:
    statistic, p_value = _mcnemar_exact(
        [0, 1, 0],
        [0, 1, 0],
    )

    assert statistic == 0
    assert p_value == 1.0


def test_holm_correction_is_monotonic() -> None:
    rows = [
        {
            "p_value": 0.01,
            "p_value_holm": "",
        },
        {
            "p_value": 0.02,
            "p_value_holm": "",
        },
        {
            "p_value": 0.20,
            "p_value_holm": "",
        },
    ]

    _apply_holm(
        rows
    )

    adjusted = [
        float(
            row[
                "p_value_holm"
            ]
        )
        for row in rows
    ]

    assert adjusted == [
        0.03,
        0.04,
        0.20,
    ]
