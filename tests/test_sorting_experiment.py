"""Tests for the FastContext sorting benchmark."""

from __future__ import annotations

import pytest

from src.evaluation.sorting_benchmark import (
    SORTING_ALGORITHMS,
    SORTING_SCENARIOS,
    calculate_input_fingerprint,
    generate_sort_keys,
    reference_sort_keys,
    run_sorting_algorithm,
)


@pytest.mark.parametrize(
    "scenario",
    SORTING_SCENARIOS,
)
def test_generate_sort_keys_has_expected_size(
    scenario: str,
) -> None:
    keys = generate_sort_keys(
        100,
        scenario=scenario,
        seed=42,
    )

    assert len(
        keys
    ) == 100

    assert len(
        set(
            keys
        )
    ) == 100


@pytest.mark.parametrize(
    "scenario",
    SORTING_SCENARIOS,
)
def test_generation_is_deterministic(
    scenario: str,
) -> None:
    first = generate_sort_keys(
        64,
        scenario=scenario,
        seed=42,
    )

    second = generate_sort_keys(
        64,
        scenario=scenario,
        seed=42,
    )

    assert first == second


def test_already_sorted_scenario_is_canonical() -> None:
    keys = generate_sort_keys(
        64,
        scenario="already_sorted",
        seed=42,
    )

    assert (
        keys
        == reference_sort_keys(
            keys
        )
    )


def test_reverse_sorted_scenario_is_reverse_canonical() -> None:
    sorted_keys = (
        generate_sort_keys(
            64,
            scenario="already_sorted",
            seed=42,
        )
    )

    reverse_keys = (
        generate_sort_keys(
            64,
            scenario="reverse_sorted",
            seed=42,
        )
    )

    assert (
        reverse_keys
        == list(
            reversed(
                sorted_keys
            )
        )
    )


def test_many_ties_contains_repeated_scores() -> None:
    keys = generate_sort_keys(
        100,
        scenario="many_ties",
        seed=42,
    )

    scores = [
        score
        for score, _ in keys
    ]

    assert (
        len(
            set(
                scores
            )
        )
        < len(
            scores
        )
    )


@pytest.mark.parametrize(
    "algorithm",
    SORTING_ALGORITHMS,
)
@pytest.mark.parametrize(
    "scenario",
    SORTING_SCENARIOS,
)
def test_sorting_algorithms_match_reference(
    algorithm: str,
    scenario: str,
) -> None:
    keys = generate_sort_keys(
        128,
        scenario=scenario,
        seed=42,
    )

    execution = (
        run_sorting_algorithm(
            algorithm,
            keys,
        )
    )

    assert (
        execution.algorithm
        == algorithm
    )

    assert (
        execution.item_count
        == 128
    )

    assert (
        execution.elapsed_time_ns
        > 0
    )

    assert (
        execution.comparisons
        > 0
    )


@pytest.mark.parametrize(
    "algorithm",
    SORTING_ALGORITHMS,
)
def test_sorting_does_not_modify_input(
    algorithm: str,
) -> None:
    keys = generate_sort_keys(
        64,
        scenario="random",
        seed=42,
    )

    original = list(
        keys
    )

    run_sorting_algorithm(
        algorithm,
        keys,
    )

    assert keys == original


def test_both_algorithms_receive_same_fingerprint() -> None:
    keys = generate_sort_keys(
        128,
        scenario="random",
        seed=99,
    )

    merge = (
        run_sorting_algorithm(
            "merge",
            keys,
        )
    )

    quick = (
        run_sorting_algorithm(
            "quick",
            keys,
        )
    )

    assert (
        merge.input_fingerprint
        == quick.input_fingerprint
    )


def test_input_fingerprint_is_deterministic() -> None:
    keys = generate_sort_keys(
        32,
        scenario="random",
        seed=42,
    )

    assert (
        calculate_input_fingerprint(
            keys
        )
        == calculate_input_fingerprint(
            keys
        )
    )


def test_invalid_scenario_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="Unsupported sorting scenario",
    ):
        generate_sort_keys(
            10,
            scenario="invalid",
            seed=42,
        )


def test_invalid_algorithm_is_rejected() -> None:
    keys = generate_sort_keys(
        10,
        scenario="random",
        seed=42,
    )

    with pytest.raises(
        ValueError,
        match="Unsupported sorting algorithm",
    ):
        run_sorting_algorithm(
            "invalid",
            keys,
        )


@pytest.mark.parametrize(
    "item_count",
    [
        0,
        -1,
        -100,
    ],
)
def test_invalid_item_count_is_rejected(
    item_count: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        generate_sort_keys(
            item_count,
            scenario="random",
            seed=42,
        )