"""Utilities for the FastContext sorting-algorithm benchmark."""

from __future__ import annotations

import hashlib
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass

from src.algorithms.merge_sort import merge_sort_keys
from src.algorithms.ordering import SortKey
from src.algorithms.quick_sort import quick_sort_keys

SORTING_ALGORITHMS = (
    "merge",
    "quick",
)

SORTING_SCENARIOS = (
    "random",
    "already_sorted",
    "reverse_sorted",
    "many_ties",
)


class SortingBenchmarkError(RuntimeError):
    """Raised when a sorting benchmark execution is invalid."""


@dataclass(frozen=True, slots=True)
class SortingExecution:
    """Store one measured sorting execution."""

    algorithm: str
    elapsed_time_ns: int
    comparisons: int
    input_fingerprint: str
    item_count: int


def generate_sort_keys(
    item_count: int,
    *,
    scenario: str,
    seed: int,
) -> list[SortKey]:
    """Generate one deterministic sorting input."""
    _validate_item_count(
        item_count
    )

    if scenario not in SORTING_SCENARIOS:
        raise ValueError(
            f"Unsupported sorting scenario: {scenario}"
        )

    random_generator = random.Random(
        seed
    )

    if scenario == "many_ties":
        keys = [
            (
                float(
                    random_generator.randrange(
                        0,
                        8,
                    )
                ),
                f"chunk-{index:06d}",
            )
            for index in range(
                item_count
            )
        ]

        random_generator.shuffle(
            keys
        )

        return keys

    keys = [
        (
            random_generator.random(),
            f"chunk-{index:06d}",
        )
        for index in range(
            item_count
        )
    ]

    if scenario == "random":
        random_generator.shuffle(
            keys
        )

        return keys

    ordered = reference_sort_keys(
        keys
    )

    if scenario == "already_sorted":
        return ordered

    if scenario == "reverse_sorted":
        return list(
            reversed(
                ordered
            )
        )

    raise ValueError(
        f"Unsupported sorting scenario: {scenario}"
    )


def run_sorting_algorithm(
    algorithm: str,
    keys: Sequence[SortKey],
) -> SortingExecution:
    """Run and measure one classical sorting algorithm."""
    if algorithm not in SORTING_ALGORITHMS:
        raise ValueError(
            f"Unsupported sorting algorithm: {algorithm}"
        )

    input_keys = list(
        keys
    )

    fingerprint = (
        calculate_input_fingerprint(
            input_keys
        )
    )

    start_time = (
        time.perf_counter_ns()
    )

    if algorithm == "merge":
        (
            ordered,
            merge_stats,
        ) = merge_sort_keys(
            list(
                input_keys
            )
        )

        comparisons = (
            merge_stats.comparisons
        )

    else:
        (
            ordered,
            quick_stats,
        ) = quick_sort_keys(
            list(
                input_keys
            )
        )

        comparisons = (
            quick_stats.comparisons
        )

    elapsed_time_ns = (
        time.perf_counter_ns()
        - start_time
    )

    expected = reference_sort_keys(
        input_keys
    )

    if ordered != expected:
        raise SortingBenchmarkError(
            f"{algorithm} produced an incorrect ranking."
        )

    if (
        isinstance(
            comparisons,
            bool,
        )
        or not isinstance(
            comparisons,
            int,
        )
        or comparisons < 0
    ):
        raise SortingBenchmarkError(
            "Sorting comparison counter is invalid."
        )

    return SortingExecution(
        algorithm=algorithm,
        elapsed_time_ns=(
            elapsed_time_ns
        ),
        comparisons=(
            comparisons
        ),
        input_fingerprint=(
            fingerprint
        ),
        item_count=len(
            input_keys
        ),
    )


def reference_sort_keys(
    keys: Sequence[SortKey],
) -> list[SortKey]:
    """Return the canonical ranking used only as a correctness oracle."""
    return sorted(
        keys,
        key=lambda item: (
            -item[0],
            item[1],
        ),
    )


def calculate_input_fingerprint(
    keys: Sequence[SortKey],
) -> str:
    """Calculate a deterministic SHA-256 fingerprint for one input."""
    digest = hashlib.sha256()

    for score, chunk_id in keys:
        digest.update(
            format(
                score,
                ".17g",
            ).encode(
                "utf-8"
            )
        )

        digest.update(
            b"\0"
        )

        digest.update(
            chunk_id.encode(
                "utf-8"
            )
        )

        digest.update(
            b"\n"
        )

    return digest.hexdigest()


def _validate_item_count(
    item_count: int,
) -> None:
    """Validate one experimental input size."""
    if (
        isinstance(
            item_count,
            bool,
        )
        or not isinstance(
            item_count,
            int,
        )
    ):
        raise TypeError(
            "item_count must be an integer."
        )

    if item_count <= 0:
        raise ValueError(
            "item_count must be greater than zero."
        )