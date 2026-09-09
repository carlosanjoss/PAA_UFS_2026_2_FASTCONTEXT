"""
Consolidated tests for manual linear and binary search implementations.

The implementations under test are the real project functions:

    binary_search(elements, target) -> (index | None, comparisons)
    linear_search(elements, target) -> (index | None, comparisons)

For binary search, ``elements`` must be sorted in non-decreasing order.

The tests cover the main required cases:

- empty collection;
- single-element collection;
- first element;
- middle element;
- last element;
- missing value;
- comparison-count behavior.
"""

import math

from src.algorithms.binary_search import binary_search
from src.algorithms.linear_search import linear_search

VOCAB = [
    "auth",
    "cors",
    "fastapi",
    "jwt",
    "security",
]


# ---------------------------------------------------------------------------
# Binary search
# ---------------------------------------------------------------------------
def test_binary_empty_collection() -> None:
    """An empty collection returns no index and performs no comparisons."""
    index, comparisons = binary_search(
        [],
        "fastapi",
    )

    assert index is None
    assert comparisons == 0


def test_binary_single_element_found() -> None:
    """A matching single element is found with one equality comparison."""
    index, comparisons = binary_search(
        ["fastapi"],
        "fastapi",
    )

    assert index == 0
    assert comparisons == 1


def test_binary_single_element_absent() -> None:
    """A missing single element requires equality and ordering comparisons."""
    index, comparisons = binary_search(
        ["fastapi"],
        "auth",
    )

    assert index is None
    assert comparisons == 2


def test_binary_found_start() -> None:
    """Binary search finds the first element."""
    index, comparisons = binary_search(
        VOCAB,
        "auth",
    )

    assert index == 0
    assert comparisons > 0


def test_binary_found_middle() -> None:
    """The middle element is found on the first equality comparison."""
    index, comparisons = binary_search(
        VOCAB,
        "fastapi",
    )

    assert index == 2
    assert comparisons == 1


def test_binary_found_end() -> None:
    """Binary search finds the final element."""
    index, comparisons = binary_search(
        VOCAB,
        "security",
    )

    assert index == 4
    assert comparisons > 0


def test_binary_absent() -> None:
    """A missing value returns no index."""
    index, comparisons = binary_search(
        VOCAB,
        "database",
    )

    assert index is None
    assert comparisons > 0


def test_binary_comparisons_logarithmic() -> None:
    """Binary-search key comparisons remain logarithmically bounded."""
    _, comparisons = binary_search(
        VOCAB,
        "database",
    )

    max_iterations = math.ceil(
        math.log2(
            len(VOCAB) + 1
        )
    )

    max_key_comparisons = (
        2 * max_iterations
    )

    assert comparisons <= max_key_comparisons


# ---------------------------------------------------------------------------
# Linear search
# ---------------------------------------------------------------------------
def test_linear_empty_collection() -> None:
    """An empty linear search performs no comparisons."""
    index, comparisons = linear_search(
        [],
        "fastapi",
    )

    assert index is None
    assert comparisons == 0


def test_linear_single_element_found() -> None:
    """Linear search finds a matching single element immediately."""
    index, comparisons = linear_search(
        ["fastapi"],
        "fastapi",
    )

    assert index == 0
    assert comparisons == 1


def test_linear_found_start() -> None:
    """Linear search finds the first item with one comparison."""
    index, comparisons = linear_search(
        VOCAB,
        "auth",
    )

    assert index == 0
    assert comparisons == 1


def test_linear_found_middle() -> None:
    """Linear search examines elements sequentially until the middle."""
    index, comparisons = linear_search(
        VOCAB,
        "fastapi",
    )

    assert index == 2
    assert comparisons == 3


def test_linear_found_end() -> None:
    """Linear search examines the complete prefix up to the last element."""
    index, comparisons = linear_search(
        VOCAB,
        "security",
    )

    assert index == 4
    assert comparisons == 5


def test_linear_absent() -> None:
    """Linear search checks every element when the target is absent."""
    index, comparisons = linear_search(
        VOCAB,
        "database",
    )

    assert index is None
    assert comparisons == len(VOCAB)


def test_linear_first_occurrence() -> None:
    """Linear search returns the first occurrence of a duplicated value."""
    elements = [
        "a",
        "b",
        "b",
        "c",
    ]

    index, comparisons = linear_search(
        elements,
        "b",
    )

    assert index == 1
    assert comparisons == 2