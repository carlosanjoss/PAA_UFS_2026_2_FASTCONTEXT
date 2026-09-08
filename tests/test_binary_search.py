"""Unit tests for the manual binary-search implementation."""

from src.algorithms.binary_search import binary_search


def test_binary_search_returns_none_for_empty_list() -> None:
    """An empty collection requires no comparisons."""
    index, comparisons = binary_search([], "fastapi")

    assert index is None
    assert comparisons == 0


def test_binary_search_finds_first_middle_and_last_elements() -> None:
    """Binary search locates values at each relevant boundary."""
    elements = ["auth", "cors", "fastapi", "jwt", "security"]

    first_index, first_comparisons = binary_search(elements, "auth")
    middle_index, middle_comparisons = binary_search(elements, "fastapi")
    last_index, last_comparisons = binary_search(elements, "security")

    assert first_index == 0
    assert middle_index == 2
    assert last_index == 4
    assert first_comparisons > 0
    assert middle_comparisons == 1
    assert last_comparisons > 0


def test_binary_search_returns_none_for_missing_value() -> None:
    """A value absent from the sorted collection is not reported as found."""
    elements = ["auth", "cors", "fastapi", "jwt", "security"]

    index, comparisons = binary_search(elements, "database")

    assert index is None
    assert comparisons > 0


def test_binary_search_uses_logarithmic_comparisons() -> None:
    """The number of iterations remains logarithmic for a sorted collection."""
    elements = list(range(1, 1025))

    index, comparisons = binary_search(elements, 1024)

    assert index == 1023
    assert comparisons <= 11