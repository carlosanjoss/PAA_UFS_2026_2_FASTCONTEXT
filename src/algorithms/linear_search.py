from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")


def linear_search(
    elements: Sequence[T],
    target: T,
) -> tuple[int | None, int]:
    """Perform manual linear search and count key comparisons."""

    comparisons = 0

    for index, item in enumerate(
        elements
    ):
        comparisons += 1

        if item == target:
            return index, comparisons

    return None, comparisons