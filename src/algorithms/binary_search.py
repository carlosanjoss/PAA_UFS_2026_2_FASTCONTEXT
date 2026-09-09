"""
Manual binary search with key-comparison instrumentation for PAA.
"""

from __future__ import annotations

from typing import Any


def binary_search(
    elements: list[Any],
    target: Any,
) -> tuple[int | None, int]:
    """Search for a target in a sorted list using binary search.

    Args:
        elements: Elements sorted in non-decreasing order.
        target: Target value to locate.

    Returns:
        A tuple containing the target index when found, otherwise None,
        and the number of key comparisons performed.
    """
    comparisons = 0
    left = 0
    right = len(elements) - 1

    while left <= right:
        mid = (
            left
            + right
        ) // 2

        comparisons += 1

        if elements[mid] == target:
            return (
                mid,
                comparisons,
            )

        comparisons += 1

        if elements[mid] < target:
            left = (
                mid
                + 1
            )
        else:
            right = (
                mid
                - 1
            )

    return (
        None,
        comparisons,
    )