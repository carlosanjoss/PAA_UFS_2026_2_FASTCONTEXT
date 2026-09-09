"""Descriptive statistics for FastContext experimental results."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SummaryStatistics:
    """Store descriptive statistics for one experimental metric."""

    count: int
    mean: float
    median: float
    std: float
    minimum: float
    maximum: float


def calculate_statistics(
    values: Sequence[int | float],
) -> SummaryStatistics:
    """Calculate descriptive statistics for numeric observations.

    The standard deviation is the sample standard deviation when at
    least two observations are available. A single observation has
    standard deviation equal to zero.
    """
    normalized = _validate_values(
        values
    )

    count = len(
        normalized
    )

    mean_value = statistics.fmean(
        normalized
    )

    median_value = float(
        statistics.median(
            normalized
        )
    )

    standard_deviation = (
        statistics.stdev(
            normalized
        )
        if count >= 2
        else 0.0
    )

    return SummaryStatistics(
        count=count,
        mean=float(
            mean_value
        ),
        median=median_value,
        std=float(
            standard_deviation
        ),
        minimum=min(
            normalized
        ),
        maximum=max(
            normalized
        ),
    )


def _validate_values(
    values: Sequence[int | float],
) -> tuple[float, ...]:
    """Validate and normalize experimental observations."""
    if not values:
        raise ValueError(
            "At least one observation is required."
        )

    normalized: list[
        float
    ] = []

    for value in values:
        if isinstance(
            value,
            bool,
        ) or not isinstance(
            value,
            (int, float),
        ):
            raise TypeError(
                "Experimental observations must be numeric."
            )

        normalized_value = float(
            value
        )

        if not math.isfinite(
            normalized_value
        ):
            raise ValueError(
                "Experimental observations must be finite."
            )

        normalized.append(
            normalized_value
        )

    return tuple(
        normalized
    )