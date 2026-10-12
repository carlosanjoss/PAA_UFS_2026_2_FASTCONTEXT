
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StringMatchResult:
    """Result produced by a string matching algorithm."""

    algorithm: str
    positions: tuple[int, ...]
    comparisons: int
    preprocessing_time_ns: int
    search_time_ns: int
    text_length: int
    pattern_length: int

    @property
    def occurrences(self) -> int:
        """Return the number of matches found."""
        return len(self.positions)

    @property
    def total_time_ns(self) -> int:
        """Return preprocessing plus search time."""
        return (
            self.preprocessing_time_ns
            + self.search_time_ns
        )
