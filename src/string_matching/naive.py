
from __future__ import annotations

from time import perf_counter_ns

from src.string_matching.base import StringMatcher
from src.string_matching.models import StringMatchResult


class NaiveMatcher(StringMatcher):
    """Naive exact string matching algorithm."""

    @property
    def name(self) -> str:
        return "naive"

    
    def search(
        self,
        text: str,
        pattern: str,
    ) -> StringMatchResult:
        self.validate_input(text, pattern)

        n = len(text)
        m = len(pattern)

        positions: list[int] = []
        comparisons = 0

        start_time = perf_counter_ns()

        for i in range(n - m + 1):
            match = True

            for j in range(m):
                comparisons += 1

                if text[i + j] != pattern[j]:
                    match = False
                    break

            if match:
                positions.append(i)

        search_time_ns = perf_counter_ns() - start_time

        return StringMatchResult(
            algorithm=self.name,
            positions=tuple(positions),
            comparisons=comparisons,
            preprocessing_time_ns=0,
            search_time_ns=search_time_ns,
            text_length=n,
            pattern_length=m,
        )
