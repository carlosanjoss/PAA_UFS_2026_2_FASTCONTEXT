from __future__ import annotations

from time import perf_counter_ns

from src.string_matching.base import StringMatcher
from src.string_matching.models import StringMatchResult


class KMPMatcher(StringMatcher):
    """Knuth-Morris-Pratt exact string matching algorithm."""

    @property
    def name(self) -> str:
        """Return the canonical algorithm name."""
        return "kmp"

    @staticmethod
    def _build_lps(pattern: str) -> list[int]:
        """Build the longest proper prefix/suffix table."""
        lps = [0] * len(pattern)
        prefix_length = 0
        index = 1

        while index < len(pattern):
            if pattern[index] == pattern[prefix_length]:
                prefix_length += 1
                lps[index] = prefix_length
                index += 1
            elif prefix_length > 0:
                prefix_length = lps[prefix_length - 1]
            else:
                lps[index] = 0
                index += 1

        return lps

    def search(
        self,
        text: str,
        pattern: str,
    ) -> StringMatchResult:
        """Find all pattern occurrences using KMP."""
        self.validate_input(text, pattern)

        text_length = len(text)
        pattern_length = len(pattern)
        positions: list[int] = []
        comparisons = 0

        preprocessing_start = perf_counter_ns()
        lps = self._build_lps(pattern)
        preprocessing_time_ns = (
            perf_counter_ns() - preprocessing_start
        )

        search_start = perf_counter_ns()

        text_index = 0
        pattern_index = 0

        while text_index < text_length:
            comparisons += 1

            if text[text_index] == pattern[pattern_index]:
                text_index += 1
                pattern_index += 1

                if pattern_index == pattern_length:
                    positions.append(text_index - pattern_length)

                    # Continue from the longest valid prefix to
                    # preserve overlapping occurrences.
                    pattern_index = lps[pattern_index - 1]

            elif pattern_index > 0:
                pattern_index = lps[pattern_index - 1]
            else:
                text_index += 1

        search_time_ns = perf_counter_ns() - search_start

        return StringMatchResult(
            algorithm=self.name,
            positions=tuple(positions),
            comparisons=comparisons,
            preprocessing_time_ns=preprocessing_time_ns,
            search_time_ns=search_time_ns,
            text_length=text_length,
            pattern_length=pattern_length,
        )