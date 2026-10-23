from __future__ import annotations

from time import perf_counter_ns

from src.string_matching.base import StringMatcher
from src.string_matching.models import StringMatchResult


class BoyerMooreMatcher(StringMatcher):
    """Boyer-Moore exact string matching (bad-character heuristic only).

    This implementation uses solely the bad-character rule. It does
    **not** implement the good-suffix heuristic, so it must not be
    described as the full Boyer-Moore algorithm. The worst-case search
    cost therefore remains O(n * m).
    """

    @property
    def name(self) -> str:
        """Return the canonical algorithm name."""
        return "boyer-moore"

    @staticmethod
    def _build_last_occurrence(pattern: str) -> dict[str, int]:
        """Build the last-occurrence table for the bad-character rule.

        Maps each character of the pattern to the index of its
        right-most occurrence. Characters absent from the pattern are
        implicitly treated as -1 by the search routine.
        """
        last_occurrence: dict[str, int] = {}

        for index, character in enumerate(pattern):
            last_occurrence[character] = index

        return last_occurrence

    def search(
        self,
        text: str,
        pattern: str,
    ) -> StringMatchResult:
        """Find all pattern occurrences using the bad-character rule."""
        self.validate_input(text, pattern)

        text_length = len(text)
        pattern_length = len(pattern)
        positions: list[int] = []
        comparisons = 0

        preprocessing_start = perf_counter_ns()
        last_occurrence = self._build_last_occurrence(pattern)
        preprocessing_time_ns = (
            perf_counter_ns() - preprocessing_start
        )

        search_start = perf_counter_ns()

        # Alignment of the pattern's left edge against the text.
        alignment = 0

        while alignment <= text_length - pattern_length:
            # Compare right-to-left within the current alignment.
            pattern_index = pattern_length - 1

            while pattern_index >= 0:
                comparisons += 1

                text_character = text[alignment + pattern_index]

                if text_character != pattern[pattern_index]:
                    break

                pattern_index -= 1

            if pattern_index < 0:
                # Full match: record and shift by one to preserve
                # overlapping occurrences.
                positions.append(alignment)
                alignment += 1
            else:
                # Bad-character shift: move the pattern so the last
                # occurrence of the conflicting text character aligns
                # with its position in the pattern. The shift is
                # clamped to at least one to guarantee progress.
                last_index = last_occurrence.get(text_character, -1)
                shift = pattern_index - last_index
                alignment += shift if shift > 0 else 1

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
