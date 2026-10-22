from __future__ import annotations

from time import perf_counter_ns

from src.string_matching.base import StringMatcher
from src.string_matching.models import StringMatchResult

# Default rolling-hash parameters. The base spans the Unicode code
# point range used by the corpus; the modulus is a large prime to keep
# collisions rare on realistic inputs. Both are configurable so tests
# can force collisions deterministically.
DEFAULT_BASE = 256
DEFAULT_MODULUS = 1_000_000_007


class RabinKarpMatcher(StringMatcher):
    """Rabin-Karp exact string matching using a manual rolling hash.

    A hash coincidence is never treated as a definitive match: every
    time the window hash equals the pattern hash, the characters are
    verified one by one. This guarantees that hash collisions cannot
    produce false positives.
    """

    def __init__(
        self,
        base: int = DEFAULT_BASE,
        modulus: int = DEFAULT_MODULUS,
    ) -> None:
        if modulus <= 1:
            raise ValueError("modulus must be greater than 1.")
        if base <= 0:
            raise ValueError("base must be positive.")

        self._base = base
        self._modulus = modulus

    @property
    def name(self) -> str:
        """Return the canonical algorithm name."""
        return "rabin-karp"

    def _char_code(self, character: str) -> int:
        """Map a character to an integer for hashing."""
        return ord(character) % self._modulus

    def search(
        self,
        text: str,
        pattern: str,
    ) -> StringMatchResult:
        """Find all pattern occurrences using a rolling hash."""
        self.validate_input(text, pattern)

        text_length = len(text)
        pattern_length = len(pattern)
        positions: list[int] = []
        comparisons = 0

        if pattern_length > text_length:
            return StringMatchResult(
                algorithm=self.name,
                positions=(),
                comparisons=0,
                preprocessing_time_ns=0,
                search_time_ns=0,
                text_length=text_length,
                pattern_length=pattern_length,
            )

        preprocessing_start = perf_counter_ns()

        # high_order = base^(m-1) % modulus, used to remove the
        # leading character when the window rolls forward.
        high_order = 1
        for _ in range(pattern_length - 1):
            high_order = (high_order * self._base) % self._modulus

        pattern_hash = 0
        window_hash = 0
        for index in range(pattern_length):
            pattern_hash = (
                self._base * pattern_hash
                + self._char_code(pattern[index])
            ) % self._modulus
            window_hash = (
                self._base * window_hash
                + self._char_code(text[index])
            ) % self._modulus

        preprocessing_time_ns = (
            perf_counter_ns() - preprocessing_start
        )

        search_start = perf_counter_ns()

        last_start = text_length - pattern_length

        for start in range(last_start + 1):
            if window_hash == pattern_hash:
                # Hash coincided: mandatory character-by-character
                # verification. Only real character comparisons are
                # counted; hash equality itself is NOT a comparison.
                match = True
                for offset in range(pattern_length):
                    comparisons += 1
                    if text[start + offset] != pattern[offset]:
                        match = False
                        break

                if match:
                    positions.append(start)

            # Roll the hash to the next window (if any remain).
            if start < last_start:
                leading_code = self._char_code(text[start])
                trailing_code = self._char_code(
                    text[start + pattern_length]
                )
                window_hash = (
                    window_hash
                    - leading_code * high_order
                ) % self._modulus
                window_hash = (
                    window_hash * self._base + trailing_code
                ) % self._modulus

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
