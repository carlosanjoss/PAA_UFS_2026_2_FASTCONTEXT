"""Equivalence tests: Boyer-Moore and Rabin-Karp vs the Naive oracle.

The Naive matcher (Wilson's baseline) is the ground truth for positions.
Boyer-Moore and Rabin-Karp must return exactly the same positions. Only
`positions` is compared: comparisons, timings and metadata are allowed
to differ between algorithms.
"""

from __future__ import annotations

import random

import pytest

from src.string_matching.boyer_moore import BoyerMooreMatcher
from src.string_matching.naive import NaiveMatcher
from src.string_matching.rabin_karp import RabinKarpMatcher

CANONICAL_CASES = [
    ("abcabc", "abc"),
    ("abcdef", "xyz"),
    ("abababab", "abab"),
    ("aaaaa", "aaa"),
    ("café café", "café"),
    ("API api API", "API"),
    ("FastAPI", "API"),
    ("banana", "ana"),
    ("abracadabra", "a"),
    ("mississippi", "issi"),
]


@pytest.mark.parametrize("text, pattern", CANONICAL_CASES)
def test_boyer_moore_matches_naive(text: str, pattern: str) -> None:
    naive = NaiveMatcher().search(text, pattern)
    bm = BoyerMooreMatcher().search(text, pattern)

    assert bm.positions == naive.positions


@pytest.mark.parametrize("text, pattern", CANONICAL_CASES)
def test_rabin_karp_matches_naive(text: str, pattern: str) -> None:
    naive = NaiveMatcher().search(text, pattern)
    rk = RabinKarpMatcher().search(text, pattern)

    assert rk.positions == naive.positions


def _random_text(rng: random.Random, alphabet: str, length: int) -> str:
    return "".join(rng.choice(alphabet) for _ in range(length))


def test_differential_random_small_alphabet() -> None:
    """Differential fuzzing with a fixed seed over a tiny alphabet.

    A small alphabet maximizes overlaps and near-matches, stressing the
    shift logic (BM) and the verification-after-hash logic (RK).
    """
    rng = random.Random(20262)
    alphabet = "ab"

    naive = NaiveMatcher()
    bm = BoyerMooreMatcher()
    rk = RabinKarpMatcher()

    for _ in range(500):
        text_length = rng.randint(0, 20)
        pattern_length = rng.randint(1, 5)

        text = _random_text(rng, alphabet, text_length)
        pattern = _random_text(rng, alphabet, pattern_length)

        expected = naive.search(text, pattern).positions

        assert bm.search(text, pattern).positions == expected
        assert rk.search(text, pattern).positions == expected


def test_differential_random_larger_alphabet() -> None:
    """Differential fuzzing with a fixed seed over a larger alphabet."""
    rng = random.Random(777)
    alphabet = "abcdefg "

    naive = NaiveMatcher()
    bm = BoyerMooreMatcher()
    rk = RabinKarpMatcher()

    for _ in range(300):
        text = _random_text(rng, alphabet, rng.randint(0, 40))
        pattern = _random_text(rng, alphabet, rng.randint(1, 6))

        expected = naive.search(text, pattern).positions

        assert bm.search(text, pattern).positions == expected
        assert rk.search(text, pattern).positions == expected


def test_determinism() -> None:
    """Repeated searches yield identical positions."""
    bm = BoyerMooreMatcher()
    rk = RabinKarpMatcher()

    first_bm = bm.search("abababab", "abab").positions
    second_bm = bm.search("abababab", "abab").positions

    first_rk = rk.search("abababab", "abab").positions
    second_rk = rk.search("abababab", "abab").positions

    assert first_bm == second_bm == (0, 2, 4)
    assert first_rk == second_rk == (0, 2, 4)
