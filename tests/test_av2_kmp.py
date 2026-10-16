from __future__ import annotations

import random

import pytest

from src.string_matching.kmp import KMPMatcher
from src.string_matching.naive import NaiveMatcher


@pytest.fixture
def matcher() -> KMPMatcher:
    """Create a KMP matcher for each test."""
    return KMPMatcher()


def test_algorithm_name(matcher: KMPMatcher) -> None:
    assert matcher.name == "kmp"


def test_single_occurrence(matcher: KMPMatcher) -> None:
    result = matcher.search("FastAPI", "API")

    assert result.positions == (4,)
    assert result.occurrences == 1


def test_multiple_occurrences(matcher: KMPMatcher) -> None:
    result = matcher.search("abcabc", "abc")

    assert result.positions == (0, 3)


def test_no_occurrence(matcher: KMPMatcher) -> None:
    result = matcher.search("FastAPI", "Django")

    assert result.positions == ()
    assert result.occurrences == 0


def test_pattern_longer_than_text(
    matcher: KMPMatcher,
) -> None:
    result = matcher.search("API", "FastAPI")

    assert result.positions == ()


def test_empty_text(matcher: KMPMatcher) -> None:
    result = matcher.search("", "API")

    assert result.positions == ()
    assert result.comparisons == 0


def test_empty_pattern(matcher: KMPMatcher) -> None:
    with pytest.raises(ValueError):
        matcher.search("FastAPI", "")


def test_invalid_text_type(matcher: KMPMatcher) -> None:
    with pytest.raises(TypeError):
        matcher.search(123, "API")  # type: ignore[arg-type]


def test_invalid_pattern_type(matcher: KMPMatcher) -> None:
    with pytest.raises(TypeError):
        matcher.search("FastAPI", 123)  # type: ignore[arg-type]


def test_case_sensitivity(matcher: KMPMatcher) -> None:
    result = matcher.search("API api API", "API")

    assert result.positions == (0, 8)


def test_overlapping_occurrences(
    matcher: KMPMatcher,
) -> None:
    result = matcher.search("ABABABA", "ABA")

    assert result.positions == (0, 2, 4)


def test_repetitive_overlapping_occurrences(
    matcher: KMPMatcher,
) -> None:
    result = matcher.search("AAAAA", "AAA")

    assert result.positions == (0, 1, 2)


def test_required_overlapping_example(
    matcher: KMPMatcher,
) -> None:
    result = matcher.search("abababab", "abab")

    assert result.positions == (0, 2, 4)


def test_unicode(matcher: KMPMatcher) -> None:
    result = matcher.search("café café", "café")

    assert result.positions == (0, 5)


def test_metrics(matcher: KMPMatcher) -> None:
    result = matcher.search("AAAA", "AA")

    assert result.algorithm == "kmp"
    assert result.positions == (0, 1, 2)
    assert result.comparisons == 4
    assert result.preprocessing_time_ns >= 0
    assert result.search_time_ns >= 0
    assert result.text_length == 4
    assert result.pattern_length == 2
    assert result.total_time_ns == (
        result.preprocessing_time_ns
        + result.search_time_ns
    )


def test_lps_table() -> None:
    assert KMPMatcher._build_lps("ABAB") == [0, 0, 1, 2]
    assert KMPMatcher._build_lps("AAAA") == [0, 1, 2, 3]
    assert KMPMatcher._build_lps("ABCDE") == [0, 0, 0, 0, 0]
    assert KMPMatcher._build_lps("AABAACAABAA") == [
        0,
        1,
        0,
        1,
        2,
        0,
        1,
        2,
        3,
        4,
        5,
    ]


def test_agreement_with_naive_fixed_cases() -> None:
    naive = NaiveMatcher()
    kmp = KMPMatcher()

    cases = [
        ("abcabc", "abc"),
        ("abcdef", "xyz"),
        ("abababab", "abab"),
        ("", "abc"),
        ("ab", "abc"),
        ("aaaaa", "aaa"),
        ("café café", "café"),
        ("API api API", "API"),
        ("banana", "ana"),
        ("ABABABA", "ABA"),
    ]

    for text, pattern in cases:
        naive_result = naive.search(text, pattern)
        kmp_result = kmp.search(text, pattern)

        assert kmp_result.positions == naive_result.positions


def test_randomized_agreement_with_naive() -> None:
    random_generator = random.Random(2026)
    alphabet = "ABCD"
    naive = NaiveMatcher()
    kmp = KMPMatcher()

    for _ in range(250):
        text_length = random_generator.randint(0, 100)
        pattern_length = random_generator.randint(1, 15)

        text = "".join(
            random_generator.choice(alphabet)
            for _ in range(text_length)
        )
        pattern = "".join(
            random_generator.choice(alphabet)
            for _ in range(pattern_length)
        )

        naive_result = naive.search(text, pattern)
        kmp_result = kmp.search(text, pattern)

        assert kmp_result.positions == naive_result.positions