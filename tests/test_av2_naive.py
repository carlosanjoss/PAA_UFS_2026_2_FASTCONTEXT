
import pytest

from src.string_matching.naive import NaiveMatcher


@pytest.fixture
def matcher() -> NaiveMatcher:
    return NaiveMatcher()


def test_single_occurrence(matcher: NaiveMatcher) -> None:
    result = matcher.search("FastAPI", "API")

    assert result.positions == (4,)
    assert result.occurrences == 1


def test_multiple_occurrences(matcher: NaiveMatcher) -> None:
    result = matcher.search("banana", "ana")

    assert result.positions == (1, 3)


def test_no_occurrence(matcher: NaiveMatcher) -> None:
    result = matcher.search("FastAPI", "Django")

    assert result.positions == ()
    assert result.occurrences == 0


def test_pattern_longer_than_text(matcher: NaiveMatcher) -> None:
    result = matcher.search("API", "FastAPI")

    assert result.positions == ()


def test_empty_text(matcher: NaiveMatcher) -> None:
    result = matcher.search("", "API")

    assert result.positions == ()


def test_empty_pattern(matcher: NaiveMatcher) -> None:
    with pytest.raises(ValueError):
        matcher.search("FastAPI", "")


def test_case_sensitivity(matcher: NaiveMatcher) -> None:
    result = matcher.search("API api API", "API")

    assert result.positions == (0, 8)


def test_overlapping_occurrences(matcher: NaiveMatcher) -> None:
    result = matcher.search("ABABABA", "ABA")

    assert result.positions == (0, 2, 4)


def test_metrics(matcher: NaiveMatcher) -> None:
    result = matcher.search("AAAA", "AA")

    assert result.algorithm == "naive"
    assert result.positions == (0, 1, 2)
    assert result.comparisons == 6

    assert result.preprocessing_time_ns == 0
    assert result.search_time_ns >= 0

    assert result.text_length == 4
    assert result.pattern_length == 2

    assert result.total_time_ns == result.search_time_ns
