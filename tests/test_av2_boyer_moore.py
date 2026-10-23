import pytest

from src.string_matching.boyer_moore import BoyerMooreMatcher


@pytest.fixture
def matcher() -> BoyerMooreMatcher:
    return BoyerMooreMatcher()


def test_name(matcher: BoyerMooreMatcher) -> None:
    assert matcher.name == "boyer-moore"


def test_single_occurrence(matcher: BoyerMooreMatcher) -> None:
    result = matcher.search("FastAPI", "API")

    assert result.positions == (4,)
    assert result.occurrences == 1


def test_non_overlapping_occurrences(matcher: BoyerMooreMatcher) -> None:
    # T01
    result = matcher.search("abcabc", "abc")

    assert result.positions == (0, 3)


def test_no_occurrence(matcher: BoyerMooreMatcher) -> None:
    # T02
    result = matcher.search("abcdef", "xyz")

    assert result.positions == ()
    assert result.occurrences == 0


def test_overlapping_occurrences(matcher: BoyerMooreMatcher) -> None:
    # T03
    result = matcher.search("abababab", "abab")

    assert result.positions == (0, 2, 4)


def test_empty_text(matcher: BoyerMooreMatcher) -> None:
    # T04
    result = matcher.search("", "abc")

    assert result.positions == ()


def test_empty_pattern_raises(matcher: BoyerMooreMatcher) -> None:
    # T05
    with pytest.raises(ValueError):
        matcher.search("abc", "")


def test_pattern_longer_than_text(matcher: BoyerMooreMatcher) -> None:
    # T06
    result = matcher.search("ab", "abc")

    assert result.positions == ()


def test_repeated_overlap(matcher: BoyerMooreMatcher) -> None:
    # T07
    result = matcher.search("aaaaa", "aaa")

    assert result.positions == (0, 1, 2)


def test_unicode(matcher: BoyerMooreMatcher) -> None:
    # T08 - café café / café
    result = matcher.search("café café", "café")

    assert result.positions == (0, 5)


def test_pattern_equals_text(matcher: BoyerMooreMatcher) -> None:
    result = matcher.search("abc", "abc")

    assert result.positions == (0,)


def test_single_character_pattern(matcher: BoyerMooreMatcher) -> None:
    result = matcher.search("abracadabra", "a")

    assert result.positions == (0, 3, 5, 7, 10)


def test_match_at_start_only(matcher: BoyerMooreMatcher) -> None:
    result = matcher.search("abcxxxx", "abc")

    assert result.positions == (0,)


def test_match_at_end_only(matcher: BoyerMooreMatcher) -> None:
    result = matcher.search("xxxxabc", "abc")

    assert result.positions == (4,)


def test_case_sensitivity(matcher: BoyerMooreMatcher) -> None:
    result = matcher.search("API api API", "API")

    assert result.positions == (0, 8)


def test_special_characters(matcher: BoyerMooreMatcher) -> None:
    result = matcher.search("a.b.c.a.b", "a.b")

    assert result.positions == (0, 6)


def test_bad_character_shift_metrics(matcher: BoyerMooreMatcher) -> None:
    # On a text with no match, the bad-character rule should let
    # Boyer-Moore use fewer comparisons than the naive worst case.
    result = matcher.search("a" * 20, "bbbb")

    assert result.positions == ()
    assert result.comparisons > 0
    assert result.preprocessing_time_ns >= 0
    assert result.search_time_ns >= 0


def test_metrics_shape(matcher: BoyerMooreMatcher) -> None:
    result = matcher.search("aaaa", "aa")

    assert result.algorithm == "boyer-moore"
    assert result.positions == (0, 1, 2)
    assert result.text_length == 4
    assert result.pattern_length == 2
    assert result.preprocessing_time_ns >= 0
    assert result.search_time_ns >= 0
    assert result.total_time_ns == (
        result.preprocessing_time_ns + result.search_time_ns
    )
