import pytest

from src.string_matching.rabin_karp import RabinKarpMatcher


@pytest.fixture
def matcher() -> RabinKarpMatcher:
    return RabinKarpMatcher()


def test_name(matcher: RabinKarpMatcher) -> None:
    assert matcher.name == "rabin-karp"


def test_single_occurrence(matcher: RabinKarpMatcher) -> None:
    result = matcher.search("FastAPI", "API")

    assert result.positions == (4,)
    assert result.occurrences == 1


def test_non_overlapping_occurrences(matcher: RabinKarpMatcher) -> None:
    # T01
    result = matcher.search("abcabc", "abc")

    assert result.positions == (0, 3)


def test_no_occurrence(matcher: RabinKarpMatcher) -> None:
    # T02
    result = matcher.search("abcdef", "xyz")

    assert result.positions == ()
    assert result.occurrences == 0


def test_overlapping_occurrences(matcher: RabinKarpMatcher) -> None:
    # T03
    result = matcher.search("abababab", "abab")

    assert result.positions == (0, 2, 4)


def test_empty_text(matcher: RabinKarpMatcher) -> None:
    # T04
    result = matcher.search("", "abc")

    assert result.positions == ()


def test_empty_pattern_raises(matcher: RabinKarpMatcher) -> None:
    # T05
    with pytest.raises(ValueError):
        matcher.search("abc", "")


def test_pattern_longer_than_text(matcher: RabinKarpMatcher) -> None:
    # T06
    result = matcher.search("ab", "abc")

    assert result.positions == ()


def test_repeated_overlap(matcher: RabinKarpMatcher) -> None:
    # T07
    result = matcher.search("aaaaa", "aaa")

    assert result.positions == (0, 1, 2)


def test_unicode(matcher: RabinKarpMatcher) -> None:
    # T08 - café café / café
    result = matcher.search("café café", "café")

    assert result.positions == (0, 5)


def test_pattern_equals_text(matcher: RabinKarpMatcher) -> None:
    result = matcher.search("abc", "abc")

    assert result.positions == (0,)


def test_single_character_pattern(matcher: RabinKarpMatcher) -> None:
    result = matcher.search("abracadabra", "a")

    assert result.positions == (0, 3, 5, 7, 10)


def test_match_at_start_only(matcher: RabinKarpMatcher) -> None:
    result = matcher.search("abcxxxx", "abc")

    assert result.positions == (0,)


def test_match_at_end_only(matcher: RabinKarpMatcher) -> None:
    result = matcher.search("xxxxabc", "abc")

    assert result.positions == (4,)


def test_case_sensitivity(matcher: RabinKarpMatcher) -> None:
    result = matcher.search("API api API", "API")

    assert result.positions == (0, 8)


def test_metrics_shape(matcher: RabinKarpMatcher) -> None:
    result = matcher.search("aaaa", "aa")

    assert result.algorithm == "rabin-karp"
    assert result.positions == (0, 1, 2)
    assert result.text_length == 4
    assert result.pattern_length == 2
    assert result.preprocessing_time_ns >= 0
    assert result.search_time_ns >= 0
    assert result.total_time_ns == (
        result.preprocessing_time_ns + result.search_time_ns
    )


# --- T10: controlled hash collision -----------------------------------

def test_hashes_collide_for_distinct_strings() -> None:
    """Sanity check: the chosen parameters produce a real collision.

    With base=10 and modulus=11, the polynomial hash of "19" equals the
    hash of "08": (1*10 + 9) % 11 == 19 % 11 == 8, and (0*10 + 8) == 8.
    The two 2-char strings are different but share the same hash.
    """
    matcher = RabinKarpMatcher(base=10, modulus=11)

    hash_19 = (
        matcher._char_code("1") * 10 + matcher._char_code("9")
    ) % 11
    hash_08 = (
        matcher._char_code("0") * 10 + matcher._char_code("8")
    ) % 11

    assert hash_19 == hash_08
    assert "19" != "08"


def test_collision_does_not_produce_false_positive() -> None:
    """A hash collision must NOT be reported as a match.

    Text "08" and pattern "19" collide in hash (see test above), but
    the characters differ, so verification must reject the window and
    no occurrence may be returned.
    """
    matcher = RabinKarpMatcher(base=10, modulus=11)

    result = matcher.search("08", "19")

    # Hash matched -> verification happened (comparisons counted),
    # but the characters differ, so there is NO occurrence.
    assert result.positions == ()
    assert result.comparisons >= 1


def test_collision_still_finds_true_match() -> None:
    """With collision-prone parameters, real matches are still found."""
    matcher = RabinKarpMatcher(base=10, modulus=11)

    result = matcher.search("xx19yy19zz", "19")

    assert result.positions == (2, 6)
