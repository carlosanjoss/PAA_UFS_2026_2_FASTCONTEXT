
from src.string_matching.models import StringMatchResult


def test_string_match_result() -> None:
    result = StringMatchResult(
        algorithm="naive",
        positions=(0, 8, 12),
        comparisons=15,
        preprocessing_time_ns=0,
        search_time_ns=1500,
        text_length=15,
        pattern_length=3,
    )

    assert result.algorithm == "naive"
    assert result.positions == (0, 8, 12)
    assert result.occurrences == 3
    assert result.total_time_ns == 1500
