
import pytest

from src.string_matching.base import StringMatcher
from src.string_matching.naive import NaiveMatcher
from src.string_matching.registry import (
    StringMatcherRegistry,
    create_default_registry,
)


def test_default_registry() -> None:
    registry = create_default_registry()

    assert registry.available_names() == ("kmp", "naive")
    assert registry.contains("naive")


def test_create_naive_matcher() -> None:
    registry = create_default_registry()

    matcher = registry.create("naive")

    assert isinstance(matcher, NaiveMatcher)
    assert matcher.name == "naive"


def test_execute_registered_matcher() -> None:
    registry = create_default_registry()

    matcher = registry.create("naive")
    result = matcher.search("ABABABA", "ABA")

    assert result.positions == (0, 2, 4)
    assert result.comparisons == 11


def test_unknown_algorithm() -> None:
    registry = create_default_registry()

    with pytest.raises(ValueError):
        registry.create("unknown")


def test_duplicate_registration() -> None:
    registry = StringMatcherRegistry()

    registry.register("naive", NaiveMatcher)

    with pytest.raises(ValueError):
        registry.register("naive", NaiveMatcher)


def test_empty_algorithm_name() -> None:
    registry = StringMatcherRegistry()

    with pytest.raises(ValueError):
        registry.register("", NaiveMatcher)


def test_case_insensitive_registry_lookup() -> None:
    registry = create_default_registry()

    matcher = registry.create("NAIVE")

    assert isinstance(matcher, NaiveMatcher)


def test_invalid_matcher_factory() -> None:
    registry = StringMatcherRegistry()

    def invalid_factory() -> StringMatcher:
        return "invalid"  # type: ignore[return-value]

    registry.register("invalid", invalid_factory)

    with pytest.raises(TypeError):
        registry.create("invalid")
