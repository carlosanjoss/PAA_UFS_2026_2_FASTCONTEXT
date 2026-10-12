
import pytest

from src.string_matching.base import StringMatcher


def test_cannot_instantiate_abstract_matcher() -> None:
    with pytest.raises(TypeError):
        StringMatcher()


def test_valid_input() -> None:
    StringMatcher.validate_input(
        text="FastAPI documentation",
        pattern="API",
    )


def test_empty_pattern() -> None:
    with pytest.raises(ValueError):
        StringMatcher.validate_input(
            text="FastAPI",
            pattern="",
        )


def test_invalid_text_type() -> None:
    with pytest.raises(TypeError):
        StringMatcher.validate_input(
            text=123,
            pattern="API",
        )


def test_invalid_pattern_type() -> None:
    with pytest.raises(TypeError):
        StringMatcher.validate_input(
            text="FastAPI",
            pattern=123,
        )
