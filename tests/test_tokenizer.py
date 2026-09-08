"""Test model-token counting behavior."""

from __future__ import annotations

import pytest

from src.preprocessing.tokenizer import count_tokens


def test_count_tokens_empty_string() -> None:
    assert count_tokens("") == 0


def test_count_tokens_single_word() -> None:
    assert count_tokens("FastAPI") > 0


def test_count_tokens_multiple_words() -> None:
    assert count_tokens("FastAPI is a web framework.") > 0


def test_count_tokens_returns_int() -> None:
    result = count_tokens("FastAPI")

    assert isinstance(result, int)


def test_count_tokens_is_deterministic() -> None:
    text = "FastAPI uses Pydantic models."

    assert count_tokens(text) == count_tokens(text)


def test_count_tokens_increases_with_more_content() -> None:
    short_text = "FastAPI"
    long_text = "FastAPI is a modern web framework for building APIs."

    assert count_tokens(long_text) > count_tokens(short_text)


def test_count_tokens_handles_punctuation() -> None:
    assert count_tokens("FastAPI, Pydantic, OAuth2!") > 0


def test_count_tokens_handles_markdown() -> None:
    text = "# FastAPI\n\nFastAPI is a web framework."

    assert count_tokens(text) > 0


def test_count_tokens_handles_code() -> None:
    text = """```python
def hello():
    return 42
```"""

    assert count_tokens(text) > 0


def test_count_tokens_handles_technical_names() -> None:
    text = "BAAI/bge-small-en-v1.5 OAuth2 JWT Pydantic"

    assert count_tokens(text) > 0


def test_count_tokens_handles_unicode() -> None:
    assert count_tokens("ação informação coração") > 0


def test_count_tokens_handles_whitespace() -> None:
    text = "FastAPI    is\t a framework"

    assert count_tokens(text) > 0


def test_count_tokens_rejects_non_string() -> None:
    with pytest.raises(TypeError):
        count_tokens(None)