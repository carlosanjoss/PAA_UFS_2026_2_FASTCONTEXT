"""Test Markdown parsing and preservation behavior."""

from __future__ import annotations

import pytest

from src.preprocessing.normalize import ParsedMarkdown, normalize_markdown


def test_normalize_markdown_returns_parsed_markdown() -> None:
    result = normalize_markdown("# FastAPI")

    assert isinstance(result, ParsedMarkdown)


def test_normalize_markdown_preserves_text() -> None:
    text = "# FastAPI\n\nFastAPI is a web framework."

    result = normalize_markdown(text)

    assert result.text == text


def test_normalize_markdown_parses_headings() -> None:
    result = normalize_markdown("# FastAPI")

    assert any(token.type == "heading_open" for token in result.tokens)


def test_normalize_markdown_parses_paragraphs() -> None:
    result = normalize_markdown("FastAPI is a web framework.")

    assert any(token.type == "paragraph_open" for token in result.tokens)


def test_normalize_markdown_preserves_code_block() -> None:
    text = "```python\nx = 10\nprint(x)\n```"

    result = normalize_markdown(text)

    assert result.text == text

    fence_tokens = [
        token
        for token in result.tokens
        if token.type == "fence"
    ]

    assert len(fence_tokens) == 1
    assert fence_tokens[0].content == "x = 10\nprint(x)\n"


def test_normalize_markdown_preserves_technical_names() -> None:
    text = "BAAI/bge-small-en-v1.5 FastAPI OAuth2 JWT Pydantic"

    result = normalize_markdown(text)

    assert result.text == text


def test_normalize_markdown_preserves_nested_headings() -> None:
    text = "# FastAPI\n\n## Security\n\n### OAuth2"

    result = normalize_markdown(text)

    headings = [
        token
        for token in result.tokens
        if token.type == "heading_open"
    ]

    assert len(headings) == 3
    assert [token.tag for token in headings] == [
        "h1",
        "h2",
        "h3",
    ]


def test_normalize_markdown_preserves_links() -> None:
    text = "[FastAPI](https://fastapi.tiangolo.com/)"

    result = normalize_markdown(text)

    assert result.text == text


def test_normalize_markdown_normalizes_empty_lines_only_by_validation() -> None:
    text = "\n\nFastAPI\n\n"

    result = normalize_markdown(text)

    assert result.text == text


def test_normalize_markdown_rejects_empty_document() -> None:
    with pytest.raises(ValueError, match="Markdown document is empty."):
        normalize_markdown("")


def test_normalize_markdown_rejects_whitespace_only_document() -> None:
    with pytest.raises(ValueError, match="Markdown document is empty."):
        normalize_markdown("   \n\t\n")