"""Test Markdown sectioning, splitting, overlap, and preservation."""

from __future__ import annotations

import pytest

from src.chunking.markdown_chunker import chunk_markdown
from src.preprocessing.normalize import normalize_markdown


def test_chunk_markdown_without_headings() -> None:
    markdown = normalize_markdown(
        "FastAPI is a modern web framework."
    )

    chunks = chunk_markdown(
        markdown,
        max_tokens=400,
        overlap_tokens=60,
    )

    assert len(chunks) == 1
    assert chunks[0]["section"] == ""
    assert chunks[0]["title"] == ""
    assert chunks[0]["content"] == markdown.text
    assert chunks[0]["token_count"] > 0


def test_chunk_markdown_creates_section() -> None:
    markdown = normalize_markdown(
        "# Security\n\n"
        "FastAPI provides security utilities."
    )

    chunks = chunk_markdown(
        markdown,
        max_tokens=400,
        overlap_tokens=60,
    )

    assert len(chunks) == 1
    assert chunks[0]["section"] == "Security"
    assert chunks[0]["title"] == "Security"


def test_chunk_markdown_builds_nested_section_hierarchy() -> None:
    markdown = normalize_markdown(
        "# Security\n\n"
        "Security content.\n\n"
        "## OAuth2\n\n"
        "OAuth2 content.\n\n"
        "### JWT Tokens\n\n"
        "JWT content."
    )

    chunks = chunk_markdown(
        markdown,
        max_tokens=400,
        overlap_tokens=60,
    )

    assert [chunk["section"] for chunk in chunks] == [
        "Security",
        "Security > OAuth2",
        "Security > OAuth2 > JWT Tokens",
    ]

    assert [chunk["title"] for chunk in chunks] == [
        "Security",
        "OAuth2",
        "JWT Tokens",
    ]


def test_chunk_markdown_handles_heading_level_decrease() -> None:
    markdown = normalize_markdown(
        "# Security\n\n"
        "Security content.\n\n"
        "## OAuth2\n\n"
        "OAuth2 content.\n\n"
        "### JWT Tokens\n\n"
        "JWT content.\n\n"
        "## API Keys\n\n"
        "API key content."
    )

    chunks = chunk_markdown(
        markdown,
        max_tokens=400,
        overlap_tokens=60,
    )

    assert [chunk["section"] for chunk in chunks] == [
        "Security",
        "Security > OAuth2",
        "Security > OAuth2 > JWT Tokens",
        "Security > API Keys",
    ]

    assert [chunk["title"] for chunk in chunks] == [
        "Security",
        "OAuth2",
        "JWT Tokens",
        "API Keys",
    ]


def test_chunk_markdown_keeps_small_section_as_one_chunk() -> None:
    markdown = normalize_markdown(
        "# Security\n\n"
        "FastAPI provides authentication and authorization."
    )

    chunks = chunk_markdown(
        markdown,
        max_tokens=400,
        overlap_tokens=60,
    )

    assert len(chunks) == 1


def test_chunk_markdown_splits_large_section() -> None:
    paragraphs = [
        " ".join(["FastAPI"] * 30)
        for _ in range(20)
    ]

    markdown = normalize_markdown(
        "# Security\n\n" + "\n\n".join(paragraphs)
    )

    chunks = chunk_markdown(
        markdown,
        max_tokens=100,
        overlap_tokens=20,
    )

    assert len(chunks) > 1


def test_chunk_markdown_splits_oversized_text_block_without_truncation(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "src.chunking.markdown_chunker.count_tokens",
        lambda text: len(text.split()),
    )
    markdown = normalize_markdown(
        "# Security\n\n" + " ".join(["FastAPI"] * 25)
    )

    chunks = chunk_markdown(
        markdown,
        max_tokens=10,
        overlap_tokens=2,
    )

    assert len(chunks) > 1
    assert all(chunk["token_count"] <= 10 for chunk in chunks)
    assert sum(
        chunk["content"].count("FastAPI")
        for chunk in chunks
    ) >= 25


def test_chunk_markdown_preserves_section_metadata_when_split() -> None:
    paragraphs = [
        " ".join(["FastAPI"] * 30)
        for _ in range(20)
    ]

    markdown = normalize_markdown(
        "# Security\n\n" + "\n\n".join(paragraphs)
    )

    chunks = chunk_markdown(
        markdown,
        max_tokens=100,
        overlap_tokens=20,
    )

    assert all(
        chunk["section"] == "Security"
        for chunk in chunks
    )

    assert all(
        chunk["title"] == "Security"
        for chunk in chunks
    )


def test_chunk_markdown_applies_overlap() -> None:
    paragraphs = [
        f"Paragraph {index} " + " ".join(["FastAPI"] * 10)
        for index in range(20)
    ]

    markdown = normalize_markdown(
        "# Security\n\n" + "\n\n".join(paragraphs)
    )

    chunks = chunk_markdown(
        markdown,
        max_tokens=50,
        overlap_tokens=10,
    )

    assert len(chunks) > 1

    first_tokens = set(
        chunks[0]["content"].split()
    )
    second_tokens = set(
        chunks[1]["content"].split()
    )

    assert first_tokens.intersection(second_tokens)


def test_chunk_markdown_overlap_does_not_exceed_limit() -> None:
    paragraphs = [
        " ".join(["FastAPI"] * 30),
        " ".join(["FastAPI"] * 30),
    ]
    markdown = normalize_markdown(
        "# Security\n\n" + "\n\n".join(paragraphs)
    )

    chunks = chunk_markdown(
        markdown,
        max_tokens=40,
        overlap_tokens=20,
    )

    assert all(chunk["token_count"] <= 40 for chunk in chunks)


def test_chunk_markdown_preserves_code_block() -> None:
    markdown = normalize_markdown(
        "# Example\n\n"
        "```python\n"
        "def hello():\n"
        "    return 42\n"
        "```"
    )

    chunks = chunk_markdown(
        markdown,
        max_tokens=400,
        overlap_tokens=60,
    )

    assert len(chunks) == 1

    content = chunks[0]["content"]

    assert "```python" in content
    assert "def hello():" in content
    assert "return 42" in content
    assert "```" in content


def test_chunk_markdown_preserves_tilde_code_block() -> None:
    markdown = normalize_markdown(
        "# Example\n\n"
        "~~~python\n"
        "print('hello')\n"
        "~~~"
    )

    chunks = chunk_markdown(
        markdown,
        max_tokens=400,
        overlap_tokens=60,
    )

    assert len(chunks) == 1

    content = chunks[0]["content"]

    assert "~~~python" in content
    assert "print('hello')" in content
    assert "~~~" in content


def test_chunk_markdown_splits_oversized_code_block(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "src.chunking.markdown_chunker.count_tokens",
        lambda text: len(text.split()),
    )
    code = "\n".join(
        f"print('line {index}')"
        for index in range(30)
    )
    markdown = normalize_markdown(
        "# Example\n\n```python\n" + code + "\n```"
    )

    chunks = chunk_markdown(
        markdown,
        max_tokens=10,
        overlap_tokens=0,
    )

    assert len(chunks) > 1
    assert all(chunk["token_count"] <= 10 for chunk in chunks)
    code_chunks = [
        chunk
        for chunk in chunks
        if "```python" in chunk["content"]
    ]
    assert code_chunks
    assert all(
        chunk["content"].startswith("```python")
        and chunk["content"].endswith("```")
        for chunk in code_chunks
    )
    assert sum(
        chunk["content"].count("print(")
        for chunk in code_chunks
    ) == 30


def test_chunk_markdown_code_limit_includes_fence_tokens(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "src.chunking.markdown_chunker.count_tokens",
        lambda text: len(text.split()),
    )
    code = "\n".join(
        f"{{'key': '{index}', 'value': 'FastAPI'}}"
        for index in range(20)
    )
    markdown = normalize_markdown(
        "# Example\n\n```JSON hl_lines=\"4-16\"\n"
        + code
        + "\n```"
    )

    chunks = chunk_markdown(
        markdown,
        max_tokens=10,
        overlap_tokens=0,
    )

    assert len(chunks) > 1
    assert all(chunk["token_count"] <= 10 for chunk in chunks)


def test_chunk_markdown_preserves_markdown_content() -> None:
    markdown = normalize_markdown(
        "# Links\n\n"
        "[FastAPI](https://fastapi.tiangolo.com/)\n\n"
        "- item one\n"
        "- item two"
    )

    chunks = chunk_markdown(
        markdown,
        max_tokens=400,
        overlap_tokens=60,
    )

    assert len(chunks) == 1

    content = chunks[0]["content"]

    assert "[FastAPI](https://fastapi.tiangolo.com/)" in content
    assert "- item one" in content
    assert "- item two" in content


def test_chunk_markdown_includes_token_count() -> None:
    markdown = normalize_markdown(
        "# Security\n\n"
        "FastAPI provides security utilities."
    )

    chunks = chunk_markdown(
        markdown,
        max_tokens=400,
        overlap_tokens=60,
    )

    assert isinstance(chunks[0]["token_count"], int)
    assert chunks[0]["token_count"] > 0


def test_chunk_markdown_rejects_non_positive_max_tokens() -> None:
    markdown = normalize_markdown("FastAPI.")

    with pytest.raises(
        ValueError,
        match=r"^max_tokens must be greater than zero\.$",
    ):
        chunk_markdown(
            markdown,
            max_tokens=0,
            overlap_tokens=60,
        )


def test_chunk_markdown_rejects_negative_overlap() -> None:
    markdown = normalize_markdown("FastAPI.")

    with pytest.raises(
        ValueError,
        match=r"^overlap_tokens cannot be negative\.$",
    ):
        chunk_markdown(
            markdown,
            max_tokens=400,
            overlap_tokens=-1,
        )


def test_chunk_markdown_rejects_overlap_equal_to_limit() -> None:
    markdown = normalize_markdown("FastAPI.")

    with pytest.raises(
        ValueError,
        match="overlap_tokens must be less than max_tokens.",
    ):
        chunk_markdown(
            markdown,
            max_tokens=60,
            overlap_tokens=60,
        )