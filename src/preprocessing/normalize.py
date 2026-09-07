"""Parse Markdown once while preserving source text and parser tokens."""

from __future__ import annotations

from dataclasses import dataclass

from markdown_it import MarkdownIt
from markdown_it.token import Token


_MARKDOWN = MarkdownIt()


@dataclass(frozen=True)
class ParsedMarkdown:
    text: str
    tokens: list[Token]


def normalize_markdown(text: str) -> ParsedMarkdown:
    if not text.strip():
        raise ValueError("Markdown document is empty.")

    tokens = _MARKDOWN.parse(text)

    return ParsedMarkdown(text=text, tokens=tokens,)