from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

INSUFFICIENT_CONTEXT_RESPONSE = (
    "The provided context is insufficient to answer this question."
)

_CITATION_PATTERN = re.compile(
    r"\[([A-Za-z0-9_.:/-]+)\]"
)


@dataclass(frozen=True, slots=True)
class CitationValidationResult:
    """Result of validating citations in an LLM answer."""

    citations: tuple[str, ...]
    valid_citations: tuple[str, ...]
    invalid_citations: tuple[str, ...]
    citation_required: bool
    citation_valid: bool


def _deduplicate(
    values: Sequence[str],
) -> tuple[str, ...]:
    """Remove duplicates while preserving order."""

    return tuple(
        dict.fromkeys(values)
    )


def extract_citations(
    answer: str,
) -> tuple[str, ...]:
    """Extract unique chunk citations from an answer."""

    matches = _CITATION_PATTERN.findall(
        answer
    )

    return _deduplicate(
        matches
    )


def validate_citations(
    answer: str,
    available_chunk_ids: Sequence[str],
) -> CitationValidationResult:
    """Validate citations against retrieved chunk identifiers."""

    citations = extract_citations(
        answer
    )

    available_ids = set(
        available_chunk_ids
    )

    valid_citations = tuple(
        citation
        for citation in citations
        if citation in available_ids
    )

    invalid_citations = tuple(
        citation
        for citation in citations
        if citation not in available_ids
    )

    normalized_answer = answer.strip()

    insufficient_context = (
        normalized_answer
        == INSUFFICIENT_CONTEXT_RESPONSE
    )

    if not available_ids:
        citation_required = False
        citation_valid = (
            insufficient_context
        )

    elif insufficient_context:
        citation_required = False
        citation_valid = True

    else:
        citation_required = True

        citation_valid = (
            bool(valid_citations)
            and not invalid_citations
        )

    return CitationValidationResult(
        citations=citations,
        valid_citations=valid_citations,
        invalid_citations=invalid_citations,
        citation_required=citation_required,
        citation_valid=citation_valid,
    )