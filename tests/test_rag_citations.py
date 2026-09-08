from src.rag.citations import (
    INSUFFICIENT_CONTEXT_RESPONSE,
    extract_citations,
    validate_citations,
)


def test_extract_single_citation() -> None:
    citations = extract_citations(
        "FastAPI supports this [chunk_001]."
    )

    assert citations == (
        "chunk_001",
    )


def test_extract_multiple_citations() -> None:
    citations = extract_citations(
        "First fact [chunk_001]. "
        "Second fact [chunk_002]."
    )

    assert citations == (
        "chunk_001",
        "chunk_002",
    )


def test_extract_citations_removes_duplicates() -> None:
    citations = extract_citations(
        "Fact [chunk_001]. "
        "Another fact [chunk_001]."
    )

    assert citations == (
        "chunk_001",
    )


def test_validate_known_citation() -> None:
    result = validate_citations(
        answer=(
            "FastAPI supports dependencies "
            "[chunk_001]."
        ),
        available_chunk_ids=[
            "chunk_001",
        ],
    )

    assert result.citation_valid is True

    assert result.valid_citations == (
        "chunk_001",
    )

    assert (
        result.invalid_citations
        == ()
    )


def test_missing_citation_is_invalid() -> None:
    result = validate_citations(
        answer=(
            "FastAPI supports dependencies."
        ),
        available_chunk_ids=[
            "chunk_001",
        ],
    )

    assert result.citation_required is True
    assert result.citation_valid is False

    assert result.citations == ()


def test_unknown_citation_is_invalid() -> None:
    result = validate_citations(
        answer=(
            "FastAPI supports dependencies "
            "[chunk_999]."
        ),
        available_chunk_ids=[
            "chunk_001",
        ],
    )

    assert result.citation_valid is False

    assert result.invalid_citations == (
        "chunk_999",
    )


def test_valid_and_invalid_citations_are_invalid() -> None:
    result = validate_citations(
        answer=(
            "Supported statement [chunk_001]. "
            "Unsupported statement [chunk_999]."
        ),
        available_chunk_ids=[
            "chunk_001",
        ],
    )

    assert result.citation_valid is False

    assert result.valid_citations == (
        "chunk_001",
    )

    assert result.invalid_citations == (
        "chunk_999",
    )


def test_insufficient_context_response_is_valid() -> None:
    result = validate_citations(
        answer=(
            INSUFFICIENT_CONTEXT_RESPONSE
        ),
        available_chunk_ids=[
            "chunk_001",
        ],
    )

    assert result.citation_required is False
    assert result.citation_valid is True


def test_no_chunks_requires_insufficient_response() -> None:
    result = validate_citations(
        answer="FastAPI does something.",
        available_chunk_ids=[],
    )

    assert result.citation_valid is False


def test_no_chunks_accepts_insufficient_response() -> None:
    result = validate_citations(
        answer=(
            INSUFFICIENT_CONTEXT_RESPONSE
        ),
        available_chunk_ids=[],
    )

    assert result.citation_valid is True