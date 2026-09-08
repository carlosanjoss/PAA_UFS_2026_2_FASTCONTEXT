import pytest

from src.rag.prompt import (
    ContextChunk,
    RAGPrompt,
    build_allowed_citations,
    build_context,
    build_rag_prompt,
)


def build_sample_chunk() -> ContextChunk:
    """Create a reusable chunk for prompt tests."""

    return ContextChunk(
        chunk_id="chunk_001",
        content=(
            "FastAPI supports dependency "
            "injection."
        ),
        source_path=(
            "tutorial/dependencies/index.md"
        ),
        section_title="Dependencies",
        score=0.95,
    )


def test_build_context_with_single_chunk() -> None:
    chunk = build_sample_chunk()

    context = build_context(
        [chunk]
    )

    assert "[chunk_001]" in context

    assert (
        "Dependencies"
        in context
    )

    assert (
        "tutorial/dependencies/index.md"
        in context
    )

    assert (
        "FastAPI supports dependency "
        "injection."
        in context
    )


def test_build_context_without_chunks() -> None:
    assert (
        build_context([])
        == "No context was retrieved."
    )


def test_build_allowed_citations() -> None:
    chunk = build_sample_chunk()

    result = (
        build_allowed_citations(
            [chunk]
        )
    )

    assert result == "[chunk_001]"


def test_build_allowed_citations_without_chunks() -> None:
    result = (
        build_allowed_citations([])
    )

    assert result == "None"


def test_build_rag_prompt_returns_structured_prompt() -> None:
    prompt = build_rag_prompt(
        query="How do dependencies work?",
        chunks=[
            build_sample_chunk()
        ],
    )

    assert isinstance(
        prompt,
        RAGPrompt,
    )


def test_system_prompt_requires_citations() -> None:
    prompt = build_rag_prompt(
        query="How do dependencies work?",
        chunks=[
            build_sample_chunk()
        ],
    )

    assert (
        "Every factual answer must contain"
        in prompt.system
    )

    assert (
        "Never invent chunk identifiers"
        in prompt.system
    )


def test_prompt_contains_query_and_context() -> None:
    prompt = build_rag_prompt(
        query="How do dependencies work?",
        chunks=[
            build_sample_chunk()
        ],
    )

    assert (
        "How do dependencies work?"
        in prompt.user
    )

    assert (
        "FastAPI supports dependency "
        "injection."
        in prompt.user
    )

    assert (
        "[chunk_001]"
        in prompt.user
    )


def test_prompt_contains_allowed_citations() -> None:
    prompt = build_rag_prompt(
        query="How do dependencies work?",
        chunks=[
            build_sample_chunk()
        ],
    )

    assert (
        "Allowed citations:"
        in prompt.user
    )

    assert (
        "[chunk_001]"
        in prompt.user
    )


def test_prompt_rejects_empty_query() -> None:
    with pytest.raises(
        ValueError,
        match="Query cannot be empty",
    ):
        build_rag_prompt(
            query="   ",
            chunks=[],
        )