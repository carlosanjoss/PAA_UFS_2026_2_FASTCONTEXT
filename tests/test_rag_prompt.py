import pytest

from src.rag.prompt import ContextChunk, build_context, build_rag_prompt


def test_build_context_with_single_chunk() -> None:
    chunk = ContextChunk(
        chunk_id="chunk_001",
        content="FastAPI supports dependency injection.",
        source_path="tutorial/dependencies/index.md",
        section_title="Dependencies",
        score=0.95,
    )

    context = build_context([chunk])

    assert "[chunk_001]" in context
    assert "Dependencies" in context
    assert "FastAPI supports dependency injection." in context


def test_build_context_without_chunks() -> None:
    assert build_context([]) == "No context was retrieved."


def test_build_rag_prompt_contains_query_and_context() -> None:
    chunk = ContextChunk(
        chunk_id="chunk_001",
        content="FastAPI supports dependency injection.",
        source_path="tutorial/dependencies/index.md",
        section_title="Dependencies",
    )

    prompt = build_rag_prompt(
        query="How do dependencies work?",
        chunks=[chunk],
    )

    assert "How do dependencies work?" in prompt
    assert "FastAPI supports dependency injection." in prompt
    assert "[chunk_001]" in prompt


def test_build_rag_prompt_rejects_empty_query() -> None:
    with pytest.raises(ValueError, match="Query cannot be empty"):
        build_rag_prompt(
            query="   ",
            chunks=[],
        )