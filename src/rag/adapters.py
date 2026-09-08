from __future__ import annotations

from src.rag.prompt import ContextChunk
from src.retrieval.models import (
    RetrievalResult,
    RetrievedChunk,
)


def retrieved_chunk_to_context_chunk(
    chunk: RetrievedChunk,
) -> ContextChunk:
    """Convert a retrieved chunk into RAG context."""

    return ContextChunk(
        chunk_id=chunk.chunk_id,
        content=chunk.content,
        source_path=chunk.source_path,
        section_title=chunk.section_title,
        score=chunk.score,
    )


def retrieval_result_to_context_chunks(
    result: RetrievalResult,
) -> tuple[ContextChunk, ...]:
    """Convert ranked retrieval results into RAG context."""

    return tuple(
        retrieved_chunk_to_context_chunk(
            chunk
        )
        for chunk in result.chunks
    )