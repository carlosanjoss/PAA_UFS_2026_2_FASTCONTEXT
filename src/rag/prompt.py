from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from src.rag.citations import (
    INSUFFICIENT_CONTEXT_RESPONSE,
)


@dataclass(frozen=True, slots=True)
class ContextChunk:
    """Chunk of retrieved documentation used as RAG context."""

    chunk_id: str
    content: str
    source_path: str
    section_title: str
    score: float | None = None


@dataclass(frozen=True, slots=True)
class RAGPrompt:
    """Structured prompt used by the RAG pipeline."""

    system: str
    user: str


SYSTEM_INSTRUCTION = f"""You are a FastAPI documentation assistant.

Answer only from the provided documentation context.

Requirements:
- Answer directly.
- Use at most three short sentences.
- Do not explain your reasoning.
- Do not describe or analyze the context.
- Do not repeat the question.
- Every factual answer must contain at least one valid chunk citation.
- Use only citations listed under "Allowed citations".
- Place citations immediately after the factual statement they support.
- Never invent chunk identifiers.
- If the context does not contain the answer, reply exactly:
  "{INSUFFICIENT_CONTEXT_RESPONSE}"
"""


def build_context(
    chunks: Sequence[ContextChunk],
) -> str:
    """Convert retrieved chunks into structured context."""

    if not chunks:
        return "No context was retrieved."

    sections: list[str] = []

    for chunk in chunks:
        sections.append(
            f"[{chunk.chunk_id}]\n"
            f"Source: {chunk.source_path}\n"
            f"Section: {chunk.section_title}\n"
            f"{chunk.content.strip()}"
        )

    return "\n\n---\n\n".join(
        sections
    )


def build_allowed_citations(
    chunks: Sequence[ContextChunk],
) -> str:
    """Build the list of chunk identifiers allowed in the answer."""

    if not chunks:
        return "None"

    return ", ".join(
        f"[{chunk.chunk_id}]"
        for chunk in chunks
    )


def build_rag_prompt(
    query: str,
    chunks: Sequence[ContextChunk],
) -> RAGPrompt:
    """Build a structured RAG prompt."""

    normalized_query = query.strip()

    if not normalized_query:
        raise ValueError(
            "Query cannot be empty."
        )

    context = build_context(
        chunks
    )

    allowed_citations = (
        build_allowed_citations(
            chunks
        )
    )

    user_message = (
        f"Context:\n{context}\n\n"
        "Allowed citations:\n"
        f"{allowed_citations}\n\n"
        f"Question:\n{normalized_query}\n\n"
        "Answer directly and include at least one "
        "allowed citation when making factual claims.\n\n"
        "Answer:"
    )

    return RAGPrompt(
        system=SYSTEM_INSTRUCTION,
        user=user_message,
    )