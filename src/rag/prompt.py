from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ContextChunk:
    """Chunk of retrieved documentation used as RAG context."""

    chunk_id: str
    content: str
    source_path: str
    section_title: str
    score: float | None = None


SYSTEM_INSTRUCTION = """You are a technical assistant answering questions about FastAPI.

Use only the provided context to answer the question.

Rules:
1. Do not invent information that is not supported by the context.
2. If the context is insufficient, explicitly say that the available context is insufficient.
3. Prefer concise and technically precise answers.
4. Cite the context chunks used in the answer using their identifiers, for example [chunk_001].
5. Do not claim that a feature exists unless it is supported by the provided context.
"""


def build_context(chunks: Sequence[ContextChunk]) -> str:
    """Convert retrieved chunks into a structured context string."""

    if not chunks:
        return "No context was retrieved."

    sections: list[str] = []

    for chunk in chunks:
        section = (
            f"[{chunk.chunk_id}]\n"
            f"Source: {chunk.source_path}\n"
            f"Section: {chunk.section_title}\n"
            f"Content:\n{chunk.content.strip()}"
        )

        sections.append(section)

    return "\n\n---\n\n".join(sections)


def build_rag_prompt(
    query: str,
    chunks: Sequence[ContextChunk],
) -> str:
    """Build a grounded RAG prompt from a query and retrieved chunks."""

    normalized_query = query.strip()

    if not normalized_query:
        raise ValueError("Query cannot be empty.")

    context = build_context(chunks)

    return (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"CONTEXT\n"
        f"=======\n"
        f"{context}\n\n"
        f"QUESTION\n"
        f"========\n"
        f"{normalized_query}\n\n"
        f"ANSWER\n"
        f"======\n"
    )