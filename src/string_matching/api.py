
"""FastAPI routes for AV2 exact string matching."""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator

from src.app.runtime import get_application
from src.rag.prompt import ContextChunk
from src.rag.providers.base import GenerationConfig, LLMProviderError
from src.services.string_matching_service import (
    ChunkMatch,
    StringMatchingService,
)

router = APIRouter(
    prefix="/api/av2/string-matching",
    tags=["AV2 - String Matching"],
)


class StringMatchingRequest(BaseModel):
    """Request for exact pattern matching over the corpus."""

    pattern: str = Field(min_length=1, max_length=2000)
    algorithm: str = Field(default="naive", min_length=1)
    max_results: int = Field(default=20, ge=1, le=50)
    use_llm: bool = False
    question: str | None = Field(default=None, max_length=2000)
    llm_context_limit: int = Field(default=5, ge=1, le=10)
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)

    @field_validator("pattern")
    @classmethod
    def reject_blank_pattern(cls, value: str) -> str:
        """Reject whitespace-only input without changing valid literals."""
        if not value.strip():
            raise ValueError("pattern cannot be blank")
        return value

    @model_validator(mode="after")
    def validate_llm_question(self) -> StringMatchingRequest:
        """Require an explicit question only when grounded generation is on."""
        if self.use_llm and not (self.question and self.question.strip()):
            raise ValueError("question cannot be blank when use_llm is enabled")
        if self.question is not None:
            self.question = self.question.strip()
        return self


@lru_cache(maxsize=1)
def get_string_matching_service() -> StringMatchingService:
    """Load the corpus once per application process."""
    return StringMatchingService()


ServiceDependency = Annotated[
    StringMatchingService,
    Depends(get_string_matching_service),
]


@router.get("/algorithms")
def list_algorithms(
    service: ServiceDependency,
) -> dict[str, Any]:
    """Return available string matching algorithms."""
    return {
        "algorithms": list(
            service.available_algorithms
        ),
        "corpus_size": service.corpus_size,
    }


@router.post("/search")
def search_pattern(
    request: StringMatchingRequest,
    service: ServiceDependency,
) -> dict[str, Any]:
    """Search an exact pattern throughout the corpus."""

    try:
        result = service.search(
            pattern=request.pattern,
            algorithm=request.algorithm,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    matches = [
        {
            "chunk_id": match.chunk_id,
            "source_path": match.source_path,
            "section_title": match.section_title,
            "content": match.content,
            "positions": list(match.positions),
            "occurrences": match.occurrences,
            "comparisons": match.comparisons,
        }
        for match in result.matches[:request.max_results]
    ]

    llm = _generate_llm_answer(request, result.matches)

    return {
        "algorithm": result.algorithm,
        "pattern": result.pattern,
        "total_chunks": result.total_chunks,
        "matching_chunks": result.matching_chunks,
        "total_occurrences": result.total_occurrences,
        "total_comparisons": result.total_comparisons,
        "preprocessing_time_ns": (
            result.preprocessing_time_ns
        ),
        "search_time_ns": result.search_time_ns,
        "total_time_ns": result.total_time_ns,
        "returned_chunks": len(matches),
        "matches": matches,
        "llm": llm,
    }


def _generate_llm_answer(
    request: StringMatchingRequest,
    matches: tuple[ChunkMatch, ...],
) -> dict[str, Any]:
    """Optionally answer a question from exact-match chunks only."""
    if not request.use_llm:
        return _llm_payload(status="disabled", question=request.question)

    question = request.question or ""
    if not matches:
        return _llm_payload(
            status="no_context",
            question=question,
            fallback_reason=(
                "Nenhum chunk contém o padrão exato; a LLM não foi executada."
            ),
        )

    application = get_application()
    if not application.provider.is_available():
        return _llm_payload(
            status="unavailable",
            question=question,
            fallback_reason="O provedor de LLM configurado está indisponível.",
        )

    context_matches = matches[: request.llm_context_limit]
    chunks = tuple(
        ContextChunk(
            chunk_id=match.chunk_id,
            content=match.content,
            source_path=match.source_path,
            section_title=match.section_title,
        )
        for match in context_matches
    )

    try:
        answer = application.rag_pipeline.answer(
            query=question,
            chunks=chunks,
            generation_config=GenerationConfig(
                temperature=request.temperature,
                max_tokens=2048,
                think=False,
            ),
        )
    except LLMProviderError as exc:
        return _llm_payload(
            status="unavailable",
            question=question,
            context_chunk_ids=[chunk.chunk_id for chunk in chunks],
            fallback_reason=str(exc) or type(exc).__name__,
        )
    except Exception as exc:
        return _llm_payload(
            status="error",
            question=question,
            context_chunk_ids=[chunk.chunk_id for chunk in chunks],
            fallback_reason=(
                "A resposta com LLM não pôde ser gerada; "
                f"a busca exata permanece válida ({type(exc).__name__})."
            ),
        )

    return _llm_payload(
        status="generated",
        question=question,
        answer=answer.answer,
        provider=answer.provider,
        model=answer.model,
        citations=list(answer.citations),
        citation_valid=answer.citation_valid,
        generation_time_ns=answer.generation_time_ns,
        context_chunk_ids=[chunk.chunk_id for chunk in chunks],
    )


def _llm_payload(
    *,
    status: str,
    question: str | None,
    answer: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    citations: list[str] | None = None,
    citation_valid: bool | None = None,
    generation_time_ns: int | None = None,
    context_chunk_ids: list[str] | None = None,
    fallback_reason: str | None = None,
) -> dict[str, Any]:
    """Build the stable frontend contract for optional AV2 generation."""
    return {
        "status": status,
        "question": question,
        "answer": answer,
        "provider": provider,
        "model": model,
        "citations": citations or [],
        "citation_valid": citation_valid,
        "generation_time_ns": generation_time_ns,
        "context_chunk_ids": context_chunk_ids or [],
        "fallback_reason": fallback_reason,
    }
