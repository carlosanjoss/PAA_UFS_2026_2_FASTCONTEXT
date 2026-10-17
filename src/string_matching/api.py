
"""FastAPI routes for AV2 exact string matching."""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.services.string_matching_service import (
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
    max_results: int = Field(default=20, ge=1, le=200)


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
            service._registry.available_names()
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
            "positions": list(match.positions),
            "occurrences": match.occurrences,
            "comparisons": match.comparisons,
        }
        for match in result.matches[:request.max_results]
    ]

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
    }
