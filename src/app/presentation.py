"""Framework-independent presentation helpers for the web API."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from src.retrieval.models import RetrievalResult


@dataclass(frozen=True, slots=True)
class RetrievalTraceStep:
    """One evidence-backed stage in a retrieval execution trace."""

    title: str
    detail: str


def build_retrieval_trace(
    result: RetrievalResult,
) -> tuple[RetrievalTraceStep, ...]:
    """Describe real stages using only metadata from a retrieval result."""

    metadata: Mapping[str, Any] = result.metadata or {}
    representation = _metadata_text(metadata, "representation")
    model = _metadata_text(metadata, "model")
    candidate_strategy = _metadata_text(metadata, "candidate_strategy")
    ranking_strategy = _metadata_text(metadata, "ranking_strategy")

    representation_detail = representation
    if model != "Not instrumented":
        representation_detail = f"{representation} · {model}"

    return (
        RetrievalTraceStep(
            title="Representation",
            detail=representation_detail,
        ),
        RetrievalTraceStep(
            title="Candidate discovery",
            detail=candidate_strategy,
        ),
        RetrievalTraceStep(
            title="Ranking",
            detail=ranking_strategy,
        ),
        RetrievalTraceStep(
            title="Top-k selection",
            detail=f"Requested {result.top_k} chunks",
        ),
        RetrievalTraceStep(
            title="Retrieved context",
            detail=f"Returned {len(result.chunks)} chunks",
        ),
    )


def _metadata_text(
    metadata: Mapping[str, Any],
    key: str,
) -> str:
    value = metadata.get(key)
    if value is None:
        return "Not instrumented"
    return str(value).replace("_", " ")
