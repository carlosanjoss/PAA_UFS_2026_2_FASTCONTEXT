from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from src.app.models import (
    ApplicationHealthReport,
)
from src.retrieval.models import (
    RetrievalResult,
)


@dataclass(frozen=True, slots=True)
class StrategyPresentation:
    """Human-readable presentation details for one real retriever."""

    identifier: str
    label: str
    description: str


@dataclass(frozen=True, slots=True)
class RetrievalTraceStep:
    """One evidence-backed stage in the retrieval execution trace."""

    title: str
    detail: str


_STRATEGY_PRESENTATIONS: dict[str, StrategyPresentation] = {
    "linear": StrategyPresentation(
        identifier="linear",
        label="Linear",
        description="TF-IDF · Sequential scan · Merge Sort",
    ),
    "indexed": StrategyPresentation(
        identifier="indexed",
        label="Indexed",
        description="Inverted index · Binary search · TF-IDF · Merge Sort",
    ),
    "optimized": StrategyPresentation(
        identifier="optimized",
        label="Optimized",
        description="Inverted index · Binary search · TF-IDF · Top-k min-heap",
    ),
    "semantic": StrategyPresentation(
        identifier="semantic",
        label="Semantic",
        description="BGE-small · FAISS IndexFlatIP",
    ),
}


def format_duration_ns(
    value: int | None,
) -> str:
    """Format a nanosecond duration for user interfaces."""

    if value is None:
        return "N/A"

    if value < 0:
        raise ValueError("Duration cannot be negative.")

    if value < 1_000:
        return f"{value} ns"

    if value < 1_000_000:
        microseconds = value / 1_000

        return f"{microseconds:.2f} µs"

    if value < 1_000_000_000:
        milliseconds = value / 1_000_000

        return f"{milliseconds:.2f} ms"

    seconds = value / 1_000_000_000

    return f"{seconds:.2f} s"


def format_optional_number(
    value: float | None,
) -> str:
    """Format an optional numeric metric."""

    if value is None:
        return "N/A"

    if isinstance(value, float):
        return f"{value:.4f}"

    return str(value)


def get_strategy_presentation(
    identifier: str,
) -> StrategyPresentation:
    """Return UI details for a registered retrieval identifier."""

    normalized_identifier = identifier.strip().lower()
    presentation = _STRATEGY_PRESENTATIONS.get(normalized_identifier)

    if presentation is not None:
        return presentation

    return StrategyPresentation(
        identifier=normalized_identifier,
        label=normalized_identifier.replace("_", " ").title(),
        description="Configured retriever",
    )


def build_retrieval_trace(
    result: RetrievalResult,
) -> tuple[RetrievalTraceStep, ...]:
    """Describe the real stages and metadata from one retrieval result."""

    metadata: Mapping[str, Any] = result.metadata or {}
    representation = _metadata_text(
        metadata,
        "representation",
    )
    model = _metadata_text(metadata, "model")
    candidate_strategy = _metadata_text(
        metadata,
        "candidate_strategy",
    )
    ranking_strategy = _metadata_text(
        metadata,
        "ranking_strategy",
    )

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


def build_metric_rows(
    result: RetrievalResult,
) -> list[dict[str, str]]:
    """Convert available retrieval metrics into display-ready rows."""

    metrics = result.metrics
    metadata: Mapping[str, Any] = result.metadata or {}

    rows = [
        {
            "Metric": "Retrieval time",
            "Value": format_duration_ns(metrics.retrieval_time_ns),
        },
        {
            "Metric": "Sorting / ranking time",
            "Value": format_duration_ns(metrics.sorting_time_ns),
        },
        {
            "Metric": "Comparisons",
            "Value": format_optional_number(metrics.comparisons),
        },
        {
            "Metric": "Chunks scored",
            "Value": format_optional_number(metrics.chunks_scored),
        },
        {
            "Metric": "Candidates found",
            "Value": format_optional_number(metrics.candidates_found),
        },
        {
            "Metric": "Index build time",
            "Value": format_duration_ns(metrics.index_build_time_ns),
        },
        {
            "Metric": "Peak memory",
            "Value": _format_memory_mb(metrics.peak_memory_mb),
        },
    ]

    for key, label in (
        ("query_embedding_time_ns", "Query embedding time"),
        ("faiss_search_time_ns", "FAISS search time"),
        ("index_load_time_ns", "Persisted index load time"),
    ):
        value = metadata.get(key)
        rows.append(
            {
                "Metric": label,
                "Value": format_duration_ns(value if isinstance(value, int) else None),
            }
        )

    for key, label in (
        ("index_source", "Semantic index source"),
        ("persistence_status", "Semantic persistence"),
    ):
        value = metadata.get(key)
        rows.append(
            {
                "Metric": label,
                "Value": str(value) if value is not None else "N/A",
            }
        )

    return rows


def build_chunk_details(
    result: RetrievalResult,
) -> list[dict[str, Any]]:
    """Build concise, metadata-rich chunk cards for the UI."""

    return [
        {
            "rank": chunk.rank,
            "score": f"{chunk.score:.4f}",
            "section_title": chunk.section_title,
            "source_path": chunk.source_path,
            "chunk_id": chunk.chunk_id,
            "token_count": (
                str(chunk.token_count) if chunk.token_count is not None else "N/A"
            ),
            "content": chunk.content,
        }
        for chunk in result.chunks
    ]


def build_retrieval_rows(
    result: RetrievalResult,
) -> list[dict[str, Any]]:
    """Convert retrieved chunks into tabular UI rows."""

    return [
        {
            "Rank": chunk.rank,
            "Chunk ID": chunk.chunk_id,
            "Score": chunk.score,
            "Section": chunk.section_title,
            "Source": chunk.source_path,
            "Tokens": chunk.token_count,
            "Content": chunk.content,
        }
        for chunk in result.chunks
    ]


def _format_memory_mb(
    value: float | None,
) -> str:
    """Format an optional memory reading without coercing None to zero."""

    if value is None:
        return "N/A"

    return f"{value:.2f} MB"


def _metadata_text(
    metadata: Mapping[str, Any],
    key: str,
) -> str:
    """Read a displayable metadata value without inventing a measurement."""

    value = metadata.get(key)

    if value is None:
        return "Not instrumented"

    return str(value).replace("_", " ")


def build_health_rows(
    report: ApplicationHealthReport,
) -> list[dict[str, str]]:
    """Convert health information into tabular UI rows."""

    return [
        {
            "Component": component.name,
            "Status": component.status,
            "Message": component.message,
        }
        for component in report.components
    ]
