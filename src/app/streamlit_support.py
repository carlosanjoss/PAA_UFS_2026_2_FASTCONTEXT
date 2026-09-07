from __future__ import annotations

from typing import Any

from src.app.models import (
    ApplicationHealthReport,
)
from src.retrieval.models import (
    RetrievalResult,
)


def format_duration_ns(
    value: int | None,
) -> str:
    """Format a nanosecond duration for user interfaces."""

    if value is None:
        return "N/A"

    if value < 0:
        raise ValueError(
            "Duration cannot be negative."
        )

    if value < 1_000:
        return f"{value} ns"

    if value < 1_000_000:
        microseconds = value / 1_000

        return (
            f"{microseconds:.2f} µs"
        )

    if value < 1_000_000_000:
        milliseconds = (
            value / 1_000_000
        )

        return (
            f"{milliseconds:.2f} ms"
        )

    seconds = (
        value / 1_000_000_000
    )

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
            "Content": chunk.content,
        }
        for chunk in result.chunks
    ]


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