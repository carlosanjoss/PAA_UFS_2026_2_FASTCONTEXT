from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """Ranked chunk returned by a retrieval algorithm."""

    chunk_id: str
    content: str
    source_path: str
    section_title: str
    score: float
    rank: int
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.chunk_id.strip():
            raise ValueError(
                "chunk_id cannot be empty."
            )

        if not self.content.strip():
            raise ValueError(
                "content cannot be empty."
            )

        if not self.source_path.strip():
            raise ValueError(
                "source_path cannot be empty."
            )

        if not self.section_title.strip():
            raise ValueError(
                "section_title cannot be empty."
            )

        if self.rank <= 0:
            raise ValueError(
                "rank must be greater than zero."
            )

        if not isfinite(self.score):
            raise ValueError(
                "score must be a finite number."
            )


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    """Performance measurements produced during retrieval."""

    retrieval_time_ns: int

    sorting_time_ns: int | None = None
    index_build_time_ns: int | None = None

    comparisons: int | None = None
    chunks_scored: int | None = None
    candidates_found: int | None = None

    peak_memory_mb: float | None = None

    def __post_init__(self) -> None:
        integer_metrics = {
            "retrieval_time_ns": self.retrieval_time_ns,
            "sorting_time_ns": self.sorting_time_ns,
            "index_build_time_ns": self.index_build_time_ns,
            "comparisons": self.comparisons,
            "chunks_scored": self.chunks_scored,
            "candidates_found": self.candidates_found,
        }

        for name, value in integer_metrics.items():
            if value is not None and value < 0:
                raise ValueError(
                    f"{name} cannot be negative."
                )

        if self.peak_memory_mb is not None and (
            self.peak_memory_mb < 0
            or not isfinite(
                self.peak_memory_mb
            )
        ):
            raise ValueError(
                "peak_memory_mb must be "
                "a finite non-negative number."
            )


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """Normalized output returned by every retriever."""

    query: str
    algorithm: str
    top_k: int

    chunks: tuple[
        RetrievedChunk,
        ...,
    ]

    metrics: RetrievalMetrics

    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError(
                "query cannot be empty."
            )

        if not self.algorithm.strip():
            raise ValueError(
                "algorithm cannot be empty."
            )

        if self.top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        if len(self.chunks) > self.top_k:
            raise ValueError(
                "The number of returned chunks "
                "cannot exceed top_k."
            )

        chunk_ids = [
            chunk.chunk_id
            for chunk in self.chunks
        ]

        if (
            len(chunk_ids)
            != len(set(chunk_ids))
        ):
            raise ValueError(
                "Retrieved chunks must have "
                "unique chunk identifiers."
            )

        expected_ranks = list(
            range(
                1,
                len(self.chunks) + 1,
            )
        )

        actual_ranks = [
            chunk.rank
            for chunk in self.chunks
        ]

        if (
            actual_ranks
            != expected_ranks
        ):
            raise ValueError(
                "Retrieved chunks must be "
                "ordered with consecutive ranks "
                "starting at 1."
            )

        candidates_found = (
            self.metrics.candidates_found
        )

        if (
            candidates_found is not None
            and candidates_found
            < len(self.chunks)
        ):
            raise ValueError(
                "candidates_found cannot be "
                "smaller than the number of "
                "returned chunks."
            )