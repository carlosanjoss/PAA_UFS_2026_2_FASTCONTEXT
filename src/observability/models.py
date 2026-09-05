from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

RunMode = Literal[
    "retrieval",
    "rag",
]

RunStatus = Literal[
    "success",
    "error",
]


@dataclass(frozen=True, slots=True)
class ExperimentContext:
    """Optional metadata identifying an experimental execution."""

    experiment_id: str | None = None
    configuration: str | None = None
    representation: str | None = None

    corpus_fraction: float | None = None
    repetition: int | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.corpus_fraction is not None:
            if not 0 < self.corpus_fraction <= 1:
                raise ValueError(
                    "corpus_fraction must be "
                    "greater than zero and "
                    "less than or equal to one."
                )

        if (
            self.repetition is not None
            and self.repetition <= 0
        ):
            raise ValueError(
                "repetition must be greater than zero."
            )

        if (
            self.seed is not None
            and self.seed < 0
        ):
            raise ValueError(
                "seed cannot be negative."
            )


@dataclass(frozen=True, slots=True)
class RunRecord:
    """Serializable record describing one FastContext execution."""

    schema_version: str

    run_id: str
    timestamp_utc: str

    mode: RunMode
    status: RunStatus

    query: str

    algorithm: str | None
    top_k: int | None

    retrieval_time_ns: int | None
    sorting_time_ns: int | None
    index_build_time_ns: int | None

    comparisons: int | None
    chunks_scored: int | None
    candidates_found: int | None

    peak_memory_mb: float | None

    returned_chunks: int | None
    chunk_ids: tuple[str, ...]

    generation_time_ns: int | None = None
    end_to_end_time_ns: int | None = None

    provider: str | None = None
    model: str | None = None

    citation_valid: bool | None = None
    citation_count: int | None = None
    citation_retry_count: int | None = None

    valid_citations: tuple[str, ...] = ()
    invalid_citations: tuple[str, ...] = ()

    error_type: str | None = None
    error_message: str | None = None

    experiment: ExperimentContext | None = None

    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.schema_version.strip():
            raise ValueError(
                "schema_version cannot be empty."
            )

        if not self.run_id.strip():
            raise ValueError(
                "run_id cannot be empty."
            )

        if not self.timestamp_utc.strip():
            raise ValueError(
                "timestamp_utc cannot be empty."
            )

        if not self.query.strip():
            raise ValueError(
                "query cannot be empty."
            )

        if (
            self.top_k is not None
            and self.top_k <= 0
        ):
            raise ValueError(
                "top_k must be greater than zero."
            )

        if (
            self.returned_chunks is not None
            and self.returned_chunks < 0
        ):
            raise ValueError(
                "returned_chunks cannot be negative."
            )

        if self.status == "success":
            if self.algorithm is None:
                raise ValueError(
                    "Successful runs must have an algorithm."
                )

            if self.top_k is None:
                raise ValueError(
                    "Successful runs must have top_k."
                )

            if self.retrieval_time_ns is None:
                raise ValueError(
                    "Successful runs must have "
                    "retrieval_time_ns."
                )

        if self.status == "error":
            if not self.error_type:
                raise ValueError(
                    "Error runs must have error_type."
                )

            if not self.error_message:
                raise ValueError(
                    "Error runs must have error_message."
                )