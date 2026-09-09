"""Performance measurement utilities for FastContext retrieval experiments."""

from __future__ import annotations

import gc
import time
import tracemalloc
from dataclasses import dataclass

from src.retrieval.base import Retriever
from src.retrieval.models import RetrievalResult

MEMORY_METHOD = "tracemalloc_incremental_python_peak"


class RetrievalDeterminismError(RuntimeError):
    """Raised when repeated retrieval produces a different ranking."""


@dataclass(frozen=True, slots=True)
class RetrievalPerformance:
    """Store performance data for one measured retrieval execution."""

    result: RetrievalResult
    total_time_ns: int
    peak_memory_bytes: int | None
    memory_method: str | None

    @property
    def peak_memory_mb(self) -> float | None:
        """Return measured peak memory in MiB."""
        if self.peak_memory_bytes is None:
            return None

        return (
            self.peak_memory_bytes
            / (1024 * 1024)
        )

    @property
    def retrieval_time_ns(self) -> int:
        """Return internal retrieval time."""
        return (
            self.result
            .metrics
            .retrieval_time_ns
        )

    @property
    def sorting_time_ns(self) -> int | None:
        """Return sorting or Top-k selection time."""
        return (
            self.result
            .metrics
            .sorting_time_ns
        )

    @property
    def index_build_time_ns(self) -> int | None:
        """Return index construction time when available."""
        return (
            self.result
            .metrics
            .index_build_time_ns
        )

    @property
    def comparisons(self) -> int | None:
        """Return the algorithm comparison counter."""
        return (
            self.result
            .metrics
            .comparisons
        )

    @property
    def chunks_scored(self) -> int | None:
        """Return the number of chunks scored."""
        return (
            self.result
            .metrics
            .chunks_scored
        )

    @property
    def candidates_found(self) -> int | None:
        """Return the number of retrieval candidates."""
        return (
            self.result
            .metrics
            .candidates_found
        )


def measure_retrieval(
    retriever: Retriever,
    query: str,
    top_k: int = 10,
    *,
    profile_memory: bool = True,
) -> RetrievalPerformance:
    """Measure one retrieval execution.

    Timing and memory profiling are deliberately separated.

    The first retrieval is the timed execution used for latency
    measurements. When memory profiling is enabled, the retrieval is
    repeated with tracemalloc enabled. The second execution is used
    only for memory measurement and is not included in timing results.
    """
    normalized_query = _validate_query(
        query
    )

    _validate_top_k(
        top_k
    )

    start_time = (
        time.perf_counter_ns()
    )

    result = retriever.retrieve(
        normalized_query,
        top_k=top_k,
    )

    total_time_ns = (
        time.perf_counter_ns()
        - start_time
    )

    peak_memory_bytes: int | None = None
    memory_method: str | None = None

    if profile_memory:
        (
            memory_result,
            peak_memory_bytes,
        ) = _profile_peak_memory(
            retriever,
            normalized_query,
            top_k,
        )

        _validate_same_ranking(
            timed_result=result,
            memory_result=memory_result,
        )

        memory_method = (
            MEMORY_METHOD
        )

    return RetrievalPerformance(
        result=result,
        total_time_ns=total_time_ns,
        peak_memory_bytes=(
            peak_memory_bytes
        ),
        memory_method=memory_method,
    )


def _profile_peak_memory(
    retriever: Retriever,
    query: str,
    top_k: int,
) -> tuple[
    RetrievalResult,
    int,
]:
    """Measure incremental Python-managed peak memory for retrieval."""
    gc.collect()

    tracing_was_active = (
        tracemalloc.is_tracing()
    )

    if not tracing_was_active:
        tracemalloc.start()

    baseline_current, _ = (
        tracemalloc.get_traced_memory()
    )

    tracemalloc.reset_peak()

    try:
        result = retriever.retrieve(
            query,
            top_k=top_k,
        )

        _, peak_memory = (
            tracemalloc.get_traced_memory()
        )
    finally:
        if not tracing_was_active:
            tracemalloc.stop()

    incremental_peak = max(
        0,
        peak_memory
        - baseline_current,
    )

    return (
        result,
        incremental_peak,
    )


def _validate_same_ranking(
    *,
    timed_result: RetrievalResult,
    memory_result: RetrievalResult,
) -> None:
    """Ensure timing and memory executions produced the same ranking."""
    if (
        timed_result.algorithm
        != memory_result.algorithm
    ):
        raise RetrievalDeterminismError(
            "Timed and memory-profiled executions "
            "used different algorithms."
        )

    timed_ids = tuple(
        chunk.chunk_id
        for chunk in timed_result.chunks
    )

    memory_ids = tuple(
        chunk.chunk_id
        for chunk in memory_result.chunks
    )

    if timed_ids != memory_ids:
        raise RetrievalDeterminismError(
            "Timed and memory-profiled executions "
            "produced different rankings."
        )


def _validate_query(
    query: str,
) -> str:
    """Validate and normalize an experimental query."""
    if not isinstance(
        query,
        str,
    ):
        raise TypeError(
            "query must be a string."
        )

    normalized = query.strip()

    if not normalized:
        raise ValueError(
            "query cannot be empty."
        )

    return normalized


def _validate_top_k(
    top_k: int,
) -> None:
    """Validate an experimental Top-k value."""
    if isinstance(
        top_k,
        bool,
    ) or not isinstance(
        top_k,
        int,
    ):
        raise TypeError(
            "top_k must be an integer."
        )

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero."
        )