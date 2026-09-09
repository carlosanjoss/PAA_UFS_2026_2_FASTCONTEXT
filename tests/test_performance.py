"""Tests for FastContext retrieval performance measurement."""

from __future__ import annotations

import pytest

from src.evaluation.performance import (
    MEMORY_METHOD,
    RetrievalDeterminismError,
    measure_retrieval,
)
from src.retrieval.base import Retriever
from src.retrieval.models import (
    RetrievalMetrics,
    RetrievalResult,
    RetrievedChunk,
)


class FakeRetriever(Retriever):
    """Deterministic retriever used by performance tests."""

    def __init__(
        self,
        *,
        reverse_on_second_call: bool = False,
    ) -> None:
        self.calls = 0
        self.reverse_on_second_call = (
            reverse_on_second_call
        )

    @property
    def name(self) -> str:
        """Return fake retriever name."""
        return "fake"

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> RetrievalResult:
        """Return deterministic fake retrieval results."""
        self.calls += 1

        allocation = bytearray(
            256 * 1024
        )

        allocation[0] = 1

        chunks = [
            RetrievedChunk(
                chunk_id="chunk-a",
                content="First chunk.",
                source_path="first.md",
                section_title="First",
                score=0.9,
                rank=1,
            ),
            RetrievedChunk(
                chunk_id="chunk-b",
                content="Second chunk.",
                source_path="second.md",
                section_title="Second",
                score=0.8,
                rank=2,
            ),
        ]

        if (
            self.reverse_on_second_call
            and self.calls >= 2
        ):
            chunks.reverse()

            chunks = [
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    content=chunk.content,
                    source_path=chunk.source_path,
                    section_title=(
                        chunk.section_title
                    ),
                    score=chunk.score,
                    rank=rank,
                )
                for rank, chunk in enumerate(
                    chunks,
                    start=1,
                )
            ]

        selected = tuple(
            chunks[
                :top_k
            ]
        )

        return RetrievalResult(
            query=query,
            algorithm=self.name,
            top_k=top_k,
            chunks=selected,
            metrics=RetrievalMetrics(
                retrieval_time_ns=1_000,
                sorting_time_ns=200,
                index_build_time_ns=300,
                comparisons=7,
                chunks_scored=8,
                candidates_found=4,
            ),
        )


def test_measure_retrieval_profiles_timing_and_memory() -> None:
    retriever = FakeRetriever()

    performance = measure_retrieval(
        retriever,
        "test query",
        top_k=2,
    )

    assert retriever.calls == 2

    assert performance.total_time_ns > 0

    assert (
        performance.peak_memory_bytes
        is not None
    )

    assert (
        performance.peak_memory_bytes
        > 0
    )

    assert (
        performance.peak_memory_mb
        is not None
    )

    assert (
        performance.peak_memory_mb
        > 0.0
    )

    assert (
        performance.memory_method
        == MEMORY_METHOD
    )


def test_measure_retrieval_can_skip_memory_profile() -> None:
    retriever = FakeRetriever()

    performance = measure_retrieval(
        retriever,
        "test query",
        top_k=2,
        profile_memory=False,
    )

    assert retriever.calls == 1

    assert (
        performance.peak_memory_bytes
        is None
    )

    assert (
        performance.peak_memory_mb
        is None
    )

    assert (
        performance.memory_method
        is None
    )


def test_performance_exposes_retrieval_metrics() -> None:
    performance = measure_retrieval(
        FakeRetriever(),
        "test query",
        top_k=2,
        profile_memory=False,
    )

    assert (
        performance.retrieval_time_ns
        == 1_000
    )

    assert (
        performance.sorting_time_ns
        == 200
    )

    assert (
        performance.index_build_time_ns
        == 300
    )

    assert (
        performance.comparisons
        == 7
    )

    assert (
        performance.chunks_scored
        == 8
    )

    assert (
        performance.candidates_found
        == 4
    )


def test_measure_retrieval_preserves_result() -> None:
    performance = measure_retrieval(
        FakeRetriever(),
        "test query",
        top_k=1,
        profile_memory=False,
    )

    assert (
        performance.result.query
        == "test query"
    )

    assert (
        performance.result.algorithm
        == "fake"
    )

    assert (
        performance.result.top_k
        == 1
    )

    assert len(
        performance.result.chunks
    ) == 1

    assert (
        performance.result
        .chunks[0]
        .chunk_id
        == "chunk-a"
    )


def test_measure_retrieval_detects_nondeterministic_ranking() -> None:
    retriever = FakeRetriever(
        reverse_on_second_call=True
    )

    with pytest.raises(
        RetrievalDeterminismError,
        match="produced different rankings",
    ):
        measure_retrieval(
            retriever,
            "test query",
            top_k=2,
        )


@pytest.mark.parametrize(
    "query",
    [
        "",
        "   ",
    ],
)
def test_measure_retrieval_rejects_empty_query(
    query: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="query cannot be empty",
    ):
        measure_retrieval(
            FakeRetriever(),
            query,
        )


def test_measure_retrieval_rejects_non_string_query() -> None:
    with pytest.raises(
        TypeError,
        match="query must be a string",
    ):
        measure_retrieval(
            FakeRetriever(),
            123,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "top_k",
    [
        0,
        -1,
        -10,
    ],
)
def test_measure_retrieval_rejects_non_positive_top_k(
    top_k: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="top_k must be greater than zero",
    ):
        measure_retrieval(
            FakeRetriever(),
            "test query",
            top_k=top_k,
        )


@pytest.mark.parametrize(
    "top_k",
    [
        1.5,
        "10",
        True,
    ],
)
def test_measure_retrieval_rejects_invalid_top_k_type(
    top_k: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="top_k must be an integer",
    ):
        measure_retrieval(
            FakeRetriever(),
            "test query",
            top_k=top_k,  
        )