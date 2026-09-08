import pytest

from src.retrieval.models import (
    RetrievalMetrics,
    RetrievalResult,
    RetrievedChunk,
)


def build_chunk(
    *,
    chunk_id: str = "chunk_001",
    score: float = 0.95,
    rank: int = 1,
) -> RetrievedChunk:
    """Create a reusable retrieved chunk."""

    return RetrievedChunk(
        chunk_id=chunk_id,
        content=(
            "FastAPI supports "
            "dependency injection."
        ),
        source_path=(
            "tutorial/dependencies/index.md"
        ),
        section_title="Dependencies",
        score=score,
        rank=rank,
    )


def build_metrics() -> RetrievalMetrics:
    """Create reusable retrieval metrics."""

    return RetrievalMetrics(
        retrieval_time_ns=1_000_000,
        sorting_time_ns=200_000,
        index_build_time_ns=0,
        comparisons=100,
        chunks_scored=50,
        candidates_found=10,
        peak_memory_mb=12.5,
    )


def test_retrieved_chunk_is_created() -> None:
    chunk = build_chunk()

    assert (
        chunk.chunk_id
        == "chunk_001"
    )

    assert chunk.rank == 1
    assert chunk.score == 0.95


def test_retrieved_chunk_rejects_empty_id() -> None:
    with pytest.raises(
        ValueError,
        match="chunk_id cannot be empty",
    ):
        RetrievedChunk(
            chunk_id=" ",
            content="Content",
            source_path="source.md",
            section_title="Section",
            score=0.5,
            rank=1,
        )


def test_retrieved_chunk_rejects_invalid_rank() -> None:
    with pytest.raises(
        ValueError,
        match="rank must be greater than zero",
    ):
        build_chunk(
            rank=0
        )


def test_metrics_are_created() -> None:
    metrics = build_metrics()

    assert (
        metrics.retrieval_time_ns
        == 1_000_000
    )

    assert (
        metrics.comparisons
        == 100
    )

    assert (
        metrics.peak_memory_mb
        == 12.5
    )


def test_metrics_reject_negative_time() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "retrieval_time_ns "
            "cannot be negative"
        ),
    ):
        RetrievalMetrics(
            retrieval_time_ns=-1
        )


def test_retrieval_result_is_created() -> None:
    result = RetrievalResult(
        query="How do dependencies work?",
        algorithm="linear",
        top_k=1,
        chunks=(
            build_chunk(),
        ),
        metrics=build_metrics(),
    )

    assert result.algorithm == "linear"
    assert result.top_k == 1
    assert len(result.chunks) == 1


def test_result_rejects_empty_query() -> None:
    with pytest.raises(
        ValueError,
        match="query cannot be empty",
    ):
        RetrievalResult(
            query=" ",
            algorithm="linear",
            top_k=1,
            chunks=(),
            metrics=build_metrics(),
        )


def test_result_rejects_more_chunks_than_top_k() -> None:
    chunks = (
        build_chunk(
            chunk_id="chunk_001",
            rank=1,
        ),
        build_chunk(
            chunk_id="chunk_002",
            rank=2,
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            "cannot exceed top_k"
        ),
    ):
        RetrievalResult(
            query="Test query",
            algorithm="linear",
            top_k=1,
            chunks=chunks,
            metrics=build_metrics(),
        )


def test_result_rejects_duplicate_chunks() -> None:
    chunks = (
        build_chunk(
            chunk_id="chunk_001",
            rank=1,
        ),
        build_chunk(
            chunk_id="chunk_001",
            rank=2,
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            "unique chunk identifiers"
        ),
    ):
        RetrievalResult(
            query="Test query",
            algorithm="linear",
            top_k=2,
            chunks=chunks,
            metrics=build_metrics(),
        )


def test_result_rejects_non_consecutive_ranks() -> None:
    chunks = (
        build_chunk(
            chunk_id="chunk_001",
            rank=1,
        ),
        build_chunk(
            chunk_id="chunk_002",
            rank=3,
        ),
    )

    with pytest.raises(
        ValueError,
        match="consecutive ranks",
    ):
        RetrievalResult(
            query="Test query",
            algorithm="linear",
            top_k=2,
            chunks=chunks,
            metrics=build_metrics(),
        )