import pytest

from src.app.models import (
    ApplicationHealthReport,
    ComponentHealth,
)
from src.app.streamlit_support import (
    build_chunk_details,
    build_health_rows,
    build_metric_rows,
    build_retrieval_rows,
    build_retrieval_trace,
    format_duration_ns,
    format_optional_number,
    get_strategy_presentation,
)
from src.retrieval.models import (
    RetrievalMetrics,
    RetrievalResult,
    RetrievedChunk,
)


def test_format_nanoseconds() -> None:
    assert format_duration_ns(500) == "500 ns"


def test_format_microseconds() -> None:
    assert format_duration_ns(5_000) == "5.00 µs"


def test_format_milliseconds() -> None:
    assert format_duration_ns(5_000_000) == "5.00 ms"


def test_format_seconds() -> None:
    assert format_duration_ns(2_000_000_000) == "2.00 s"


def test_format_missing_duration() -> None:
    assert format_duration_ns(None) == "N/A"


def test_negative_duration_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="cannot be negative",
    ):
        format_duration_ns(-1)


def test_format_optional_number() -> None:
    assert format_optional_number(None) == "N/A"

    assert format_optional_number(10) == "10"

    assert format_optional_number(0.95) == "0.9500"


def test_build_retrieval_rows() -> None:
    chunk = RetrievedChunk(
        chunk_id="chunk_001",
        content="Example content.",
        source_path="example.md",
        section_title="Example",
        score=0.95,
        rank=1,
    )

    result = RetrievalResult(
        query="Example query",
        algorithm="linear",
        top_k=1,
        chunks=(chunk,),
        metrics=RetrievalMetrics(
            retrieval_time_ns=100,
        ),
    )

    rows = build_retrieval_rows(result)

    assert len(rows) == 1

    assert rows[0]["Chunk ID"] == "chunk_001"

    assert rows[0]["Rank"] == 1

    assert rows[0]["Score"] == 0.95


def test_build_health_rows() -> None:
    report = ApplicationHealthReport(
        status="degraded",
        components=(
            ComponentHealth(
                name="Retrieval",
                status="degraded",
                message=("No algorithms registered."),
            ),
        ),
    )

    rows = build_health_rows(report)

    assert rows == [
        {
            "Component": "Retrieval",
            "Status": "degraded",
            "Message": ("No algorithms registered."),
        }
    ]


def test_strategy_presentation_uses_real_optimized_components() -> None:
    presentation = get_strategy_presentation("optimized")

    assert presentation.label == "Optimized"
    assert "Top-k min-heap" in presentation.description


def test_metric_rows_preserve_missing_values() -> None:
    result = RetrievalResult(
        query="Example query",
        algorithm="semantic",
        top_k=1,
        chunks=(),
        metrics=RetrievalMetrics(
            retrieval_time_ns=2_000_000,
            comparisons=None,
        ),
        metadata={
            "query_embedding_time_ns": 1_000_000,
            "faiss_search_time_ns": None,
            "index_source": "persisted",
            "persistence_status": "loaded",
        },
    )

    values = {row["Metric"]: row["Value"] for row in build_metric_rows(result)}

    assert values["Comparisons"] == "N/A"
    assert values["Query embedding time"] == "1.00 ms"
    assert values["FAISS search time"] == "N/A"
    assert values["Semantic index source"] == "persisted"


def test_chunk_details_include_token_count_without_coercing_none() -> None:
    chunk = RetrievedChunk(
        chunk_id="chunk_001",
        content="Example content.",
        source_path="example.md",
        section_title="Example",
        score=0.95,
        rank=1,
        token_count=None,
    )
    result = RetrievalResult(
        query="Example query",
        algorithm="linear",
        top_k=1,
        chunks=(chunk,),
        metrics=RetrievalMetrics(retrieval_time_ns=100),
    )

    details = build_chunk_details(result)

    assert details[0]["rank"] == 1
    assert details[0]["token_count"] == "N/A"


def test_retrieval_trace_uses_result_metadata_without_fabricating_values() -> None:
    result = RetrievalResult(
        query="Example query",
        algorithm="semantic",
        top_k=3,
        chunks=(),
        metrics=RetrievalMetrics(retrieval_time_ns=100),
        metadata={
            "representation": "bge_embeddings",
            "model": "BAAI/bge-small-en-v1.5",
            "candidate_strategy": "faiss_index_flat_ip",
            "ranking_strategy": "faiss_top_k_with_chunk_id_tie_break",
        },
    )

    trace = build_retrieval_trace(result)

    assert trace[0].title == "Representation"
    assert trace[0].detail == "bge embeddings · BAAI/bge-small-en-v1.5"
    assert trace[1].detail == "faiss index flat ip"
    assert trace[3].detail == "Requested 3 chunks"
