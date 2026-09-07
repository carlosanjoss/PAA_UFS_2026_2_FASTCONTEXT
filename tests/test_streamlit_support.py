import pytest

from src.app.models import (
    ApplicationHealthReport,
    ComponentHealth,
)
from src.app.streamlit_support import (
    build_health_rows,
    build_retrieval_rows,
    format_duration_ns,
    format_optional_number,
)
from src.retrieval.models import (
    RetrievalMetrics,
    RetrievalResult,
    RetrievedChunk,
)


def test_format_nanoseconds() -> None:
    assert (
        format_duration_ns(500)
        == "500 ns"
    )


def test_format_microseconds() -> None:
    assert (
        format_duration_ns(5_000)
        == "5.00 µs"
    )


def test_format_milliseconds() -> None:
    assert (
        format_duration_ns(
            5_000_000
        )
        == "5.00 ms"
    )


def test_format_seconds() -> None:
    assert (
        format_duration_ns(
            2_000_000_000
        )
        == "2.00 s"
    )


def test_format_missing_duration() -> None:
    assert (
        format_duration_ns(None)
        == "N/A"
    )


def test_negative_duration_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="cannot be negative",
    ):
        format_duration_ns(-1)


def test_format_optional_number() -> None:
    assert (
        format_optional_number(None)
        == "N/A"
    )

    assert (
        format_optional_number(10)
        == "10"
    )

    assert (
        format_optional_number(0.95)
        == "0.9500"
    )


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

    rows = build_retrieval_rows(
        result
    )

    assert len(rows) == 1

    assert (
        rows[0]["Chunk ID"]
        == "chunk_001"
    )

    assert (
        rows[0]["Rank"]
        == 1
    )

    assert (
        rows[0]["Score"]
        == 0.95
    )


def test_build_health_rows() -> None:
    report = ApplicationHealthReport(
        status="degraded",
        components=(
            ComponentHealth(
                name="Retrieval",
                status="degraded",
                message=(
                    "No algorithms registered."
                ),
            ),
        ),
    )

    rows = build_health_rows(
        report
    )

    assert rows == [
        {
            "Component": "Retrieval",
            "Status": "degraded",
            "Message": (
                "No algorithms registered."
            ),
        }
    ]