from pathlib import Path

from src.observability.jsonl import (
    JsonlRunWriter,
    read_jsonl,
)
from src.observability.records import (
    create_retrieval_record,
)
from src.retrieval.models import (
    RetrievalMetrics,
    RetrievalResult,
)


def build_record():
    """Create a deterministic record for persistence tests."""

    result = RetrievalResult(
        query="Test query",
        algorithm="linear",
        top_k=5,
        chunks=(),
        metrics=RetrievalMetrics(
            retrieval_time_ns=100,
            comparisons=10,
            chunks_scored=10,
            candidates_found=0,
        ),
    )

    return create_retrieval_record(
        result,
        run_id="run-test",
        timestamp_utc=(
            "2026-09-05T12:00:00+00:00"
        ),
    )


def test_writer_creates_parent_directory(
    tmp_path: Path,
) -> None:
    output_path = (
        tmp_path
        / "nested"
        / "runs.jsonl"
    )

    writer = JsonlRunWriter(
        output_path
    )

    writer.write(
        build_record()
    )

    assert (
        output_path.exists()
    )


def test_writer_appends_json_lines(
    tmp_path: Path,
) -> None:
    output_path = (
        tmp_path / "runs.jsonl"
    )

    writer = JsonlRunWriter(
        output_path
    )

    writer.write(
        build_record()
    )

    writer.write(
        build_record()
    )

    records = read_jsonl(
        output_path
    )

    assert len(records) == 2


def test_persisted_record_contains_metrics(
    tmp_path: Path,
) -> None:
    output_path = (
        tmp_path / "runs.jsonl"
    )

    writer = JsonlRunWriter(
        output_path
    )

    writer.write(
        build_record()
    )

    records = read_jsonl(
        output_path
    )

    record = records[0]

    assert (
        record["algorithm"]
        == "linear"
    )

    assert (
        record["retrieval_time_ns"]
        == 100
    )

    assert (
        record["comparisons"]
        == 10
    )


def test_read_missing_file_returns_empty_list(
    tmp_path: Path,
) -> None:
    result = read_jsonl(
        tmp_path / "missing.jsonl"
    )

    assert result == []