from src.observability.models import (
    ExperimentContext,
)
from src.observability.records import (
    create_error_record,
    create_retrieval_record,
)
from src.retrieval.models import (
    RetrievalMetrics,
    RetrievalResult,
    RetrievedChunk,
)


def build_retrieval_result() -> RetrievalResult:
    """Create a deterministic retrieval result."""

    chunk = RetrievedChunk(
        chunk_id="chunk_001",
        content="Example FastAPI content.",
        source_path="example.md",
        section_title="Example",
        score=0.95,
        rank=1,
    )

    return RetrievalResult(
        query="How does FastAPI work?",
        algorithm="linear",
        top_k=5,
        chunks=(chunk,),
        metrics=RetrievalMetrics(
            retrieval_time_ns=1_000,
            sorting_time_ns=200,
            index_build_time_ns=300,
            comparisons=50,
            chunks_scored=25,
            candidates_found=10,
            peak_memory_mb=12.5,
        ),
        metadata={
            "source": "test",
        },
    )


def test_create_retrieval_record() -> None:
    result = (
        build_retrieval_result()
    )

    experiment = ExperimentContext(
        experiment_id="exp-001",
        configuration="baseline",
        representation="tfidf",
        corpus_fraction=0.25,
        repetition=1,
        seed=42,
    )

    record = create_retrieval_record(
        result,
        experiment=experiment,
        run_id="run-001",
        timestamp_utc=(
            "2026-09-05T12:00:00+00:00"
        ),
    )

    assert (
        record.run_id
        == "run-001"
    )

    assert (
        record.mode
        == "retrieval"
    )

    assert (
        record.status
        == "success"
    )

    assert (
        record.algorithm
        == "linear"
    )

    assert (
        record.retrieval_time_ns
        == 1_000
    )

    assert (
        record.comparisons
        == 50
    )

    assert (
        record.returned_chunks
        == 1
    )

    assert (
        record.chunk_ids
        == ("chunk_001",)
    )

    assert (
        record.experiment
        == experiment
    )


def test_create_error_record() -> None:
    error = RuntimeError(
        "Simulated failure"
    )

    record = create_error_record(
        query="Test query",
        error=error,
        mode="retrieval",
        algorithm="linear",
        top_k=5,
        run_id="error-001",
        timestamp_utc=(
            "2026-09-05T12:00:00+00:00"
        ),
    )

    assert (
        record.status
        == "error"
    )

    assert (
        record.error_type
        == "RuntimeError"
    )

    assert (
        record.error_message
        == "Simulated failure"
    )


def test_experiment_context_accepts_expected_values() -> None:
    context = ExperimentContext(
        experiment_id="exp-001",
        corpus_fraction=1.0,
        repetition=5,
        seed=42,
    )

    assert (
        context.corpus_fraction
        == 1.0
    )

    assert (
        context.repetition
        == 5
    )