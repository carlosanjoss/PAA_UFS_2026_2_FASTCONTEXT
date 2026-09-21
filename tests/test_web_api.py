from __future__ import annotations

from src.app.presentation import build_retrieval_trace
from src.retrieval.models import RetrievalMetrics, RetrievalResult, RetrievedChunk
from src.web_api import (
    QueryRequest,
    _coerce_csv_value,
    _serialize_retrieval_result,
    metrics,
)


def _retrieval_result() -> RetrievalResult:
    return RetrievalResult(
        query="How does dependency injection work?",
        algorithm="indexed",
        top_k=3,
        chunks=(
            RetrievedChunk(
                chunk_id="tutorial/dependencies-0",
                content="FastAPI resolves declared dependencies.",
                source_path="docs/tutorial/dependencies.md",
                section_title="Dependencies",
                score=0.91,
                rank=1,
                token_count=12,
            ),
        ),
        metrics=RetrievalMetrics(
            retrieval_time_ns=2_500_000,
            sorting_time_ns=500_000,
            comparisons=42,
            chunks_scored=18,
            candidates_found=7,
            peak_memory_mb=1.25,
        ),
        metadata={
            "representation": "tfidf",
            "candidate_strategy": "inverted_index",
            "ranking_strategy": "merge_sort",
        },
    )


def test_query_request_defaults_to_indexed_rag() -> None:
    request = QueryRequest(query="What is FastAPI?")

    assert request.algorithm == "indexed"
    assert request.top_k == 5
    assert request.use_rag is True


def test_csv_values_preserve_missing_metrics() -> None:
    assert _coerce_csv_value("comparisons_mean", "") is None
    assert _coerce_csv_value("corpus_chunks", "1305") == 1305
    assert _coerce_csv_value("algorithm", "semantic") == "semantic"


def test_metrics_reads_all_persisted_report_families() -> None:
    response = metrics()

    assert response["performance"]
    assert response["quality"]
    assert response["sorting"]
    assert response["generated_from"] == [
        "performance_by_algorithm.csv",
        "quality_by_algorithm.csv",
        "sorting_comparison.csv",
    ]


def test_retrieval_response_contains_metrics_trace_and_evidence() -> None:
    response = _serialize_retrieval_result(_retrieval_result())

    assert response["metrics"]["retrieval_time_ms"] == 2.5
    assert response["metrics"]["comparisons"] == 42
    assert response["trace"][1] == {
        "title": "Candidate discovery",
        "detail": "inverted index",
    }
    assert response["chunks"][0]["chunk_id"] == "tutorial/dependencies-0"


def test_trace_marks_uninstrumented_metadata_without_inventing_values() -> None:
    result = _retrieval_result()
    result_without_metadata = RetrievalResult(
        query=result.query,
        algorithm=result.algorithm,
        top_k=result.top_k,
        chunks=result.chunks,
        metrics=result.metrics,
        metadata=None,
    )

    trace = build_retrieval_trace(result_without_metadata)

    assert trace[0].detail == "Not instrumented"
    assert trace[1].detail == "Not instrumented"
