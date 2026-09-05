from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from src.observability.models import (
    ExperimentContext,
    RunRecord,
)
from src.retrieval.models import RetrievalResult
from src.services.fastcontext import FastContextResult

SCHEMA_VERSION = "1.0"


def create_retrieval_record(
    result: RetrievalResult,
    *,
    experiment: ExperimentContext | None = None,
    run_id: str | None = None,
    timestamp_utc: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> RunRecord:
    """Create a structured record from retrieval output."""

    resolved_run_id = (
        run_id
        if run_id is not None
        else _create_run_id()
    )

    resolved_timestamp = (
        timestamp_utc
        if timestamp_utc is not None
        else _create_timestamp()
    )

    metrics = result.metrics

    return RunRecord(
        schema_version=SCHEMA_VERSION,
        run_id=resolved_run_id,
        timestamp_utc=resolved_timestamp,
        mode="retrieval",
        status="success",
        query=result.query,
        algorithm=result.algorithm,
        top_k=result.top_k,
        retrieval_time_ns=(
            metrics.retrieval_time_ns
        ),
        sorting_time_ns=(
            metrics.sorting_time_ns
        ),
        index_build_time_ns=(
            metrics.index_build_time_ns
        ),
        comparisons=metrics.comparisons,
        chunks_scored=metrics.chunks_scored,
        candidates_found=(
            metrics.candidates_found
        ),
        peak_memory_mb=(
            metrics.peak_memory_mb
        ),
        returned_chunks=len(
            result.chunks
        ),
        chunk_ids=tuple(
            chunk.chunk_id
            for chunk in result.chunks
        ),
        experiment=experiment,
        metadata=_merge_metadata(
            result.metadata,
            metadata,
        ),
    )


def create_rag_record(
    result: FastContextResult,
    *,
    experiment: ExperimentContext | None = None,
    run_id: str | None = None,
    timestamp_utc: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> RunRecord:
    """Create a structured record from a complete RAG execution."""

    resolved_run_id = (
        run_id
        if run_id is not None
        else _create_run_id()
    )

    resolved_timestamp = (
        timestamp_utc
        if timestamp_utc is not None
        else _create_timestamp()
    )

    retrieval = result.retrieval
    retrieval_metrics = (
        retrieval.metrics
    )
    rag = result.rag

    combined_metadata = (
        _merge_metadata(
            retrieval.metadata,
            metadata,
        )
    )

    combined_metadata = (
        _merge_metadata(
            combined_metadata,
            rag.metadata,
        )
    )

    return RunRecord(
        schema_version=SCHEMA_VERSION,
        run_id=resolved_run_id,
        timestamp_utc=resolved_timestamp,
        mode="rag",
        status="success",
        query=result.query,
        algorithm=retrieval.algorithm,
        top_k=retrieval.top_k,
        retrieval_time_ns=(
            retrieval_metrics
            .retrieval_time_ns
        ),
        sorting_time_ns=(
            retrieval_metrics
            .sorting_time_ns
        ),
        index_build_time_ns=(
            retrieval_metrics
            .index_build_time_ns
        ),
        comparisons=(
            retrieval_metrics
            .comparisons
        ),
        chunks_scored=(
            retrieval_metrics
            .chunks_scored
        ),
        candidates_found=(
            retrieval_metrics
            .candidates_found
        ),
        peak_memory_mb=(
            retrieval_metrics
            .peak_memory_mb
        ),
        returned_chunks=len(
            retrieval.chunks
        ),
        chunk_ids=tuple(
            chunk.chunk_id
            for chunk in retrieval.chunks
        ),
        generation_time_ns=(
            rag.generation_time_ns
        ),
        end_to_end_time_ns=(
            result.end_to_end_time_ns
        ),
        provider=rag.provider,
        model=rag.model,
        citation_valid=(
            rag.citation_valid
        ),
        citation_count=(
            rag.citation_count
        ),
        citation_retry_count=(
            rag.citation_retry_count
        ),
        valid_citations=(
            rag.valid_citations
        ),
        invalid_citations=(
            rag.invalid_citations
        ),
        experiment=experiment,
        metadata=combined_metadata,
    )


def create_error_record(
    *,
    query: str,
    error: Exception,
    mode: str,
    algorithm: str | None = None,
    top_k: int | None = None,
    experiment: ExperimentContext | None = None,
    run_id: str | None = None,
    timestamp_utc: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> RunRecord:
    """Create a structured record for a failed execution."""

    if mode not in {
        "retrieval",
        "rag",
    }:
        raise ValueError(
            "mode must be 'retrieval' or 'rag'."
        )

    normalized_query = query.strip()

    if not normalized_query:
        raise ValueError(
            "query cannot be empty."
        )

    resolved_run_id = (
        run_id
        if run_id is not None
        else _create_run_id()
    )

    resolved_timestamp = (
        timestamp_utc
        if timestamp_utc is not None
        else _create_timestamp()
    )

    return RunRecord(
        schema_version=SCHEMA_VERSION,
        run_id=resolved_run_id,
        timestamp_utc=resolved_timestamp,
        mode=mode,
        status="error",
        query=normalized_query,
        algorithm=algorithm,
        top_k=top_k,
        retrieval_time_ns=None,
        sorting_time_ns=None,
        index_build_time_ns=None,
        comparisons=None,
        chunks_scored=None,
        candidates_found=None,
        peak_memory_mb=None,
        returned_chunks=None,
        chunk_ids=(),
        error_type=type(error).__name__,
        error_message=str(error),
        experiment=experiment,
        metadata=metadata,
    )


def _create_run_id() -> str:
    """Create a globally unique run identifier."""

    return str(
        uuid4()
    )


def _create_timestamp() -> str:
    """Create an ISO 8601 UTC timestamp."""

    return (
        datetime.now(
            UTC
        )
        .isoformat()
    )


def _merge_metadata(
    first: dict[str, Any] | None,
    second: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Merge optional metadata dictionaries."""

    if first is None and second is None:
        return None

    merged: dict[str, Any] = {}

    if first is not None:
        merged.update(
            first
        )

    if second is not None:
        merged.update(
            second
        )

    return merged