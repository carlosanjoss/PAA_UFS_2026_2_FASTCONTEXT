"""Tests for the FastContext RAG experiment runner."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from experiments.run_rag_experiments import (
    CONDITIONS,
    CSV_FIELDS,
    EvaluationQuery,
    _build_run_id,
    _citation_relevance,
    _condition_order_for_query,
    _expected_run_ids,
    _read_completed_runs,
    _retrieval_quality,
    _to_csv_row,
    build_parser,
)


def _query(
    query_id: str = "q01",
) -> EvaluationQuery:
    return EvaluationQuery(
        query_id=query_id,
        category="tutorial",
        question=(
            f"Question {query_id}?"
        ),
        relevant_chunks=(
            "chunk-1",
            "chunk-3",
        ),
    )


def test_parser_accepts_smoke() -> None:
    args = build_parser().parse_args(
        [
            "--smoke",
        ]
    )

    assert args.smoke is True
    assert args.force is False
    assert args.resume is False


def test_force_and_resume_are_mutually_exclusive() -> None:
    parser = build_parser()

    with pytest.raises(
        SystemExit
    ):
        parser.parse_args(
            [
                "--force",
                "--resume",
            ]
        )


def test_build_run_id() -> None:
    assert (
        _build_run_id(
            condition="semantic",
            query_id="q07",
        )
        == "semantic|q07"
    )


def test_expected_run_ids_build_five_conditions() -> None:
    expected = _expected_run_ids(
        conditions=CONDITIONS,
        queries=(
            _query(),
        ),
    )

    assert len(
        expected
    ) == 5

    assert (
        "no_rag|q01"
        in expected
    )

    assert (
        "semantic|q01"
        in expected
    )


def test_condition_rotation_is_balanced() -> None:
    orders = [
        _condition_order_for_query(
            CONDITIONS,
            index,
        )
        for index in range(
            len(
                CONDITIONS
            )
        )
    ]

    first_conditions = [
        order[
            0
        ]
        for order in orders
    ]

    assert (
        tuple(
            first_conditions
        )
        == CONDITIONS
    )

    assert all(
        set(
            order
        )
        == set(
            CONDITIONS
        )
        for order in orders
    )


def test_retrieval_quality() -> None:
    (
        precision,
        recall,
        relevant_retrieved,
    ) = _retrieval_quality(
        retrieved_chunk_ids=(
            "chunk-1",
            "chunk-2",
            "chunk-3",
            "chunk-4",
            "chunk-5",
        ),
        relevant_chunk_ids=(
            "chunk-1",
            "chunk-3",
            "chunk-9",
            "chunk-10",
        ),
        k=5,
    )

    assert precision == pytest.approx(
        0.4
    )

    assert recall == pytest.approx(
        0.5
    )

    assert relevant_retrieved == [
        "chunk-1",
        "chunk-3",
    ]


def test_retrieval_quality_rejects_zero_k() -> None:
    with pytest.raises(
        ValueError
    ):
        _retrieval_quality(
            retrieved_chunk_ids=(),
            relevant_chunk_ids=(
                "chunk-1",
            ),
            k=0,
        )


def test_citation_relevance() -> None:
    count, score = _citation_relevance(
        valid_citations=(
            "chunk-1",
            "chunk-2",
        ),
        relevant_chunk_ids=(
            "chunk-1",
            "chunk-3",
        ),
    )

    assert count == 1
    assert score == pytest.approx(
        0.5
    )


def test_citation_relevance_without_citations() -> None:
    count, score = _citation_relevance(
        valid_citations=(),
        relevant_chunk_ids=(
            "chunk-1",
        ),
    )

    assert count == 0
    assert score is None


def test_csv_serialization_keeps_none_empty() -> None:
    row = {
        field: None
        for field in CSV_FIELDS
    }

    row[
        "run_id"
    ] = "no_rag|q01"

    row[
        "retrieved_chunk_ids"
    ] = []

    row[
        "generation_metadata"
    ] = {}

    serialized = _to_csv_row(
        row
    )

    assert (
        serialized[
            "retrieval_time_ns"
        ]
        == ""
    )

    assert (
        serialized[
            "retrieved_chunk_ids"
        ]
        == "[]"
    )

    assert (
        serialized[
            "generation_metadata"
        ]
        == "{}"
    )


def test_read_completed_runs_rejects_duplicate_ids(
    tmp_path: Path,
) -> None:
    path = (
        tmp_path
        / "results.csv"
    )

    row = {
        field: ""
        for field in CSV_FIELDS
    }

    row[
        "experiment_signature"
    ] = "signature"

    row[
        "run_id"
    ] = "linear|q01"

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=CSV_FIELDS,
        )

        writer.writeheader()
        writer.writerow(
            row
        )
        writer.writerow(
            row
        )

    with pytest.raises(
        ValueError,
        match="Duplicate run_id",
    ):
        _read_completed_runs(
            path=path,
            expected_run_ids={
                "linear|q01",
            },
            expected_signature=(
                "signature"
            ),
        )


def test_read_completed_runs_rejects_signature_change(
    tmp_path: Path,
) -> None:
    path = (
        tmp_path
        / "results.csv"
    )

    row = {
        field: ""
        for field in CSV_FIELDS
    }

    row[
        "experiment_signature"
    ] = "old"

    row[
        "run_id"
    ] = "linear|q01"

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=CSV_FIELDS,
        )

        writer.writeheader()
        writer.writerow(
            row
        )

    with pytest.raises(
        ValueError,
        match="different experiment signature",
    ):
        _read_completed_runs(
            path=path,
            expected_run_ids={
                "linear|q01",
            },
            expected_signature=(
                "new"
            ),
        )
