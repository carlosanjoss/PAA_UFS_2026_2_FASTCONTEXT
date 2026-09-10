"""Tests for the FastContext experiment runner."""

from __future__ import annotations

import os

import pytest

import experiments.run_experiments as experiment_runner

from experiments.run_experiments import (
    CHECKPOINT_SCHEMA_VERSION,
    EvaluationQuery,
    _build_nested_subsets,
    _checkpoint_due,
    _build_run_id,
    _calculate_signature,
    _expected_run_ids,
    _load_checkpoint,
    _quality_fields,
    _replace_checkpoint_with_retry,
    _raw_fieldnames,
    _read_raw_results,
    _summarize,
    _validate_checkpoint,
    _validate_existing_rows,
    _write_checkpoint,
    build_parser,
)


def _chunks(
    count: int,
) -> list[
    dict[
        str,
        object,
    ]
]:
    return [
        {
            "chunk_id": (
                f"chunk-{index}"
            ),
            "content": (
                f"content {index}"
            ),
        }
        for index in range(
            count
        )
    ]


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
        ),
    )


def _resume_row(
    *,
    algorithm: str = "linear",
    fraction: float = 1.0,
    query_id: str = "q01",
    repetition: int = 1,
    fingerprint: str = "fingerprint",
) -> dict[str, str]:
    return {
        "run_id": _build_run_id(
            algorithm,
            fraction,
            query_id,
            repetition,
        ),
        "algorithm": algorithm,
        "corpus_fraction": str(
            fraction
        ),
        "corpus_chunks": "20",
        "full_corpus_chunks": "20",
        "corpus_fingerprint": fingerprint,
        "query_id": query_id,
        "category": "tutorial",
        "question": (
            f"Question {query_id}?"
        ),
        "repetition": str(
            repetition
        ),
        "retrieval_top_k": "10",
        "warmup_repetitions": "1",
        "random_seed": "42",
        "result_count": "2",
        "retrieved_chunk_ids": (
            '["chunk-1", "chunk-2"]'
        ),
    }


def _validate_rows(
    rows: list[
        dict[
            str,
            str,
        ]
    ],
    *,
    repetitions: int = 1,
) -> set[str]:
    queries = (
        _query(),
    )

    expected = _expected_run_ids(
        algorithms=(
            "linear",
        ),
        fractions=(
            1.0,
        ),
        queries=queries,
        repetitions=repetitions,
    )

    return _validate_existing_rows(
        rows,
        expected_run_ids=expected,
        algorithms=(
            "linear",
        ),
        fractions=(
            1.0,
        ),
        queries=queries,
        repetitions=repetitions,
        subset_metadata={
            1.0: (
                20,
                "fingerprint",
            )
        },
        full_corpus_chunks=20,
        top_k=10,
        warmup_repetitions=1,
        seed=42,
    )


def test_nested_subsets_have_expected_sizes() -> None:
    chunks = _chunks(
        20
    )

    subsets = (
        _build_nested_subsets(
            chunks,
            (
                0.25,
                0.50,
                0.75,
                1.0,
            ),
            seed=42,
        )
    )

    assert len(
        subsets[
            0.25
        ]
    ) == 5

    assert len(
        subsets[
            0.50
        ]
    ) == 10

    assert len(
        subsets[
            0.75
        ]
    ) == 15

    assert len(
        subsets[
            1.0
        ]
    ) == 20


def test_nested_subsets_are_nested() -> None:
    chunks = _chunks(
        20
    )

    subsets = (
        _build_nested_subsets(
            chunks,
            (
                0.25,
                0.50,
                0.75,
                1.0,
            ),
            seed=42,
        )
    )

    ids_25 = {
        chunk[
            "chunk_id"
        ]
        for chunk
        in subsets[
            0.25
        ]
    }

    ids_50 = {
        chunk[
            "chunk_id"
        ]
        for chunk
        in subsets[
            0.50
        ]
    }

    ids_75 = {
        chunk[
            "chunk_id"
        ]
        for chunk
        in subsets[
            0.75
        ]
    }

    ids_100 = {
        chunk[
            "chunk_id"
        ]
        for chunk
        in subsets[
            1.0
        ]
    }

    assert ids_25 < ids_50
    assert ids_50 < ids_75
    assert ids_75 < ids_100


def test_full_subset_preserves_original_order() -> None:
    chunks = _chunks(
        20
    )

    subsets = (
        _build_nested_subsets(
            chunks,
            (
                1.0,
            ),
            seed=42,
        )
    )

    assert (
        subsets[
            1.0
        ]
        == chunks
    )


def test_quality_is_empty_while_ground_truth_is_pending() -> None:
    fields = (
        _quality_fields(
            retrieved_ids=(
                "chunk-a",
                "chunk-b",
            ),
            relevant_ids=(),
            available_ids={
                "chunk-a",
                "chunk-b",
            },
            k_values=(
                1,
                2,
            ),
            ground_truth_complete=False,
        )
    )

    assert (
        fields[
            "quality_available"
        ]
        is False
    )

    assert (
        fields[
            "precision_at_1"
        ]
        is None
    )

    assert (
        fields[
            "recall_at_2"
        ]
        is None
    )


def test_quality_uses_relevant_chunks_available_in_subset() -> None:
    fields = (
        _quality_fields(
            retrieved_ids=(
                "chunk-a",
                "chunk-b",
                "chunk-c",
            ),
            relevant_ids=(
                "chunk-b",
                "chunk-x",
            ),
            available_ids={
                "chunk-a",
                "chunk-b",
                "chunk-c",
            },
            k_values=(
                1,
                3,
            ),
            ground_truth_complete=True,
        )
    )

    assert (
        fields[
            "quality_available"
        ]
        is True
    )

    assert (
        fields[
            "relevant_total_full"
        ]
        == 2
    )

    assert (
        fields[
            "relevant_total_available"
        ]
        == 1
    )

    assert (
        fields[
            "precision_at_1"
        ]
        == 0.0
    )

    assert (
        fields[
            "recall_at_1"
        ]
        == 0.0
    )

    assert (
        fields[
            "precision_at_3"
        ]
        == pytest.approx(
            1 / 3
        )
    )

    assert (
        fields[
            "recall_at_3"
        ]
        == 1.0
    )

    assert (
        fields[
            "mrr_at_3"
        ]
        == pytest.approx(
            0.5
        )
    )

    assert (
        fields[
            "hit_rate_at_3"
        ]
        == 1.0
    )


def test_summary_uses_all_repetitions_for_performance() -> None:
    rows = [
        {
            "algorithm": "linear",
            "corpus_fraction": "1.0",
            "corpus_chunks": "1305",
            "corpus_fingerprint": "abc",
            "repetition": "1",
            "quality_available": "False",
            "total_time_ns": "10",
        },
        {
            "algorithm": "linear",
            "corpus_fraction": "1.0",
            "corpus_chunks": "1305",
            "corpus_fingerprint": "abc",
            "repetition": "2",
            "quality_available": "False",
            "total_time_ns": "20",
        },
    ]

    summary = _summarize(
        rows,
        (
            1,
        ),
    )

    total_time = next(
        row
        for row in summary
        if row[
            "metric"
        ]
        == "total_time_ns"
    )

    assert (
        total_time[
            "count"
        ]
        == 2
    )

    assert (
        total_time[
            "mean"
        ]
        == pytest.approx(
            15.0
        )
    )


def test_summary_does_not_duplicate_quality_across_repetitions() -> None:
    rows = [
        {
            "algorithm": "semantic",
            "corpus_fraction": "1.0",
            "corpus_chunks": "1305",
            "corpus_fingerprint": "abc",
            "repetition": "1",
            "quality_available": "True",
            "precision_at_1": "1.0",
        },
        {
            "algorithm": "semantic",
            "corpus_fraction": "1.0",
            "corpus_chunks": "1305",
            "corpus_fingerprint": "abc",
            "repetition": "2",
            "quality_available": "True",
            "precision_at_1": "1.0",
        },
    ]

    summary = _summarize(
        rows,
        (
            1,
        ),
    )

    precision = next(
        row
        for row in summary
        if row[
            "metric"
        ]
        == "precision_at_1"
    )

    assert (
        precision[
            "count"
        ]
        == 1
    )

    assert (
        precision[
            "mean"
        ]
        == 1.0
    )


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


def test_expected_final_matrix_contains_2400_measured_runs() -> None:
    queries = tuple(
        _query(
            f"q{index:02d}"
        )
        for index in range(
            1,
            31,
        )
    )

    run_ids = _expected_run_ids(
        algorithms=(
            "linear",
            "indexed",
            "optimized",
            "semantic",
        ),
        fractions=(
            0.25,
            0.50,
            0.75,
            1.0,
        ),
        queries=queries,
        repetitions=5,
    )

    assert len(
        run_ids
    ) == 2400


def test_run_id_is_deterministic() -> None:
    assert _build_run_id(
        "semantic",
        0.5,
        "q12",
        4,
    ) == (
        "semantic|0.500000|q12|4"
    )


def test_resume_row_validation_accepts_one_exact_run() -> None:
    completed = _validate_rows(
        [
            _resume_row()
        ]
    )

    assert completed == {
        "linear|1.000000|q01|1"
    }


def test_resume_row_validation_rejects_duplicate_run_id() -> None:
    row = _resume_row()

    with pytest.raises(
        ValueError,
        match="Duplicate run_id",
    ):
        _validate_rows(
            [
                row,
                dict(
                    row
                ),
            ]
        )


def test_resume_row_validation_rejects_incompatible_fingerprint() -> None:
    with pytest.raises(
        ValueError,
        match="fingerprint mismatch",
    ):
        _validate_rows(
            [
                _resume_row(
                    fingerprint="changed"
                )
            ]
        )


def test_read_raw_results_rejects_truncated_row(
    tmp_path,
) -> None:
    path = (
        tmp_path
        / "results.csv"
    )

    fields = _raw_fieldnames(
        (
            1,
        )
    )

    path.write_text(
        ",".join(
            fields
        )
        + "\n"
        + "linear|1.000000|q01|1,linear\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="truncated or incomplete",
    ):
        _read_raw_results(
            path,
            expected_fieldnames=fields,
        )


def test_read_raw_results_rejects_schema_change(
    tmp_path,
) -> None:
    path = (
        tmp_path
        / "results.csv"
    )

    path.write_text(
        "run_id,algorithm\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="schema does not match",
    ):
        _read_raw_results(
            path,
            expected_fieldnames=(
                "run_id",
                "algorithm",
                "query_id",
            ),
        )


def test_signature_is_deterministic_and_sensitive_to_changes() -> None:
    first = {
        "algorithms": [
            "linear",
            "semantic",
        ],
        "memory_profiling": True,
        "seed": 42,
    }

    reordered = {
        "seed": 42,
        "memory_profiling": True,
        "algorithms": [
            "linear",
            "semantic",
        ],
    }

    changed = {
        "seed": 42,
        "memory_profiling": False,
        "algorithms": [
            "linear",
            "semantic",
        ],
    }

    assert (
        _calculate_signature(
            first
        )
        == _calculate_signature(
            reordered
        )
    )

    assert (
        _calculate_signature(
            first
        )
        != _calculate_signature(
            changed
        )
    )


def test_checkpoint_round_trip(
    tmp_path,
) -> None:
    path = (
        tmp_path
        / "results.csv.checkpoint.json"
    )

    raw_output = (
        tmp_path
        / "results.csv"
    )

    summary_output = (
        tmp_path
        / "summary.csv"
    )

    completed = {
        "linear|1.000000|q01|1"
    }

    _write_checkpoint(
        path,
        signature="abc",
        signature_payload={
            "mode": "test"
        },
        expected_runs=2,
        completed_run_ids=(
            completed
        ),
        status="in_progress",
        raw_output=raw_output,
        summary_output=(
            summary_output
        ),
    )

    checkpoint = (
        _load_checkpoint(
            path
        )
    )

    assert checkpoint[
        "schema_version"
    ] == CHECKPOINT_SCHEMA_VERSION

    assert checkpoint[
        "signature"
    ] == "abc"

    assert checkpoint[
        "completed_runs"
    ] == 1

    assert checkpoint[
        "completed_run_ids"
    ] == sorted(
        completed
    )


def test_checkpoint_may_lag_raw_results_by_one_or_more_rows() -> None:
    checkpoint = {
        "schema_version": (
            CHECKPOINT_SCHEMA_VERSION
        ),
        "signature": "abc",
        "status": "in_progress",
        "expected_runs": 2,
        "completed_runs": 1,
        "completed_run_ids": [
            "run-1"
        ],
    }

    _validate_checkpoint(
        checkpoint,
        signature="abc",
        expected_runs=2,
        raw_completed_run_ids={
            "run-1",
            "run-2",
        },
    )


def test_checkpoint_cannot_be_ahead_of_raw_results() -> None:
    checkpoint = {
        "schema_version": (
            CHECKPOINT_SCHEMA_VERSION
        ),
        "signature": "abc",
        "status": "in_progress",
        "expected_runs": 2,
        "completed_runs": 2,
        "completed_run_ids": [
            "run-1",
            "run-2",
        ],
    }

    with pytest.raises(
        ValueError,
        match="missing from raw results",
    ):
        _validate_checkpoint(
            checkpoint,
            signature="abc",
            expected_runs=2,
            raw_completed_run_ids={
                "run-1"
            },
        )


def test_checkpoint_rejects_incompatible_signature() -> None:
    checkpoint = {
        "schema_version": (
            CHECKPOINT_SCHEMA_VERSION
        ),
        "signature": "old",
        "status": "in_progress",
        "expected_runs": 1,
        "completed_runs": 0,
        "completed_run_ids": [],
    }

    with pytest.raises(
        ValueError,
        match="signature does not match",
    ):
        _validate_checkpoint(
            checkpoint,
            signature="new",
            expected_runs=1,
            raw_completed_run_ids=set(),
        )


def test_complete_checkpoint_requires_complete_raw_results() -> None:
    checkpoint = {
        "schema_version": (
            CHECKPOINT_SCHEMA_VERSION
        ),
        "signature": "abc",
        "status": "complete",
        "expected_runs": 2,
        "completed_runs": 1,
        "completed_run_ids": [
            "run-1"
        ],
    }

    with pytest.raises(
        ValueError,
        match="marked complete",
    ):
        _validate_checkpoint(
            checkpoint,
            signature="abc",
            expected_runs=2,
            raw_completed_run_ids={
                "run-1"
            },
        )


def test_checkpoint_due_every_configured_interval() -> None:
    assert _checkpoint_due(
        1,
        2400,
    ) is False
    assert _checkpoint_due(
        24,
        2400,
    ) is False
    assert _checkpoint_due(
        25,
        2400,
    ) is True
    assert _checkpoint_due(
        26,
        2400,
    ) is False
    assert _checkpoint_due(
        2400,
        2400,
    ) is True
    assert _checkpoint_due(
        7,
        7,
    ) is True


def test_checkpoint_replace_retries_transient_permission_error(
    tmp_path,
    monkeypatch,
) -> None:
    checkpoint_path = (
        tmp_path
        / "checkpoint.json"
    )
    temporary_path = (
        tmp_path
        / "checkpoint.tmp"
    )

    checkpoint_path.write_text(
        "old",
        encoding="utf-8",
    )
    temporary_path.write_text(
        "new",
        encoding="utf-8",
    )

    real_replace = os.replace
    attempts = 0

    def flaky_replace(
        source,
        target,
    ) -> None:
        nonlocal attempts
        attempts += 1

        if attempts < 3:
            raise PermissionError(
                "transient lock"
            )

        real_replace(
            source,
            target,
        )

    monkeypatch.setattr(
        experiment_runner.os,
        "replace",
        flaky_replace,
    )
    monkeypatch.setattr(
        experiment_runner.time,
        "sleep",
        lambda _: None,
    )

    replaced = (
        _replace_checkpoint_with_retry(
            temporary_path,
            checkpoint_path,
        )
    )

    assert replaced is True
    assert attempts == 3
    assert checkpoint_path.read_text(
        encoding="utf-8"
    ) == "new"
    assert not temporary_path.exists()


def test_checkpoint_replace_retains_existing_checkpoint_after_lock(
    tmp_path,
    monkeypatch,
) -> None:
    checkpoint_path = (
        tmp_path
        / "checkpoint.json"
    )
    temporary_path = (
        tmp_path
        / "checkpoint.tmp"
    )

    checkpoint_path.write_text(
        "old",
        encoding="utf-8",
    )
    temporary_path.write_text(
        "new",
        encoding="utf-8",
    )

    def locked_replace(
        source,
        target,
    ) -> None:
        del source, target
        raise PermissionError(
            "persistent lock"
        )

    monkeypatch.setattr(
        experiment_runner.os,
        "replace",
        locked_replace,
    )
    monkeypatch.setattr(
        experiment_runner.time,
        "sleep",
        lambda _: None,
    )

    replaced = (
        _replace_checkpoint_with_retry(
            temporary_path,
            checkpoint_path,
        )
    )

    assert replaced is False
    assert checkpoint_path.read_text(
        encoding="utf-8"
    ) == "old"
    assert not temporary_path.exists()


def test_checkpoint_replace_requires_initial_checkpoint_on_persistent_lock(
    tmp_path,
    monkeypatch,
) -> None:
    checkpoint_path = (
        tmp_path
        / "checkpoint.json"
    )
    temporary_path = (
        tmp_path
        / "checkpoint.tmp"
    )

    temporary_path.write_text(
        "new",
        encoding="utf-8",
    )

    def locked_replace(
        source,
        target,
    ) -> None:
        del source, target
        raise PermissionError(
            "persistent lock"
        )

    monkeypatch.setattr(
        experiment_runner.os,
        "replace",
        locked_replace,
    )
    monkeypatch.setattr(
        experiment_runner.time,
        "sleep",
        lambda _: None,
    )

    with pytest.raises(
        PermissionError,
        match="persistent lock",
    ):
        _replace_checkpoint_with_retry(
            temporary_path,
            checkpoint_path,
        )

    assert not temporary_path.exists()
