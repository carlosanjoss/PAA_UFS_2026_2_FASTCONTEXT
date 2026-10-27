
"""Offline tests for the AV2 x RAG retrieval comparison runner."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from experiments.av2.run_retrieval_comparison import (
    EvaluationQuery,
    evaluate_one,
    load_evaluation_data,
    prepare_output,
    run_comparison,
    summarize_results,
    write_csv,
)
from src.retrieval.base import Retriever
from src.retrieval.models import (
    RetrievedChunk,
    RetrievalMetrics,
    RetrievalResult,
)


class FixedRetriever(Retriever):
    """Provide deterministic Top-k lists without an embedding model."""

    def __init__(
        self,
        name: str,
        chunk_ids: tuple[str, ...],
    ) -> None:
        self._name = name
        self._ids = chunk_ids
        self.calls = 0

    @property
    def name(self) -> str:
        return self._name

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> RetrievalResult:
        self.calls += 1

        chunks = tuple(
            RetrievedChunk(
                chunk_id=chunk_id,
                content=chunk_id,
                source_path=f"{chunk_id}.md",
                section_title="Test",
                score=1.0 / rank,
                rank=rank,
            )
            for rank, chunk_id in enumerate(
                self._ids[:top_k],
                start=1,
            )
        )

        return RetrievalResult(
            query=query,
            algorithm=self.name,
            top_k=top_k,
            chunks=chunks,
            metrics=RetrievalMetrics(
                retrieval_time_ns=1_000_000,
            ),
        )


def test_known_positive_and_unjudged_are_separate() -> None:
    """Unjudged results must not become confirmed negatives."""

    query = EvaluationQuery(
        "q01",
        "Question?",
        "test",
        ("A", "B"),
    )

    result = FixedRetriever(
        "literal",
        ("A", "X", "C"),
    ).retrieve(query.question)

    row = evaluate_one(
        query,
        "literal",
        result,
        {"A", "B", "C"},
        1,
        5,
    )

    assert row["judged_count"] == 2
    assert row["unjudged_count"] == 1

    assert json.loads(
        row["unjudged_chunk_ids"]
    ) == ["X"]

    assert row["pooled_precision_at_k"] == pytest.approx(0.2)
    assert row["precision_lower_bound"] == pytest.approx(0.2)
    assert row["precision_upper_bound"] == pytest.approx(0.4)
    assert row["pooled_recall_at_k"] == pytest.approx(0.5)
    assert row["known_mrr_at_k"] == pytest.approx(1.0)
    assert row["known_hit_rate_at_k"] == 1.0


def test_paired_matrix_and_macro_aggregation() -> None:
    """Every condition must be evaluated repeatedly per question."""

    query = EvaluationQuery(
        "q01",
        "Question?",
        "test",
        ("A",),
    )

    retrievers: dict[str, Retriever] = {
        "semantic": FixedRetriever(
            "semantic",
            ("A", "B"),
        ),
        "literal": FixedRetriever(
            "literal",
            ("B", "A"),
        ),
        "hybrid": FixedRetriever(
            "hybrid",
            ("A", "C"),
        ),
    }

    raw = run_comparison(
        retrievers,
        [query],
        {"q01": {"A", "B"}},
        top_k=2,
        repetitions=3,
        warmup=1,
    )

    assert len(raw) == 9
    assert {
        row["repetition"]
        for row in raw
    } == {1, 2, 3}

    assert all(
        retriever.calls == 4
        for retriever in retrievers.values()
        if isinstance(retriever, FixedRetriever)
    )

    per_query, totals = summarize_results(raw)

    assert len(per_query) == 3
    assert len(totals) == 3

    by_condition = {
        row["condition"]: row
        for row in totals
    }

    assert by_condition["semantic"][
        "known_mrr_at_k"
    ] == pytest.approx(1.0)

    assert by_condition["literal"][
        "known_mrr_at_k"
    ] == pytest.approx(0.5)

    assert by_condition["hybrid"]["unjudged_results"] == 1


def test_wrong_algorithm_response_rejected() -> None:
    """A mislabeled retrieval result must cause an error."""

    query = EvaluationQuery(
        "q01",
        "Question?",
        "test",
        ("A",),
    )

    result = FixedRetriever(
        "semantic",
        ("A",),
    ).retrieve(query.question)

    with pytest.raises(
        ValueError,
        match="does not match",
    ):
        evaluate_one(
            query,
            "literal",
            result,
            {"A"},
            1,
            5,
        )


def test_missing_judgment_pool_rejected() -> None:
    """All evaluated questions need judgment metadata."""

    retrievers: dict[str, Retriever] = {
        name: FixedRetriever(name, ("A",))
        for name in (
            "semantic",
            "literal",
            "hybrid",
        )
    }

    query = EvaluationQuery(
        "q01",
        "Question?",
        "test",
        ("A",),
    )

    with pytest.raises(
        ValueError,
        match="Missing judgments",
    ):
        run_comparison(
            retrievers,
            [query],
            {},
        )


def test_existing_results_not_replaced(tmp_path: Path) -> None:
    """The runner must not overwrite completed results by default."""

    path = tmp_path / "raw_results.csv"
    path.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError):
        prepare_output(
            tmp_path,
            force=False,
        )

    assert path.read_text(
        encoding="utf-8"
    ) == "existing"

    prepare_output(
        tmp_path,
        force=True,
    )

    assert path.read_text(
        encoding="utf-8"
    ) == "existing"


def test_csv_round_trip(tmp_path: Path) -> None:
    """CSV exports must preserve rows and values."""

    path = tmp_path / "rows.csv"

    write_csv(
        path,
        [{
            "condition": "semantic",
            "value": 1.0,
        }],
    )

    with path.open(
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))

    assert rows == [{
        "condition": "semantic",
        "value": "1.0",
    }]


def test_load_judged_universe_and_validate_labels(
    tmp_path: Path,
) -> None:
    """Ground truth and judgment pool must remain consistent."""

    gt_path = tmp_path / "ground_truth.json"
    pool_path = tmp_path / "pool.csv"

    queries = [
        {
            "query_id": f"q{i:02d}",
            "question": f"Question {i}?",
            "category": "test",
            "relevant_chunks": [f"positive_{i}"],
        }
        for i in range(1, 31)
    ]

    ground_truth = {
        "metadata": {
            "corpus_fingerprint": "fixture",
            "annotation_status": "complete",
            "annotation_protocol": {
                "total_candidate_judgments": 30,
            },
        },
        "queries": queries,
    }

    gt_path.write_text(
        json.dumps(ground_truth),
        encoding="utf-8",
    )

    with pool_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "query_id",
                "question",
                "chunk_id",
            ],
        )

        writer.writeheader()

        for item in queries:
            writer.writerow({
                "query_id": item["query_id"],
                "question": item["question"],
                "chunk_id": item["relevant_chunks"][0],
            })

    loaded, judged = load_evaluation_data(
        gt_path,
        pool_path,
        expected_fingerprint="fixture",
    )

    assert len(loaded) == 30
    assert len(judged) == 30
    assert judged["q01"] == {"positive_1"}

    with pool_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "query_id",
                "question",
                "chunk_id",
            ],
        )
        writer.writeheader()

    with pytest.raises(
        ValueError,
        match="Relevant chunks missing",
    ):
        load_evaluation_data(
            gt_path,
            pool_path,
            expected_fingerprint="fixture",
        )
