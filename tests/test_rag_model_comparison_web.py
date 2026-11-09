"""Regression tests for the aggregate-only RAG model comparison dashboard."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.rag.model_comparison_api import load_rag_model_comparison
from src.web_api import app


@pytest.fixture(scope="module")
def comparison_payload() -> dict[str, object]:
    return load_rag_model_comparison().model_dump(mode="json")


def test_model_comparison_router_is_registered() -> None:
    schema = app.openapi()

    assert "/api/rag/model-comparison" in schema["paths"]


def test_audited_totals_are_loaded_from_persisted_artifacts(
    comparison_payload: dict[str, object],
) -> None:
    metadata = comparison_payload["metadata"]

    assert isinstance(metadata, dict)
    assert metadata == {
        **metadata,
        "attempts": 180,
        "complete_answers": 179,
        "truncated_answers": 1,
        "frozen_prompts": 90,
        "human_judgments": 596,
        "valid_criterion_comparisons": 415,
        "planned_criterion_comparisons": 420,
        "missing_criterion_comparisons": 5,
        "reviewers": 5,
        "descriptive_only": True,
    }


def test_model_conditions_and_retrieval_method_are_exact(
    comparison_payload: dict[str, object],
) -> None:
    models = comparison_payload["models"]
    retrieval = comparison_payload["retrieval"]

    assert isinstance(models, list)
    assert {(row["role"], row["model"]) for row in models} == {
        ("baseline", "qwen2.5:3b"),
        ("candidate", "llama3.2:3b"),
    }
    assert comparison_payload["conditions"] == ["no_rag", "optimized", "semantic"]
    assert isinstance(retrieval, dict)
    assert retrieval["model"] == "BAAI/bge-small-en-v1.5"
    assert retrieval["embedding_dimension"] == 384
    assert retrieval["normalized"] is True
    assert retrieval["index_type"] == "IndexFlatIP"
    assert retrieval["top_k"] == 5


def test_automatic_human_and_agreement_aggregates_are_preserved(
    comparison_payload: dict[str, object],
) -> None:
    automatic = comparison_payload["automatic_metrics"]
    human = comparison_payload["human_metrics"]
    paired = comparison_payload["paired_human_metrics"]
    agreement = comparison_payload["agreement_metrics"]

    assert isinstance(automatic, list)
    assert isinstance(human, list)
    assert isinstance(paired, list)
    assert isinstance(agreement, list)
    assert len(automatic) == 6
    assert len(human) == 28
    assert len(paired) == 14
    assert len(agreement) == 5
    assert sum(row["both_complete"] for row in paired) == 415

    qwen_optimized = next(
        row
        for row in automatic
        if row["role"] == "baseline" and row["condition"] == "optimized"
    )
    llama_optimized = next(
        row
        for row in automatic
        if row["role"] == "candidate" and row["condition"] == "optimized"
    )
    assert qwen_optimized["truncated"] == 1
    assert qwen_optimized["citation_accepted_n"] == 23
    assert llama_optimized["citation_accepted_n"] == 28

    hallucination_agreement = next(
        row for row in agreement if row["criterion"] == "hallucination_present"
    )
    assert hallucination_agreement["item_pairs"] == 179
    assert hallucination_agreement["exact_agreement_rate"] == pytest.approx(
        0.921788,
    )


def test_public_api_exposes_aggregates_without_reviewer_records() -> None:
    response = TestClient(app).get("/api/rag/model-comparison")

    assert response.status_code == 200
    body = response.text.lower()
    for private_field in (
        "reviewer_id",
        "blind_id",
        "reviewer_notes",
        "allocation",
        "answer_text",
        "run_key",
    ):
        assert private_field not in body


def test_missing_experimental_artifact_returns_an_informative_failure(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="generation_status_summary.json"):
        load_rag_model_comparison(tmp_path)
