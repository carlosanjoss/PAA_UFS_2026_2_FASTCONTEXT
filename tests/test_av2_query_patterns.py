
"""Validate the predefined AV2 literal patterns for AV1 questions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PATTERNS_PATH = (
    PROJECT_ROOT / "data" / "av2" / "query_patterns.json"
)

GROUND_TRUTH_PATH = (
    PROJECT_ROOT
    / "data"
    / "ground_truth"
    / "ground_truth.final.json"
)

EXPECTED_FINGERPRINT = (
    "e08b47b55d7237f38bcaba48c79a468a40b92e4f3120e968ae9ece999f5e8c2d"
)


@pytest.fixture(scope="module")
def datasets() -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the pattern catalog and frozen AV1 ground truth."""

    patterns = json.loads(
        PATTERNS_PATH.read_text(encoding="utf-8")
    )

    ground_truth = json.loads(
        GROUND_TRUTH_PATH.read_text(encoding="utf-8")
    )

    assert isinstance(patterns, dict)
    assert isinstance(ground_truth, dict)

    return patterns, ground_truth


def test_expected_query_count(
    datasets: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    """The catalog must cover exactly the 30 AV1 questions."""

    patterns, ground_truth = datasets

    assert patterns["schema_version"] == 1
    assert patterns["expected_queries"] == 30

    assert len(patterns["queries"]) == 30
    assert len(ground_truth["queries"]) == 30


def test_query_ids_match_ground_truth(
    datasets: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    """No evaluated question may be missing or duplicated."""

    patterns, ground_truth = datasets

    expected_ids = [
        entry["query_id"]
        for entry in ground_truth["queries"]
    ]

    actual_ids = [
        entry["query_id"]
        for entry in patterns["queries"]
    ]

    assert len(actual_ids) == len(set(actual_ids))
    assert set(actual_ids) == set(expected_ids)


def test_patterns_are_exact_question_substrings(
    datasets: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    """Every pattern must occur literally in its own question."""

    patterns, ground_truth = datasets

    questions = {
        entry["query_id"]: entry["question"]
        for entry in ground_truth["queries"]
    }

    for entry in patterns["queries"]:
        query_id = entry["query_id"]
        question = questions[query_id]
        selected_patterns = entry["patterns"]

        assert isinstance(selected_patterns, list)
        assert selected_patterns

        assert len(selected_patterns) == len(
            set(selected_patterns)
        )

        for pattern in selected_patterns:
            assert isinstance(pattern, str)
            assert pattern
            assert pattern == pattern.strip()
            assert pattern in question, (
                f"{query_id}: pattern {pattern!r} "
                "is not an exact question substring"
            )


def test_corpus_fingerprint_matches(
    datasets: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    """The catalog must reference the frozen AV1 corpus."""

    patterns, ground_truth = datasets

    actual_fingerprint = ground_truth[
        "metadata"
    ]["corpus_fingerprint"]

    assert actual_fingerprint == EXPECTED_FINGERPRINT
    assert patterns["corpus_fingerprint"] == actual_fingerprint

    assert ground_truth["metadata"]["corpus_chunk_count"] == 1305
    assert ground_truth["metadata"]["annotation_status"] == "complete"


def test_selection_policy(
    datasets: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    """Document the rules enforced by the literal matching protocol."""

    patterns, _ = datasets
    policy = patterns["selection_policy"]

    assert policy["method"] == "question_only_predefined"
    assert policy["exact_substring"] is True
    assert policy["case_sensitive"] is True
    assert policy["allow_overlaps"] is True
    assert policy["normalization"] == "none"

    assert policy["ground_truth_relevant_chunks_used"] is False

    for entry in patterns["queries"]:
        assert set(entry) == {"query_id", "patterns"}


def test_no_relevance_labels_in_pattern_catalog(
    datasets: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    """Keep relevance labels outside the pattern input file."""

    patterns, _ = datasets

    forbidden_fields = {
        "relevant_chunks",
        "relevant_chunk_ids",
        "expected_relevant",
        "retrieved_chunks",
        "precision",
        "recall",
        "mrr",
    }

    for entry in patterns["queries"]:
        assert forbidden_fields.isdisjoint(entry.keys())
