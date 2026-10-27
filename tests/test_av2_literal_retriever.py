
"""Tests for the AV2 literal retrieval adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.retrieval.base import Retriever
from src.retrieval.literal_retriever import (
    LiteralRetriever,
    load_question_patterns,
)

QUESTION = "How does OAuth2 API work?"


@pytest.fixture
def chunks() -> list[dict[str, Any]]:
    """Create a small corpus with controlled occurrence counts."""

    return [
        {
            "chunk_id": "chunk-003",
            "content": "API API API",
            "source_path": "third.md",
            "section_title": "Third",
            "token_count": 3,
        },
        {
            "chunk_id": "chunk-002",
            "content": "OAuth2 API",
            "source_path": "second.md",
            "section_title": "Second",
            "token_count": 2,
        },
        {
            "chunk_id": "chunk-001",
            "content": "OAuth2 OAuth2 API",
            "source_path": "first.md",
            "section_title": "First",
            "token_count": 3,
        },
        {
            "chunk_id": "chunk-004",
            "content": "Unrelated documentation",
            "source_path": "fourth.md",
            "section_title": "Fourth",
            "token_count": 2,
        },
    ]


@pytest.fixture
def retriever(
    chunks: list[dict[str, Any]],
) -> LiteralRetriever:
    return LiteralRetriever(
        chunks,
        question_patterns={
            QUESTION: ("OAuth2", "API"),
        },
    )


def test_implements_retriever_contract(
    retriever: LiteralRetriever,
) -> None:
    assert isinstance(retriever, Retriever)
    assert retriever.name == "literal"
    assert retriever.matcher_algorithm == "boyer-moore"


def test_ranking_uses_distinct_patterns_then_occurrences(
    retriever: LiteralRetriever,
) -> None:
    result = retriever.retrieve(QUESTION, top_k=3)

    assert result.algorithm == "literal"
    assert result.query == QUESTION

    assert [chunk.chunk_id for chunk in result.chunks] == [
        "chunk-001",
        "chunk-002",
        "chunk-003",
    ]

    assert [chunk.rank for chunk in result.chunks] == [
        1,
        2,
        3,
    ]


def test_top_k_limits_only_returned_chunks(
    retriever: LiteralRetriever,
) -> None:
    result = retriever.retrieve(QUESTION, top_k=1)

    assert len(result.chunks) == 1
    assert result.chunks[0].chunk_id == "chunk-001"

    assert result.metrics.candidates_found == 3
    assert result.metrics.chunks_scored == 4


def test_literal_metadata_contains_exact_positions(
    retriever: LiteralRetriever,
) -> None:
    result = retriever.retrieve(QUESTION)

    first = result.chunks[0]

    assert first.metadata is not None

    assert first.metadata["matched_pattern_count"] == 2
    assert first.metadata["total_occurrences"] == 3

    assert first.metadata["positions_by_pattern"] == {
        "OAuth2": [0, 7],
        "API": [14],
    }


def test_metrics_are_recorded(
    retriever: LiteralRetriever,
) -> None:
    result = retriever.retrieve(QUESTION)

    assert result.metrics.retrieval_time_ns >= 0
    assert result.metrics.sorting_time_ns is not None
    assert result.metrics.sorting_time_ns >= 0
    assert result.metrics.comparisons is not None
    assert result.metrics.comparisons > 0

    assert result.metadata is not None
    assert result.metadata["uses_relevance_labels"] is False
    assert result.metadata["matching_algorithm"] == "boyer-moore"


def test_unknown_question_is_rejected(
    retriever: LiteralRetriever,
) -> None:
    with pytest.raises(
        ValueError,
        match="No predefined literal patterns",
    ):
        retriever.retrieve("A different question")


def test_empty_question_is_rejected(
    retriever: LiteralRetriever,
) -> None:
    with pytest.raises(ValueError):
        retriever.retrieve("")


def test_case_sensitive_matching(
    chunks: list[dict[str, Any]],
) -> None:
    retriever = LiteralRetriever(
        chunks,
        question_patterns={
            "Find oauth2": ("oauth2",),
        },
    )

    result = retriever.retrieve("Find oauth2")

    assert result.chunks == ()
    assert result.metrics.candidates_found == 0


def test_zero_top_k(
    retriever: LiteralRetriever,
) -> None:
    result = retriever.retrieve(QUESTION, top_k=0)

    assert result.chunks == ()
    assert result.metrics.chunks_scored == 0
    assert result.metrics.candidates_found == 0


def test_unknown_matcher_is_rejected(
    chunks: list[dict[str, Any]],
) -> None:
    with pytest.raises(ValueError):
        LiteralRetriever(
            chunks,
            question_patterns={
                QUESTION: ("OAuth2",),
            },
            matcher_algorithm="unknown",
        )


def test_invalid_pattern_is_rejected(
    chunks: list[dict[str, Any]],
) -> None:
    with pytest.raises(ValueError):
        LiteralRetriever(
            chunks,
            question_patterns={
                QUESTION: ("DoesNotAppearInQuestion",),
            },
        )


def test_all_matchers_produce_same_ranking(
    chunks: list[dict[str, Any]],
) -> None:
    algorithms = (
        "naive",
        "kmp",
        "boyer-moore",
        "rabin-karp",
    )

    rankings: list[tuple[str, ...]] = []

    for algorithm in algorithms:
        retriever = LiteralRetriever(
            chunks,
            question_patterns={
                QUESTION: ("OAuth2", "API"),
            },
            matcher_algorithm=algorithm,
        )

        result = retriever.retrieve(QUESTION, top_k=3)

        rankings.append(
            tuple(chunk.chunk_id for chunk in result.chunks)
        )

    assert len(set(rankings)) == 1


def test_deterministic_tie_breaking() -> None:
    chunks = [
        {
            "chunk_id": "z",
            "content": "OAuth2",
            "source_path": "z.md",
            "section_title": "Z",
        },
        {
            "chunk_id": "a",
            "content": "OAuth2",
            "source_path": "a.md",
            "section_title": "A",
        },
    ]

    retriever = LiteralRetriever(
        chunks,
        question_patterns={
            "Explain OAuth2": ("OAuth2",),
        },
    )

    result = retriever.retrieve("Explain OAuth2")

    assert [chunk.chunk_id for chunk in result.chunks] == [
        "a",
        "z",
    ]


def test_load_question_patterns(tmp_path: Path) -> None:
    """The catalog joins to question text, not relevance labels."""

    catalog_path = tmp_path / "patterns.json"
    ground_truth_path = tmp_path / "ground_truth.json"

    catalog = {
        "schema_version": 1,
        "expected_queries": 2,
        "corpus_fingerprint": "test-fingerprint",
        "selection_policy": {
            "method": "question_only_predefined",
            "case_sensitive": True,
            "exact_substring": True,
            "allow_overlaps": True,
            "normalization": "none",
            "ground_truth_relevant_chunks_used": False,
        },
        "queries": [
            {
                "query_id": "q01",
                "patterns": ["OAuth2"],
            },
            {
                "query_id": "q02",
                "patterns": ["WebSocket"],
            },
        ],
    }

    ground_truth = {
        "metadata": {
            "corpus_fingerprint": "test-fingerprint",
        },
        "queries": [
            {
                "query_id": "q01",
                "question": "Explain OAuth2",
                "relevant_chunks": ["hidden-chunk"],
            },
            {
                "query_id": "q02",
                "question": "Explain WebSocket",
                "relevant_chunks": [],
            },
        ],
    }

    catalog_path.write_text(
        json.dumps(catalog),
        encoding="utf-8",
    )

    ground_truth_path.write_text(
        json.dumps(ground_truth),
        encoding="utf-8",
    )

    loaded = load_question_patterns(
        catalog_path,
        ground_truth_path,
    )

    assert loaded == {
        "Explain OAuth2": ("OAuth2",),
        "Explain WebSocket": ("WebSocket",),
    }


def test_real_catalog_loads() -> None:
    """Validate the catalog against the actual AV1 questions."""

    mapping = load_question_patterns()

    assert len(mapping) == 30
    assert all(mapping.values())
    assert all(
        pattern in question
        for question, patterns in mapping.items()
        for pattern in patterns
    )
