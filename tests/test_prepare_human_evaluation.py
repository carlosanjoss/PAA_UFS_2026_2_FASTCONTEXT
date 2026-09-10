"""Tests for blinded FastContext human-evaluation preparation."""

from __future__ import annotations

from collections import Counter

from scripts.prepare_human_evaluation import (
    GROUNDEDNESS_TASK,
    QUALITY_TASK,
    EvaluationItem,
    _assign_balanced_pairs,
    _blind_id,
    _generation_outcome,
    _strip_chunk_citations,
)

ANNOTATORS = (
    "carlos",
    "yann",
    "wilson",
    "matheus",
    "gabriela",
)


def _item(
    index: int,
    task_type: str,
) -> EvaluationItem:
    return EvaluationItem(
        task_type=task_type,
        blind_id=(
            f"{task_type}-{index}"
        ),
        run_id=f"run-{index}",
        query_id=f"q{index:02d}",
        condition="optimized",
        question="Question?",
        answer="Answer.",
        retrieved_context="Context.",
        citation_valid="true",
        citation_retry_count="0",
        valid_citations=(
            "chunk-1",
        ),
        generation_outcome="cited_answer",
    )


def test_quality_pair_assignment_is_exactly_balanced() -> None:
    items = tuple(
        _item(
            index,
            QUALITY_TASK,
        )
        for index in range(
            90
        )
    )

    assignments = (
        _assign_balanced_pairs(
            items=items,
            annotators=ANNOTATORS,
            seed=143,
        )
    )

    counts = Counter(
        annotator
        for pair
        in assignments.values()
        for annotator in pair
    )

    assert len(
        assignments
    ) == 90

    assert set(
        counts.values()
    ) == {
        36
    }


def test_groundedness_pair_assignment_is_exactly_balanced() -> None:
    items = tuple(
        _item(
            index,
            GROUNDEDNESS_TASK,
        )
        for index in range(
            60
        )
    )

    assignments = (
        _assign_balanced_pairs(
            items=items,
            annotators=ANNOTATORS,
            seed=244,
        )
    )

    counts = Counter(
        annotator
        for pair
        in assignments.values()
        for annotator in pair
    )

    assert len(
        assignments
    ) == 60

    assert set(
        counts.values()
    ) == {
        24
    }


def test_blind_id_is_deterministic_and_task_specific() -> None:
    quality_id = _blind_id(
        task_type=QUALITY_TASK,
        run_id="optimized|q01",
        seed=42,
    )

    repeated = _blind_id(
        task_type=QUALITY_TASK,
        run_id="optimized|q01",
        seed=42,
    )

    groundedness_id = _blind_id(
        task_type=GROUNDEDNESS_TASK,
        run_id="optimized|q01",
        seed=42,
    )

    assert quality_id == repeated
    assert quality_id.startswith(
        "Q-"
    )
    assert groundedness_id.startswith(
        "G-"
    )
    assert quality_id != groundedness_id


def test_strip_chunk_citations_removes_only_exact_tokens() -> None:
    answer = (
        "FastAPI supports this [chunk-a]. "
        "Keep [not-a-retrieved-chunk]."
    )

    result = _strip_chunk_citations(
        answer,
        (
            "chunk-a",
        ),
    )

    assert (
        result
        == (
            "FastAPI supports this. "
            "Keep [not-a-retrieved-chunk]."
        )
    )


def test_generation_outcome_cited_answer() -> None:
    row = {
        "condition": "optimized",
        "citation_valid": "true",
        "valid_citations": (
            '["chunk-1"]'
        ),
        "answer": (
            "Supported answer [chunk-1]"
        ),
    }

    assert (
        _generation_outcome(
            row
        )
        == "cited_answer"
    )


def test_generation_outcome_valid_abstention() -> None:
    row = {
        "condition": "optimized",
        "citation_valid": "true",
        "valid_citations": "[]",
        "answer": (
            "The provided context is insufficient "
            "to answer this question."
        ),
    }

    assert (
        _generation_outcome(
            row
        )
        == "valid_abstention"
    )


def test_generation_outcome_invalid_abstention() -> None:
    row = {
        "condition": "optimized",
        "citation_valid": "false",
        "valid_citations": "[]",
        "answer": (
            "The provided context is insufficient "
            "to answer."
        ),
    }

    assert (
        _generation_outcome(
            row
        )
        == "invalid_abstention"
    )


def test_generation_outcome_invalid_uncited_answer() -> None:
    row = {
        "condition": "semantic",
        "citation_valid": "false",
        "valid_citations": "[]",
        "answer": (
            "A factual answer without citation."
        ),
    }

    assert (
        _generation_outcome(
            row
        )
        == "invalid_uncited_answer"
    )
