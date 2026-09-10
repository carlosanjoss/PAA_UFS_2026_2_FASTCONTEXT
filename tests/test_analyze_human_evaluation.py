"""Tests for human-evaluation agreement and adjudication helpers."""

from __future__ import annotations

from scripts.analyze_human_evaluation import (
    GROUNDEDNESS_TASK,
    QUALITY_TASK,
    Disagreement,
    _assign_adjudicators,
    _cohen_kappa,
)

ANNOTATORS = (
    "carlos",
    "yann",
    "wilson",
    "matheus",
    "gabriela",
)


def test_unweighted_kappa_is_one_for_perfect_agreement() -> None:
    pairs = [
        (0, 0),
        (1, 1),
        (0, 0),
        (1, 1),
    ]

    result = _cohen_kappa(
        pairs,
        values=(0, 1),
        ordinal=False,
    )

    assert result == 1.0


def test_linear_weighted_kappa_is_one_for_perfect_agreement() -> None:
    pairs = [
        (0, 0),
        (1, 1),
        (2, 2),
        (1, 1),
    ]

    result = _cohen_kappa(
        pairs,
        values=(0, 1, 2),
        ordinal=True,
    )

    assert result == 1.0


def test_kappa_returns_none_when_expected_agreement_is_one() -> None:
    pairs = [
        (2, 2),
        (2, 2),
        (2, 2),
    ]

    result = _cohen_kappa(
        pairs,
        values=(0, 1, 2),
        ordinal=True,
    )

    assert result is None


def _disagreement(
    index: int,
    task_type: str,
    annotator_1: str,
    annotator_2: str,
) -> Disagreement:
    return Disagreement(
        blind_id=f"B-{index}",
        task_type=task_type,
        query_id=f"q{index:02d}",
        condition="optimized",
        question="Question?",
        answer="Answer.",
        retrieved_context="Context.",
        annotator_1=annotator_1,
        annotator_2=annotator_2,
        scores_1={
            "correctness": 1,
        },
        scores_2={
            "correctness": 2,
        },
        metrics=(
            "correctness",
        ),
    )


def test_adjudicator_is_never_one_of_original_raters() -> None:
    disagreements = [
        _disagreement(
            index=1,
            task_type=QUALITY_TASK,
            annotator_1="carlos",
            annotator_2="yann",
        ),
        _disagreement(
            index=2,
            task_type=GROUNDEDNESS_TASK,
            annotator_1="wilson",
            annotator_2="matheus",
        ),
    ]

    assignments = _assign_adjudicators(
        disagreements=disagreements,
        annotators=ANNOTATORS,
        seed=345,
    )

    for item in disagreements:
        assert (
            assignments[
                item.blind_id
            ]
            not in {
                item.annotator_1,
                item.annotator_2,
            }
        )


def test_adjudicator_assignment_is_deterministic() -> None:
    disagreements = [
        _disagreement(
            index=index,
            task_type=QUALITY_TASK,
            annotator_1="carlos",
            annotator_2="yann",
        )
        for index in range(
            1,
            11,
        )
    ]

    first = _assign_adjudicators(
        disagreements=disagreements,
        annotators=ANNOTATORS,
        seed=345,
    )

    second = _assign_adjudicators(
        disagreements=disagreements,
        annotators=ANNOTATORS,
        seed=345,
    )

    assert first == second
