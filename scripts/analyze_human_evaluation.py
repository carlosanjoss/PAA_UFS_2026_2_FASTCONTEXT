"""Validate completed human ratings and prepare blinded adjudication."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.utils.config import PROJECT_ROOT

CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "human_evaluation.yaml"
)

QUALITY_TASK = "quality"
GROUNDEDNESS_TASK = "groundedness"

QUALITY_FIELDS = (
    "correctness",
    "completeness",
    "clarity",
    "hallucination_present",
)

GROUNDEDNESS_FIELDS = (
    "groundedness",
)

TASK_SCORE_FIELDS = {
    QUALITY_TASK: QUALITY_FIELDS,
    GROUNDEDNESS_TASK: GROUNDEDNESS_FIELDS,
}

SCORE_VALUES = {
    "correctness": {0, 1, 2},
    "completeness": {0, 1, 2},
    "clarity": {0, 1, 2},
    "hallucination_present": {0, 1},
    "groundedness": {0, 1, 2},
}

IMMUTABLE_FIELDS = {
    QUALITY_TASK: (
        "blind_id",
        "question",
        "answer",
    ),
    GROUNDEDNESS_TASK: (
        "blind_id",
        "question",
        "answer",
        "retrieved_context",
    ),
}

AGREEMENT_FIELDS = (
    "task_type",
    "metric",
    "items",
    "exact_agreements",
    "exact_agreement_rate",
    "mean_absolute_difference",
    "kappa_type",
    "kappa",
)

PAIRWISE_FIELDS = (
    "task_type",
    "metric",
    "annotator_a",
    "annotator_b",
    "items",
    "exact_agreements",
    "exact_agreement_rate",
)

ADJUDICATION_FIELDS = (
    "blind_id",
    "task_type",
    "question",
    "answer",
    "retrieved_context",
    "metrics_to_adjudicate",
    "correctness",
    "completeness",
    "clarity",
    "hallucination_present",
    "groundedness",
    "adjudication_notes",
)

HIDDEN_ADJUDICATION_FIELDS = (
    "blind_id",
    "task_type",
    "query_id",
    "condition",
    "original_annotator_1",
    "original_annotator_2",
    "assigned_adjudicator",
    "metrics_to_adjudicate",
    "original_scores_1",
    "original_scores_2",
)


@dataclass(frozen=True, slots=True)
class Rating:
    """One completed blinded rating."""

    blind_id: str
    task_type: str
    annotator: str
    scores: dict[str, int]
    notes: str


@dataclass(frozen=True, slots=True)
class Disagreement:
    """One item requiring third-person adjudication."""

    blind_id: str
    task_type: str
    query_id: str
    condition: str
    question: str
    answer: str
    retrieved_context: str
    annotator_1: str
    annotator_2: str
    scores_1: dict[str, int]
    scores_2: dict[str, int]
    metrics: tuple[str, ...]


def build_parser() -> argparse.ArgumentParser:
    """Build the completed-rating analysis CLI."""

    parser = argparse.ArgumentParser(
        description=(
            "Validate completed FastContext ratings, compute "
            "agreement, and prepare blinded adjudication."
        )
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Replace only derived analysis/adjudication outputs."
        ),
    )

    return parser


def main() -> int:
    """Validate 300 judgments and create adjudication material."""

    args = build_parser().parse_args()

    config = _load_config()
    evaluation = config[
        "human_evaluation"
    ]

    annotators = tuple(
        str(value).strip()
        for value in evaluation[
            "annotators"
        ]
    )

    seed = int(
        evaluation[
            "seed"
        ]
    )

    output_directory = _project_path(
        evaluation[
            "output"
        ][
            "directory"
        ]
    )

    completed_directory = (
        output_directory
        / "completed"
    )

    assignments_directory = (
        output_directory
        / "assignments"
    )

    manifest_path = (
        output_directory
        / "hidden_manifest.csv"
    )

    analysis_directory = (
        output_directory
        / "analysis"
    )

    adjudication_directory = (
        output_directory
        / "adjudication"
    )

    _prepare_derived_directories(
        analysis_directory=analysis_directory,
        adjudication_directory=(
            adjudication_directory
        ),
        force=args.force,
    )

    manifest_rows = _load_csv(
        manifest_path
    )

    manifest_by_key = {
        (
            row[
                "task_type"
            ],
            row[
                "blind_id"
            ],
        ): row
        for row in manifest_rows
    }

    if len(
        manifest_by_key
    ) != len(
        manifest_rows
    ):
        raise ValueError(
            "Duplicate task/blind_id entries in hidden manifest."
        )

    ratings: list[
        Rating
    ] = []

    file_hash_rows: list[
        dict[
            str,
            str,
        ]
    ] = []

    for annotator in annotators:
        for task_type in (
            QUALITY_TASK,
            GROUNDEDNESS_TASK,
        ):
            completed_path = (
                _resolve_completed_file(
                    completed_directory=(
                        completed_directory
                    ),
                    annotator=annotator,
                    task_type=task_type,
                )
            )

            original_path = (
                assignments_directory
                / (
                    f"{annotator}_"
                    f"{task_type}.csv"
                )
            )

            file_hash_rows.append(
                {
                    "annotator": annotator,
                    "task_type": task_type,
                    "filename": (
                        completed_path.name
                    ),
                    "sha256": _sha256_file(
                        completed_path
                    ),
                }
            )

            ratings.extend(
                _load_and_validate_completed_file(
                    completed_path=(
                        completed_path
                    ),
                    original_path=original_path,
                    annotator=annotator,
                    task_type=task_type,
                )
            )

    _validate_global_coverage(
        ratings=ratings,
        manifest_rows=manifest_rows,
        annotators=annotators,
    )

    ratings_by_item = (
        _group_ratings_by_item(
            ratings
        )
    )

    agreement_rows = (
        _build_agreement_summary(
            ratings_by_item
        )
    )

    pairwise_rows = (
        _build_pairwise_summary(
            ratings_by_item
        )
    )

    disagreements = (
        _find_disagreements(
            ratings_by_item=ratings_by_item,
            manifest_by_key=manifest_by_key,
            assignments_directory=(
                assignments_directory
            ),
        )
    )

    adjudicator_assignments = (
        _assign_adjudicators(
            disagreements=disagreements,
            annotators=annotators,
            seed=seed + 303,
        )
    )

    _write_csv(
        analysis_directory
        / "agreement_summary.csv",
        AGREEMENT_FIELDS,
        agreement_rows,
    )

    _write_csv(
        analysis_directory
        / "pairwise_agreement.csv",
        PAIRWISE_FIELDS,
        pairwise_rows,
    )

    _write_csv(
        analysis_directory
        / "completed_file_hashes.csv",
        (
            "annotator",
            "task_type",
            "filename",
            "sha256",
        ),
        file_hash_rows,
    )

    _write_validated_judgments(
        analysis_directory
        / "validated_judgments.csv",
        ratings,
    )

    _write_adjudication_files(
        adjudication_directory=(
            adjudication_directory
        ),
        disagreements=disagreements,
        assignments=(
            adjudicator_assignments
        ),
        annotators=annotators,
        seed=seed,
    )

    _write_adjudication_summary(
        adjudication_directory
        / "adjudication_summary.csv",
        disagreements=disagreements,
        assignments=(
            adjudicator_assignments
        ),
        annotators=annotators,
    )

    print(
        "FastContext Human Evaluation Audit"
    )
    print(
        "=" * 60
    )
    print(
        f"Completed item-level judgments: {len(ratings)}"
    )
    print(
        "Expected item-level judgments: 300"
    )
    print(
        "Quality judgments: "
        f"{sum(r.task_type == QUALITY_TASK for r in ratings)}"
    )
    print(
        "Groundedness judgments: "
        f"{sum(r.task_type == GROUNDEDNESS_TASK for r in ratings)}"
    )
    print()

    _print_agreement_summary(
        agreement_rows
    )

    quality_disagreements = sum(
        item.task_type
        == QUALITY_TASK
        for item in disagreements
    )

    groundedness_disagreements = sum(
        item.task_type
        == GROUNDEDNESS_TASK
        for item in disagreements
    )

    print()
    print(
        "Items requiring adjudication: "
        f"{len(disagreements)}"
    )
    print(
        "  Quality items: "
        f"{quality_disagreements}"
    )
    print(
        "  Groundedness items: "
        f"{groundedness_disagreements}"
    )
    print()

    counts = Counter(
        adjudicator_assignments.values()
    )

    print(
        "Adjudication workload:"
    )

    for annotator in annotators:
        print(
            f"  {annotator}: "
            f"{counts[annotator]}"
        )

    print()
    print(
        "Analysis directory: "
        f"{analysis_directory}"
    )
    print(
        "Adjudication directory: "
        f"{adjudication_directory}"
    )
    print()
    print(
        "IMPORTANT: do not share "
        "adjudication_manifest.csv with adjudicators."
    )

    return 0


def _load_and_validate_completed_file(
    *,
    completed_path: Path,
    original_path: Path,
    annotator: str,
    task_type: str,
) -> list[Rating]:
    """Validate one evaluator file against its frozen assignment."""

    completed_rows = _load_csv(
        completed_path
    )

    original_rows = _load_csv(
        original_path
    )

    completed_by_id = {
        row[
            "blind_id"
        ]: row
        for row in completed_rows
    }

    original_by_id = {
        row[
            "blind_id"
        ]: row
        for row in original_rows
    }

    if len(
        completed_by_id
    ) != len(
        completed_rows
    ):
        raise ValueError(
            f"Duplicate blind_id in {completed_path.name}."
        )

    if set(
        completed_by_id
    ) != set(
        original_by_id
    ):
        raise ValueError(
            f"Blind ID coverage changed in {completed_path.name}."
        )

    expected_count = (
        36
        if task_type
        == QUALITY_TASK
        else 24
    )

    if len(
        completed_rows
    ) != expected_count:
        raise ValueError(
            f"{completed_path.name} must contain "
            f"{expected_count} rows."
        )

    result: list[
        Rating
    ] = []

    for blind_id, original in original_by_id.items():
        completed = completed_by_id[
            blind_id
        ]

        for field in IMMUTABLE_FIELDS[
            task_type
        ]:
            if completed.get(
                field,
                "",
            ) != original.get(
                field,
                "",
            ):
                raise ValueError(
                    "Immutable evaluation content changed: "
                    f"{completed_path.name} / {blind_id} / "
                    f"{field}."
                )

        scores: dict[
            str,
            int,
        ] = {}

        for metric in TASK_SCORE_FIELDS[
            task_type
        ]:
            raw_value = completed.get(
                metric,
                "",
            ).strip()

            if not raw_value:
                raise ValueError(
                    "Missing score: "
                    f"{completed_path.name} / "
                    f"{blind_id} / {metric}."
                )

            try:
                score = int(
                    raw_value
                )
            except ValueError as error:
                raise ValueError(
                    "Non-integer score: "
                    f"{completed_path.name} / "
                    f"{blind_id} / {metric}."
                ) from error

            if score not in SCORE_VALUES[
                metric
            ]:
                raise ValueError(
                    "Out-of-range score: "
                    f"{completed_path.name} / "
                    f"{blind_id} / {metric}={score}."
                )

            scores[
                metric
            ] = score

        result.append(
            Rating(
                blind_id=blind_id,
                task_type=task_type,
                annotator=annotator,
                scores=scores,
                notes=completed.get(
                    "notes",
                    "",
                ),
            )
        )

    return result


def _validate_global_coverage(
    *,
    ratings: Sequence[Rating],
    manifest_rows: Sequence[Mapping[str, str]],
    annotators: Sequence[str],
) -> None:
    """Require exactly two independent ratings per blinded item."""

    if len(
        ratings
    ) != 300:
        raise ValueError(
            "Expected exactly 300 completed item-level judgments."
        )

    by_item = _group_ratings_by_item(
        ratings
    )

    if len(
        by_item
    ) != 150:
        raise ValueError(
            "Expected exactly 150 unique blinded evaluation items."
        )

    for key, item_ratings in by_item.items():
        if len(
            item_ratings
        ) != 2:
            raise ValueError(
                "Every blinded item must have exactly "
                f"two ratings: {key}."
            )

        raters = {
            rating.annotator
            for rating in item_ratings
        }

        if len(
            raters
        ) != 2:
            raise ValueError(
                "An item cannot be rated twice by "
                f"the same annotator: {key}."
            )

    expected_manifest_keys = {
        (
            row[
                "task_type"
            ],
            row[
                "blind_id"
            ],
        )
        for row in manifest_rows
    }

    if set(
        by_item
    ) != expected_manifest_keys:
        raise ValueError(
            "Completed ratings do not match hidden manifest coverage."
        )

    counts = Counter(
        rating.annotator
        for rating in ratings
    )

    if {
        annotator: counts[
            annotator
        ]
        for annotator in annotators
    } != {
        annotator: 60
        for annotator in annotators
    }:
        raise ValueError(
            "Completed annotator workload is not 60 each."
        )


def _group_ratings_by_item(
    ratings: Sequence[Rating],
) -> dict[
    tuple[
        str,
        str,
    ],
    list[Rating],
]:
    """Group ratings by task and blind ID."""

    grouped: dict[
        tuple[
            str,
            str,
        ],
        list[Rating],
    ] = defaultdict(
        list
    )

    for rating in ratings:
        grouped[
            (
                rating.task_type,
                rating.blind_id,
            )
        ].append(
            rating
        )

    return dict(
        grouped
    )


def _build_agreement_summary(
    ratings_by_item: Mapping[
        tuple[str, str],
        Sequence[Rating],
    ],
) -> list[dict[str, object]]:
    """Compute pooled agreement for every human-evaluation metric."""

    rows: list[
        dict[
            str,
            object,
        ]
    ] = []

    for task_type in (
        QUALITY_TASK,
        GROUNDEDNESS_TASK,
    ):
        for metric in TASK_SCORE_FIELDS[
            task_type
        ]:
            pairs = _metric_pairs(
                ratings_by_item=(
                    ratings_by_item
                ),
                task_type=task_type,
                metric=metric,
            )

            values = sorted(
                SCORE_VALUES[
                    metric
                ]
            )

            ordinal = (
                metric
                != "hallucination_present"
            )

            kappa = _cohen_kappa(
                pairs,
                values=values,
                ordinal=ordinal,
            )

            exact = sum(
                left == right
                for left, right
                in pairs
            )

            if ordinal:
                mean_absolute_difference = (
                    statistics.fmean(
                        abs(
                            left
                            - right
                        )
                        for left, right
                        in pairs
                    )
                )
                kappa_type = (
                    "linear_weighted_cohen"
                )
            else:
                mean_absolute_difference = ""
                kappa_type = (
                    "unweighted_cohen"
                )

            rows.append(
                {
                    "task_type": task_type,
                    "metric": metric,
                    "items": len(
                        pairs
                    ),
                    "exact_agreements": exact,
                    "exact_agreement_rate": (
                        exact
                        / len(
                            pairs
                        )
                    ),
                    "mean_absolute_difference": (
                        mean_absolute_difference
                    ),
                    "kappa_type": kappa_type,
                    "kappa": (
                        ""
                        if kappa is None
                        else kappa
                    ),
                }
            )

    return rows


def _build_pairwise_summary(
    ratings_by_item: Mapping[
        tuple[str, str],
        Sequence[Rating],
    ],
) -> list[dict[str, object]]:
    """Compute exact agreement separately for each annotator pair."""

    rows: list[
        dict[
            str,
            object,
        ]
    ] = []

    grouped: dict[
        tuple[
            str,
            str,
            str,
            str,
        ],
        list[
            tuple[
                int,
                int,
            ]
        ],
    ] = defaultdict(
        list
    )

    for (
        task_type,
        _blind_id,
    ), item_ratings in ratings_by_item.items():
        ordered = sorted(
            item_ratings,
            key=lambda value: (
                value.annotator
            ),
        )

        left, right = ordered

        for metric in TASK_SCORE_FIELDS[
            task_type
        ]:
            grouped[
                (
                    task_type,
                    metric,
                    left.annotator,
                    right.annotator,
                )
            ].append(
                (
                    left.scores[
                        metric
                    ],
                    right.scores[
                        metric
                    ],
                )
            )

    for (
        task_type,
        metric,
        annotator_a,
        annotator_b,
    ), pairs in sorted(
        grouped.items()
    ):
        exact = sum(
            left == right
            for left, right
            in pairs
        )

        rows.append(
            {
                "task_type": task_type,
                "metric": metric,
                "annotator_a": annotator_a,
                "annotator_b": annotator_b,
                "items": len(
                    pairs
                ),
                "exact_agreements": exact,
                "exact_agreement_rate": (
                    exact
                    / len(
                        pairs
                    )
                ),
            }
        )

    return rows


def _metric_pairs(
    *,
    ratings_by_item: Mapping[
        tuple[str, str],
        Sequence[Rating],
    ],
    task_type: str,
    metric: str,
) -> list[
    tuple[
        int,
        int,
    ]
]:
    """Collect deterministically oriented rating pairs for one metric."""

    pairs: list[
        tuple[
            int,
            int,
        ]
    ] = []

    for (
        item_task,
        _blind_id,
    ), item_ratings in sorted(
        ratings_by_item.items()
    ):
        if item_task != task_type:
            continue

        ordered = sorted(
            item_ratings,
            key=lambda value: (
                value.annotator
            ),
        )

        left, right = ordered

        pairs.append(
            (
                left.scores[
                    metric
                ],
                right.scores[
                    metric
                ],
            )
        )

    return pairs


def _cohen_kappa(
    pairs: Sequence[
        tuple[
            int,
            int,
        ]
    ],
    *,
    values: Sequence[int],
    ordinal: bool,
) -> float | None:
    """Calculate unweighted or linearly weighted Cohen's kappa."""

    if not pairs:
        return None

    index = {
        value: position
        for position, value
        in enumerate(
            values
        )
    }

    size = len(
        values
    )

    observed = [
        [
            0.0
            for _ in range(
                size
            )
        ]
        for _ in range(
            size
        )
    ]

    left_counts = [
        0.0
        for _ in range(
            size
        )
    ]

    right_counts = [
        0.0
        for _ in range(
            size
        )
    ]

    for left, right in pairs:
        left_index = index[
            left
        ]
        right_index = index[
            right
        ]

        observed[
            left_index
        ][
            right_index
        ] += 1.0

        left_counts[
            left_index
        ] += 1.0

        right_counts[
            right_index
        ] += 1.0

    total = float(
        len(
            pairs
        )
    )

    if ordinal:
        denominator = max(
            size - 1,
            1,
        )

        weights = [
            [
                1.0
                - (
                    abs(
                        row
                        - column
                    )
                    / denominator
                )
                for column in range(
                    size
                )
            ]
            for row in range(
                size
            )
        ]
    else:
        weights = [
            [
                1.0
                if row == column
                else 0.0
                for column in range(
                    size
                )
            ]
            for row in range(
                size
            )
        ]

    observed_agreement = sum(
        weights[
            row
        ][
            column
        ]
        * (
            observed[
                row
            ][
                column
            ]
            / total
        )
        for row in range(
            size
        )
        for column in range(
            size
        )
    )

    expected_agreement = sum(
        weights[
            row
        ][
            column
        ]
        * (
            left_counts[
                row
            ]
            / total
        )
        * (
            right_counts[
                column
            ]
            / total
        )
        for row in range(
            size
        )
        for column in range(
            size
        )
    )

    denominator = (
        1.0
        - expected_agreement
    )

    if abs(
        denominator
    ) < 1e-12:
        return None

    return (
        observed_agreement
        - expected_agreement
    ) / denominator


def _find_disagreements(
    *,
    ratings_by_item: Mapping[
        tuple[str, str],
        Sequence[Rating],
    ],
    manifest_by_key: Mapping[
        tuple[str, str],
        Mapping[str, str],
    ],
    assignments_directory: Path,
) -> list[Disagreement]:
    """Identify items with one or more metric disagreements."""

    original_content = (
        _load_original_content(
            assignments_directory
        )
    )

    disagreements: list[
        Disagreement
    ] = []

    for key, item_ratings in sorted(
        ratings_by_item.items()
    ):
        task_type, blind_id = key

        ordered = sorted(
            item_ratings,
            key=lambda value: (
                value.annotator
            ),
        )

        left, right = ordered

        metrics = tuple(
            metric
            for metric in TASK_SCORE_FIELDS[
                task_type
            ]
            if left.scores[
                metric
            ]
            != right.scores[
                metric
            ]
        )

        if not metrics:
            continue

        manifest = manifest_by_key[
            key
        ]

        content = original_content[
            key
        ]

        disagreements.append(
            Disagreement(
                blind_id=blind_id,
                task_type=task_type,
                query_id=manifest[
                    "query_id"
                ],
                condition=manifest[
                    "condition"
                ],
                question=content[
                    "question"
                ],
                answer=content[
                    "answer"
                ],
                retrieved_context=(
                    content.get(
                        "retrieved_context",
                        "",
                    )
                ),
                annotator_1=left.annotator,
                annotator_2=right.annotator,
                scores_1=dict(
                    left.scores
                ),
                scores_2=dict(
                    right.scores
                ),
                metrics=metrics,
            )
        )

    return disagreements


def _load_original_content(
    assignments_directory: Path,
) -> dict[
    tuple[
        str,
        str,
    ],
    dict[str, str],
]:
    """Load one canonical blinded copy of each evaluation item."""

    result: dict[
        tuple[
            str,
            str,
        ],
        dict[str, str],
    ] = {}

    for path in assignments_directory.glob(
        "*.csv"
    ):
        name = path.stem

        if name.endswith(
            "_quality"
        ):
            task_type = (
                QUALITY_TASK
            )
        elif name.endswith(
            "_groundedness"
        ):
            task_type = (
                GROUNDEDNESS_TASK
            )
        else:
            continue

        for row in _load_csv(
            path
        ):
            key = (
                task_type,
                row[
                    "blind_id"
                ],
            )

            candidate = {
                field: row.get(
                    field,
                    "",
                )
                for field in IMMUTABLE_FIELDS[
                    task_type
                ]
            }

            previous = result.get(
                key
            )

            if (
                previous is not None
                and previous
                != candidate
            ):
                raise ValueError(
                    "Original assignment content "
                    f"differs across raters: {key}."
                )

            result[
                key
            ] = candidate

    return result


def _assign_adjudicators(
    *,
    disagreements: Sequence[Disagreement],
    annotators: Sequence[str],
    seed: int,
) -> dict[str, str]:
    """Assign each disagreement to a third eligible evaluator."""

    rng = random.Random(
        seed
    )

    shuffled = list(
        disagreements
    )

    rng.shuffle(
        shuffled
    )

    counts = Counter()

    assignments: dict[
        str,
        str,
    ] = {}

    for item in shuffled:
        excluded = {
            item.annotator_1,
            item.annotator_2,
        }

        candidates = [
            annotator
            for annotator in annotators
            if annotator not in excluded
        ]

        rng.shuffle(
            candidates
        )

        selected = min(
            candidates,
            key=lambda annotator: (
                counts[
                    annotator
                ],
                annotator,
            ),
        )

        assignments[
            item.blind_id
        ] = selected

        counts[
            selected
        ] += 1

    return assignments


def _write_adjudication_files(
    *,
    adjudication_directory: Path,
    disagreements: Sequence[Disagreement],
    assignments: Mapping[str, str],
    annotators: Sequence[str],
    seed: int,
) -> None:
    """Write blinded per-adjudicator files plus an admin manifest."""

    assignment_directory = (
        adjudication_directory
        / "assignments"
    )

    assignment_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    by_adjudicator: dict[
        str,
        list[Disagreement],
    ] = {
        annotator: []
        for annotator in annotators
    }

    for item in disagreements:
        by_adjudicator[
            assignments[
                item.blind_id
            ]
        ].append(
            item
        )

    for index, annotator in enumerate(
        annotators
    ):
        items = list(
            by_adjudicator[
                annotator
            ]
        )

        random.Random(
            seed
            + 3000
            + index
        ).shuffle(
            items
        )

        rows = [
            _blinded_adjudication_row(
                item
            )
            for item in items
        ]

        _write_csv(
            assignment_directory
            / (
                f"{annotator}_"
                "adjudication.csv"
            ),
            ADJUDICATION_FIELDS,
            rows,
        )

    hidden_rows = [
        {
            "blind_id": item.blind_id,
            "task_type": item.task_type,
            "query_id": item.query_id,
            "condition": item.condition,
            "original_annotator_1": (
                item.annotator_1
            ),
            "original_annotator_2": (
                item.annotator_2
            ),
            "assigned_adjudicator": (
                assignments[
                    item.blind_id
                ]
            ),
            "metrics_to_adjudicate": (
                json.dumps(
                    list(
                        item.metrics
                    ),
                    ensure_ascii=False,
                )
            ),
            "original_scores_1": (
                json.dumps(
                    item.scores_1,
                    ensure_ascii=False,
                    sort_keys=True,
                )
            ),
            "original_scores_2": (
                json.dumps(
                    item.scores_2,
                    ensure_ascii=False,
                    sort_keys=True,
                )
            ),
        }
        for item in disagreements
    ]

    _write_csv(
        adjudication_directory
        / "adjudication_manifest.csv",
        HIDDEN_ADJUDICATION_FIELDS,
        hidden_rows,
    )

    _write_adjudication_instructions(
        adjudication_directory
        / "ADJUDICATION_CODEBOOK.md"
    )


def _blinded_adjudication_row(
    item: Disagreement,
) -> dict[str, str]:
    """Build one condition-blind third-rating row."""

    return {
        "blind_id": item.blind_id,
        "task_type": item.task_type,
        "question": item.question,
        "answer": item.answer,
        "retrieved_context": (
            item.retrieved_context
        ),
        "metrics_to_adjudicate": (
            ", ".join(
                item.metrics
            )
        ),
        "correctness": "",
        "completeness": "",
        "clarity": "",
        "hallucination_present": "",
        "groundedness": "",
        "adjudication_notes": "",
    }


def _write_adjudication_instructions(
    path: Path,
) -> None:
    """Write the third-rater protocol."""

    content = """# FastContext — Adjudicação cega

Cada item desta etapa apresentou divergência entre os dois avaliadores
independentes originais.

## Regras

- Não tente descobrir a condição experimental.
- Não consulte `adjudication_manifest.csv`.
- Não solicite os escores dos dois avaliadores anteriores.
- Avalie novamente o item de forma independente.
- Preencha somente as métricas listadas em `metrics_to_adjudicate`.
- Use exatamente as mesmas escalas do `evaluation_codebook.md`.
- Não altere `blind_id`, pergunta, resposta ou contexto.
- Use `adjudication_notes` somente quando necessário.

A pontuação do terceiro avaliador será usada como decisão de adjudicação
para a métrica em que houve divergência.
"""

    path.write_text(
        content,
        encoding="utf-8",
    )


def _write_adjudication_summary(
    path: Path,
    *,
    disagreements: Sequence[Disagreement],
    assignments: Mapping[str, str],
    annotators: Sequence[str],
) -> None:
    """Write administrative counts for the adjudication phase."""

    total_by_annotator = Counter(
        assignments.values()
    )

    quality_by_annotator = Counter(
        assignments[
            item.blind_id
        ]
        for item in disagreements
        if item.task_type
        == QUALITY_TASK
    )

    groundedness_by_annotator = Counter(
        assignments[
            item.blind_id
        ]
        for item in disagreements
        if item.task_type
        == GROUNDEDNESS_TASK
    )

    rows = [
        {
            "annotator": annotator,
            "quality_items": (
                quality_by_annotator[
                    annotator
                ]
            ),
            "groundedness_items": (
                groundedness_by_annotator[
                    annotator
                ]
            ),
            "total_items": (
                total_by_annotator[
                    annotator
                ]
            ),
        }
        for annotator in annotators
    ]

    _write_csv(
        path,
        (
            "annotator",
            "quality_items",
            "groundedness_items",
            "total_items",
        ),
        rows,
    )


def _write_validated_judgments(
    path: Path,
    ratings: Sequence[Rating],
) -> None:
    """Write normalized completed ratings for later finalization."""

    rows = []

    for rating in sorted(
        ratings,
        key=lambda value: (
            value.task_type,
            value.blind_id,
            value.annotator,
        ),
    ):
        row: dict[
            str,
            object,
        ] = {
            "task_type": (
                rating.task_type
            ),
            "blind_id": (
                rating.blind_id
            ),
            "annotator": (
                rating.annotator
            ),
            "correctness": "",
            "completeness": "",
            "clarity": "",
            "hallucination_present": "",
            "groundedness": "",
            "notes": rating.notes,
        }

        for metric, score in rating.scores.items():
            row[
                metric
            ] = score

        rows.append(
            row
        )

    _write_csv(
        path,
        (
            "task_type",
            "blind_id",
            "annotator",
            "correctness",
            "completeness",
            "clarity",
            "hallucination_present",
            "groundedness",
            "notes",
        ),
        rows,
    )


def _print_agreement_summary(
    rows: Sequence[Mapping[str, object]],
) -> None:
    """Print concise metric-level agreement results."""

    print(
        "Agreement summary:"
    )

    for row in rows:
        agreement = float(
            row[
                "exact_agreement_rate"
            ]
        )

        kappa = row[
            "kappa"
        ]

        kappa_text = (
            "N/A"
            if kappa == ""
            else f"{float(kappa):.4f}"
        )

        print(
            "  "
            f"{row['metric']}: "
            f"exact={agreement:.2%}, "
            f"kappa={kappa_text}"
        )


def _resolve_completed_file(
    *,
    completed_directory: Path,
    annotator: str,
    task_type: str,
) -> Path:
    """Resolve a completed evaluator file without guessing its content."""

    candidates = (
        completed_directory
        / (
            f"{annotator}_"
            f"{task_type}_completed.csv"
        ),
        completed_directory
        / (
            f"{annotator}_"
            f"{task_type}.csv"
        ),
    )

    existing = [
        path
        for path in candidates
        if path.exists()
    ]

    if len(
        existing
    ) == 1:
        return existing[
            0
        ]

    if not existing:
        raise FileNotFoundError(
            "Missing completed evaluator file for "
            f"{annotator}/{task_type}. Expected one of: "
            + ", ".join(
                path.name
                for path in candidates
            )
        )

    raise ValueError(
        "Ambiguous completed files for "
        f"{annotator}/{task_type}: "
        + ", ".join(
            path.name
            for path in existing
        )
    )


def _prepare_derived_directories(
    *,
    analysis_directory: Path,
    adjudication_directory: Path,
    force: bool,
) -> None:
    """Prepare derived outputs without touching completed ratings."""

    existing = [
        path
        for path in (
            analysis_directory,
            adjudication_directory,
        )
        if path.exists()
    ]

    if existing and not force:
        raise FileExistsError(
            "Derived analysis/adjudication output already exists. "
            "Use --force only to rebuild these derived outputs."
        )

    for path in existing:
        import shutil

        shutil.rmtree(
            path
        )

    analysis_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    adjudication_directory.mkdir(
        parents=True,
        exist_ok=True,
    )


def _load_config() -> dict[str, Any]:
    """Load the frozen human-evaluation configuration."""

    data = yaml.safe_load(
        CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        data,
        dict,
    ):
        raise TypeError(
            "Human-evaluation config must be a mapping."
        )

    return data


def _load_csv(
    path: Path,
) -> list[dict[str, str]]:
    """Read one UTF-8 CSV."""

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        return list(
            csv.DictReader(
                file
            )
        )


def _write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, object]],
) -> None:
    """Write one deterministic UTF-8 CSV."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    field: row.get(
                        field,
                        "",
                    )
                    for field in fieldnames
                }
            )


def _project_path(
    value: object,
) -> Path:
    """Resolve a project-relative path."""

    path = Path(
        str(
            value
        )
    )

    if path.is_absolute():
        return path

    return (
        PROJECT_ROOT
        / path
    )


def _sha256_file(
    path: Path,
) -> str:
    """Calculate SHA-256 for one completed evaluator file."""

    hasher = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:
        for block in iter(
            lambda: file.read(
                1024
                * 1024
            ),
            b"",
        ):
            hasher.update(
                block
            )

    return hasher.hexdigest()


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
