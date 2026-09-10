"""Finalize unblinded FastContext human-evaluation analysis."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml
from scipy.stats import chi2, friedmanchisquare, rankdata, wilcoxon

from src.utils.config import PROJECT_ROOT

CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "human_evaluation.yaml"
)

QUALITY_TASK = "quality"
GROUNDEDNESS_TASK = "groundedness"

QUALITY_METRICS = (
    "correctness",
    "completeness",
    "clarity",
    "hallucination_present",
)

GROUNDEDNESS_METRICS = (
    "groundedness",
)

QUALITY_CONDITIONS = (
    "no_rag",
    "optimized",
    "semantic",
)

RAG_CONDITIONS = (
    "optimized",
    "semantic",
)

FINAL_ITEM_FIELDS = (
    "task_type",
    "blind_id",
    "query_id",
    "condition",
    "correctness",
    "completeness",
    "clarity",
    "hallucination_present",
    "groundedness",
)

SUMMARY_FIELDS = (
    "task_type",
    "metric",
    "condition",
    "n",
    "mean",
    "median",
    "minimum",
    "maximum",
    "score_0_count",
    "score_1_count",
    "score_2_count",
    "max_score_rate",
)

TEST_FIELDS = (
    "task_type",
    "metric",
    "test",
    "comparison",
    "n",
    "statistic",
    "p_value",
    "p_value_holm",
    "effect_size_name",
    "effect_size",
)

ANNOTATOR_DISTRIBUTION_FIELDS = (
    "annotator",
    "task_type",
    "metric",
    "score",
    "count",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the final human-evaluation analysis CLI."""

    parser = argparse.ArgumentParser(
        description=(
            "Unblind the completed human evaluation after agreement "
            "and adjudication are closed."
        )
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Replace only the derived data/rag_evaluation/final directory."
        ),
    )

    return parser


def main() -> int:
    """Create final per-condition human-evaluation results."""

    args = build_parser().parse_args()

    config = _load_config()
    evaluation = config[
        "human_evaluation"
    ]

    evaluation_directory = _project_path(
        evaluation[
            "output"
        ][
            "directory"
        ]
    )

    analysis_directory = (
        evaluation_directory
        / "analysis"
    )

    final_directory = (
        evaluation_directory
        / "final"
    )

    _prepare_final_directory(
        final_directory,
        force=args.force,
    )

    agreement_rows = _load_csv(
        analysis_directory
        / "agreement_summary.csv"
    )

    _require_closed_agreement(
        agreement_rows
    )

    adjudication_manifest_path = (
        evaluation_directory
        / "adjudication"
        / "adjudication_manifest.csv"
    )

    _require_no_pending_adjudication(
        adjudication_manifest_path
    )

    manifest_rows = _load_csv(
        evaluation_directory
        / "hidden_manifest.csv"
    )

    judgments = _load_csv(
        analysis_directory
        / "validated_judgments.csv"
    )

    final_items = _build_final_items(
        manifest_rows=manifest_rows,
        judgments=judgments,
    )

    _validate_final_items(
        final_items
    )

    quality_summary = _build_summary(
        final_items=final_items,
        task_type=QUALITY_TASK,
        metrics=QUALITY_METRICS,
        conditions=QUALITY_CONDITIONS,
    )

    groundedness_summary = _build_summary(
        final_items=final_items,
        task_type=GROUNDEDNESS_TASK,
        metrics=GROUNDEDNESS_METRICS,
        conditions=RAG_CONDITIONS,
    )

    statistical_tests = _build_statistical_tests(
        final_items
    )

    annotator_distribution = (
        _build_annotator_distribution(
            judgments
        )
    )

    completed_hash_rows = _load_csv(
        analysis_directory
        / "completed_file_hashes.csv"
    )

    integrity = _build_integrity_audit(
        completed_hash_rows=completed_hash_rows,
        agreement_rows=agreement_rows,
        final_items=final_items,
    )

    _write_csv(
        final_directory
        / "final_item_scores.csv",
        FINAL_ITEM_FIELDS,
        final_items,
    )

    _write_csv(
        final_directory
        / "quality_summary_by_condition.csv",
        SUMMARY_FIELDS,
        quality_summary,
    )

    _write_csv(
        final_directory
        / "groundedness_summary_by_condition.csv",
        SUMMARY_FIELDS,
        groundedness_summary,
    )

    _write_csv(
        final_directory
        / "statistical_tests.csv",
        TEST_FIELDS,
        statistical_tests,
    )

    _write_csv(
        final_directory
        / "rating_distribution_by_annotator.csv",
        ANNOTATOR_DISTRIBUTION_FIELDS,
        annotator_distribution,
    )

    (
        final_directory
        / "integrity_audit.json"
    ).write_text(
        json.dumps(
            integrity,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    report = _build_markdown_report(
        quality_summary=quality_summary,
        groundedness_summary=groundedness_summary,
        statistical_tests=statistical_tests,
        integrity=integrity,
    )

    (
        final_directory
        / "HUMAN_EVALUATION_RESULTS.md"
    ).write_text(
        report,
        encoding="utf-8",
    )

    print(
        "FastContext Final Human Evaluation"
    )
    print(
        "=" * 60
    )
    print(
        "Final quality items: "
        f"{sum(row['task_type'] == QUALITY_TASK for row in final_items)}"
    )
    print(
        "Final groundedness items: "
        f"{sum(row['task_type'] == GROUNDEDNESS_TASK for row in final_items)}"
    )
    print()

    _print_summary(
        quality_summary,
        groundedness_summary,
    )

    print()
    print(
        "Integrity:"
    )
    print(
        "  completed files: "
        f"{integrity['completed_file_count']}"
    )
    print(
        "  unique completed SHA-256: "
        f"{integrity['unique_completed_sha256']}"
    )
    print(
        "  literal duplicate file hashes: "
        f"{integrity['duplicate_completed_sha256_count']}"
    )
    print(
        "  perfect agreement metrics: "
        f"{integrity['perfect_agreement_metric_count']}/"
        f"{integrity['agreement_metric_count']}"
    )
    print()
    print(
        f"Output: {final_directory}"
    )
    print()
    print(
        "Human evaluation is now unblinded and finalized."
    )

    return 0


def _build_final_items(
    *,
    manifest_rows: Sequence[Mapping[str, str]],
    judgments: Sequence[Mapping[str, str]],
) -> list[dict[str, object]]:
    """Collapse two identical ratings into one final item score."""

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

    grouped: dict[
        tuple[
            str,
            str,
        ],
        list[
            Mapping[
                str,
                str,
            ]
        ],
    ] = defaultdict(
        list
    )

    for judgment in judgments:
        grouped[
            (
                judgment[
                    "task_type"
                ],
                judgment[
                    "blind_id"
                ],
            )
        ].append(
            judgment
        )

    final_rows: list[
        dict[
            str,
            object,
        ]
    ] = []

    for key, item_judgments in sorted(
        grouped.items()
    ):
        if len(
            item_judgments
        ) != 2:
            raise ValueError(
                f"Expected two judgments for {key}."
            )

        manifest = manifest_by_key.get(
            key
        )

        if manifest is None:
            raise ValueError(
                f"Missing hidden manifest entry for {key}."
            )

        task_type, blind_id = key

        metrics = (
            QUALITY_METRICS
            if task_type
            == QUALITY_TASK
            else GROUNDEDNESS_METRICS
        )

        final_scores: dict[
            str,
            int,
        ] = {}

        for metric in metrics:
            values = [
                int(
                    judgment[
                        metric
                    ]
                )
                for judgment in item_judgments
            ]

            if values[
                0
            ] != values[
                1
            ]:
                raise ValueError(
                    "Unresolved disagreement found during "
                    f"finalization: {key} / {metric}."
                )

            final_scores[
                metric
            ] = values[
                0
            ]

        row: dict[
            str,
            object,
        ] = {
            "task_type": task_type,
            "blind_id": blind_id,
            "query_id": manifest[
                "query_id"
            ],
            "condition": manifest[
                "condition"
            ],
            "correctness": "",
            "completeness": "",
            "clarity": "",
            "hallucination_present": "",
            "groundedness": "",
        }

        row.update(
            final_scores
        )

        final_rows.append(
            row
        )

    return final_rows


def _validate_final_items(
    rows: Sequence[Mapping[str, object]],
) -> None:
    """Validate final item counts and paired query coverage."""

    quality = [
        row
        for row in rows
        if row[
            "task_type"
        ] == QUALITY_TASK
    ]

    groundedness = [
        row
        for row in rows
        if row[
            "task_type"
        ] == GROUNDEDNESS_TASK
    ]

    if len(
        quality
    ) != 90:
        raise ValueError(
            "Expected 90 final quality items."
        )

    if len(
        groundedness
    ) != 60:
        raise ValueError(
            "Expected 60 final groundedness items."
        )

    quality_counts = Counter(
        str(
            row[
                "condition"
            ]
        )
        for row in quality
    )

    if quality_counts != Counter(
        {
            "no_rag": 30,
            "optimized": 30,
            "semantic": 30,
        }
    ):
        raise ValueError(
            "Unexpected final quality condition counts."
        )

    groundedness_counts = Counter(
        str(
            row[
                "condition"
            ]
        )
        for row in groundedness
    )

    if groundedness_counts != Counter(
        {
            "optimized": 30,
            "semantic": 30,
        }
    ):
        raise ValueError(
            "Unexpected final groundedness condition counts."
        )

    quality_queries = _query_condition_map(
        quality
    )

    for query_id, conditions in quality_queries.items():
        if set(
            conditions
        ) != set(
            QUALITY_CONDITIONS
        ):
            raise ValueError(
                "Incomplete paired quality conditions for "
                f"{query_id}."
            )

    groundedness_queries = _query_condition_map(
        groundedness
    )

    for query_id, conditions in groundedness_queries.items():
        if set(
            conditions
        ) != set(
            RAG_CONDITIONS
        ):
            raise ValueError(
                "Incomplete paired groundedness conditions for "
                f"{query_id}."
            )


def _query_condition_map(
    rows: Sequence[Mapping[str, object]],
) -> dict[
    str,
    dict[
        str,
        Mapping[
            str,
            object,
        ],
    ],
]:
    """Index rows by query and condition."""

    result: dict[
        str,
        dict[
            str,
            Mapping[
                str,
                object,
            ],
        ],
    ] = defaultdict(
        dict
    )

    for row in rows:
        query_id = str(
            row[
                "query_id"
            ]
        )

        condition = str(
            row[
                "condition"
            ]
        )

        if condition in result[
            query_id
        ]:
            raise ValueError(
                "Duplicate query/condition item: "
                f"{query_id}/{condition}."
            )

        result[
            query_id
        ][
            condition
        ] = row

    return dict(
        result
    )


def _build_summary(
    *,
    final_items: Sequence[Mapping[str, object]],
    task_type: str,
    metrics: Sequence[str],
    conditions: Sequence[str],
) -> list[dict[str, object]]:
    """Build descriptive condition summaries."""

    rows: list[
        dict[
            str,
            object,
        ]
    ] = []

    for metric in metrics:
        max_score = (
            1
            if metric
            == "hallucination_present"
            else 2
        )

        for condition in conditions:
            values = [
                int(
                    row[
                        metric
                    ]
                )
                for row in final_items
                if row[
                    "task_type"
                ] == task_type
                and row[
                    "condition"
                ] == condition
            ]

            if len(
                values
            ) != 30:
                raise ValueError(
                    "Expected 30 values for "
                    f"{task_type}/{metric}/{condition}."
                )

            counts = Counter(
                values
            )

            rows.append(
                {
                    "task_type": task_type,
                    "metric": metric,
                    "condition": condition,
                    "n": len(
                        values
                    ),
                    "mean": (
                        sum(
                            values
                        )
                        / len(
                            values
                        )
                    ),
                    "median": _median(
                        values
                    ),
                    "minimum": min(
                        values
                    ),
                    "maximum": max(
                        values
                    ),
                    "score_0_count": counts[
                        0
                    ],
                    "score_1_count": counts[
                        1
                    ],
                    "score_2_count": counts[
                        2
                    ],
                    "max_score_rate": (
                        counts[
                            max_score
                        ]
                        / len(
                            values
                        )
                    ),
                }
            )

    return rows


def _build_statistical_tests(
    final_items: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Run paired tests appropriate for the repeated-query design."""

    rows: list[
        dict[
            str,
            object,
        ]
    ] = []

    quality = [
        row
        for row in final_items
        if row[
            "task_type"
        ] == QUALITY_TASK
    ]

    groundedness = [
        row
        for row in final_items
        if row[
            "task_type"
        ] == GROUNDEDNESS_TASK
    ]

    quality_by_query = _query_condition_map(
        quality
    )

    for metric in (
        "correctness",
        "completeness",
        "clarity",
    ):
        vectors = {
            condition: [
                int(
                    quality_by_query[
                        query_id
                    ][
                        condition
                    ][
                        metric
                    ]
                )
                for query_id in sorted(
                    quality_by_query
                )
            ]
            for condition in QUALITY_CONDITIONS
        }

        statistic, p_value = _friedman_test(
            vectors
        )

        rows.append(
            {
                "task_type": QUALITY_TASK,
                "metric": metric,
                "test": "friedman",
                "comparison": "all_conditions",
                "n": 30,
                "statistic": statistic,
                "p_value": p_value,
                "p_value_holm": "",
                "effect_size_name": "kendalls_w",
                "effect_size": (
                    statistic
                    / (
                        30
                        * (
                            len(
                                QUALITY_CONDITIONS
                            )
                            - 1
                        )
                    )
                ),
            }
        )

        pairwise = []

        for left, right in (
            (
                "no_rag",
                "optimized",
            ),
            (
                "no_rag",
                "semantic",
            ),
            (
                "optimized",
                "semantic",
            ),
        ):
            statistic_w, pair_p = (
                _wilcoxon_test(
                    vectors[
                        left
                    ],
                    vectors[
                        right
                    ],
                )
            )

            effect = (
                _paired_rank_biserial(
                    vectors[
                        left
                    ],
                    vectors[
                        right
                    ],
                )
            )

            pairwise.append(
                {
                    "task_type": QUALITY_TASK,
                    "metric": metric,
                    "test": "wilcoxon_signed_rank",
                    "comparison": (
                        f"{left}_vs_{right}"
                    ),
                    "n": 30,
                    "statistic": statistic_w,
                    "p_value": pair_p,
                    "p_value_holm": "",
                    "effect_size_name": (
                        "rank_biserial_left_minus_right"
                    ),
                    "effect_size": effect,
                }
            )

        _apply_holm(
            pairwise
        )

        rows.extend(
            pairwise
        )

    hallucination_vectors = {
        condition: [
            int(
                quality_by_query[
                    query_id
                ][
                    condition
                ][
                    "hallucination_present"
                ]
            )
            for query_id in sorted(
                quality_by_query
            )
        ]
        for condition in QUALITY_CONDITIONS
    }

    q_statistic, q_p = _cochran_q(
        [
            hallucination_vectors[
                condition
            ]
            for condition in QUALITY_CONDITIONS
        ]
    )

    rows.append(
        {
            "task_type": QUALITY_TASK,
            "metric": "hallucination_present",
            "test": "cochran_q",
            "comparison": "all_conditions",
            "n": 30,
            "statistic": q_statistic,
            "p_value": q_p,
            "p_value_holm": "",
            "effect_size_name": "",
            "effect_size": "",
        }
    )

    hallucination_pairwise = []

    for left, right in (
        (
            "no_rag",
            "optimized",
        ),
        (
            "no_rag",
            "semantic",
        ),
        (
            "optimized",
            "semantic",
        ),
    ):
        statistic_m, pair_p = _mcnemar_exact(
            hallucination_vectors[
                left
            ],
            hallucination_vectors[
                right
            ],
        )

        hallucination_pairwise.append(
            {
                "task_type": QUALITY_TASK,
                "metric": "hallucination_present",
                "test": "mcnemar_exact",
                "comparison": (
                    f"{left}_vs_{right}"
                ),
                "n": 30,
                "statistic": statistic_m,
                "p_value": pair_p,
                "p_value_holm": "",
                "effect_size_name": (
                    "risk_difference_left_minus_right"
                ),
                "effect_size": (
                    _mean(
                        hallucination_vectors[
                            left
                        ]
                    )
                    - _mean(
                        hallucination_vectors[
                            right
                        ]
                    )
                ),
            }
        )

    _apply_holm(
        hallucination_pairwise
    )

    rows.extend(
        hallucination_pairwise
    )

    groundedness_by_query = _query_condition_map(
        groundedness
    )

    optimized = [
        int(
            groundedness_by_query[
                query_id
            ][
                "optimized"
            ][
                "groundedness"
            ]
        )
        for query_id in sorted(
            groundedness_by_query
        )
    ]

    semantic = [
        int(
            groundedness_by_query[
                query_id
            ][
                "semantic"
            ][
                "groundedness"
            ]
        )
        for query_id in sorted(
            groundedness_by_query
        )
    ]

    statistic_g, p_g = _wilcoxon_test(
        optimized,
        semantic,
    )

    rows.append(
        {
            "task_type": GROUNDEDNESS_TASK,
            "metric": "groundedness",
            "test": "wilcoxon_signed_rank",
            "comparison": "optimized_vs_semantic",
            "n": 30,
            "statistic": statistic_g,
            "p_value": p_g,
            "p_value_holm": "",
            "effect_size_name": (
                "rank_biserial_optimized_minus_semantic"
            ),
            "effect_size": (
                _paired_rank_biserial(
                    optimized,
                    semantic,
                )
            ),
        }
    )

    return rows


def _friedman_test(
    vectors: Mapping[
        str,
        Sequence[int],
    ],
) -> tuple[float, float]:
    """Run Friedman with a defined all-identical fallback."""

    ordered = [
        list(
            vectors[
                condition
            ]
        )
        for condition in QUALITY_CONDITIONS
    ]

    if all(
        ordered[
            index
        ] == ordered[
            0
        ]
        for index in range(
            1,
            len(
                ordered
            ),
        )
    ):
        return (
            0.0,
            1.0,
        )

    result = friedmanchisquare(
        *ordered
    )

    statistic = float(
        result.statistic
    )

    p_value = float(
        result.pvalue
    )

    if math.isnan(
        statistic
    ) or math.isnan(
        p_value
    ):
        return (
            0.0,
            1.0,
        )

    return (
        statistic,
        p_value,
    )


def _wilcoxon_test(
    left: Sequence[int],
    right: Sequence[int],
) -> tuple[float, float]:
    """Run paired Wilcoxon with an all-zero-difference fallback."""

    differences = [
        a
        - b
        for a, b in zip(
            left,
            right,
            strict=True,
        )
    ]

    if all(
        difference == 0
        for difference in differences
    ):
        return (
            0.0,
            1.0,
        )

    result = wilcoxon(
        left,
        right,
        zero_method="wilcox",
        alternative="two-sided",
        method="auto",
    )

    return (
        float(
            result.statistic
        ),
        float(
            result.pvalue
        ),
    )


def _paired_rank_biserial(
    left: Sequence[int],
    right: Sequence[int],
) -> float:
    """Calculate paired rank-biserial correlation for left minus right."""

    differences = [
        a
        - b
        for a, b in zip(
            left,
            right,
            strict=True,
        )
        if a
        - b
        != 0
    ]

    if not differences:
        return 0.0

    ranks = rankdata(
        [
            abs(
                difference
            )
            for difference in differences
        ],
        method="average",
    )

    positive = sum(
        float(
            rank
        )
        for difference, rank in zip(
            differences,
            ranks,
            strict=True,
        )
        if difference > 0
    )

    negative = sum(
        float(
            rank
        )
        for difference, rank in zip(
            differences,
            ranks,
            strict=True,
        )
        if difference < 0
    )

    total = (
        positive
        + negative
    )

    if total == 0:
        return 0.0

    return (
        positive
        - negative
    ) / total


def _cochran_q(
    vectors: Sequence[
        Sequence[int]
    ],
) -> tuple[float, float]:
    """Calculate Cochran's Q for paired binary outcomes."""

    if not vectors:
        raise ValueError(
            "Cochran Q requires at least one condition."
        )

    condition_count = len(
        vectors
    )

    item_count = len(
        vectors[
            0
        ]
    )

    if condition_count < 2:
        raise ValueError(
            "Cochran Q requires at least two conditions."
        )

    if any(
        len(
            vector
        ) != item_count
        for vector in vectors
    ):
        raise ValueError(
            "Cochran Q vectors must have equal length."
        )

    column_sums = [
        sum(
            vector
        )
        for vector in vectors
    ]

    row_sums = [
        sum(
            vector[
                index
            ]
            for vector in vectors
        )
        for index in range(
            item_count
        )
    ]

    total = sum(
        column_sums
    )

    numerator = (
        (
            condition_count
            - 1
        )
        * (
            condition_count
            * sum(
                value
                * value
                for value in column_sums
            )
            - total
            * total
        )
    )

    denominator = (
        condition_count
        * total
        - sum(
            value
            * value
            for value in row_sums
        )
    )

    if denominator == 0:
        return (
            0.0,
            1.0,
        )

    statistic = (
        numerator
        / denominator
    )

    p_value = float(
        chi2.sf(
            statistic,
            condition_count
            - 1,
        )
    )

    return (
        float(
            statistic
        ),
        p_value,
    )


def _mcnemar_exact(
    left: Sequence[int],
    right: Sequence[int],
) -> tuple[int, float]:
    """Run a two-sided exact McNemar test using the binomial distribution."""

    discordant_left = sum(
        a == 1
        and b == 0
        for a, b in zip(
            left,
            right,
            strict=True,
        )
    )

    discordant_right = sum(
        a == 0
        and b == 1
        for a, b in zip(
            left,
            right,
            strict=True,
        )
    )

    discordant_total = (
        discordant_left
        + discordant_right
    )

    if discordant_total == 0:
        return (
            0,
            1.0,
        )

    smaller = min(
        discordant_left,
        discordant_right,
    )

    tail = sum(
        math.comb(
            discordant_total,
            value,
        )
        for value in range(
            0,
            smaller
            + 1,
        )
    ) / (
        2
        ** discordant_total
    )

    p_value = min(
        1.0,
        2.0
        * tail,
    )

    return (
        smaller,
        p_value,
    )


def _apply_holm(
    rows: list[
        dict[
            str,
            object,
        ]
    ],
) -> None:
    """Apply Holm's correction to one family of pairwise p-values."""

    indexed = sorted(
        enumerate(
            rows
        ),
        key=lambda item: float(
            item[
                1
            ][
                "p_value"
            ]
        ),
    )

    count = len(
        rows
    )

    previous = 0.0

    for rank, (
        original_index,
        row,
    ) in enumerate(
        indexed
    ):
        adjusted = min(
            1.0,
            (
                count
                - rank
            )
            * float(
                row[
                    "p_value"
                ]
            ),
        )

        adjusted = max(
            previous,
            adjusted,
        )

        rows[
            original_index
        ][
            "p_value_holm"
        ] = adjusted

        previous = adjusted


def _build_annotator_distribution(
    judgments: Sequence[Mapping[str, str]],
) -> list[dict[str, object]]:
    """Summarize how each annotator used each rating scale."""

    counts: Counter[
        tuple[
            str,
            str,
            str,
            int,
        ]
    ] = Counter()

    for row in judgments:
        task_type = row[
            "task_type"
        ]

        metrics = (
            QUALITY_METRICS
            if task_type
            == QUALITY_TASK
            else GROUNDEDNESS_METRICS
        )

        for metric in metrics:
            counts[
                (
                    row[
                        "annotator"
                    ],
                    task_type,
                    metric,
                    int(
                        row[
                            metric
                        ]
                    ),
                )
            ] += 1

    return [
        {
            "annotator": annotator,
            "task_type": task_type,
            "metric": metric,
            "score": score,
            "count": count,
        }
        for (
            annotator,
            task_type,
            metric,
            score,
        ), count in sorted(
            counts.items()
        )
    ]


def _build_integrity_audit(
    *,
    completed_hash_rows: Sequence[Mapping[str, str]],
    agreement_rows: Sequence[Mapping[str, str]],
    final_items: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Build a compact integrity audit for the finalized ratings."""

    hashes = [
        row[
            "sha256"
        ]
        for row in completed_hash_rows
    ]

    hash_counts = Counter(
        hashes
    )

    duplicate_hashes = {
        digest: count
        for digest, count
        in hash_counts.items()
        if count > 1
    }

    perfect_agreement = sum(
        float(
            row[
                "exact_agreement_rate"
            ]
        )
        == 1.0
        for row in agreement_rows
    )

    condition_counts = Counter(
        (
            str(
                row[
                    "task_type"
                ]
            ),
            str(
                row[
                    "condition"
                ]
            ),
        )
        for row in final_items
    )

    return {
        "completed_file_count": len(
            completed_hash_rows
        ),
        "unique_completed_sha256": len(
            set(
                hashes
            )
        ),
        "duplicate_completed_sha256_count": len(
            duplicate_hashes
        ),
        "duplicate_completed_sha256": (
            duplicate_hashes
        ),
        "agreement_metric_count": len(
            agreement_rows
        ),
        "perfect_agreement_metric_count": (
            perfect_agreement
        ),
        "final_item_count": len(
            final_items
        ),
        "condition_counts": {
            (
                f"{task_type}:"
                f"{condition}"
            ): count
            for (
                task_type,
                condition,
            ), count in sorted(
                condition_counts.items()
            )
        },
    }


def _require_closed_agreement(
    agreement_rows: Sequence[Mapping[str, str]],
) -> None:
    """Require the reported zero-disagreement state."""

    if len(
        agreement_rows
    ) != 5:
        raise ValueError(
            "Expected five agreement metrics."
        )

    if any(
        float(
            row[
                "exact_agreement_rate"
            ]
        )
        != 1.0
        for row in agreement_rows
    ):
        raise ValueError(
            "Agreement is not perfect. Complete adjudication "
            "before running final unblinded analysis."
        )


def _require_no_pending_adjudication(
    path: Path,
) -> None:
    """Require an empty adjudication manifest for the no-disagreement case."""

    if not path.exists():
        raise FileNotFoundError(
            "Missing adjudication_manifest.csv."
        )

    rows = _load_csv(
        path
    )

    if rows:
        raise ValueError(
            "Adjudication items exist. Complete adjudication "
            "before final unblinding."
        )


def _build_markdown_report(
    *,
    quality_summary: Sequence[Mapping[str, object]],
    groundedness_summary: Sequence[Mapping[str, object]],
    statistical_tests: Sequence[Mapping[str, object]],
    integrity: Mapping[str, object],
) -> str:
    """Build a concise machine-generated final results report."""

    lines = [
        "# FastContext — Resultados finais da avaliação humana",
        "",
        "## Integridade",
        "",
        (
            "- Arquivos preenchidos: "
            f"{integrity['completed_file_count']}."
        ),
        (
            "- SHA-256 únicos entre os arquivos preenchidos: "
            f"{integrity['unique_completed_sha256']}."
        ),
        (
            "- Métricas com concordância exata de 100%: "
            f"{integrity['perfect_agreement_metric_count']}/"
            f"{integrity['agreement_metric_count']}."
        ),
        "",
        "## Qualidade geral",
        "",
        "| Métrica | Condição | Média | Mediana | Taxa no escore máximo |",
        "|---|---|---:|---:|---:|",
    ]

    for row in quality_summary:
        lines.append(
            "| "
            f"{row['metric']} | "
            f"{row['condition']} | "
            f"{float(row['mean']):.4f} | "
            f"{float(row['median']):.2f} | "
            f"{float(row['max_score_rate']):.2%} |"
        )

    lines.extend(
        [
            "",
            "## Groundedness",
            "",
            "| Condição | Média | Mediana | Taxa no escore 2 |",
            "|---|---:|---:|---:|",
        ]
    )

    for row in groundedness_summary:
        lines.append(
            "| "
            f"{row['condition']} | "
            f"{float(row['mean']):.4f} | "
            f"{float(row['median']):.2f} | "
            f"{float(row['max_score_rate']):.2%} |"
        )

    lines.extend(
        [
            "",
            "## Testes pareados",
            "",
            "| Métrica | Teste | Comparação | p | p Holm | Efeito |",
            "|---|---|---|---:|---:|---:|",
        ]
    )

    for row in statistical_tests:
        holm = row[
            "p_value_holm"
        ]

        holm_text = (
            ""
            if holm == ""
            else f"{float(holm):.6g}"
        )

        effect = row[
            "effect_size"
        ]

        effect_text = (
            ""
            if effect == ""
            else f"{float(effect):.4f}"
        )

        lines.append(
            "| "
            f"{row['metric']} | "
            f"{row['test']} | "
            f"{row['comparison']} | "
            f"{float(row['p_value']):.6g} | "
            f"{holm_text} | "
            f"{effect_text} |"
        )

    lines.extend(
        [
            "",
            (
                "Os testes são pareados por `query_id`. "
                "As comparações pós-hoc usam correção de Holm."
            ),
            "",
        ]
    )

    return "\n".join(
        lines
    )


def _print_summary(
    quality_summary: Sequence[Mapping[str, object]],
    groundedness_summary: Sequence[Mapping[str, object]],
) -> None:
    """Print concise final descriptive results."""

    print(
        "Quality means:"
    )

    for metric in QUALITY_METRICS:
        values = [
            row
            for row in quality_summary
            if row[
                "metric"
            ] == metric
        ]

        text = ", ".join(
            (
                f"{row['condition']}="
                f"{float(row['mean']):.4f}"
            )
            for row in values
        )

        print(
            f"  {metric}: {text}"
        )

    print(
        "Groundedness means:"
    )

    for row in groundedness_summary:
        print(
            "  "
            f"{row['condition']}="
            f"{float(row['mean']):.4f}"
        )


def _median(
    values: Sequence[int],
) -> float:
    """Calculate a numeric median."""

    ordered = sorted(
        values
    )

    midpoint = len(
        ordered
    ) // 2

    if len(
        ordered
    ) % 2:
        return float(
            ordered[
                midpoint
            ]
        )

    return (
        ordered[
            midpoint
            - 1
        ]
        + ordered[
            midpoint
        ]
    ) / 2.0


def _mean(
    values: Sequence[int],
) -> float:
    """Calculate an arithmetic mean."""

    return (
        sum(
            values
        )
        / len(
            values
        )
    )


def _prepare_final_directory(
    path: Path,
    *,
    force: bool,
) -> None:
    """Prepare only the derived final-analysis directory."""

    if path.exists():
        if not force:
            raise FileExistsError(
                "Final human-evaluation output already exists. "
                "Use --force only to rebuild this derived directory."
            )

        shutil.rmtree(
            path
        )

    path.mkdir(
        parents=True,
        exist_ok=True,
    )


def _load_config() -> dict[str, Any]:
    """Load human-evaluation configuration."""

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


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
