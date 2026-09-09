"""Prepare and finalize FastContext human relevance adjudication."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.evaluation.agreement import (
    AgreementMetrics,
    calculate_agreement,
)
from src.utils.config import PROJECT_ROOT

GROUND_TRUTH_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "ground_truth"
)

ANNOTATION_DIRECTORY = (
    GROUND_TRUTH_DIRECTORY
    / "annotations"
)

ASSIGNMENT_SUMMARY_PATH = (
    ANNOTATION_DIRECTORY
    / "assignment_summary.json"
)

GROUND_TRUTH_PATH = (
    GROUND_TRUTH_DIRECTORY
    / "ground_truth.json"
)

ADJUDICATION_PATH = (
    GROUND_TRUTH_DIRECTORY
    / "adjudication.csv"
)

AGREEMENT_REPORT_PATH = (
    GROUND_TRUTH_DIRECTORY
    / "agreement_report.json"
)

FINAL_GROUND_TRUTH_PATH = (
    GROUND_TRUTH_DIRECTORY
    / "ground_truth.final.json"
)

ANNOTATORS = {
    "carlos",
    "yann",
    "wilson",
    "matheus",
    "gabriela",
}

ANNOTATION_FIELDS = {
    "query_id",
    "category",
    "question",
    "candidate_order",
    "chunk_id",
    "source_path",
    "section_title",
    "content",
    "relevance",
    "annotation_notes",
}

ADJUDICATION_FIELDS = (
    "query_id",
    "category",
    "question",
    "candidate_order",
    "chunk_id",
    "source_path",
    "section_title",
    "content",
    "adjudicator",
    "adjudicated_relevance",
    "adjudication_notes",
)

PROHIBITED_ADJUDICATION_FIELDS = {
    "annotator_a",
    "label_a",
    "annotator_b",
    "label_b",
}


@dataclass(frozen=True, slots=True)
class ComparedAnnotation:
    """Represent one paired binary relevance judgment."""

    query_id: str
    category: str
    question: str
    candidate_order: int
    chunk_id: str
    source_path: str
    section_title: str
    content: str
    annotator_a: str
    label_a: int
    annotator_b: str
    label_b: int

    @property
    def agreed(self) -> bool:
        """Return whether both annotators assigned the same label."""
        return (
            self.label_a
            == self.label_b
        )

    @property
    def key(self) -> tuple[str, str]:
        """Return the unique query/chunk judgment key."""
        return (
            self.query_id,
            self.chunk_id,
        )


def build_parser() -> argparse.ArgumentParser:
    """Create adjudication CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Prepare and finalize FastContext "
            "human relevance adjudication."
        )
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    prepare_parser = subparsers.add_parser(
        "prepare",
        help=(
            "Compare the two independent "
            "annotations and create adjudication.csv."
        ),
    )

    prepare_parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Overwrite an existing adjudication file."
        ),
    )

    subparsers.add_parser(
        "finalize",
        help=(
            "Use completed adjudications to create "
            "ground_truth.final.json."
        ),
    )

    return parser


def main(
    argv: Sequence[str] | None = None,
) -> int:
    """Run adjudication preparation or finalization."""
    args = build_parser().parse_args(
        argv
    )

    assignments = (
        _load_query_assignments(
            ASSIGNMENT_SUMMARY_PATH
        )
    )

    annotation_rows = (
        _load_all_annotations(
            assignments
        )
    )

    comparisons = (
        _compare_annotations(
            assignments,
            annotation_rows,
        )
    )

    report = (
        _build_agreement_report(
            comparisons
        )
    )

    _write_json(
        AGREEMENT_REPORT_PATH,
        report,
    )

    if args.command == "prepare":
        _prepare_adjudication(
            comparisons,
            force=args.force,
        )

        _print_prepare_summary(
            comparisons,
            report,
        )

        return 0

    if args.command == "finalize":
        _finalize_ground_truth(
            comparisons,
            report,
        )

        return 0

    raise ValueError(
        "Unsupported adjudication command."
    )


def _load_query_assignments(
    path: Path,
) -> dict[
    str,
    tuple[str, str],
]:
    """Load the query-to-annotator assignment table."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Assignment summary not found: {path}"
        )

    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        data,
        dict,
    ):
        raise TypeError(
            "Assignment summary root "
            "must be an object."
        )

    raw_assignments = data.get(
        "query_assignments"
    )

    if not isinstance(
        raw_assignments,
        dict,
    ):
        raise TypeError(
            "query_assignments "
            "must be an object."
        )

    assignments: dict[
        str,
        tuple[str, str],
    ] = {}

    for query_id, annotators in (
        raw_assignments.items()
    ):
        if not isinstance(
            query_id,
            str,
        ):
            raise TypeError(
                "query_id must be a string."
            )

        if not isinstance(
            annotators,
            list,
        ):
            raise TypeError(
                "Query annotators must be a list."
            )

        if len(annotators) != 2:
            raise ValueError(
                "Every query must have exactly "
                "two primary annotators."
            )

        if not all(
            isinstance(
                annotator,
                str,
            )
            for annotator in annotators
        ):
            raise TypeError(
                "Annotator names must be strings."
            )

        first = annotators[
            0
        ]

        second = annotators[
            1
        ]

        if (
            first not in ANNOTATORS
            or second not in ANNOTATORS
        ):
            raise ValueError(
                "Unknown annotator in assignment."
            )

        if first == second:
            raise ValueError(
                "Primary annotators must be different."
            )

        assignments[
            query_id
        ] = (
            first,
            second,
        )

    expected_query_ids = {
        f"q{index:02d}"
        for index in range(
            1,
            31,
        )
    }

    if (
        set(assignments)
        != expected_query_ids
    ):
        raise ValueError(
            "Assignment summary must contain "
            "exactly q01 through q30."
        )

    return assignments


def _load_all_annotations(
    assignments: Mapping[
        str,
        tuple[str, str],
    ],
) -> dict[
    str,
    dict[
        tuple[str, str],
        dict[str, str],
    ],
]:
    """Load and validate all annotator CSV files."""
    queries_by_annotator: dict[
        str,
        set[str],
    ] = defaultdict(
        set
    )

    for query_id, annotators in (
        assignments.items()
    ):
        for annotator in annotators:
            queries_by_annotator[
                annotator
            ].add(
                query_id
            )

    output: dict[
        str,
        dict[
            tuple[str, str],
            dict[str, str],
        ],
    ] = {}

    for annotator in sorted(
        ANNOTATORS
    ):
        path = (
            ANNOTATION_DIRECTORY
            / f"annotation_{annotator}.csv"
        )

        output[
            annotator
        ] = _load_annotation_file(
            path,
            expected_queries=(
                queries_by_annotator[
                    annotator
                ]
            ),
        )

    return output


def _load_annotation_file(
    path: Path,
    *,
    expected_queries: set[str],
) -> dict[
    tuple[str, str],
    dict[str, str],
]:
    """Load one completed blinded annotation file."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Annotation file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(
            file
        )

        fieldnames = (
            set(
                reader.fieldnames
                or []
            )
        )

        missing = (
            ANNOTATION_FIELDS
            - fieldnames
        )

        if missing:
            raise ValueError(
                "Annotation file is missing fields: "
                + ", ".join(
                    sorted(
                        missing
                    )
                )
            )

        rows: dict[
            tuple[str, str],
            dict[str, str],
        ] = {}

        found_queries: set[
            str
        ] = set()

        for row in reader:
            query_id = (
                row[
                    "query_id"
                ].strip()
            )

            chunk_id = (
                row[
                    "chunk_id"
                ].strip()
            )

            if (
                query_id
                not in expected_queries
            ):
                raise ValueError(
                    f"Unexpected query {query_id} "
                    f"in {path.name}."
                )

            if not chunk_id:
                raise ValueError(
                    "chunk_id cannot be empty."
                )

            _parse_binary_label(
                row[
                    "relevance"
                ]
            )

            key = (
                query_id,
                chunk_id,
            )

            if key in rows:
                raise ValueError(
                    "Duplicate annotation row: "
                    f"{query_id}/{chunk_id}"
                )

            rows[
                key
            ] = dict(
                row
            )

            found_queries.add(
                query_id
            )

        if (
            found_queries
            != expected_queries
        ):
            missing_queries = (
                expected_queries
                - found_queries
            )

            raise ValueError(
                "Annotation file is missing "
                "assigned queries: "
                + ", ".join(
                    sorted(
                        missing_queries
                    )
                )
            )

    return rows


def _compare_annotations(
    assignments: Mapping[
        str,
        tuple[str, str],
    ],
    annotations: Mapping[
        str,
        Mapping[
            tuple[str, str],
            dict[str, str],
        ],
    ],
) -> tuple[
    ComparedAnnotation,
    ...,
]:
    """Pair the two independent judgments for each query."""
    comparisons: list[
        ComparedAnnotation
    ] = []

    for query_id in sorted(
        assignments
    ):
        (
            annotator_a,
            annotator_b,
        ) = assignments[
            query_id
        ]

        rows_a = {
            key: row
            for key, row
            in annotations[
                annotator_a
            ].items()
            if key[
                0
            ]
            == query_id
        }

        rows_b = {
            key: row
            for key, row
            in annotations[
                annotator_b
            ].items()
            if key[
                0
            ]
            == query_id
        }

        if (
            set(rows_a)
            != set(rows_b)
        ):
            raise ValueError(
                "Primary annotators received "
                "different candidate sets for "
                f"{query_id}."
            )

        ordered_keys = sorted(
            rows_a,
            key=lambda key: int(
                rows_a[
                    key
                ][
                    "candidate_order"
                ]
            ),
        )

        for key in ordered_keys:
            row_a = rows_a[
                key
            ]

            row_b = rows_b[
                key
            ]

            _validate_candidate_metadata(
                row_a,
                row_b,
            )

            comparisons.append(
                ComparedAnnotation(
                    query_id=query_id,
                    category=(
                        row_a[
                            "category"
                        ]
                    ),
                    question=(
                        row_a[
                            "question"
                        ]
                    ),
                    candidate_order=int(
                        row_a[
                            "candidate_order"
                        ]
                    ),
                    chunk_id=key[
                        1
                    ],
                    source_path=(
                        row_a[
                            "source_path"
                        ]
                    ),
                    section_title=(
                        row_a[
                            "section_title"
                        ]
                    ),
                    content=(
                        row_a[
                            "content"
                        ]
                    ),
                    annotator_a=(
                        annotator_a
                    ),
                    label_a=(
                        _parse_binary_label(
                            row_a[
                                "relevance"
                            ]
                        )
                    ),
                    annotator_b=(
                        annotator_b
                    ),
                    label_b=(
                        _parse_binary_label(
                            row_b[
                                "relevance"
                            ]
                        )
                    ),
                )
            )

    return tuple(
        comparisons
    )


def _validate_candidate_metadata(
    row_a: Mapping[
        str,
        str,
    ],
    row_b: Mapping[
        str,
        str,
    ],
) -> None:
    """Ensure both annotators judged the same candidate."""
    fields = (
        "query_id",
        "category",
        "question",
        "candidate_order",
        "chunk_id",
        "source_path",
        "section_title",
        "content",
    )

    for field in fields:
        if (
            row_a[
                field
            ]
            != row_b[
                field
            ]
        ):
            raise ValueError(
                "Annotation candidate metadata "
                f"differs in field '{field}'."
            )


def _build_agreement_report(
    comparisons: Sequence[
        ComparedAnnotation
    ],
) -> dict[
    str,
    Any,
]:
    """Calculate pooled and annotator-pair agreement statistics."""
    overall = (
        _agreement_for(
            comparisons
        )
    )

    by_pair: dict[
        tuple[str, str],
        list[
            ComparedAnnotation
        ],
    ] = defaultdict(
        list
    )

    by_query: dict[
        str,
        list[
            ComparedAnnotation
        ],
    ] = defaultdict(
        list
    )

    for comparison in comparisons:
        pair = tuple(
            sorted(
                (
                    comparison.annotator_a,
                    comparison.annotator_b,
                )
            )
        )

        by_pair[
            pair
        ].append(
            comparison
        )

        by_query[
            comparison.query_id
        ].append(
            comparison
        )

    pair_report: dict[
        str,
        dict[
            str,
            int | float | None,
        ],
    ] = {}

    for pair, items in sorted(
        by_pair.items()
    ):
        metrics = (
            _agreement_for(
                items
            )
        )

        pair_report[
            " + ".join(
                pair
            )
        ] = _metrics_to_dict(
            metrics
        )

    query_report: dict[
        str,
        dict[
            str,
            int | float | None,
        ],
    ] = {}

    for query_id, items in sorted(
        by_query.items()
    ):
        metrics = (
            _agreement_for(
                items
            )
        )

        query_report[
            query_id
        ] = _metrics_to_dict(
            metrics
        )

    return {
        "schema_version": 1,
        "overall": (
            _metrics_to_dict(
                overall
            )
        ),
        "by_pair": (
            pair_report
        ),
        "by_query": (
            query_report
        ),
    }


def _agreement_for(
    comparisons: Sequence[
        ComparedAnnotation
    ],
) -> AgreementMetrics:
    """Calculate agreement for a comparison collection."""
    return calculate_agreement(
        [
            item.label_a
            for item
            in comparisons
        ],
        [
            item.label_b
            for item
            in comparisons
        ],
    )


def _metrics_to_dict(
    metrics: AgreementMetrics,
) -> dict[
    str,
    int | float | None,
]:
    """Serialize agreement metrics."""
    return {
        "count": metrics.count,
        "agreements": (
            metrics.agreements
        ),
        "disagreements": (
            metrics.disagreements
        ),
        "agreement_rate": (
            metrics.agreement_rate
        ),
        "cohen_kappa": (
            metrics.cohen_kappa
        ),
    }


def _prepare_adjudication(
    comparisons: Sequence[
        ComparedAnnotation
    ],
    *,
    force: bool,
) -> None:
    """Write blinded disagreement cases for third-person adjudication."""
    if (
        ADJUDICATION_PATH.exists()
        and not force
    ):
        raise FileExistsError(
            "adjudication.csv already exists. "
            "Use --force only if it is safe "
            "to overwrite it."
        )

    disagreements = [
        comparison
        for comparison
        in comparisons
        if not comparison.agreed
    ]

    with ADJUDICATION_PATH.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                ADJUDICATION_FIELDS
            ),
        )

        writer.writeheader()

        for item in disagreements:
            writer.writerow(
                {
                    "query_id": (
                        item.query_id
                    ),
                    "category": (
                        item.category
                    ),
                    "question": (
                        item.question
                    ),
                    "candidate_order": (
                        item.candidate_order
                    ),
                    "chunk_id": (
                        item.chunk_id
                    ),
                    "source_path": (
                        item.source_path
                    ),
                    "section_title": (
                        item.section_title
                    ),
                    "content": (
                        item.content
                    ),
                    "adjudicator": "",
                    "adjudicated_relevance": "",
                    "adjudication_notes": "",
                }
            )


def _finalize_ground_truth(
    comparisons: Sequence[
        ComparedAnnotation
    ],
    report: Mapping[
        str,
        Any,
    ],
) -> None:
    """Create final ground truth after all disagreements are adjudicated."""
    resolved: dict[
        tuple[str, str],
        int,
    ] = {}

    disagreement_map = {
        item.key: item
        for item
        in comparisons
        if not item.agreed
    }

    for item in comparisons:
        if item.agreed:
            resolved[
                item.key
            ] = item.label_a

    adjudications = (
        _load_completed_adjudications(
            disagreement_map
        )
    )

    resolved.update(
        adjudications
    )

    if (
        len(resolved)
        != len(comparisons)
    ):
        raise ValueError(
            "Not all pooled candidate judgments "
            "have a final label."
        )

    ground_truth = json.loads(
        GROUND_TRUTH_PATH.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        ground_truth,
        dict,
    ):
        raise TypeError(
            "Ground-truth root must be an object."
        )

    raw_queries = ground_truth.get(
        "queries"
    )

    metadata = ground_truth.get(
        "metadata"
    )

    if not isinstance(
        raw_queries,
        list,
    ):
        raise TypeError(
            "Ground-truth queries must be a list."
        )

    if not isinstance(
        metadata,
        dict,
    ):
        raise TypeError(
            "Ground-truth metadata must be an object."
        )

    disagreements_by_query: dict[
        str,
        int,
    ] = defaultdict(
        int
    )

    for item in disagreement_map.values():
        disagreements_by_query[
            item.query_id
        ] += 1

    total_relevant = 0

    for query in raw_queries:
        if not isinstance(
            query,
            dict,
        ):
            raise TypeError(
                "Every ground-truth query "
                "must be an object."
            )

        query_id = query.get(
            "query_id"
        )

        if not isinstance(
            query_id,
            str,
        ):
            raise TypeError(
                "query_id must be a string."
            )

        relevant_chunks = sorted(
            chunk_id
            for (
                resolved_query_id,
                chunk_id,
            ), label in resolved.items()
            if (
                resolved_query_id
                == query_id
                and label == 1
            )
        )

        if not relevant_chunks:
            raise ValueError(
                "No relevant chunk was identified "
                f"for {query_id}. Human review "
                "is required before finalization."
            )

        query[
            "relevant_chunks"
        ] = relevant_chunks

        query[
            "annotation_notes"
        ] = (
            "Binary pooled relevance assessment; "
            "two independent primary annotators; "
            f"{disagreements_by_query[query_id]} "
            "disagreement(s) adjudicated; "
            f"{len(relevant_chunks)} relevant "
            "chunk(s)."
        )

        total_relevant += len(
            relevant_chunks
        )

    overall = report.get(
        "overall"
    )

    metadata[
        "annotation_status"
    ] = "complete"

    metadata[
        "annotation_protocol"
    ] = {
        "relevance_type": "binary",
        "primary_annotators_per_query": 2,
        "adjudication": (
            "third_annotator_on_disagreement"
        ),
        "pool_depth": 20,
        "total_candidate_judgments": (
            len(comparisons)
        ),
        "total_relevant_chunks": (
            total_relevant
        ),
        "overall_agreement": (
            overall
            if isinstance(
                overall,
                Mapping,
            )
            else {}
        ),
    }

    _write_json(
        FINAL_GROUND_TRUTH_PATH,
        ground_truth,
    )

    print(
        "FastContext Ground Truth Finalized"
    )

    print(
        "=" * 60
    )

    print(
        f"Candidate judgments: {len(comparisons)}"
    )

    print(
        "Adjudicated disagreements: "
        f"{len(disagreement_map)}"
    )

    print(
        f"Relevant chunk assignments: {total_relevant}"
    )

    print(
        "Output: "
        f"{FINAL_GROUND_TRUTH_PATH}"
    )


def _load_completed_adjudications(
    disagreements: Mapping[
        tuple[str, str],
        ComparedAnnotation,
    ],
) -> dict[
    tuple[str, str],
    int,
]:
    """Load and validate blinded third-annotator decisions."""
    if not disagreements:
        return {}

    if not ADJUDICATION_PATH.is_file():
        raise FileNotFoundError(
            "adjudication.csv does not exist."
        )

    resolved: dict[
        tuple[str, str],
        int,
    ] = {}

    with ADJUDICATION_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(
            file
        )

        fieldnames = set(
            reader.fieldnames
            or []
        )

        missing = (
            set(
                ADJUDICATION_FIELDS
            )
            - fieldnames
        )

        if missing:
            raise ValueError(
                "Adjudication file is missing fields: "
                + ", ".join(
                    sorted(
                        missing
                    )
                )
            )

        leaked_fields = (
            PROHIBITED_ADJUDICATION_FIELDS
            & fieldnames
        )

        if leaked_fields:
            raise ValueError(
                "Adjudication file exposes primary "
                "annotation information: "
                + ", ".join(
                    sorted(
                        leaked_fields
                    )
                )
            )

        for row in reader:
            key = (
                row[
                    "query_id"
                ].strip(),
                row[
                    "chunk_id"
                ].strip(),
            )

            if key not in disagreements:
                raise ValueError(
                    "Unexpected adjudication case: "
                    f"{key[0]}/{key[1]}"
                )

            if key in resolved:
                raise ValueError(
                    "Duplicate adjudication case: "
                    f"{key[0]}/{key[1]}"
                )

            item = disagreements[
                key
            ]

            _validate_adjudication_metadata(
                row,
                item,
            )

            adjudicator = (
                row[
                    "adjudicator"
                ].strip()
                .lower()
            )

            if not adjudicator:
                raise ValueError(
                    "Every disagreement must "
                    "have an adjudicator."
                )

            if adjudicator not in ANNOTATORS:
                raise ValueError(
                    "Unknown adjudicator: "
                    f"{adjudicator}"
                )

            if adjudicator in {
                item.annotator_a,
                item.annotator_b,
            }:
                raise ValueError(
                    "The adjudicator must be "
                    "a third team member."
                )

            resolved[
                key
            ] = _parse_binary_label(
                row[
                    "adjudicated_relevance"
                ]
            )

    if (
        set(resolved)
        != set(disagreements)
    ):
        missing = (
            set(disagreements)
            - set(resolved)
        )

        raise ValueError(
            "Missing adjudications: "
            + ", ".join(
                f"{query_id}/{chunk_id}"
                for query_id, chunk_id
                in sorted(
                    missing
                )
            )
        )

    return resolved


def _validate_adjudication_metadata(
    row: Mapping[
        str,
        str,
    ],
    item: ComparedAnnotation,
) -> None:
    """Ensure a blinded adjudication row still identifies the same case."""
    expected = {
        "query_id": item.query_id,
        "category": item.category,
        "question": item.question,
        "candidate_order": str(
            item.candidate_order
        ),
        "chunk_id": item.chunk_id,
        "source_path": item.source_path,
        "section_title": item.section_title,
        "content": item.content,
    }

    for field, expected_value in (
        expected.items()
    ):
        if (
            row[
                field
            ]
            != expected_value
        ):
            raise ValueError(
                "Adjudication candidate metadata "
                f"differs in field '{field}'."
            )


def _parse_binary_label(
    value: str,
) -> int:
    """Parse a required 0/1 human relevance label."""
    normalized = value.strip()

    if normalized not in {
        "0",
        "1",
    }:
        raise ValueError(
            "Every relevance label must "
            "be exactly 0 or 1."
        )

    return int(
        normalized
    )


def _write_json(
    path: Path,
    data: Mapping[
        str,
        Any,
    ],
) -> None:
    """Write deterministic UTF-8 JSON."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _format_kappa(
    value: Any,
) -> str:
    """Format Cohen's kappa, including mathematically undefined cases."""
    if value is None:
        return (
            "undefined "
            "(degenerate marginals)"
        )

    if isinstance(
        value,
        bool,
    ) or not isinstance(
        value,
        (int, float),
    ):
        raise TypeError(
            "Invalid Cohen's kappa value."
        )

    return f"{float(value):.4f}"


def _print_prepare_summary(
    comparisons: Sequence[
        ComparedAnnotation
    ],
    report: Mapping[
        str,
        Any,
    ],
) -> None:
    """Print agreement and adjudication preparation summary."""
    overall = report[
        "overall"
    ]

    if not isinstance(
        overall,
        Mapping,
    ):
        raise TypeError(
            "Invalid agreement report."
        )

    print(
        "FastContext Annotation Agreement"
    )

    print(
        "=" * 60
    )

    print(
        f"Paired judgments: {len(comparisons)}"
    )

    print(
        "Agreements: "
        f"{overall['agreements']}"
    )

    print(
        "Disagreements: "
        f"{overall['disagreements']}"
    )

    print(
        "Agreement rate: "
        f"{float(overall['agreement_rate']) * 100:.2f}%"
    )

    print(
        "Pooled Cohen's kappa "
        "(descriptive): "
        f"{_format_kappa(overall.get('cohen_kappa'))}"
    )

    print()

    print(
        "Agreement report: "
        f"{AGREEMENT_REPORT_PATH}"
    )

    print(
        "Blinded adjudication file: "
        f"{ADJUDICATION_PATH}"
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
