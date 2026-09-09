"""Create balanced blinded annotation files for the FastContext team."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.utils.config import PROJECT_ROOT

ANNOTATION_TEMPLATE_PATH = (
    PROJECT_ROOT
    / "data"
    / "ground_truth"
    / "annotation_template.csv"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "ground_truth"
    / "annotations"
)

ASSIGNMENT_SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "assignment_summary.json"
)

ANNOTATORS = (
    "carlos",
    "yann",
    "wilson",
    "matheus",
    "gabriela",
)

PAIR_ASSIGNMENTS: dict[
    tuple[str, str],
    tuple[str, ...],
] = {
    (
        "carlos",
        "yann",
    ): (
        "q01",
        "q02",
        "q03",
    ),
    (
        "carlos",
        "wilson",
    ): (
        "q04",
        "q05",
        "q06",
    ),
    (
        "carlos",
        "matheus",
    ): (
        "q07",
        "q08",
        "q09",
    ),
    (
        "carlos",
        "gabriela",
    ): (
        "q10",
        "q11",
        "q12",
    ),
    (
        "yann",
        "wilson",
    ): (
        "q13",
        "q14",
        "q15",
    ),
    (
        "yann",
        "matheus",
    ): (
        "q16",
        "q17",
        "q18",
    ),
    (
        "yann",
        "gabriela",
    ): (
        "q19",
        "q20",
        "q21",
    ),
    (
        "wilson",
        "matheus",
    ): (
        "q22",
        "q23",
        "q24",
    ),
    (
        "wilson",
        "gabriela",
    ): (
        "q25",
        "q26",
        "q27",
    ),
    (
        "matheus",
        "gabriela",
    ): (
        "q28",
        "q29",
        "q30",
    ),
}


def main() -> int:
    """Create one blinded annotation CSV per team member."""
    rows = _load_template(
        ANNOTATION_TEMPLATE_PATH
    )

    query_assignments = (
        _build_query_assignments()
    )

    _validate_assignments(
        rows,
        query_assignments,
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows_by_annotator: dict[
        str,
        list[dict[str, str]],
    ] = {
        annotator: []
        for annotator in ANNOTATORS
    }

    for row in rows:
        query_id = row[
            "query_id"
        ]

        annotators = (
            query_assignments[
                query_id
            ]
        )

        for annotator in annotators:
            rows_by_annotator[
                annotator
            ].append(
                dict(row)
            )

    for annotator in ANNOTATORS:
        output_path = (
            OUTPUT_DIRECTORY
            / f"annotation_{annotator}.csv"
        )

        _write_annotation_file(
            output_path,
            rows_by_annotator[
                annotator
            ],
        )

    summary = (
        _build_summary(
            rows_by_annotator,
            query_assignments,
        )
    )

    ASSIGNMENT_SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    _print_summary(
        summary
    )

    return 0


def _load_template(
    path: Path,
) -> list[dict[str, str]]:
    """Load the blinded annotation template."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Annotation template not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(
            file
        )

        if reader.fieldnames is None:
            raise ValueError(
                "Annotation template has no header."
            )

        required_fields = {
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

        missing_fields = (
            required_fields
            - set(
                reader.fieldnames
            )
        )

        if missing_fields:
            raise ValueError(
                "Annotation template is missing fields: "
                + ", ".join(
                    sorted(
                        missing_fields
                    )
                )
            )

        rows = [
            dict(row)
            for row in reader
        ]

    if not rows:
        raise ValueError(
            "Annotation template is empty."
        )

    return rows


def _build_query_assignments(
    *,
    pair_assignments: dict[
        tuple[str, str],
        tuple[str, ...],
    ] = PAIR_ASSIGNMENTS,
) -> dict[
    str,
    tuple[str, str],
]:
    """Convert pair assignments into a query-to-annotator mapping."""
    query_assignments: dict[
        str,
        tuple[str, str],
    ] = {}

    for pair, query_ids in (
        pair_assignments.items()
    ):
        first_annotator, second_annotator = (
            pair
        )

        if (
            first_annotator
            == second_annotator
        ):
            raise ValueError(
                "A query must have two "
                "different annotators."
            )

        for query_id in query_ids:
            if (
                query_id
                in query_assignments
            ):
                raise ValueError(
                    "Duplicate query assignment: "
                    f"{query_id}"
                )

            query_assignments[
                query_id
            ] = (
                first_annotator,
                second_annotator,
            )

    return query_assignments


def _validate_assignments(
    rows: list[dict[str, str]],
    query_assignments: dict[
        str,
        tuple[str, str],
    ],
) -> None:
    """Validate coverage and annotation multiplicity."""
    template_query_ids = {
        row["query_id"]
        for row in rows
    }

    expected_query_ids = {
        f"q{index:02d}"
        for index in range(
            1,
            31,
        )
    }

    if (
        template_query_ids
        != expected_query_ids
    ):
        missing = (
            expected_query_ids
            - template_query_ids
        )

        extra = (
            template_query_ids
            - expected_query_ids
        )

        raise ValueError(
            "Unexpected query IDs in annotation template. "
            f"Missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )

    if (
        set(query_assignments)
        != expected_query_ids
    ):
        raise ValueError(
            "Assignment table must cover "
            "exactly q01 through q30."
        )

    annotation_counts = Counter()

    for annotators in (
        query_assignments.values()
    ):
        for annotator in annotators:
            if annotator not in ANNOTATORS:
                raise ValueError(
                    "Unknown annotator: "
                    f"{annotator}"
                )

        annotation_counts.update(
            annotators
        )

    if any(
        count != 12
        for count in annotation_counts.values()
    ):
        raise ValueError(
            "Every annotator must receive "
            "exactly 12 queries."
        )


def _write_annotation_file(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    """Write one annotator-specific blinded CSV."""
    if not rows:
        raise ValueError(
            "Cannot create an empty annotation file."
        )

    fieldnames = list(
        rows[0]
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
        writer.writerows(
            rows
        )


def _build_summary(
    rows_by_annotator: dict[
        str,
        list[dict[str, str]],
    ],
    query_assignments: dict[
        str,
        tuple[str, str],
    ],
) -> dict[str, object]:
    """Build assignment statistics."""
    assignments_by_annotator: dict[
        str,
        list[str],
    ] = defaultdict(
        list
    )

    for query_id, annotators in (
        query_assignments.items()
    ):
        for annotator in annotators:
            assignments_by_annotator[
                annotator
            ].append(
                query_id
            )

    annotator_summary: dict[
        str,
        dict[str, object],
    ] = {}

    for annotator in ANNOTATORS:
        query_ids = sorted(
            assignments_by_annotator[
                annotator
            ]
        )

        rows = rows_by_annotator[
            annotator
        ]

        annotator_summary[
            annotator
        ] = {
            "query_count": len(
                query_ids
            ),
            "judgment_count": len(
                rows
            ),
            "query_ids": query_ids,
            "file": (
                f"annotation_{annotator}.csv"
            ),
        }

    return {
        "schema_version": 1,
        "annotation_policy": {
            "relevance": {
                "0": "not_relevant",
                "1": "relevant",
            },
            "annotators_per_query": 2,
            "adjudication": (
                "third_annotator_on_disagreement"
            ),
        },
        "annotators": (
            annotator_summary
        ),
        "query_assignments": {
            query_id: list(
                annotators
            )
            for query_id, annotators
            in sorted(
                query_assignments.items()
            )
        },
    }


def _print_summary(
    summary: dict[str, object],
) -> None:
    """Print assignment statistics."""
    raw_annotators = summary[
        "annotators"
    ]

    if not isinstance(
        raw_annotators,
        dict,
    ):
        raise TypeError(
            "Invalid assignment summary."
        )

    print(
        "FastContext Annotation Assignments"
    )
    print("=" * 60)

    for annotator in ANNOTATORS:
        raw_data = raw_annotators[
            annotator
        ]

        if not isinstance(
            raw_data,
            dict,
        ):
            raise TypeError(
                "Invalid annotator summary."
            )

        print(
            f"{annotator.capitalize():10s} "
            f"queries={raw_data['query_count']:2d} "
            f"judgments={raw_data['judgment_count']:3d}"
        )

    print()
    print(
        f"Output directory: {OUTPUT_DIRECTORY}"
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )