"""Create blinded third-annotator assignments for retrieval disagreements."""

from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "ground_truth"
    / "adjudication.csv"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "ground_truth"
    / "adjudication_assignments"
)

SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "assignment_summary.json"
)

RANDOM_SEED = 42

MEMBERS = (
    "carlos",
    "gabriela",
    "matheus",
    "wilson",
    "yann",
)

FIELDNAMES = (
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

PROHIBITED_FIELDS = {
    "annotator_a",
    "annotator_b",
    "label_a",
    "label_b",
}

PRIMARY_PAIRS = {
    "q01": ("carlos", "yann"),
    "q02": ("carlos", "yann"),
    "q03": ("carlos", "yann"),

    "q04": ("carlos", "wilson"),
    "q05": ("carlos", "wilson"),
    "q06": ("carlos", "wilson"),

    "q07": ("carlos", "matheus"),
    "q08": ("carlos", "matheus"),
    "q09": ("carlos", "matheus"),

    "q10": ("carlos", "gabriela"),
    "q11": ("carlos", "gabriela"),
    "q12": ("carlos", "gabriela"),

    "q13": ("yann", "wilson"),
    "q14": ("yann", "wilson"),
    "q15": ("yann", "wilson"),

    "q16": ("yann", "matheus"),
    "q17": ("yann", "matheus"),
    "q18": ("yann", "matheus"),

    "q19": ("yann", "gabriela"),
    "q20": ("yann", "gabriela"),
    "q21": ("yann", "gabriela"),

    "q22": ("wilson", "matheus"),
    "q23": ("wilson", "matheus"),
    "q24": ("wilson", "matheus"),

    "q25": ("wilson", "gabriela"),
    "q26": ("wilson", "gabriela"),
    "q27": ("wilson", "gabriela"),

    "q28": ("matheus", "gabriela"),
    "q29": ("matheus", "gabriela"),
    "q30": ("matheus", "gabriela"),
}


def load_rows() -> list[dict[str, str]]:
    """Load and validate the blinded adjudication file."""
    with INPUT_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        fieldnames = tuple(
            reader.fieldnames or ()
        )

        if fieldnames != FIELDNAMES:
            raise ValueError(
                "Unexpected adjudication schema. "
                f"Expected {FIELDNAMES}, "
                f"found {fieldnames}."
            )

        leaked_fields = (
            set(fieldnames)
            & PROHIBITED_FIELDS
        )

        if leaked_fields:
            raise ValueError(
                "Blinded adjudication file "
                "contains prohibited fields: "
                f"{sorted(leaked_fields)}"
            )

        rows = list(reader)

    if not rows:
        raise ValueError(
            "The adjudication file is empty."
        )

    for row_number, row in enumerate(
        rows,
        start=2,
    ):
        query_id = (
            row["query_id"].strip()
        )

        if query_id not in PRIMARY_PAIRS:
            raise ValueError(
                f"Row {row_number}: "
                f"unknown query_id {query_id!r}."
            )

        if row[
            "adjudicator"
        ].strip():
            raise ValueError(
                f"Row {row_number}: "
                "adjudicator is already filled."
            )

        if row[
            "adjudicated_relevance"
        ].strip():
            raise ValueError(
                f"Row {row_number}: "
                "adjudicated_relevance "
                "is already filled."
            )

    return rows


def assign_rows(
    rows: list[dict[str, str]],
) -> tuple[
    dict[int, str],
    Counter[str],
]:
    """Assign every disagreement to an eligible third annotator."""
    rng = random.Random(
        RANDOM_SEED
    )

    row_indexes = list(
        range(len(rows))
    )

    rng.shuffle(
        row_indexes
    )

    loads: Counter[str] = Counter(
        {
            member: 0
            for member in MEMBERS
        }
    )

    assignments: dict[int, str] = {}

    for index in row_indexes:
        row = rows[index]

        query_id = (
            row["query_id"].strip()
        )

        primary_pair = set(
            PRIMARY_PAIRS[
                query_id
            ]
        )

        eligible = [
            member
            for member in MEMBERS
            if member
            not in primary_pair
        ]

        if len(eligible) != 3:
            raise ValueError(
                f"{query_id}: expected "
                "exactly three eligible "
                "adjudicators."
            )

        minimum_load = min(
            loads[member]
            for member in eligible
        )

        least_loaded = [
            member
            for member in eligible
            if loads[member]
            == minimum_load
        ]

        adjudicator = rng.choice(
            least_loaded
        )

        if adjudicator in primary_pair:
            raise AssertionError(
                "Primary annotator was "
                "selected as adjudicator."
            )

        assignments[index] = (
            adjudicator
        )

        loads[
            adjudicator
        ] += 1

    return assignments, loads


def validate_assignments(
    rows: list[dict[str, str]],
    assignments: dict[int, str],
) -> None:
    """Validate third-annotator eligibility and assignment completeness."""
    if len(assignments) != len(rows):
        raise ValueError(
            "Not every disagreement "
            "received an adjudicator."
        )

    for index, row in enumerate(rows):
        if index not in assignments:
            raise ValueError(
                f"Row index {index} "
                "has no assignment."
            )

        query_id = (
            row["query_id"].strip()
        )

        adjudicator = (
            assignments[index]
        )

        primary_pair = set(
            PRIMARY_PAIRS[
                query_id
            ]
        )

        if adjudicator in primary_pair:
            raise ValueError(
                f"{query_id}: "
                f"{adjudicator} was one "
                "of the primary annotators."
            )

        if adjudicator not in MEMBERS:
            raise ValueError(
                f"Unknown adjudicator: "
                f"{adjudicator}"
            )


def write_assignment_files(
    rows: list[dict[str, str]],
    assignments: dict[int, str],
) -> dict[str, int]:
    """Write one blinded adjudication file per team member."""
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows_by_member = {
        member: []
        for member in MEMBERS
    }

    for index, row in enumerate(rows):
        adjudicator = (
            assignments[index]
        )

        output_row = dict(row)

        output_row[
            "adjudicator"
        ] = adjudicator

        output_row[
            "adjudicated_relevance"
        ] = ""

        output_row[
            "adjudication_notes"
        ] = ""

        rows_by_member[
            adjudicator
        ].append(
            output_row
        )

    counts = {}

    for member in MEMBERS:
        output_path = (
            OUTPUT_DIRECTORY
            / f"adjudication_{member}.csv"
        )

        member_rows = (
            rows_by_member[
                member
            ]
        )

        with output_path.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=FIELDNAMES,
            )

            writer.writeheader()

            writer.writerows(
                member_rows
            )

        counts[
            member
        ] = len(
            member_rows
        )

    return counts


def write_summary(
    rows: list[dict[str, str]],
    counts: dict[str, int],
) -> None:
    """Write the reproducibility summary for adjudication assignment."""
    query_counts = Counter(
        row[
            "query_id"
        ].strip()
        for row in rows
    )

    summary = {
        "schema_version": 1,
        "random_seed": RANDOM_SEED,
        "total_disagreements": len(
            rows
        ),
        "assignment_rule": (
            "Each disagreement is assigned "
            "to one team member who was not "
            "one of its two primary annotators. "
            "Among eligible members, the "
            "least-loaded adjudicator is "
            "preferred; ties are resolved "
            "deterministically using the "
            "configured random seed."
        ),
        "counts_by_adjudicator": counts,
        "disagreements_by_query": dict(
            sorted(
                query_counts.items()
            )
        ),
    }

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Create balanced blinded adjudication assignments."""
    rows = load_rows()

    assignments, loads = (
        assign_rows(rows)
    )

    validate_assignments(
        rows,
        assignments,
    )

    counts = write_assignment_files(
        rows,
        assignments,
    )

    write_summary(
        rows,
        counts,
    )

    print(
        "FastContext Adjudication Assignment"
    )
    print(
        "=" * 60
    )

    print(
        "Disagreements:",
        len(rows),
    )

    print(
        "Random seed:",
        RANDOM_SEED,
    )

    print()

    for member in MEMBERS:
        print(
            f"{member}: "
            f"{counts[member]}"
        )

    print()

    print(
        "Total assigned:",
        sum(
            counts.values()
        ),
    )

    print(
        "Minimum load:",
        min(
            loads.values()
        ),
    )

    print(
        "Maximum load:",
        max(
            loads.values()
        ),
    )

    print()

    print(
        "Output directory:",
        OUTPUT_DIRECTORY,
    )

    print(
        "Summary:",
        SUMMARY_PATH,
    )


if __name__ == "__main__":
    main()