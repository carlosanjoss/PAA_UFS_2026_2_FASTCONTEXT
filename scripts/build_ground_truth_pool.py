"""Build pooled retrieval candidates for human relevance annotation."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.app.bootstrap import create_application
from src.evaluation.pooling import (
    DEFAULT_POOL_DEPTH,
    DEFAULT_POOL_SEED,
    QueryPool,
    build_annotation_pools,
    query_pool_to_dict,
)
from src.retrieval.base import Retriever
from src.utils.config import PROJECT_ROOT

DEFAULT_GROUND_TRUTH_PATH = (
    PROJECT_ROOT
    / "data"
    / "ground_truth"
    / "ground_truth.json"
)

DEFAULT_POOL_PATH = (
    PROJECT_ROOT
    / "data"
    / "ground_truth"
    / "pool.json"
)

DEFAULT_ANNOTATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "ground_truth"
    / "annotation_template.csv"
)

POOL_ALGORITHMS = (
    "linear",
    "indexed",
    "optimized",
    "semantic",
)


def build_parser() -> argparse.ArgumentParser:
    """Create the ground-truth pool command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Build a pooled set of retrieval candidates "
            "for blinded human relevance annotation."
        )
    )

    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=DEFAULT_GROUND_TRUTH_PATH,
        help=(
            "Path to ground_truth.json."
        ),
    )

    parser.add_argument(
        "--pool-output",
        type=Path,
        default=DEFAULT_POOL_PATH,
        help=(
            "Path to the complete pool JSON."
        ),
    )

    parser.add_argument(
        "--annotation-output",
        type=Path,
        default=DEFAULT_ANNOTATION_PATH,
        help=(
            "Path to the blinded annotation CSV."
        ),
    )

    parser.add_argument(
        "--depth",
        type=int,
        default=DEFAULT_POOL_DEPTH,
        help=(
            "Top-k depth collected from each retriever."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_POOL_SEED,
        help=(
            "Seed used for deterministic blinded ordering."
        ),
    )

    return parser


def main(
    argv: Sequence[str] | None = None,
) -> int:
    """Build pool artifacts for human annotation."""
    parser = build_parser()

    args = parser.parse_args(
        argv
    )

    if args.depth <= 0:
        parser.error(
            "--depth must be greater than zero."
        )

    ground_truth_path = _resolve_path(
        args.ground_truth
    )

    pool_path = _resolve_path(
        args.pool_output
    )

    annotation_path = _resolve_path(
        args.annotation_output
    )

    ground_truth = (
        _load_ground_truth(
            ground_truth_path
        )
    )

    queries = ground_truth[
        "queries"
    ]

    print(
        "FastContext Ground Truth Pool"
    )
    print("=" * 60)
    print(
        f"Queries: {len(queries)}"
    )
    print(
        f"Pool depth per retriever: {args.depth}"
    )
    print(
        "Algorithms: "
        + ", ".join(
            POOL_ALGORITHMS
        )
    )
    print(
        f"Blind ordering seed: {args.seed}"
    )
    print()

    print(
        "[1/4] Loading FastContext application..."
    )

    application = (
        create_application()
    )

    retrievers = (
        _build_retrievers(
            application.registry
        )
    )

    print(
        "[2/4] Building candidate pools..."
    )

    pools = build_annotation_pools(
        queries,
        retrievers=retrievers,
        pool_depth=args.depth,
        seed=args.seed,
    )

    for pool in pools:
        print(
            f"      {pool.query_id}: "
            f"{len(pool.candidates)} "
            "unique candidates"
        )

    print(
        "[3/4] Writing complete pool..."
    )

    _write_pool_json(
        pool_path,
        ground_truth=ground_truth,
        pools=pools,
        pool_depth=args.depth,
        seed=args.seed,
    )

    print(
        "[4/4] Writing blinded annotation template..."
    )

    _write_annotation_csv(
        annotation_path,
        pools=pools,
    )

    counts = [
        len(pool.candidates)
        for pool in pools
    ]

    total_candidates = sum(
        counts
    )

    print()
    print(
        "Pool summary"
    )
    print("-" * 60)
    print(
        f"Queries: {len(pools)}"
    )
    print(
        "Candidate judgments per annotator: "
        f"{total_candidates}"
    )

    if counts:
        print(
            "Candidates per query: "
            f"min={min(counts)}, "
            f"mean={total_candidates / len(counts):.2f}, "
            f"max={max(counts)}"
        )

    category_counts = Counter(
        pool.category
        for pool in pools
    )

    print(
        "Categories:"
    )

    for category in sorted(
        category_counts
    ):
        print(
            f"  {category}: "
            f"{category_counts[category]}"
        )

    print()
    print(
        f"Pool JSON: {pool_path}"
    )
    print(
        f"Annotation CSV: {annotation_path}"
    )

    return 0


def _build_retrievers(
    registry: Any,
) -> dict[str, Retriever]:
    """Create one reusable retriever instance per pooling algorithm."""
    retrievers: dict[
        str,
        Retriever,
    ] = {}

    for algorithm in POOL_ALGORITHMS:
        print(
            f"      Initializing {algorithm}..."
        )

        retriever = registry.create(
            algorithm
        )

        retrievers[
            algorithm
        ] = retriever

    return retrievers


def _load_ground_truth(
    path: Path,
) -> dict[str, Any]:
    """Load and validate the ground-truth query definition."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Ground truth file not found: {path}"
        )

    try:
        data = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except json.JSONDecodeError as error:
        raise ValueError(
            "ground_truth.json contains invalid JSON."
        ) from error

    if not isinstance(
        data,
        dict,
    ):
        raise TypeError(
            "Ground truth root must be a JSON object."
        )

    queries = data.get(
        "queries"
    )

    if not isinstance(
        queries,
        list,
    ):
        raise TypeError(
            "Ground truth must contain a queries list."
        )

    if len(queries) != 30:
        raise ValueError(
            "Ground truth must contain exactly 30 queries."
        )

    for query in queries:
        if not isinstance(
            query,
            Mapping,
        ):
            raise TypeError(
                "Every ground-truth query must be an object."
            )

    return data


def _write_pool_json(
    path: Path,
    *,
    ground_truth: Mapping[str, Any],
    pools: Sequence[QueryPool],
    pool_depth: int,
    seed: int,
) -> None:
    """Write the complete pooled candidate artifact."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = ground_truth.get(
        "metadata"
    )

    payload = {
        "metadata": {
            "schema_version": 1,
            "pool_depth": pool_depth,
            "pool_seed": seed,
            "algorithms": list(
                POOL_ALGORITHMS
            ),
            "corpus_metadata": (
                metadata
                if isinstance(
                    metadata,
                    Mapping,
                )
                else {}
            ),
        },
        "queries": [
            query_pool_to_dict(
                pool
            )
            for pool in pools
        ],
    }

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_annotation_csv(
    path: Path,
    *,
    pools: Sequence[QueryPool],
) -> None:
    """Write a blinded CSV for independent human annotation."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
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
    ]

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

        for pool in pools:
            for order, candidate in enumerate(
                pool.candidates,
                start=1,
            ):
                writer.writerow(
                    {
                        "query_id": (
                            pool.query_id
                        ),
                        "category": (
                            pool.category
                        ),
                        "question": (
                            pool.question
                        ),
                        "candidate_order": (
                            order
                        ),
                        "chunk_id": (
                            candidate.chunk_id
                        ),
                        "source_path": (
                            candidate.source_path
                        ),
                        "section_title": (
                            candidate.section_title
                        ),
                        "content": (
                            candidate.content
                        ),
                        "relevance": "",
                        "annotation_notes": "",
                    }
                )


def _resolve_path(
    path: Path,
) -> Path:
    """Resolve project-relative or absolute paths."""
    if path.is_absolute():
        return path

    return (
        PROJECT_ROOT
        / path
    ).resolve()


if __name__ == "__main__":
    raise SystemExit(
        main()
    )