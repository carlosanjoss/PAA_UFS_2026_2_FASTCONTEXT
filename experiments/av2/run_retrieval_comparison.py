
"""Compare semantic, literal and hybrid Top-k retrieval without an LLM."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.evaluation.retrieval_metrics import evaluate_retrieval
from src.ingestion.chunk_loader import (
    DEFAULT_CHUNKS_PATH,
    load_chunks_jsonl,
)
from src.retrieval.base import Retriever
from src.retrieval.models import RetrievalResult
from src.semantic.persistence import calculate_corpus_fingerprint
from src.utils.config import PROJECT_ROOT


GROUND_TRUTH_PATH = (
    PROJECT_ROOT / "data" / "ground_truth" / "ground_truth.final.json"
)

JUDGED_POOL_PATH = (
    PROJECT_ROOT / "data" / "ground_truth" / "annotation_template.csv"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT / "experiments" / "av2" / "retrieval_comparison"
)

PATTERN_CATALOG_PATH = (
    PROJECT_ROOT / "data" / "av2" / "query_patterns.json"
)

CONDITIONS = ("semantic", "literal", "hybrid")

FROZEN_FINGERPRINT = (
    "e08b47b55d7237f38bcaba48c79a468a40b92e4f3120e968ae9ece999f5e8c2d"
)


@dataclass(frozen=True)
class EvaluationQuery:
    """One adjudicated question from the AV1 ground truth."""

    query_id: str
    question: str
    category: str
    relevant_chunks: tuple[str, ...]


def load_evaluation_data(
    ground_truth_path: Path = GROUND_TRUTH_PATH,
    pool_path: Path = JUDGED_POOL_PATH,
    *,
    expected_fingerprint: str = FROZEN_FINGERPRINT,
) -> tuple[list[EvaluationQuery], dict[str, set[str]]]:
    """Load positives and the universe of previously judged candidates."""

    ground_truth = json.loads(
        ground_truth_path.read_text(encoding="utf-8-sig")
    )

    metadata = ground_truth["metadata"]

    if (
        metadata.get("corpus_fingerprint") != expected_fingerprint
        or metadata.get("annotation_status") != "complete"
    ):
        raise ValueError(
            "Ground truth is not the frozen, adjudicated corpus."
        )

    queries: list[EvaluationQuery] = []

    for item in ground_truth["queries"]:
        relevant = tuple(item["relevant_chunks"])

        if not relevant or len(relevant) != len(set(relevant)):
            raise ValueError(
                f"Invalid positive judgments for {item['query_id']}"
            )

        queries.append(
            EvaluationQuery(
                query_id=item["query_id"],
                question=item["question"],
                category=item["category"],
                relevant_chunks=relevant,
            )
        )

    ids = [item.query_id for item in queries]

    if len(ids) != len(set(ids)) or len(queries) != 30:
        raise ValueError(
            "Expected 30 distinct queries from the AV1 ground truth."
        )

    judged: dict[str, set[str]] = {
        item.query_id: set()
        for item in queries
    }

    with pool_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)

        required_columns = {
            "query_id",
            "chunk_id",
            "question",
        }

        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(
                "The annotation pool has an unexpected CSV schema."
            )

        questions = {
            item.query_id: item.question
            for item in queries
        }

        for row in reader:
            query_id = row["query_id"]
            chunk_id = row["chunk_id"]

            if (
                query_id not in judged
                or row["question"] != questions[query_id]
            ):
                raise ValueError(
                    "Annotation pool and ground truth do not agree."
                )

            if not chunk_id or chunk_id in judged[query_id]:
                raise ValueError(
                    f"Missing or duplicate judged chunk: {query_id}"
                )

            judged[query_id].add(chunk_id)

    for item in queries:
        if not set(item.relevant_chunks) <= judged[item.query_id]:
            raise ValueError(
                f"Relevant chunks missing from judged pool: "
                f"{item.query_id}"
            )

    expected_count = metadata[
        "annotation_protocol"
    ]["total_candidate_judgments"]

    if sum(len(ids) for ids in judged.values()) != expected_count:
        raise ValueError(
            "The judged pool does not match the ground-truth manifest."
        )

    return queries, judged


def evaluate_one(
    query: EvaluationQuery,
    condition: str,
    result: RetrievalResult,
    judged_ids: set[str],
    repetition: int,
    top_k: int,
) -> dict[str, Any]:
    """Measure known positives without labeling unseen hits as negative."""

    if (
        result.algorithm != condition
        or result.query != query.question
    ):
        raise ValueError(
            "Retriever output does not match its experimental input."
        )

    retrieved = [
        chunk.chunk_id
        for chunk in result.chunks
    ]

    if (
        len(retrieved) != len(set(retrieved))
        or len(retrieved) > top_k
    ):
        raise ValueError("Invalid retrieved Top-k.")

    positive = set(query.relevant_chunks)

    unjudged = [
        chunk_id
        for chunk_id in retrieved
        if chunk_id not in judged_ids
    ]

    known_positive = [
        chunk_id
        for chunk_id in retrieved
        if chunk_id in positive
    ]

    metrics = evaluate_retrieval(
        retrieved,
        query.relevant_chunks,
        top_k,
    )

    return {
        "query_id": query.query_id,
        "category": query.category,
        "condition": condition,
        "repetition": repetition,
        "top_k": top_k,
        "retrieved_count": len(retrieved),
        "judged_count": len(retrieved) - len(unjudged),
        "unjudged_count": len(unjudged),
        "known_positive_count": len(known_positive),
        "pooled_precision_at_k": metrics.precision,
        "pooled_recall_at_k": metrics.recall,
        "known_mrr_at_k": metrics.reciprocal_rank,
        "known_hit_rate_at_k": metrics.hit_rate,
        "precision_lower_bound": len(known_positive) / top_k,
        "precision_upper_bound": (
            len(known_positive) + len(unjudged)
        ) / top_k,
        "retrieval_time_ms": (
            result.metrics.retrieval_time_ns / 1_000_000
        ),
        "retrieved_chunk_ids": json.dumps(retrieved),
        "unjudged_chunk_ids": json.dumps(unjudged),
        "known_positive_chunk_ids": json.dumps(known_positive),
    }


def run_comparison(
    retrievers: Mapping[str, Retriever],
    queries: Sequence[EvaluationQuery],
    judged: Mapping[str, set[str]],
    *,
    top_k: int = 5,
    repetitions: int = 5,
    warmup: int = 1,
) -> list[dict[str, Any]]:
    """Run paired queries while rotating retrieval execution order."""

    if (
        isinstance(top_k, bool)
        or not isinstance(top_k, int)
        or top_k <= 0
    ):
        raise ValueError("top_k must be a positive integer.")

    if (
        isinstance(repetitions, bool)
        or not isinstance(repetitions, int)
        or repetitions < 1
        or isinstance(warmup, bool)
        or not isinstance(warmup, int)
        or warmup < 0
    ):
        raise ValueError("Invalid repetitions or warmup.")

    if set(retrievers) != set(CONDITIONS):
        raise ValueError(
            "Exactly semantic, literal and hybrid are required."
        )

    if len({query.query_id for query in queries}) != len(queries):
        raise ValueError("Duplicate query IDs.")

    rows: list[dict[str, Any]] = []

    for query_index, query in enumerate(queries):
        if query.query_id not in judged:
            raise ValueError(
                f"Missing judgments for {query.query_id}."
            )

        for repetition_index in range(-warmup, repetitions):
            shift = (
                query_index + max(repetition_index, 0)
            ) % len(CONDITIONS)

            order = CONDITIONS[shift:] + CONDITIONS[:shift]

            for condition in order:
                result = retrievers[condition].retrieve(
                    query.question,
                    top_k=top_k,
                )

                if repetition_index < 0:
                    continue

                rows.append(
                    evaluate_one(
                        query=query,
                        condition=condition,
                        result=result,
                        judged_ids=judged[query.query_id],
                        repetition=repetition_index + 1,
                        top_k=top_k,
                    )
                )

    return rows


def summarize_results(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Aggregate repetitions and macro-average the query results."""

    groups: dict[
        tuple[str, str],
        list[Mapping[str, Any]],
    ] = defaultdict(list)

    for row in rows:
        key = (
            str(row["query_id"]),
            str(row["condition"]),
        )

        groups[key].append(row)

    per_query: list[dict[str, Any]] = []

    measures = (
        "pooled_precision_at_k",
        "pooled_recall_at_k",
        "known_mrr_at_k",
        "known_hit_rate_at_k",
        "precision_lower_bound",
        "precision_upper_bound",
        "unjudged_count",
        "judged_count",
        "known_positive_count",
    )

    for (query_id, condition), records in sorted(groups.items()):
        first = records[0]

        for row in records[1:]:
            if any(
                row[key] != first[key]
                for key in measures
            ):
                raise ValueError(
                    "Retrieval quality changed between repetitions: "
                    f"{query_id}"
                )

            if (
                row["retrieved_chunk_ids"]
                != first["retrieved_chunk_ids"]
            ):
                raise ValueError(
                    "Retrieval ranking changed between repetitions: "
                    f"{query_id}"
                )

        times = [
            float(row["retrieval_time_ms"])
            for row in records
        ]

        per_query.append(
            {
                "query_id": query_id,
                "category": first["category"],
                "condition": condition,
                "repetitions": len(records),
                "mean_time_ms": statistics.mean(times),
                "std_time_ms": (
                    statistics.stdev(times)
                    if len(times) > 1
                    else 0.0
                ),
                **{
                    key: first[key]
                    for key in measures
                },
                "retrieved_chunk_ids": first["retrieved_chunk_ids"],
                "unjudged_chunk_ids": first["unjudged_chunk_ids"],
            }
        )

    by_condition: list[dict[str, Any]] = []

    for condition in CONDITIONS:
        subset = [
            row
            for row in per_query
            if row["condition"] == condition
        ]

        if not subset:
            raise ValueError(
                f"No results for {condition}."
            )

        by_condition.append(
            {
                "condition": condition,
                "queries": len(subset),
                "repetitions_per_query": (
                    subset[0]["repetitions"]
                ),
                "mean_time_ms": statistics.mean(
                    float(row["mean_time_ms"])
                    for row in subset
                ),
                "pooled_precision_at_k": statistics.mean(
                    float(row["pooled_precision_at_k"])
                    for row in subset
                ),
                "pooled_recall_at_k": statistics.mean(
                    float(row["pooled_recall_at_k"])
                    for row in subset
                ),
                "known_mrr_at_k": statistics.mean(
                    float(row["known_mrr_at_k"])
                    for row in subset
                ),
                "known_hit_rate_at_k": statistics.mean(
                    float(row["known_hit_rate_at_k"])
                    for row in subset
                ),
                "mean_precision_lower_bound": statistics.mean(
                    float(row["precision_lower_bound"])
                    for row in subset
                ),
                "mean_precision_upper_bound": statistics.mean(
                    float(row["precision_upper_bound"])
                    for row in subset
                ),
                "unjudged_results": sum(
                    int(row["unjudged_count"])
                    for row in subset
                ),
                "queries_with_unjudged_results": sum(
                    int(row["unjudged_count"]) > 0
                    for row in subset
                ),
            }
        )

    return per_query, by_condition


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """Write a complete CSV table."""

    if not rows:
        raise ValueError(
            "Cannot save an empty result table."
        )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
        )

        writer.writeheader()
        writer.writerows(rows)


def prepare_output(
    directory: Path,
    *,
    force: bool,
) -> None:
    """Prevent accidental replacement of existing results."""

    targets = (
        "raw_results.csv",
        "per_query.csv",
        "by_condition.csv",
        "manifest.json",
    )

    if (
        not force
        and any(
            (directory / filename).exists()
            for filename in targets
        )
    ):
        raise FileExistsError(
            f"Results already exist in {directory}. "
            "Specify --force to replace them."
        )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


def file_hash(path: Path) -> str:
    """Compute a SHA-256 checksum for a source artifact."""

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def main() -> int:
    """Execute the complete AV2 retrieval comparison."""

    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--candidate-depth",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--force",
        action="store_true",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
    )

    args = parser.parse_args()

    if (
        args.top_k < 1
        or args.repetitions < 1
        or args.warmup < 0
    ):
        parser.error(
            "top-k and repetitions must be positive; warmup >= 0"
        )

    if args.candidate_depth < args.top_k:
        parser.error(
            "candidate-depth must be >= top-k"
        )

    directory = args.output_dir or (
        DEFAULT_OUTPUT / "smoke"
        if args.smoke
        else DEFAULT_OUTPUT / "final"
    )

    prepare_output(
        directory,
        force=args.force,
    )

    chunks = load_chunks_jsonl(DEFAULT_CHUNKS_PATH)

    fingerprint = calculate_corpus_fingerprint(chunks)

    if (
        len(chunks) != 1305
        or fingerprint != FROZEN_FINGERPRINT
    ):
        raise ValueError(
            "The AV1 corpus is not the expected frozen corpus."
        )

    queries, judged = load_evaluation_data()

    if args.smoke:
        queries = queries[:1]

    # Imports are delayed to keep offline unit tests lightweight.
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.retrieval.literal_retriever import LiteralRetriever
    from src.retrieval.semantic_retriever import (
        DEFAULT_SEMANTIC_DIRECTORY,
        SemanticRetriever,
    )

    for filename in (
        "index.faiss",
        "chunk_ids.txt",
        "metadata.json",
    ):
        path = DEFAULT_SEMANTIC_DIRECTORY / filename

        if not path.is_file():
            raise FileNotFoundError(
                "The persisted semantic index is missing "
                f"{filename}."
            )

    semantic = SemanticRetriever(
        chunks,
        use_persisted_index=True,
    )

    if semantic.index_source != "persisted":
        raise RuntimeError(
            "The semantic index was rebuilt instead of reused; "
            "aborting comparison."
        )

    literal = LiteralRetriever(chunks)

    retrievers: dict[str, Retriever] = {
        "semantic": semantic,
        "literal": literal,
        "hybrid": HybridRetriever(
            semantic,
            literal,
            candidate_depth=args.candidate_depth,
        ),
    }

    rows = run_comparison(
        retrievers,
        queries,
        judged,
        top_k=args.top_k,
        repetitions=args.repetitions,
        warmup=args.warmup,
    )

    per_query, by_condition = summarize_results(rows)

    write_csv(
        directory / "raw_results.csv",
        rows,
    )
    write_csv(
        directory / "per_query.csv",
        per_query,
    )
    write_csv(
        directory / "by_condition.csv",
        by_condition,
    )

    manifest = {
        "corpus_fingerprint": fingerprint,
        "corpus_chunks": len(chunks),
        "query_count": len(queries),
        "top_k": args.top_k,
        "candidate_depth": args.candidate_depth,
        "repetitions": args.repetitions,
        "warmup": args.warmup,
        "conditions": CONDITIONS,
        "semantic_index_source": semantic.index_source,
        "judgment_pool": (
            "data/ground_truth/annotation_template.csv"
        ),
        "judgment_pool_sha256": file_hash(
            JUDGED_POOL_PATH
        ),
        "ground_truth_sha256": file_hash(
            GROUND_TRUTH_PATH
        ),
        "pattern_catalog_sha256": file_hash(
            PATTERN_CATALOG_PATH
        ),
        "quality_interpretation": (
            "Pooled positive-label estimates. Unjudged hits are "
            "not known negatives. Precision bounds vary their "
            "relevance. Recall, MRR and Hit Rate use only "
            "known positives."
        ),
        "latency_interpretation": (
            "Warm runs only. Semantic and hybrid reuse the "
            "persisted index. Hybrid time includes semantic "
            "and literal retrieval plus fusion."
        ),
    }

    (directory / "manifest.json").write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    print("AV2 × RAG — RETRIEVAL COMPARISON COMPLETED")
    print(f"Measured runs: {len(rows)}")
    print(f"Per-query records: {len(per_query)}")
    print(f"Output: {directory}")

    for row in by_condition:
        print(
            f"{row['condition']:8s} "
            f"P@{args.top_k}="
            f"{row['pooled_precision_at_k']:.3f} "
            f"MRR={row['known_mrr_at_k']:.3f} "
            f"unjudged={row['unjudged_results']} "
            f"time={row['mean_time_ms']:.2f}ms"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
