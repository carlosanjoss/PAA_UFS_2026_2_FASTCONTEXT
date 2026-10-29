"""Typed loaders for the AV2 retrieval, RAG, and human-evaluation artifacts."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Literal, cast

from pydantic import BaseModel

RetrievalCondition = Literal["semantic", "literal", "hybrid"]


class AV2RetrievalMetadata(BaseModel):
    query_count: int
    top_k: int
    candidate_depth: int
    repetitions: int
    warmup: int
    raw_measurements: int
    conditions: list[RetrievalCondition]
    semantic_index_source: str
    quality_interpretation: str
    latency_interpretation: str


class AV2RetrievalCondition(BaseModel):
    condition: RetrievalCondition
    queries: int
    repetitions_per_query: int
    mean_time_ms: float
    pooled_precision_at_k: float
    pooled_recall_at_k: float
    known_mrr_at_k: float
    known_hit_rate_at_k: float
    mean_precision_lower_bound: float
    mean_precision_upper_bound: float
    unjudged_results: int
    queries_with_unjudged_results: int


class AV2RetrievalQuery(BaseModel):
    query_id: str
    category: str
    condition: RetrievalCondition
    repetitions: int
    mean_time_ms: float
    std_time_ms: float
    pooled_precision_at_k: float
    pooled_recall_at_k: float
    known_mrr_at_k: float
    known_hit_rate_at_k: float
    precision_lower_bound: float
    precision_upper_bound: float
    unjudged_count: int
    judged_count: int
    known_positive_count: int


class AV2RetrievalComparison(BaseModel):
    metadata: AV2RetrievalMetadata
    conditions: list[AV2RetrievalCondition]
    queries: list[AV2RetrievalQuery]


class AV2RAGMetadata(BaseModel):
    query_count: int
    generation_count: int
    model: str
    top_k: int
    temperature: float
    seed: int
    conditions: list[RetrievalCondition]
    evaluation_note: str


class AV2RAGCondition(BaseModel):
    condition: RetrievalCondition
    generations: int
    mean_generation_time_ms: float
    std_generation_time_ms: float
    citation_valid_count: int
    citation_valid_rate: float
    mean_citation_count: float
    mean_known_relevant_citations: float
    citation_retry_count: int


class AV2RAGQuery(BaseModel):
    query_id: str
    category: str
    condition: RetrievalCondition
    generation_time_ms: float
    citation_valid: bool
    citation_count: int
    citation_retry_count: int
    known_relevant_citations: int
    answer_length_chars: int


class AV2RAGComparison(BaseModel):
    metadata: AV2RAGMetadata
    conditions: list[AV2RAGCondition]
    queries: list[AV2RAGQuery]


class AV2HumanCondition(BaseModel):
    condition: RetrievalCondition
    responses: int
    quality_ratings: int
    groundedness_ratings: int
    mean_correctness: float
    mean_completeness: float
    mean_clarity: float
    hallucination_rate: float
    mean_groundedness: float


class AV2HumanAgreement(BaseModel):
    metric: str
    exact_agreement_rate: float
    mean_absolute_difference: float


class AV2HumanEvaluation(BaseModel):
    status: Literal["complete"]
    scale_min: int
    scale_max: int
    quality_ratings: int
    groundedness_ratings: int
    effective_quality_judgments: int
    effective_groundedness_judgments: int
    unique_responses: int
    ratings_per_response: int
    duplicate_score_pairs: int
    duplicate_note_pairs: int
    independent_ratings_confirmed: bool
    conditions: list[AV2HumanCondition]
    agreement: list[AV2HumanAgreement]
    source_archive_sha256: str
    methodology_note: str


class AV2ComparisonArtifacts(BaseModel):
    retrieval: AV2RetrievalComparison
    rag: AV2RAGComparison
    human_evaluation: AV2HumanEvaluation
    generated_from: list[str]


def load_av2_comparison_artifacts(repository_root: Path) -> AV2ComparisonArtifacts:
    """Load persisted comparisons without exposing answers or evaluator identity."""
    human_directory = (
        repository_root / "experiments/av2/human_evaluation/completed_anonymous"
    )
    paths = {
        "retrieval_conditions": repository_root
        / "experiments/av2/retrieval_comparison/final/by_condition.csv",
        "retrieval_queries": repository_root
        / "experiments/av2/retrieval_comparison/final/per_query.csv",
        "retrieval_raw": repository_root
        / "experiments/av2/retrieval_comparison/final/raw_results.csv",
        "retrieval_manifest": repository_root
        / "experiments/av2/retrieval_comparison/final/manifest.json",
        "rag_generations": repository_root
        / "experiments/av2/rag_comparison/final/generations.csv",
        "rag_manifest": repository_root
        / "experiments/av2/rag_comparison/final/manifest.json",
        "human_manifest": repository_root
        / "experiments/av2/human_evaluation/hidden_manifest.csv",
        "human_source_hash": human_directory / "SOURCE_SHA256.txt",
    }
    quality_files = sorted(human_directory.glob("conjunto_*_qualidade.csv"))
    groundedness_files = sorted(
        human_directory.glob("conjunto_*_fundamentacao.csv")
    )
    required_paths = [*paths.values(), *quality_files, *groundedness_files]
    missing = [
        str(path.relative_to(repository_root))
        for path in required_paths
        if not path.is_file()
    ]
    if missing or len(quality_files) != 5 or len(groundedness_files) != 5:
        detail = missing or ["expected five anonymous file pairs"]
        raise FileNotFoundError(", ".join(detail))

    retrieval_conditions = _load_retrieval_conditions(
        _read_csv(paths["retrieval_conditions"])
    )
    retrieval_queries = _load_retrieval_queries(
        _read_csv(paths["retrieval_queries"])
    )
    retrieval_raw_count = len(_read_csv(paths["retrieval_raw"]))
    retrieval_manifest = _read_json(paths["retrieval_manifest"])
    retrieval = _build_retrieval_comparison(
        retrieval_conditions,
        retrieval_queries,
        retrieval_raw_count,
        retrieval_manifest,
    )

    generation_rows = _read_csv(paths["rag_generations"])
    rag_manifest = _read_json(paths["rag_manifest"])
    rag = _build_rag_comparison(generation_rows, rag_manifest)

    manifest_rows = _read_csv(paths["human_manifest"])
    quality_rows = _read_anonymous_rows(quality_files)
    groundedness_rows = _read_anonymous_rows(groundedness_files)
    source_hash = paths["human_source_hash"].read_text(encoding="utf-8-sig").strip()
    human_evaluation = _build_human_evaluation(
        manifest_rows,
        quality_rows,
        groundedness_rows,
        source_hash.split()[-1],
    )

    return AV2ComparisonArtifacts(
        retrieval=retrieval,
        rag=rag,
        human_evaluation=human_evaluation,
        generated_from=[
            str(path.relative_to(repository_root)).replace("\\", "/")
            for path in paths.values()
        ]
        + [
            str(path.relative_to(repository_root)).replace("\\", "/")
            for path in (*quality_files, *groundedness_files)
        ],
    )


def _build_retrieval_comparison(
    conditions: list[AV2RetrievalCondition],
    queries: list[AV2RetrievalQuery],
    raw_count: int,
    manifest: dict[str, Any],
) -> AV2RetrievalComparison:
    condition_names = [
        cast(RetrievalCondition, item) for item in manifest["conditions"]
    ]
    expected_queries = int(manifest["query_count"])
    expected_rows = expected_queries * len(condition_names)
    expected_raw = expected_rows * int(manifest["repetitions"])
    if len(conditions) != len(condition_names) or len(queries) != expected_rows:
        raise ValueError("retrieval comparison dimensions do not match manifest")
    if raw_count != expected_raw:
        raise ValueError("retrieval raw measurement count does not match manifest")
    return AV2RetrievalComparison(
        metadata=AV2RetrievalMetadata(
            query_count=expected_queries,
            top_k=int(manifest["top_k"]),
            candidate_depth=int(manifest["candidate_depth"]),
            repetitions=int(manifest["repetitions"]),
            warmup=int(manifest["warmup"]),
            raw_measurements=raw_count,
            conditions=condition_names,
            semantic_index_source=str(manifest["semantic_index_source"]),
            quality_interpretation=str(manifest["quality_interpretation"]),
            latency_interpretation=str(manifest["latency_interpretation"]),
        ),
        conditions=conditions,
        queries=queries,
    )


def _build_rag_comparison(
    rows: list[dict[str, str]],
    manifest: dict[str, Any],
) -> AV2RAGComparison:
    condition_names = [
        cast(RetrievalCondition, item) for item in manifest["conditions"]
    ]
    if len(rows) != int(manifest["generations"]):
        raise ValueError("RAG generation count does not match manifest")
    queries = [
        AV2RAGQuery(
            query_id=row["query_id"],
            category=row["category"],
            condition=cast(RetrievalCondition, row["condition"]),
            generation_time_ms=float(row["generation_time_ms"]),
            citation_valid=_parse_bool(row["citation_valid"]),
            citation_count=int(row["citation_count"]),
            citation_retry_count=int(row["citation_retry_count"]),
            known_relevant_citations=int(row["known_relevant_citations"]),
            answer_length_chars=len(row["answer"]),
        )
        for row in rows
    ]
    grouped: dict[RetrievalCondition, list[AV2RAGQuery]] = defaultdict(list)
    for row in queries:
        grouped[row.condition].append(row)
    conditions = []
    for condition in condition_names:
        group = grouped[condition]
        times = [row.generation_time_ms for row in group]
        valid_count = sum(row.citation_valid for row in group)
        conditions.append(
            AV2RAGCondition(
                condition=condition,
                generations=len(group),
                mean_generation_time_ms=fmean(times),
                std_generation_time_ms=pstdev(times),
                citation_valid_count=valid_count,
                citation_valid_rate=valid_count / len(group),
                mean_citation_count=fmean(row.citation_count for row in group),
                mean_known_relevant_citations=fmean(
                    row.known_relevant_citations for row in group
                ),
                citation_retry_count=sum(row.citation_retry_count for row in group),
            )
        )
    return AV2RAGComparison(
        metadata=AV2RAGMetadata(
            query_count=int(manifest["queries"]),
            generation_count=int(manifest["generations"]),
            model=str(manifest["model"]),
            top_k=int(manifest["top_k"]),
            temperature=float(manifest["temperature"]),
            seed=int(manifest["seed"]),
            conditions=condition_names,
            evaluation_note=str(manifest["evaluation_note"]),
        ),
        conditions=conditions,
        queries=queries,
    )


def _build_human_evaluation(
    manifest_rows: list[dict[str, str]],
    quality_rows: list[dict[str, str]],
    groundedness_rows: list[dict[str, str]],
    source_hash: str,
) -> AV2HumanEvaluation:
    manifest = {row["blind_id"]: row for row in manifest_rows}
    _validate_human_rows(quality_rows, manifest, "quality")
    _validate_human_rows(groundedness_rows, manifest, "groundedness")

    condition_scores: dict[RetrievalCondition, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    quality_by_id: dict[str, list[dict[str, str]]] = defaultdict(list)
    groundedness_by_id: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in quality_rows:
        condition = cast(RetrievalCondition, manifest[row["blind_id"]]["condition"])
        quality_by_id[row["blind_id"]].append(row)
        for metric in (
            "correctness",
            "completeness",
            "clarity",
            "hallucination_present",
        ):
            condition_scores[condition][metric].append(int(row[metric]))
    for row in groundedness_rows:
        condition = cast(RetrievalCondition, manifest[row["blind_id"]]["condition"])
        groundedness_by_id[row["blind_id"]].append(row)
        condition_scores[condition]["groundedness"].append(int(row["groundedness"]))

    conditions = []
    for condition in ("semantic", "literal", "hybrid"):
        typed_condition = cast(RetrievalCondition, condition)
        scores = condition_scores[typed_condition]
        quality_count = len(scores["correctness"])
        ground_count = len(scores["groundedness"])
        conditions.append(
            AV2HumanCondition(
                condition=typed_condition,
                responses=quality_count // 2,
                quality_ratings=quality_count,
                groundedness_ratings=ground_count,
                mean_correctness=fmean(scores["correctness"]),
                mean_completeness=fmean(scores["completeness"]),
                mean_clarity=fmean(scores["clarity"]),
                hallucination_rate=fmean(scores["hallucination_present"]),
                mean_groundedness=fmean(scores["groundedness"]),
            )
        )

    agreement = [
        _agreement(metric, quality_by_id)
        for metric in (
            "correctness",
            "completeness",
            "clarity",
            "hallucination_present",
        )
    ]
    agreement.append(_agreement("groundedness", groundedness_by_id))
    quality_duplicate_scores = _duplicate_score_pairs(
        quality_by_id,
        ("correctness", "completeness", "clarity", "hallucination_present"),
    )
    groundedness_duplicate_scores = _duplicate_score_pairs(
        groundedness_by_id,
        ("groundedness",),
    )
    duplicate_notes = _duplicate_note_pairs(quality_by_id) + _duplicate_note_pairs(
        groundedness_by_id
    )
    return AV2HumanEvaluation(
        status="complete",
        scale_min=1,
        scale_max=5,
        quality_ratings=len(quality_rows),
        groundedness_ratings=len(groundedness_rows),
        effective_quality_judgments=len(quality_by_id),
        effective_groundedness_judgments=len(groundedness_by_id),
        unique_responses=len(quality_by_id),
        ratings_per_response=2,
        duplicate_score_pairs=(
            quality_duplicate_scores + groundedness_duplicate_scores
        ),
        duplicate_note_pairs=duplicate_notes,
        independent_ratings_confirmed=False,
        conditions=conditions,
        agreement=agreement,
        source_archive_sha256=source_hash,
        methodology_note=(
            "Anonymous completed export received on a 1–5 ordinal scale; "
            "hallucination_present is binary (0/1). Both submitted rows for "
            "every blind_id contain identical scores and notes, so the package "
            "does not provide evidence of independent inter-rater agreement."
        ),
    )


def _validate_human_rows(
    rows: list[dict[str, str]],
    manifest: dict[str, dict[str, str]],
    task_type: str,
) -> None:
    expected_metrics = (
        ("correctness", "completeness", "clarity")
        if task_type == "quality"
        else ("groundedness",)
    )
    counts = Counter(row["blind_id"] for row in rows)
    if len(rows) != 180 or len(counts) != 90 or set(counts.values()) != {2}:
        raise ValueError(f"{task_type} ratings must contain two ratings for 90 items")
    for row in rows:
        blind_id = row["blind_id"]
        if blind_id not in manifest or manifest[blind_id]["task_type"] != task_type:
            raise ValueError(f"unknown {task_type} blind_id")
        for metric in expected_metrics:
            value = int(row[metric])
            if not 1 <= value <= 5:
                raise ValueError(f"{metric} must use the 1–5 scale")
        if task_type == "quality" and int(row["hallucination_present"]) not in {0, 1}:
            raise ValueError("hallucination_present must be binary")


def _agreement(
    metric: str,
    rows_by_id: dict[str, list[dict[str, str]]],
) -> AV2HumanAgreement:
    pairs = [
        (int(rows[0][metric]), int(rows[1][metric])) for rows in rows_by_id.values()
    ]
    return AV2HumanAgreement(
        metric=metric,
        exact_agreement_rate=sum(left == right for left, right in pairs) / len(pairs),
        mean_absolute_difference=fmean(abs(left - right) for left, right in pairs),
    )


def _duplicate_score_pairs(
    rows_by_id: dict[str, list[dict[str, str]]],
    metrics: tuple[str, ...],
) -> int:
    return sum(
        all(rows[0][metric] == rows[1][metric] for metric in metrics)
        for rows in rows_by_id.values()
    )


def _duplicate_note_pairs(
    rows_by_id: dict[str, list[dict[str, str]]],
) -> int:
    return sum(
        rows[0].get("notes", "") == rows[1].get("notes", "")
        for rows in rows_by_id.values()
    )


def _load_retrieval_conditions(
    rows: list[dict[str, str]],
) -> list[AV2RetrievalCondition]:
    return [
        AV2RetrievalCondition(
            condition=cast(RetrievalCondition, row["condition"]),
            queries=int(row["queries"]),
            repetitions_per_query=int(row["repetitions_per_query"]),
            mean_time_ms=float(row["mean_time_ms"]),
            pooled_precision_at_k=float(row["pooled_precision_at_k"]),
            pooled_recall_at_k=float(row["pooled_recall_at_k"]),
            known_mrr_at_k=float(row["known_mrr_at_k"]),
            known_hit_rate_at_k=float(row["known_hit_rate_at_k"]),
            mean_precision_lower_bound=float(row["mean_precision_lower_bound"]),
            mean_precision_upper_bound=float(row["mean_precision_upper_bound"]),
            unjudged_results=int(row["unjudged_results"]),
            queries_with_unjudged_results=int(row["queries_with_unjudged_results"]),
        )
        for row in rows
    ]


def _load_retrieval_queries(
    rows: list[dict[str, str]],
) -> list[AV2RetrievalQuery]:
    integer_fields = (
        "repetitions",
        "unjudged_count",
        "judged_count",
        "known_positive_count",
    )
    float_fields = (
        "mean_time_ms",
        "std_time_ms",
        "pooled_precision_at_k",
        "pooled_recall_at_k",
        "known_mrr_at_k",
        "known_hit_rate_at_k",
        "precision_lower_bound",
        "precision_upper_bound",
    )
    result = []
    for row in rows:
        payload: dict[str, Any] = {
            "query_id": row["query_id"],
            "category": row["category"],
            "condition": cast(RetrievalCondition, row["condition"]),
        }
        payload.update({field: int(row[field]) for field in integer_fields})
        payload.update({field: float(row[field]) for field in float_fields})
        result.append(AV2RetrievalQuery(**payload))
    return result


def _read_anonymous_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows = []
    for path in paths:
        rows.extend(_read_csv(path))
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain a JSON object")
    return value


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"invalid boolean value: {value}")
    return normalized == "true"
