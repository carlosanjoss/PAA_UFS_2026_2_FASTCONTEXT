
"""Literal chunk retrieval using the FastContext AV2 matchers."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter_ns
from typing import Any

from src.retrieval.base import Retriever
from src.retrieval.models import (
    RetrievalMetrics,
    RetrievalResult,
    RetrievedChunk,
)
from src.string_matching.registry import create_default_registry
from src.utils.config import PROJECT_ROOT

DEFAULT_PATTERN_CATALOG = (
    PROJECT_ROOT / "data" / "av2" / "query_patterns.json"
)

DEFAULT_GROUND_TRUTH = (
    PROJECT_ROOT
    / "data"
    / "ground_truth"
    / "ground_truth.final.json"
)


def load_question_patterns(
    catalog_path: str | Path = DEFAULT_PATTERN_CATALOG,
    ground_truth_path: str | Path = DEFAULT_GROUND_TRUTH,
) -> dict[str, tuple[str, ...]]:
    """Associate predefined literal patterns with AV1 questions.

    Only query IDs and question texts are read from the ground truth.
    Relevance judgments are never used to choose search patterns.
    """

    catalog = json.loads(
        Path(catalog_path).read_text(encoding="utf-8")
    )
    ground_truth = json.loads(
        Path(ground_truth_path).read_text(encoding="utf-8")
    )

    if not isinstance(catalog, dict):
        raise ValueError("Pattern catalog must be a JSON object.")

    if not isinstance(ground_truth, dict):
        raise ValueError("Ground truth must be a JSON object.")

    if catalog.get("schema_version") != 1:
        raise ValueError("Unsupported pattern catalog schema.")

    policy = catalog.get("selection_policy")

    if not isinstance(policy, dict):
        raise ValueError("Missing pattern selection policy.")

    expected_policy = {
        "method": "question_only_predefined",
        "case_sensitive": True,
        "exact_substring": True,
        "allow_overlaps": True,
        "normalization": "none",
        "ground_truth_relevant_chunks_used": False,
    }

    for key, expected in expected_policy.items():
        if policy.get(key) != expected:
            raise ValueError(
                f"Unexpected pattern selection policy: {key}"
            )

    metadata = ground_truth.get("metadata", {})

    if not isinstance(metadata, dict):
        raise ValueError("Invalid ground truth metadata.")

    if (
        catalog.get("corpus_fingerprint")
        != metadata.get("corpus_fingerprint")
    ):
        raise ValueError(
            "Pattern catalog and ground truth fingerprints differ."
        )

    ground_truth_queries = ground_truth.get("queries")
    catalog_queries = catalog.get("queries")

    if not isinstance(ground_truth_queries, list):
        raise ValueError("Invalid ground truth query list.")

    if not isinstance(catalog_queries, list):
        raise ValueError("Invalid pattern catalog query list.")

    if len(catalog_queries) != catalog.get("expected_queries"):
        raise ValueError(
            "Pattern catalog query count is inconsistent."
        )

    questions_by_id: dict[str, str] = {}

    for entry in ground_truth_queries:
        if not isinstance(entry, dict):
            raise ValueError("Invalid ground truth query entry.")

        query_id = entry.get("query_id")
        question = entry.get("question")

        if (
            not isinstance(query_id, str)
            or not query_id
            or not isinstance(question, str)
            or not question.strip()
        ):
            raise ValueError("Invalid ground truth question.")

        if query_id in questions_by_id:
            raise ValueError("Duplicate ground truth query ID.")

        questions_by_id[query_id] = question

    seen_ids: set[str] = set()
    patterns_by_question: dict[str, tuple[str, ...]] = {}

    for entry in catalog_queries:
        if not isinstance(entry, dict):
            raise ValueError("Invalid pattern catalog entry.")

        query_id = entry.get("query_id")
        patterns = entry.get("patterns")

        if not isinstance(query_id, str):
            raise ValueError("Invalid pattern query ID.")

        if query_id in seen_ids:
            raise ValueError("Duplicate pattern query ID.")

        seen_ids.add(query_id)

        if query_id not in questions_by_id:
            raise ValueError(
                f"Unknown ground truth query ID: {query_id}"
            )

        if not isinstance(patterns, list) or not patterns:
            raise ValueError(
                f"No literal patterns for query {query_id}."
            )

        question = questions_by_id[query_id]
        validated_patterns: list[str] = []

        for pattern in patterns:
            if (
                not isinstance(pattern, str)
                or not pattern
                or pattern != pattern.strip()
                or pattern not in question
            ):
                raise ValueError(
                    f"Invalid literal pattern for {query_id}: "
                    f"{pattern!r}"
                )

            if pattern in validated_patterns:
                raise ValueError(
                    f"Duplicate literal pattern for {query_id}."
                )

            validated_patterns.append(pattern)

        if question in patterns_by_question:
            raise ValueError("Duplicate ground truth question text.")

        patterns_by_question[question] = tuple(
            validated_patterns
        )

    if seen_ids != set(questions_by_id):
        raise ValueError(
            "Pattern catalog does not cover all questions."
        )

    return patterns_by_question


class LiteralRetriever(Retriever):
    """Rank corpus chunks using exact AV2 pattern occurrences."""

    def __init__(
        self,
        corpus_chunks: Sequence[Mapping[str, Any]],
        *,
        question_patterns: Mapping[str, Sequence[str]] | None = None,
        matcher_algorithm: str = "boyer-moore",
    ) -> None:
        """Initialize the corpus, query patterns, and matcher."""

        self._corpus = [
            dict(chunk) for chunk in corpus_chunks
        ]

        raw_patterns = (
            question_patterns
            if question_patterns is not None
            else load_question_patterns()
        )

        self._patterns: dict[str, tuple[str, ...]] = {}

        for question, patterns in raw_patterns.items():
            if not isinstance(question, str) or not question.strip():
                raise ValueError("Question must be a non-empty string.")

            if isinstance(patterns, str):
                raise TypeError(
                    "Question patterns must be a sequence of strings."
                )

            selected = tuple(patterns)

            if not selected:
                raise ValueError(
                    f"No patterns provided for question: {question}"
                )

            if len(set(selected)) != len(selected):
                raise ValueError(
                    "Duplicate patterns are not allowed."
                )

            for pattern in selected:
                if (
                    not isinstance(pattern, str)
                    or not pattern
                    or pattern != pattern.strip()
                    or pattern not in question
                ):
                    raise ValueError(
                        f"Pattern {pattern!r} is not an exact "
                        "substring of its question."
                    )

            normalized_question = question.strip()

            if normalized_question in self._patterns:
                raise ValueError("Duplicate normalized question.")

            self._patterns[normalized_question] = selected

        self._matcher = (
            create_default_registry().create(matcher_algorithm)
        )

        chunk_ids: set[str] = set()

        for chunk in self._corpus:
            chunk_id = chunk.get("chunk_id")
            content = chunk.get("content")
            source_path = chunk.get("source_path")

            if (
                not isinstance(chunk_id, str)
                or not chunk_id.strip()
                or not isinstance(content, str)
                or not content.strip()
                or not isinstance(source_path, str)
                or not source_path.strip()
            ):
                raise ValueError("Invalid corpus chunk.")

            if chunk_id in chunk_ids:
                raise ValueError(
                    f"Duplicate corpus chunk ID: {chunk_id}"
                )

            chunk_ids.add(chunk_id)

    @property
    def name(self) -> str:
        """Return the retrieval strategy name."""
        return "literal"

    @property
    def matcher_algorithm(self) -> str:
        """Return the AV2 matching implementation."""
        return self._matcher.name

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> RetrievalResult:
        """Select the best chunks using predefined literal patterns."""

        normalized_query = self._validate_request(
            query=query,
            top_k=top_k,
        )

        if normalized_query not in self._patterns:
            raise ValueError(
                "No predefined literal patterns for this question."
            )

        patterns = self._patterns[normalized_query]
        start_ns = perf_counter_ns()

        candidates: list[
            tuple[int, int, str, dict[str, Any], dict[str, Any]]
        ] = []

        comparisons = 0
        preprocessing_ns = 0
        matching_search_ns = 0

        if top_k > 0:
            for chunk in self._corpus:
                content = str(chunk["content"])

                matched_pattern_count = 0
                total_occurrences = 0
                positions_by_pattern: dict[str, list[int]] = {}

                for pattern in patterns:
                    match = self._matcher.search(
                        content,
                        pattern,
                    )

                    comparisons += match.comparisons
                    preprocessing_ns += (
                        match.preprocessing_time_ns
                    )
                    matching_search_ns += match.search_time_ns

                    if match.positions:
                        matched_pattern_count += 1
                        total_occurrences += match.occurrences

                        positions_by_pattern[pattern] = list(
                            match.positions
                        )

                if matched_pattern_count == 0:
                    continue

                candidates.append(
                    (
                        matched_pattern_count,
                        total_occurrences,
                        str(chunk["chunk_id"]),
                        chunk,
                        positions_by_pattern,
                    )
                )

        sorting_start_ns = perf_counter_ns()

        candidates.sort(
            key=lambda item: (
                -item[0],
                -item[1],
                item[2],
            )
        )

        selected = candidates[:top_k]
        sorting_time_ns = perf_counter_ns() - sorting_start_ns

        retrieved: list[RetrievedChunk] = []

        for rank, candidate in enumerate(selected, start=1):
            (
                matched_count,
                occurrences,
                _,
                chunk,
                positions_by_pattern,
            ) = candidate

            # A bounded deterministic heuristic, not a semantic score.
            score = float(matched_count) + (
                occurrences / (occurrences + 1)
            )

            raw_token_count = chunk.get("token_count")

            token_count = (
                raw_token_count
                if isinstance(raw_token_count, int)
                and not isinstance(raw_token_count, bool)
                and raw_token_count >= 0
                else None
            )

            retrieved.append(
                RetrievedChunk(
                    chunk_id=str(chunk["chunk_id"]),
                    content=str(chunk["content"]),
                    source_path=str(chunk["source_path"]),
                    section_title=str(
                        chunk.get("section_title")
                        or chunk.get("section")
                        or "Untitled"
                    ),
                    score=score,
                    rank=rank,
                    token_count=token_count,
                    metadata={
                        "matched_pattern_count": matched_count,
                        "total_occurrences": occurrences,
                        "positions_by_pattern": positions_by_pattern,
                        "score_type": "literal_heuristic",
                    },
                )
            )

        total_time_ns = perf_counter_ns() - start_ns

        return RetrievalResult(
            query=normalized_query,
            algorithm=self.name,
            top_k=top_k,
            chunks=tuple(retrieved),
            metrics=RetrievalMetrics(
                retrieval_time_ns=total_time_ns,
                sorting_time_ns=sorting_time_ns,
                comparisons=comparisons,
                chunks_scored=(
                    len(self._corpus) if top_k > 0 else 0
                ),
                candidates_found=len(candidates),
            ),
            metadata={
                "matching_algorithm": self._matcher.name,
                "ranking_policy": (
                    "distinct_patterns_desc,"
                    "occurrences_desc,"
                    "chunk_id_asc"
                ),
                "patterns": list(patterns),
                "preprocessing_time_ns": preprocessing_ns,
                "matching_search_time_ns": matching_search_ns,
                "candidate_count": len(candidates),
                "pattern_source": "predefined_question_only",
                "uses_relevance_labels": False,
            },
        )
