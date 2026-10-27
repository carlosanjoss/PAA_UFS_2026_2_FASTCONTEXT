
"""Hybrid retrieval using semantic and literal reciprocal rank fusion."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns
from typing import Any

from src.retrieval.base import Retriever
from src.retrieval.models import (
    RetrievedChunk,
    RetrievalMetrics,
    RetrievalResult,
)


@dataclass(slots=True)
class _FusionCandidate:
    """Internal candidate used by reciprocal rank fusion."""

    chunk: RetrievedChunk
    score: float = 0.0
    semantic_rank: int | None = None
    literal_rank: int | None = None
    semantic_score: float | None = None
    literal_score: float | None = None


class HybridRetriever(Retriever):
    """Combine semantic and literal rankings with RRF."""

    def __init__(
        self,
        semantic_retriever: Retriever,
        literal_retriever: Retriever,
        *,
        rrf_k: int = 60,
        candidate_depth: int = 20,
    ) -> None:
        if semantic_retriever.name != "semantic":
            raise ValueError(
                "The semantic retriever must be named 'semantic'."
            )

        if literal_retriever.name != "literal":
            raise ValueError(
                "The literal retriever must be named 'literal'."
            )

        if (
            isinstance(rrf_k, bool)
            or not isinstance(rrf_k, int)
            or rrf_k <= 0
        ):
            raise ValueError("rrf_k must be a positive integer.")

        if (
            isinstance(candidate_depth, bool)
            or not isinstance(candidate_depth, int)
            or candidate_depth <= 0
        ):
            raise ValueError(
                "candidate_depth must be a positive integer."
            )

        self._semantic = semantic_retriever
        self._literal = literal_retriever
        self._rrf_k = rrf_k
        self._candidate_depth = candidate_depth

    @property
    def name(self) -> str:
        return "hybrid"

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> RetrievalResult:
        """Retrieve and merge ranked evidence from both strategies."""

        normalized_query = self._validate_request(
            query,
            top_k,
        )

        start_ns = perf_counter_ns()

        if top_k == 0:
            return RetrievalResult(
                query=normalized_query,
                algorithm=self.name,
                top_k=0,
                chunks=(),
                metrics=RetrievalMetrics(
                    retrieval_time_ns=(
                        perf_counter_ns() - start_ns
                    ),
                    sorting_time_ns=0,
                    comparisons=0,
                    candidates_found=0,
                    chunks_scored=0,
                ),
                metadata={
                    "fusion": "reciprocal_rank_fusion",
                    "rrf_k": self._rrf_k,
                    "candidate_depth": 0,
                    "semantic_candidates": 0,
                    "literal_candidates": 0,
                },
            )

        depth = max(top_k, self._candidate_depth)

        semantic_result = self._semantic.retrieve(
            normalized_query,
            top_k=depth,
        )

        literal_result = self._literal.retrieve(
            normalized_query,
            top_k=depth,
        )

        if (
            semantic_result.query != normalized_query
            or literal_result.query != normalized_query
        ):
            raise ValueError(
                "Source retrievers returned different queries."
            )

        candidates: dict[str, _FusionCandidate] = {}

        fusion_start_ns = perf_counter_ns()

        for chunk in semantic_result.chunks:
            candidates[chunk.chunk_id] = _FusionCandidate(
                chunk=chunk,
                score=1.0 / (self._rrf_k + chunk.rank),
                semantic_rank=chunk.rank,
                semantic_score=chunk.score,
            )

        for chunk in literal_result.chunks:
            candidate = candidates.get(chunk.chunk_id)

            if candidate is None:
                candidates[chunk.chunk_id] = _FusionCandidate(
                    chunk=chunk,
                    score=1.0 / (self._rrf_k + chunk.rank),
                    literal_rank=chunk.rank,
                    literal_score=chunk.score,
                )
                continue

            if (
                candidate.chunk.content != chunk.content
                or candidate.chunk.source_path != chunk.source_path
            ):
                raise ValueError(
                    "The source retrievers disagree about "
                    f"chunk {chunk.chunk_id!r}."
                )

            candidate.score += (
                1.0 / (self._rrf_k + chunk.rank)
            )
            candidate.literal_rank = chunk.rank
            candidate.literal_score = chunk.score

        ordered = sorted(
            candidates.values(),
            key=lambda item: (
                -item.score,
                item.chunk.chunk_id,
            ),
        )

        selected = ordered[:top_k]

        sorting_time_ns = (
            perf_counter_ns() - fusion_start_ns
        )

        retrieved: list[RetrievedChunk] = []

        for rank, candidate in enumerate(
            selected,
            start=1,
        ):
            original = candidate.chunk

            metadata: dict[str, Any] = {
                "fusion": "reciprocal_rank_fusion",
                "rrf_k": self._rrf_k,
                "semantic_rank": candidate.semantic_rank,
                "literal_rank": candidate.literal_rank,
                "semantic_original_score": (
                    candidate.semantic_score
                ),
                "literal_original_score": (
                    candidate.literal_score
                ),
            }

            retrieved.append(
                RetrievedChunk(
                    chunk_id=original.chunk_id,
                    content=original.content,
                    source_path=original.source_path,
                    section_title=original.section_title,
                    score=candidate.score,
                    rank=rank,
                    token_count=original.token_count,
                    metadata=metadata,
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
                index_build_time_ns=None,
                comparisons=(
                    literal_result.metrics.comparisons
                ),
                chunks_scored=None,
                candidates_found=len(candidates),
            ),
            metadata={
                "fusion": "reciprocal_rank_fusion",
                "rrf_k": self._rrf_k,
                "candidate_depth": depth,
                "semantic_candidates": len(
                    semantic_result.chunks
                ),
                "literal_candidates": len(
                    literal_result.chunks
                ),
                "union_candidates": len(candidates),
                "semantic_retrieval_time_ns": (
                    semantic_result.metrics.retrieval_time_ns
                ),
                "literal_retrieval_time_ns": (
                    literal_result.metrics.retrieval_time_ns
                ),
                "fusion_sorting_time_ns": sorting_time_ns,
                "semantic_algorithm": (
                    semantic_result.algorithm
                ),
                "literal_algorithm": (
                    literal_result.algorithm
                ),
                "score_type": "rrf",
                "source_scores_are_not_directly_combined": True,
            },
        )
