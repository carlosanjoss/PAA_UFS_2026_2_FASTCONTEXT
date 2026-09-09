"""Build deterministic candidate pools for retrieval relevance annotation."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from src.retrieval.base import Retriever
from src.retrieval.models import RetrievalResult

DEFAULT_POOL_DEPTH = 20
DEFAULT_POOL_SEED = 42


class GroundTruthPoolError(ValueError):
    """Base error raised while constructing an annotation pool."""


@dataclass(frozen=True, slots=True)
class PoolRetrievalRecord:
    """Record how one retriever returned a candidate chunk."""

    algorithm: str
    rank: int
    score: float

    def __post_init__(self) -> None:
        if not self.algorithm.strip():
            raise ValueError(
                "algorithm cannot be empty."
            )

        if self.rank <= 0:
            raise ValueError(
                "rank must be greater than zero."
            )


@dataclass(frozen=True, slots=True)
class PoolCandidate:
    """Represent one unique candidate chunk in an annotation pool."""

    chunk_id: str
    source_path: str
    section_title: str
    content: str
    token_count: int | None
    retrieved_by: tuple[
        PoolRetrievalRecord,
        ...,
    ]

    def __post_init__(self) -> None:
        if not self.chunk_id.strip():
            raise ValueError(
                "chunk_id cannot be empty."
            )

        if not self.content.strip():
            raise ValueError(
                "content cannot be empty."
            )

        if not self.retrieved_by:
            raise ValueError(
                "retrieved_by cannot be empty."
            )


@dataclass(frozen=True, slots=True)
class QueryPool:
    """Represent the candidate pool for one evaluation query."""

    query_id: str
    category: str
    question: str
    candidates: tuple[
        PoolCandidate,
        ...,
    ]

    def __post_init__(self) -> None:
        if not self.query_id.strip():
            raise ValueError(
                "query_id cannot be empty."
            )

        if not self.question.strip():
            raise ValueError(
                "question cannot be empty."
            )


def build_query_pool(
    *,
    query_id: str,
    category: str,
    question: str,
    retrievers: Mapping[str, Retriever],
    pool_depth: int = DEFAULT_POOL_DEPTH,
    seed: int = DEFAULT_POOL_SEED,
) -> QueryPool:
    """Build a deterministic pooled candidate set for one query."""
    normalized_query_id = (
        _require_non_empty_string(
            query_id,
            "query_id",
        )
    )

    normalized_category = (
        _require_non_empty_string(
            category,
            "category",
        )
    )

    normalized_question = (
        _require_non_empty_string(
            question,
            "question",
        )
    )

    if pool_depth <= 0:
        raise GroundTruthPoolError(
            "pool_depth must be greater than zero."
        )

    if not retrievers:
        raise GroundTruthPoolError(
            "At least one retriever is required."
        )

    candidate_data: dict[
        str,
        dict[str, Any],
    ] = {}

    for algorithm in sorted(
        retrievers
    ):
        retriever = retrievers[
            algorithm
        ]

        result = retriever.retrieve(
            normalized_question,
            top_k=pool_depth,
        )

        _validate_result_algorithm(
            result,
            expected_algorithm=algorithm,
        )

        _merge_result(
            candidate_data,
            result,
        )

    ordered_chunk_ids = sorted(
        candidate_data,
        key=lambda chunk_id: (
            _blind_order_key(
                query_id=normalized_query_id,
                chunk_id=chunk_id,
                seed=seed,
            ),
            chunk_id,
        ),
    )

    candidates = tuple(
        _build_candidate(
            candidate_data[
                chunk_id
            ]
        )
        for chunk_id in ordered_chunk_ids
    )

    return QueryPool(
        query_id=normalized_query_id,
        category=normalized_category,
        question=normalized_question,
        candidates=candidates,
    )


def build_annotation_pools(
    queries: Sequence[
        Mapping[str, Any]
    ],
    *,
    retrievers: Mapping[str, Retriever],
    pool_depth: int = DEFAULT_POOL_DEPTH,
    seed: int = DEFAULT_POOL_SEED,
) -> tuple[QueryPool, ...]:
    """Build annotation pools for all evaluation queries."""
    pools: list[
        QueryPool
    ] = []

    seen_query_ids: set[str] = set()

    for query in queries:
        query_id = _mapping_string(
            query,
            "query_id",
        )

        if query_id in seen_query_ids:
            raise GroundTruthPoolError(
                f"Duplicate query_id: {query_id}"
            )

        seen_query_ids.add(
            query_id
        )

        pools.append(
            build_query_pool(
                query_id=query_id,
                category=_mapping_string(
                    query,
                    "category",
                ),
                question=_mapping_string(
                    query,
                    "question",
                ),
                retrievers=retrievers,
                pool_depth=pool_depth,
                seed=seed,
            )
        )

    return tuple(
        pools
    )


def query_pool_to_dict(
    pool: QueryPool,
) -> dict[str, Any]:
    """Serialize one query pool with retrieval provenance."""
    return {
        "query_id": pool.query_id,
        "category": pool.category,
        "question": pool.question,
        "candidate_count": len(
            pool.candidates
        ),
        "candidates": [
            {
                "candidate_order": order,
                "chunk_id": candidate.chunk_id,
                "source_path": candidate.source_path,
                "section_title": candidate.section_title,
                "content": candidate.content,
                "token_count": candidate.token_count,
                "retrieved_by": [
                    {
                        "algorithm": record.algorithm,
                        "rank": record.rank,
                        "score": record.score,
                    }
                    for record in candidate.retrieved_by
                ],
            }
            for order, candidate in enumerate(
                pool.candidates,
                start=1,
            )
        ],
    }


def _merge_result(
    candidate_data: dict[
        str,
        dict[str, Any],
    ],
    result: RetrievalResult,
) -> None:
    """Merge one retriever result into a unique candidate dictionary."""
    for chunk in result.chunks:
        entry = candidate_data.setdefault(
            chunk.chunk_id,
            {
                "chunk_id": chunk.chunk_id,
                "source_path": (
                    chunk.source_path
                ),
                "section_title": (
                    chunk.section_title
                ),
                "content": chunk.content,
                "token_count": (
                    chunk.token_count
                ),
                "retrieved_by": [],
            },
        )

        retrieved_by = entry[
            "retrieved_by"
        ]

        if not isinstance(
            retrieved_by,
            list,
        ):
            raise GroundTruthPoolError(
                "Invalid internal retrieval provenance."
            )

        retrieved_by.append(
            PoolRetrievalRecord(
                algorithm=(
                    result.algorithm
                ),
                rank=chunk.rank,
                score=chunk.score,
            )
        )


def _build_candidate(
    data: Mapping[str, Any],
) -> PoolCandidate:
    """Create one validated candidate from pooled internal data."""
    raw_retrieved_by = data.get(
        "retrieved_by"
    )

    if not isinstance(
        raw_retrieved_by,
        list,
    ):
        raise GroundTruthPoolError(
            "Candidate retrieval provenance "
            "must be a list."
        )

    retrieval_records = tuple(
        sorted(
            (
                record
                for record in raw_retrieved_by
                if isinstance(
                    record,
                    PoolRetrievalRecord,
                )
            ),
            key=lambda record: (
                record.algorithm,
                record.rank,
            ),
        )
    )

    token_count = data.get(
        "token_count"
    )

    normalized_token_count = (
        token_count
        if isinstance(
            token_count,
            int,
        )
        and not isinstance(
            token_count,
            bool,
        )
        else None
    )

    return PoolCandidate(
        chunk_id=str(
            data.get(
                "chunk_id",
                "",
            )
        ),
        source_path=str(
            data.get(
                "source_path",
                "",
            )
        ),
        section_title=str(
            data.get(
                "section_title",
                "",
            )
        ),
        content=str(
            data.get(
                "content",
                "",
            )
        ),
        token_count=(
            normalized_token_count
        ),
        retrieved_by=(
            retrieval_records
        ),
    )


def _blind_order_key(
    *,
    query_id: str,
    chunk_id: str,
    seed: int,
) -> str:
    """Return a deterministic pseudorandom key for blinded ordering."""
    payload = (
        f"{seed}|{query_id}|{chunk_id}"
    )

    return hashlib.sha256(
        payload.encode(
            "utf-8"
        )
    ).hexdigest()


def _validate_result_algorithm(
    result: RetrievalResult,
    *,
    expected_algorithm: str,
) -> None:
    """Ensure registry name and returned algorithm remain consistent."""
    if (
        result.algorithm
        != expected_algorithm
    ):
        raise GroundTruthPoolError(
            "Retriever result algorithm "
            f"'{result.algorithm}' does not match "
            f"registry name '{expected_algorithm}'."
        )


def _mapping_string(
    data: Mapping[str, Any],
    field: str,
) -> str:
    """Read a required non-empty string from a mapping."""
    value = data.get(
        field
    )

    if not isinstance(
        value,
        str,
    ):
        raise GroundTruthPoolError(
            f"{field} must be a string."
        )

    return _require_non_empty_string(
        value,
        field,
    )


def _require_non_empty_string(
    value: str,
    field: str,
) -> str:
    """Normalize one required string."""
    normalized = value.strip()

    if not normalized:
        raise GroundTruthPoolError(
            f"{field} cannot be empty."
        )

    return normalized