"""Information-retrieval quality metrics for FastContext experiments."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetrievalQualityMetrics:
    """Store retrieval-quality metrics for one query at one k value."""

    k: int
    precision: float
    recall: float
    reciprocal_rank: float
    hit_rate: float
    relevant_retrieved: int
    relevant_total: int
    retrieved_count: int


def precision_at_k(
    retrieved_chunk_ids: Sequence[str],
    relevant_chunk_ids: Sequence[str],
    k: int,
) -> float:
    """Calculate Precision@k.

    Precision@k is the fraction of the first k ranking positions
    occupied by relevant chunks.
    """
    _validate_k(
        k
    )

    retrieved = _validate_retrieved_ids(
        retrieved_chunk_ids
    )

    relevant = _normalize_relevant_ids(
        relevant_chunk_ids
    )

    top_k = retrieved[
        :k
    ]

    relevant_retrieved = sum(
        chunk_id in relevant
        for chunk_id in top_k
    )

    return (
        relevant_retrieved
        / k
    )


def recall_at_k(
    retrieved_chunk_ids: Sequence[str],
    relevant_chunk_ids: Sequence[str],
    k: int,
) -> float:
    """Calculate Recall@k.

    Recall@k is the fraction of all known relevant chunks
    retrieved within the first k ranking positions.
    """
    _validate_k(
        k
    )

    retrieved = _validate_retrieved_ids(
        retrieved_chunk_ids
    )

    relevant = _normalize_relevant_ids(
        relevant_chunk_ids
    )

    if not relevant:
        return 0.0

    top_k = retrieved[
        :k
    ]

    relevant_retrieved = sum(
        chunk_id in relevant
        for chunk_id in top_k
    )

    return (
        relevant_retrieved
        / len(relevant)
    )


def reciprocal_rank(
    retrieved_chunk_ids: Sequence[str],
    relevant_chunk_ids: Sequence[str],
    k: int | None = None,
) -> float:
    """Return the reciprocal rank of the first relevant chunk.

    When k is provided, only the first k positions are considered.
    """
    if k is not None:
        _validate_k(
            k
        )

    retrieved = _validate_retrieved_ids(
        retrieved_chunk_ids
    )

    relevant = _normalize_relevant_ids(
        relevant_chunk_ids
    )

    if not relevant:
        return 0.0

    ranking = (
        retrieved
        if k is None
        else retrieved[:k]
    )

    for rank, chunk_id in enumerate(
        ranking,
        start=1,
    ):
        if chunk_id in relevant:
            return (
                1.0
                / rank
            )

    return 0.0


def hit_rate_at_k(
    retrieved_chunk_ids: Sequence[str],
    relevant_chunk_ids: Sequence[str],
    k: int,
) -> float:
    """Calculate Hit Rate@k.

    The result is 1.0 when at least one relevant chunk occurs
    in the first k positions and 0.0 otherwise.
    """
    _validate_k(
        k
    )

    retrieved = _validate_retrieved_ids(
        retrieved_chunk_ids
    )

    relevant = _normalize_relevant_ids(
        relevant_chunk_ids
    )

    if not relevant:
        return 0.0

    return float(
        any(
            chunk_id in relevant
            for chunk_id in retrieved[
                :k
            ]
        )
    )


def evaluate_retrieval(
    retrieved_chunk_ids: Sequence[str],
    relevant_chunk_ids: Sequence[str],
    k: int,
) -> RetrievalQualityMetrics:
    """Calculate all FastContext retrieval metrics for one query."""
    _validate_k(
        k
    )

    retrieved = _validate_retrieved_ids(
        retrieved_chunk_ids
    )

    relevant = _normalize_relevant_ids(
        relevant_chunk_ids
    )

    top_k = retrieved[
        :k
    ]

    relevant_retrieved = sum(
        chunk_id in relevant
        for chunk_id in top_k
    )

    precision = (
        relevant_retrieved
        / k
    )

    recall = (
        relevant_retrieved
        / len(relevant)
        if relevant
        else 0.0
    )

    reciprocal = (
        _reciprocal_rank_from_validated(
            top_k,
            relevant,
        )
    )

    hit_rate = float(
        relevant_retrieved > 0
    )

    return RetrievalQualityMetrics(
        k=k,
        precision=precision,
        recall=recall,
        reciprocal_rank=reciprocal,
        hit_rate=hit_rate,
        relevant_retrieved=(
            relevant_retrieved
        ),
        relevant_total=len(
            relevant
        ),
        retrieved_count=len(
            top_k
        ),
    )


def _reciprocal_rank_from_validated(
    ranking: Sequence[str],
    relevant: set[str],
) -> float:
    """Calculate reciprocal rank from already validated inputs."""
    for rank, chunk_id in enumerate(
        ranking,
        start=1,
    ):
        if chunk_id in relevant:
            return (
                1.0
                / rank
            )

    return 0.0


def _validate_retrieved_ids(
    chunk_ids: Sequence[str],
) -> tuple[str, ...]:
    """Validate and normalize a ranked retrieval sequence."""
    normalized = tuple(
        _normalize_chunk_id(
            chunk_id
        )
        for chunk_id in chunk_ids
    )

    if (
        len(normalized)
        != len(
            set(normalized)
        )
    ):
        raise ValueError(
            "Retrieved chunk IDs must be unique."
        )

    return normalized


def _normalize_relevant_ids(
    chunk_ids: Sequence[str],
) -> set[str]:
    """Normalize ground-truth relevant chunk identifiers."""
    normalized = [
        _normalize_chunk_id(
            chunk_id
        )
        for chunk_id in chunk_ids
    ]

    if (
        len(normalized)
        != len(
            set(normalized)
        )
    ):
        raise ValueError(
            "Relevant chunk IDs must be unique."
        )

    return set(
        normalized
    )


def _normalize_chunk_id(
    chunk_id: str,
) -> str:
    """Validate one chunk identifier."""
    if not isinstance(
        chunk_id,
        str,
    ):
        raise TypeError(
            "Chunk IDs must be strings."
        )

    normalized = (
        chunk_id.strip()
    )

    if not normalized:
        raise ValueError(
            "Chunk IDs cannot be empty."
        )

    return normalized


def _validate_k(
    k: int,
) -> None:
    """Validate a Top-k value."""
    if isinstance(
        k,
        bool,
    ) or not isinstance(
        k,
        int,
    ):
        raise TypeError(
            "k must be an integer."
        )

    if k <= 0:
        raise ValueError(
            "k must be greater than zero."
        )