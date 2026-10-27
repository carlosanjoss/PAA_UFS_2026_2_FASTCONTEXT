
"""Tests for AV2 hybrid semantic and literal retrieval."""

from __future__ import annotations

from typing import Any

import pytest

from src.retrieval.base import Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.models import (
    RetrievedChunk,
    RetrievalMetrics,
    RetrievalResult,
)


QUESTION = "Explain OAuth2"


class StubRetriever(Retriever):
    """Controlled retriever for testing deterministic fusion."""

    def __init__(
        self,
        algorithm: str,
        chunk_ids: tuple[str, ...],
        *,
        content_overrides: dict[str, str] | None = None,
    ) -> None:
        self._algorithm = algorithm
        self._chunk_ids = chunk_ids
        self._content_overrides = content_overrides or {}
        self.calls = 0
        self.last_top_k: int | None = None

    @property
    def name(self) -> str:
        return self._algorithm

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> RetrievalResult:
        self.calls += 1
        self.last_top_k = top_k

        selected = self._chunk_ids[:top_k]

        chunks = tuple(
            RetrievedChunk(
                chunk_id=chunk_id,
                content=self._content_overrides.get(
                    chunk_id,
                    f"Content for {chunk_id}",
                ),
                source_path=f"{chunk_id}.md",
                section_title=f"Section {chunk_id}",
                score=float(len(selected) - index + 1),
                rank=index,
            )
            for index, chunk_id in enumerate(
                selected,
                start=1,
            )
        )

        return RetrievalResult(
            query=query,
            algorithm=self.name,
            top_k=top_k,
            chunks=chunks,
            metrics=RetrievalMetrics(
                retrieval_time_ns=100,
                candidates_found=len(chunks),
                comparisons=10,
            ),
        )


@pytest.fixture
def retrievers() -> tuple[StubRetriever, StubRetriever]:
    semantic = StubRetriever(
        "semantic",
        ("B", "A", "C"),
    )

    literal = StubRetriever(
        "literal",
        ("A", "D", "B"),
    )

    return semantic, literal


def test_hybrid_contract(
    retrievers: tuple[StubRetriever, StubRetriever],
) -> None:
    semantic, literal = retrievers

    hybrid = HybridRetriever(
        semantic,
        literal,
    )

    assert isinstance(hybrid, Retriever)
    assert hybrid.name == "hybrid"


def test_rrf_ranking(
    retrievers: tuple[StubRetriever, StubRetriever],
) -> None:
    semantic, literal = retrievers

    hybrid = HybridRetriever(
        semantic,
        literal,
        candidate_depth=3,
    )

    result = hybrid.retrieve(
        QUESTION,
        top_k=3,
    )

    assert result.algorithm == "hybrid"
    assert result.top_k == 3

    assert [
        chunk.chunk_id for chunk in result.chunks
    ] == ["A", "B", "D"]

    assert [
        chunk.rank for chunk in result.chunks
    ] == [1, 2, 3]


def test_rrf_score_is_sum_of_rank_contributions(
    retrievers: tuple[StubRetriever, StubRetriever],
) -> None:
    semantic, literal = retrievers

    hybrid = HybridRetriever(
        semantic,
        literal,
        candidate_depth=3,
    )

    result = hybrid.retrieve(QUESTION, top_k=3)

    first = result.chunks[0]

    assert first.chunk_id == "A"
    assert first.score == pytest.approx(
        1 / 62 + 1 / 61
    )

    assert first.metadata is not None
    assert first.metadata["semantic_rank"] == 2
    assert first.metadata["literal_rank"] == 1


def test_union_deduplicates_chunks(
    retrievers: tuple[StubRetriever, StubRetriever],
) -> None:
    semantic, literal = retrievers

    hybrid = HybridRetriever(
        semantic,
        literal,
        candidate_depth=3,
    )

    result = hybrid.retrieve(QUESTION, top_k=5)

    ids = [
        chunk.chunk_id for chunk in result.chunks
    ]

    assert len(ids) == len(set(ids))
    assert result.metrics.candidates_found == 4


def test_candidate_depth_is_forwarded() -> None:
    semantic = StubRetriever(
        "semantic",
        ("A", "B", "C", "D"),
    )

    literal = StubRetriever(
        "literal",
        ("E", "F", "G", "H"),
    )

    hybrid = HybridRetriever(
        semantic,
        literal,
        candidate_depth=3,
    )

    hybrid.retrieve(QUESTION, top_k=2)

    assert semantic.last_top_k == 3
    assert literal.last_top_k == 3


def test_requested_top_k_can_exceed_candidate_depth() -> None:
    semantic = StubRetriever(
        "semantic",
        ("A", "B", "C", "D"),
    )

    literal = StubRetriever(
        "literal",
        ("E", "F", "G", "H"),
    )

    hybrid = HybridRetriever(
        semantic,
        literal,
        candidate_depth=2,
    )

    result = hybrid.retrieve(QUESTION, top_k=4)

    assert semantic.last_top_k == 4
    assert literal.last_top_k == 4
    assert len(result.chunks) == 4


def test_empty_literal_falls_back_to_semantic() -> None:
    semantic = StubRetriever(
        "semantic",
        ("A", "B", "C"),
    )

    literal = StubRetriever(
        "literal",
        (),
    )

    hybrid = HybridRetriever(semantic, literal)

    result = hybrid.retrieve(QUESTION)

    assert [
        chunk.chunk_id for chunk in result.chunks
    ] == ["A", "B", "C"]


def test_empty_semantic_uses_literal() -> None:
    semantic = StubRetriever(
        "semantic",
        (),
    )

    literal = StubRetriever(
        "literal",
        ("A", "B"),
    )

    hybrid = HybridRetriever(semantic, literal)

    result = hybrid.retrieve(QUESTION)

    assert [
        chunk.chunk_id for chunk in result.chunks
    ] == ["A", "B"]


def test_both_empty_returns_no_chunks() -> None:
    hybrid = HybridRetriever(
        StubRetriever("semantic", ()),
        StubRetriever("literal", ()),
    )

    result = hybrid.retrieve(QUESTION)

    assert result.chunks == ()
    assert result.metrics.candidates_found == 0


def test_tie_breaking_by_chunk_id() -> None:
    hybrid = HybridRetriever(
        StubRetriever("semantic", ("Z",)),
        StubRetriever("literal", ("A",)),
    )

    result = hybrid.retrieve(QUESTION)

    assert [
        chunk.chunk_id for chunk in result.chunks
    ] == ["A", "Z"]


def test_zero_top_k_does_not_call_sources() -> None:
    semantic = StubRetriever(
        "semantic",
        ("A",),
    )

    literal = StubRetriever(
        "literal",
        ("B",),
    )

    hybrid = HybridRetriever(
        semantic,
        literal,
    )

    result = hybrid.retrieve(
        QUESTION,
        top_k=0,
    )

    assert result.chunks == ()
    assert semantic.calls == 0
    assert literal.calls == 0


def test_retrieval_metrics() -> None:
    hybrid = HybridRetriever(
        StubRetriever("semantic", ("A",)),
        StubRetriever("literal", ("A", "B")),
    )

    result = hybrid.retrieve(QUESTION)

    assert result.metrics.retrieval_time_ns >= 0
    assert result.metrics.sorting_time_ns is not None
    assert result.metrics.sorting_time_ns >= 0
    assert result.metrics.candidates_found == 2

    assert result.metadata is not None
    assert result.metadata["rrf_k"] == 60
    assert result.metadata["fusion"] == "reciprocal_rank_fusion"
    assert result.metadata["semantic_retrieval_time_ns"] == 100
    assert result.metadata["literal_retrieval_time_ns"] == 100


def test_reject_invalid_rrf_k() -> None:
    with pytest.raises(ValueError):
        HybridRetriever(
            StubRetriever("semantic", ()),
            StubRetriever("literal", ()),
            rrf_k=0,
        )


def test_reject_invalid_candidate_depth() -> None:
    with pytest.raises(ValueError):
        HybridRetriever(
            StubRetriever("semantic", ()),
            StubRetriever("literal", ()),
            candidate_depth=0,
        )


def test_reject_wrong_source_types() -> None:
    with pytest.raises(ValueError):
        HybridRetriever(
            StubRetriever("literal", ()),
            StubRetriever("semantic", ()),
        )


def test_reject_inconsistent_chunk_content() -> None:
    semantic = StubRetriever(
        "semantic",
        ("A",),
    )

    literal = StubRetriever(
        "literal",
        ("A",),
        content_overrides={
            "A": "Different content",
        },
    )

    hybrid = HybridRetriever(
        semantic,
        literal,
    )

    with pytest.raises(
        ValueError,
        match="disagree",
    ):
        hybrid.retrieve(QUESTION)


def test_reject_empty_query() -> None:
    hybrid = HybridRetriever(
        StubRetriever("semantic", ()),
        StubRetriever("literal", ()),
    )

    with pytest.raises(ValueError):
        hybrid.retrieve("   ")


def test_reject_negative_top_k() -> None:
    hybrid = HybridRetriever(
        StubRetriever("semantic", ()),
        StubRetriever("literal", ()),
    )

    with pytest.raises(ValueError):
        hybrid.retrieve(QUESTION, top_k=-1)
