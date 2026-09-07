from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from src.representations.embeddings import (
    EmbeddingEncoder,
)
from src.retrieval.semantic_retriever import (
    SemanticRetriever,
)


class FakeSentenceEncoder:
    """Deterministic semantic encoder used by retriever tests."""

    def encode(
        self,
        sentences: Sequence[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> Any:
        vectors: list[
            list[float]
        ] = []

        for sentence in sentences:
            normalized = (
                sentence.lower()
            )

            if (
                "depend" in normalized
                or "reuse" in normalized
                or "application logic"
                in normalized
            ):
                vector = [
                    1.0,
                    0.0,
                    0.0,
                ]
            elif (
                "oauth" in normalized
                or "security" in normalized
                or "authenticat" in normalized
            ):
                vector = [
                    0.0,
                    1.0,
                    0.0,
                ]
            elif (
                "cors" in normalized
                or "origin" in normalized
                or "middleware" in normalized
            ):
                vector = [
                    0.0,
                    0.0,
                    1.0,
                ]
            else:
                vector = [
                    1.0,
                    1.0,
                    1.0,
                ]

            vectors.append(
                vector
            )

        return np.asarray(
            vectors,
            dtype=np.float32,
        )

    def get_embedding_dimension(
        self,
    ) -> int:
        return 3


class IdenticalSentenceEncoder:
    """Encoder that maps every input to the same vector."""

    def encode(
        self,
        sentences: Sequence[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> Any:
        return np.asarray(
            [
                [
                    1.0,
                    0.0,
                    0.0,
                ]
                for _ in sentences
            ],
            dtype=np.float32,
        )

    def get_embedding_dimension(
        self,
    ) -> int:
        return 3


def build_encoder() -> EmbeddingEncoder:
    """Create a deterministic embedding encoder."""

    return EmbeddingEncoder(
        model=FakeSentenceEncoder()
    )


def build_corpus() -> list[
    dict[str, Any]
]:
    """Create the semantic retrieval test corpus."""

    return [
        {
            "chunk_id": "dependencies",
            "content": (
                "Use Depends to declare reusable "
                "dependencies in path operations."
            ),
            "source_path": (
                "docs/dependencies.md"
            ),
            "section_title": (
                "Dependencies"
            ),
            "token_count": 10,
            "metadata": {
                "version": "0.141.0"
            },
        },
        {
            "chunk_id": "security",
            "content": (
                "OAuth2 bearer tokens can be "
                "used for authentication."
            ),
            "source_path": (
                "docs/security.md"
            ),
            "section_title": (
                "Security"
            ),
            "token_count": 9,
        },
        {
            "chunk_id": "cors",
            "content": (
                "CORS middleware controls "
                "access from different origins."
            ),
            "source_path": (
                "docs/cors.md"
            ),
            "section_title": (
                "CORS"
            ),
            "token_count": 8,
        },
    ]


def test_name_is_semantic() -> None:
    retriever = SemanticRetriever(
        build_corpus(),
        encoder=build_encoder(),
    )

    assert retriever.name == "semantic"


def test_index_contains_all_corpus_chunks() -> None:
    retriever = SemanticRetriever(
        build_corpus(),
        encoder=build_encoder(),
    )

    assert retriever.corpus_size == 3
    assert retriever.index_size == 3
    assert retriever.embedding_dimension == 3


def test_semantic_query_returns_relevant_chunk_first() -> None:
    retriever = SemanticRetriever(
        build_corpus(),
        encoder=build_encoder(),
    )

    result = retriever.retrieve(
        (
            "How can I reuse application "
            "logic across endpoints?"
        ),
        top_k=3,
    )

    assert result.algorithm == "semantic"
    assert result.top_k == 3
    assert len(result.chunks) == 3

    assert (
        result.chunks[0].chunk_id
        == "dependencies"
    )

    assert result.chunks[0].rank == 1


def test_result_preserves_chunk_metadata() -> None:
    retriever = SemanticRetriever(
        build_corpus(),
        encoder=build_encoder(),
    )

    result = retriever.retrieve(
        "How do dependencies work?",
        top_k=1,
    )

    chunk = result.chunks[0]

    assert chunk.chunk_id == "dependencies"
    assert (
        chunk.source_path
        == "docs/dependencies.md"
    )
    assert (
        chunk.section_title
        == "Dependencies"
    )
    assert chunk.token_count == 10
    assert chunk.metadata == {
        "version": "0.141.0"
    }


def test_semantic_metrics_are_reported() -> None:
    retriever = SemanticRetriever(
        build_corpus(),
        encoder=build_encoder(),
    )

    result = retriever.retrieve(
        "OAuth2 authentication",
        top_k=2,
    )

    assert (
        result.metrics.retrieval_time_ns
        >= 0
    )

    assert (
        result.metrics.index_build_time_ns
        is not None
    )

    assert (
        result.metrics.index_build_time_ns
        >= 0
    )

    assert (
        result.metrics.sorting_time_ns
        is not None
    )

    assert (
        result.metrics.sorting_time_ns
        >= 0
    )

    assert (
        result.metrics.comparisons
        is None
    )

    assert (
        result.metrics.chunks_scored
        == 3
    )

    assert (
        result.metrics.candidates_found
        == 2
    )


def test_metadata_contains_semantic_timings() -> None:
    retriever = SemanticRetriever(
        build_corpus(),
        encoder=build_encoder(),
    )

    result = retriever.retrieve(
        "OAuth2 authentication",
        top_k=1,
    )

    assert result.metadata is not None

    assert (
        result.metadata[
            "representation"
        ]
        == "bge_embeddings"
    )

    assert (
        result.metadata[
            "candidate_strategy"
        ]
        == "faiss_index_flat_ip"
    )

    assert (
        result.metadata[
            "embedding_build_time_ns"
        ]
        >= 0
    )

    assert (
        result.metadata[
            "faiss_build_time_ns"
        ]
        >= 0
    )

    assert (
        result.metadata[
            "query_embedding_time_ns"
        ]
        >= 0
    )

    assert (
        result.metadata[
            "faiss_search_time_ns"
        ]
        >= 0
    )


def test_top_k_larger_than_corpus_is_supported() -> None:
    retriever = SemanticRetriever(
        build_corpus(),
        encoder=build_encoder(),
    )

    result = retriever.retrieve(
        "FastAPI dependencies",
        top_k=10,
    )

    assert len(
        result.chunks
    ) == 3


def test_zero_top_k_returns_empty_result() -> None:
    retriever = SemanticRetriever(
        build_corpus(),
        encoder=build_encoder(),
    )

    result = retriever.retrieve(
        "FastAPI dependencies",
        top_k=0,
    )

    assert result.chunks == ()
    assert (
        result.metrics.chunks_scored
        == 0
    )
    assert (
        result.metrics.candidates_found
        == 0
    )


def test_empty_corpus_returns_empty_result() -> None:
    retriever = SemanticRetriever(
        [],
        encoder=build_encoder(),
    )

    result = retriever.retrieve(
        "FastAPI dependencies",
        top_k=5,
    )

    assert retriever.index_size == 0
    assert result.chunks == ()


def test_duplicate_chunk_ids_are_rejected() -> None:
    corpus = build_corpus()

    corpus.append(
        {
            "chunk_id": "dependencies",
            "content": (
                "Duplicate dependencies chunk."
            ),
        }
    )

    with pytest.raises(
        ValueError,
        match="must be unique",
    ):
        SemanticRetriever(
            corpus,
            encoder=build_encoder(),
        )


def test_empty_chunk_content_is_rejected() -> None:
    corpus = [
        {
            "chunk_id": "empty",
            "content": "   ",
        }
    ]

    with pytest.raises(
        ValueError,
        match="non-empty content",
    ):
        SemanticRetriever(
            corpus,
            encoder=build_encoder(),
        )


def test_equal_scores_use_chunk_id_tie_break() -> None:
    encoder = EmbeddingEncoder(
        model=(
            IdenticalSentenceEncoder()
        )
    )

    corpus = [
        {
            "chunk_id": "chunk_b",
            "content": "Document B",
        },
        {
            "chunk_id": "chunk_a",
            "content": "Document A",
        },
    ]

    retriever = SemanticRetriever(
        corpus,
        encoder=encoder,
    )

    result = retriever.retrieve(
        "Any query",
        top_k=2,
    )

    assert [
        chunk.chunk_id
        for chunk in result.chunks
    ] == [
        "chunk_a",
        "chunk_b",
    ]