"""Test persisted semantic index reuse at retrieval runtime."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.representations.embeddings import EmbeddingConfig
from src.retrieval.semantic_retriever import SemanticRetriever
from src.semantic.faiss_index import FaissIndex
from src.semantic.persistence import (
    build_semantic_index_metadata,
    save_semantic_index_metadata,
)

FloatMatrix = NDArray[np.float32]
FloatVector = NDArray[np.float32]


class FakeEncoder:
    """Provide deterministic embeddings without loading a real model."""

    def __init__(self) -> None:
        self.config = EmbeddingConfig(
            model_name="test-model",
            normalize_embeddings=True,
            query_instruction="",
        )

        self.document_encode_calls = 0
        self.query_encode_calls = 0

    @property
    def model_name(self) -> str:
        """Return the fake embedding model identifier."""
        return self.config.model_name

    @property
    def dimension(self) -> int:
        """Return the fake embedding dimension."""
        return 3

    def encode_documents(
        self,
        texts: list[str],
    ) -> FloatMatrix:
        """Generate deterministic fake document embeddings."""
        self.document_encode_calls += 1

        vectors = [
            (
                [1.0, 0.0, 0.0]
                if index == 0
                else [0.0, 1.0, 0.0]
            )
            for index, _ in enumerate(
                texts
            )
        ]

        return np.asarray(
            vectors,
            dtype=np.float32,
        )

    def encode_query(
        self,
        query: str,
    ) -> FloatVector:
        """Generate one deterministic fake query embedding."""
        self.query_encode_calls += 1

        return np.asarray(
            [1.0, 0.0, 0.0],
            dtype=np.float32,
        )


def _build_corpus() -> list[dict]:
    """Create a small deterministic semantic corpus."""
    return [
        {
            "chunk_id": "chunk-a",
            "content": "FastAPI dependency injection.",
            "source_path": "a.md",
            "section_title": "Dependencies",
            "token_count": 4,
            "version": "0.141.0",
        },
        {
            "chunk_id": "chunk-b",
            "content": "FastAPI CORS configuration.",
            "source_path": "b.md",
            "section_title": "CORS",
            "token_count": 4,
            "version": "0.141.0",
        },
    ]


def _persist_index(
    directory: Path,
    corpus: list[dict],
) -> None:
    """Create compatible persisted semantic artifacts."""
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    embeddings = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    index = FaissIndex(
        dimension=3
    )

    index.build(
        embeddings,
        [
            "chunk-a",
            "chunk-b",
        ],
    )

    index.save(
        directory / "index.faiss",
        directory / "chunk_ids.txt",
    )

    metadata = (
        build_semantic_index_metadata(
            corpus,
            model_name="test-model",
            embedding_dimension=3,
            normalize_embeddings=True,
            faiss_index_type="IndexFlatIP",
            embedding_build_time_ns=123,
            faiss_build_time_ns=456,
        )
    )

    save_semantic_index_metadata(
        metadata,
        directory / "metadata.json",
    )


def test_semantic_retriever_reuses_persisted_index(
    tmp_path: Path,
) -> None:
    """Persisted index must avoid document re-encoding."""
    corpus = _build_corpus()

    semantic_directory = (
        tmp_path
        / "semantic"
    )

    _persist_index(
        semantic_directory,
        corpus,
    )

    encoder = FakeEncoder()

    retriever = SemanticRetriever(
        corpus,
        encoder=encoder,  # type: ignore[arg-type]
        persistence_directory=(
            semantic_directory
        ),
        use_persisted_index=True,
    )

    assert (
        encoder.document_encode_calls
        == 0
    )

    assert (
        retriever.index_source
        == "persisted"
    )

    assert (
        retriever.persistence_status
        == "loaded"
    )

    assert retriever.index_size == 2

    result = retriever.retrieve(
        "dependency injection",
        top_k=1,
    )

    assert (
        encoder.query_encode_calls
        == 1
    )

    assert (
        result.chunks[0].chunk_id
        == "chunk-a"
    )

    assert (
        result.metadata[
            "index_source"
        ]
        == "persisted"
    )

    assert (
        result.metadata[
            "persisted_embedding_build_time_ns"
        ]
        == 123
    )

    assert (
        result.metadata[
            "persisted_faiss_build_time_ns"
        ]
        == 456
    )


def test_semantic_retriever_rebuilds_when_cache_is_incompatible(
    tmp_path: Path,
) -> None:
    """Corpus changes must invalidate the persisted semantic index."""
    persisted_corpus = (
        _build_corpus()
    )

    semantic_directory = (
        tmp_path
        / "semantic"
    )

    _persist_index(
        semantic_directory,
        persisted_corpus,
    )

    current_corpus = (
        _build_corpus()
    )

    current_corpus[0][
        "content"
    ] = (
        "Changed dependency content."
    )

    encoder = FakeEncoder()

    retriever = SemanticRetriever(
        current_corpus,
        encoder=encoder,  # type: ignore[arg-type]
        persistence_directory=(
            semantic_directory
        ),
        use_persisted_index=True,
    )

    assert (
        encoder.document_encode_calls
        == 1
    )

    assert (
        retriever.index_source
        == "built"
    )

    assert (
        retriever.persistence_status
        == "incompatible"
    )

    assert retriever.index_size == 2