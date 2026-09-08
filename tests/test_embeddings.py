from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from src.representations.embeddings import (
    DEFAULT_MODEL_NAME,
    DEFAULT_QUERY_INSTRUCTION,
    EmbeddingConfig,
    EmbeddingEncoder,
    EmbeddingOutputError,
)


class FakeSentenceEncoder:
    """Deterministic embedding model used by unit tests."""

    def __init__(self) -> None:
        self.received_texts: list[str] = []

    def encode(
        self,
        sentences: Sequence[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> Any:
        self.received_texts = list(sentences)

        vectors = []

        for index, _ in enumerate(sentences):
            vectors.append(
                [
                    float(index + 1),
                    float(index + 2),
                    float(index + 3),
                ]
            )

        return np.asarray(
            vectors,
            dtype=np.float32,
        )

    def get_sentence_embedding_dimension(
        self,
    ) -> int:
        return 3


class ZeroVectorEncoder:
    """Embedding model that always produces zero vectors."""

    def encode(
        self,
        sentences: Sequence[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> Any:
        return np.zeros(
            (len(sentences), 3),
            dtype=np.float32,
        )

    def get_sentence_embedding_dimension(
        self,
    ) -> int:
        return 3


class InvalidRowCountEncoder:
    """Embedding model that returns an invalid number of vectors."""

    def encode(
        self,
        sentences: Sequence[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> Any:
        return np.ones(
            (len(sentences) + 1, 3),
            dtype=np.float32,
        )

    def get_sentence_embedding_dimension(
        self,
    ) -> int:
        return 3


def test_default_config_uses_bge_model() -> None:
    config = EmbeddingConfig()

    assert config.model_name == DEFAULT_MODEL_NAME
    assert config.normalize_embeddings is True
    assert config.batch_size == 32


def test_encoder_reports_dimension() -> None:
    encoder = EmbeddingEncoder(
        model=FakeSentenceEncoder()
    )

    assert encoder.dimension == 3


def test_encode_documents_returns_normalized_vectors() -> None:
    encoder = EmbeddingEncoder(
        model=FakeSentenceEncoder()
    )

    embeddings = encoder.encode_documents(
        [
            "FastAPI dependencies",
            "OAuth2 authentication",
        ]
    )

    assert embeddings.shape == (2, 3)
    assert embeddings.dtype == np.float32

    norms = np.linalg.norm(
        embeddings,
        axis=1,
    )

    assert np.allclose(
        norms,
        np.ones(2),
    )


def test_encode_query_applies_bge_instruction() -> None:
    model = FakeSentenceEncoder()

    encoder = EmbeddingEncoder(
        model=model
    )

    vector = encoder.encode_query(
        "How do dependencies work?"
    )

    assert vector.shape == (3,)

    assert model.received_texts == [
        (
            f"{DEFAULT_QUERY_INSTRUCTION}"
            "How do dependencies work?"
        )
    ]


def test_query_instruction_can_be_disabled() -> None:
    model = FakeSentenceEncoder()

    encoder = EmbeddingEncoder(
        config=EmbeddingConfig(
            query_instruction=""
        ),
        model=model,
    )

    encoder.encode_query(
        "FastAPI middleware"
    )

    assert model.received_texts == [
        "FastAPI middleware"
    ]


def test_normalization_can_be_disabled() -> None:
    encoder = EmbeddingEncoder(
        config=EmbeddingConfig(
            normalize_embeddings=False
        ),
        model=FakeSentenceEncoder(),
    )

    embeddings = encoder.encode_documents(
        ["FastAPI"]
    )

    expected = np.asarray(
        [[1.0, 2.0, 3.0]],
        dtype=np.float32,
    )

    assert np.array_equal(
        embeddings,
        expected,
    )


def test_empty_document_collection_returns_empty_matrix() -> None:
    encoder = EmbeddingEncoder(
        model=FakeSentenceEncoder()
    )

    embeddings = encoder.encode_documents([])

    assert embeddings.shape == (0, 3)
    assert embeddings.dtype == np.float32


def test_blank_query_is_rejected() -> None:
    encoder = EmbeddingEncoder(
        model=FakeSentenceEncoder()
    )

    with pytest.raises(
        ValueError,
        match="query cannot be empty",
    ):
        encoder.encode_query("   ")


def test_blank_document_is_rejected() -> None:
    encoder = EmbeddingEncoder(
        model=FakeSentenceEncoder()
    )

    with pytest.raises(
        ValueError,
        match=r"texts\[1\] cannot be empty",
    ):
        encoder.encode_documents(
            [
                "valid document",
                "   ",
            ]
        )


def test_invalid_batch_size_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="batch_size must be greater than zero",
    ):
        EmbeddingConfig(
            batch_size=0
        )


def test_zero_vector_cannot_be_normalized() -> None:
    encoder = EmbeddingEncoder(
        model=ZeroVectorEncoder()
    )

    with pytest.raises(
        EmbeddingOutputError,
        match="Zero-length embedding",
    ):
        encoder.encode_documents(
            ["FastAPI"]
        )


def test_invalid_output_row_count_is_rejected() -> None:
    encoder = EmbeddingEncoder(
        model=InvalidRowCountEncoder()
    )

    with pytest.raises(
        EmbeddingOutputError,
        match="row count",
    ):
        encoder.encode_documents(
            ["FastAPI"]
        )