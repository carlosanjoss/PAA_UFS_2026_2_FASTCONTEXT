from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEFAULT_QUERY_INSTRUCTION = (
    "Represent this sentence for searching relevant passages: "
)

FloatMatrix = NDArray[np.float32]
FloatVector = NDArray[np.float32]


class EmbeddingError(RuntimeError):
    """Base exception raised by the embedding representation layer."""


class EmbeddingDependencyError(EmbeddingError):
    """Raised when the sentence-transformers dependency is unavailable."""


class EmbeddingOutputError(EmbeddingError):
    """Raised when the embedding model returns invalid output."""


class SentenceEncoder(Protocol):
    """Minimal interface required from a sentence embedding model."""

    def encode(
        self,
        sentences: Sequence[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> Any:
        """Encode a sequence of texts."""


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    """Configuration for the semantic embedding representation."""

    model_name: str = DEFAULT_MODEL_NAME
    normalize_embeddings: bool = True
    batch_size: int = 32
    device: str | None = None
    query_instruction: str = DEFAULT_QUERY_INSTRUCTION

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError(
                "model_name cannot be empty."
            )

        if self.batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero."
            )


class EmbeddingEncoder:
    """Generate document and query embeddings for semantic retrieval."""

    def __init__(
        self,
        config: EmbeddingConfig | None = None,
        *,
        model: SentenceEncoder | None = None,
    ) -> None:
        self._config = (
            config
            if config is not None
            else EmbeddingConfig()
        )

        self._model = (
            model
            if model is not None
            else self._load_model()
        )

    @property
    def config(self) -> EmbeddingConfig:
        """Return the embedding configuration."""

        return self._config

    @property
    def model_name(self) -> str:
        """Return the configured model identifier."""

        return self._config.model_name

    @property
    def dimension(self) -> int:
        """Return the model embedding dimension."""

        dimension = self._read_embedding_dimension()

        if dimension <= 0:
            raise EmbeddingOutputError(
                "The embedding model did not report "
                "a valid dimension."
            )

        return dimension

    def encode_documents(
        self,
        texts: Sequence[str],
    ) -> FloatMatrix:
        """Encode corpus documents or chunks into dense vectors."""

        normalized_texts = self._validate_texts(
            texts
        )

        if not normalized_texts:
            empty_matrix = np.empty(
                (
                    0,
                    self.dimension,
                ),
                dtype=np.float32,
            )

            return cast(
                FloatMatrix,
                empty_matrix,
            )

        return self._encode(
            normalized_texts
        )

    def encode_query(
        self,
        query: str,
    ) -> FloatVector:
        """Encode one retrieval query into a dense vector."""

        normalized_query = self._validate_text(
            query,
            field_name="query",
        )

        query_instruction = (
            self._config.query_instruction
        )

        if query_instruction:
            prepared_query = (
                f"{query_instruction}"
                f"{normalized_query}"
            )
        else:
            prepared_query = normalized_query

        embeddings = self._encode(
            [prepared_query]
        )

        vector = embeddings[0]

        return cast(
            FloatVector,
            vector,
        )

    def _encode(
        self,
        texts: Sequence[str],
    ) -> FloatMatrix:
        """Run the model and validate its output matrix."""

        raw_embeddings = self._model.encode(
            texts,
            batch_size=self._config.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )

        raw_array = np.asarray(
            raw_embeddings,
            dtype=np.float32,
        )

        embeddings = cast(
            FloatMatrix,
            raw_array,
        )

        if embeddings.ndim == 1:
            reshaped = embeddings.reshape(
                1,
                -1,
            )

            embeddings = cast(
                FloatMatrix,
                reshaped,
            )

        if embeddings.ndim != 2:
            raise EmbeddingOutputError(
                "Embedding output must be "
                "a two-dimensional matrix."
            )

        if embeddings.shape[0] != len(texts):
            raise EmbeddingOutputError(
                "Embedding output row count does "
                "not match input text count."
            )

        if embeddings.shape[1] <= 0:
            raise EmbeddingOutputError(
                "Embedding output must contain "
                "at least one dimension."
            )

        if not np.all(
            np.isfinite(
                embeddings
            )
        ):
            raise EmbeddingOutputError(
                "Embedding output contains "
                "non-finite values."
            )

        if self._config.normalize_embeddings:
            embeddings = self._normalize_rows(
                embeddings
            )

        float_embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        return cast(
            FloatMatrix,
            float_embeddings,
        )

    @staticmethod
    def _normalize_rows(
        embeddings: FloatMatrix,
    ) -> FloatMatrix:
        """Apply L2 normalization independently to each embedding."""

        norms = np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True,
        )

        if np.any(
            norms == 0
        ):
            raise EmbeddingOutputError(
                "Zero-length embedding "
                "cannot be normalized."
            )

        normalized = (
            embeddings
            / norms
        )

        normalized_array = np.asarray(
            normalized,
            dtype=np.float32,
        )

        return cast(
            FloatMatrix,
            normalized_array,
        )

    @classmethod
    def _validate_texts(
        cls,
        texts: Sequence[str],
    ) -> list[str]:
        """Validate and normalize document text inputs."""

        return [
            cls._validate_text(
                text,
                field_name=(
                    f"texts[{index}]"
                ),
            )
            for index, text
            in enumerate(texts)
        ]

    @staticmethod
    def _validate_text(
        text: str,
        *,
        field_name: str,
    ) -> str:
        """Validate one embedding input string."""

        if not isinstance(
            text,
            str,
        ):
            raise TypeError(
                f"{field_name} "
                "must be a string."
            )

        normalized_text = (
            text.strip()
        )

        if not normalized_text:
            raise ValueError(
                f"{field_name} "
                "cannot be empty."
            )

        return normalized_text

    def _read_embedding_dimension(
        self,
    ) -> int:
        """Read the embedding dimension across model API versions."""

        new_method = getattr(
            self._model,
            "get_embedding_dimension",
            None,
        )

        if callable(new_method):
            dimension = new_method()

            if dimension is None:
                raise EmbeddingOutputError(
                    "The embedding model did not report "
                    "a valid dimension."
                )

            return int(dimension)

        legacy_method = getattr(
            self._model,
            "get_sentence_embedding_dimension",
            None,
        )

        if callable(legacy_method):
            dimension = legacy_method()

            if dimension is None:
                raise EmbeddingOutputError(
                    "The embedding model did not report "
                    "a valid dimension."
                )

            return int(dimension)

        raise EmbeddingOutputError(
            "The embedding model does not expose "
            "an embedding dimension method."
        )

    def _load_model(
        self,
    ) -> SentenceEncoder:
        """Load SentenceTransformer without importing it at module import time."""

        try:
            module = import_module(
                "sentence_transformers"
            )
        except ModuleNotFoundError as exc:
            raise EmbeddingDependencyError(
                "sentence-transformers is required "
                "to generate embeddings."
            ) from exc

        model_class = getattr(
            module,
            "SentenceTransformer",
            None,
        )

        if model_class is None:
            raise EmbeddingDependencyError(
                "sentence-transformers does not "
                "expose SentenceTransformer."
            )

        model = model_class(
            self._config.model_name,
            device=self._config.device,
        )

        return cast(
            SentenceEncoder,
            model,
        )