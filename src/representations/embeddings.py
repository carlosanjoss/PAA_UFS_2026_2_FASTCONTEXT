"""
src/representations/embeddings.py
Geracao e normalizacao L2 de embeddings densos com BAAI/bge-small-en-v1.5 para PAA.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, List, Sequence, cast
import numpy as np
from numpy.typing import NDArray

FloatMatrix = NDArray[np.float32]
FloatVector = NDArray[np.float32]

DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEFAULT_EMBEDDING_DIM = 384


class EmbeddingDependencyError(RuntimeError):
    """Lancada quando sentence-transformers ou dependencias de ML nao estao instaladas."""


class EmbeddingGenerator:
    """
    Gera embeddings densos normalizados utilizando SentenceTransformers.
    Emprega BAAI/bge-small-en-v1.5 como modelo padrao do projeto FastContext.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        normalize: bool = True,
    ) -> None:
        self.model_name = model_name
        self.normalize = normalize
        self._model = self._load_model()
        self._dimension = int(
            self._model.get_sentence_embedding_dimension()
            if hasattr(self._model, "get_sentence_embedding_dimension")
            else DEFAULT_EMBEDDING_DIM
        )

    @property
    def dimension(self) -> int:
        """Retorna a dimensao esperada dos vetores densos."""
        return self._dimension

    def encode_corpus(self, texts: Sequence[str]) -> FloatMatrix:
        """
        Converte uma lista de textos do corpus em uma matriz de embeddings (N x D).
        """
        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)

        raw_embeddings = self._model.encode(
            list(texts),
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        matrix = np.asarray(raw_embeddings, dtype=np.float32)
        if self.normalize:
            matrix = self._apply_l2_normalization(matrix)

        return cast(FloatMatrix, np.ascontiguousarray(matrix, dtype=np.float32))

    def encode_query(self, query: str) -> FloatVector:
        """
        Converte uma consulta de busca textual em um vetor denso 1D (D,).
        """
        clean_query = query.strip()
        if not clean_query:
            return np.zeros(self._dimension, dtype=np.float32)

        raw_embedding = self._model.encode(
            clean_query,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        vector = np.asarray(raw_embedding, dtype=np.float32).flatten()
        if self.normalize:
            vector = self._apply_l2_normalization_vector(vector)

        return cast(FloatVector, np.ascontiguousarray(vector, dtype=np.float32))

    @staticmethod
    def _apply_l2_normalization(matrix: FloatMatrix) -> FloatMatrix:
        """Aplica normalizacao Euclidiana L2 linha a linha."""
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return matrix / norms

    @staticmethod
    def _apply_l2_normalization_vector(vector: FloatVector) -> FloatVector:
        """Aplica normalizacao Euclidiana L2 a um vetor unidimensional."""
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return vector
        return vector / norm

    def _load_model(self) -> Any:
        """Importa a biblioteca sentence_transformers de forma defensiva."""
        try:
            st_module = import_module("sentence_transformers")
            sentence_transformer_cls = getattr(st_module, "SentenceTransformer")
            return sentence_transformer_cls(self.model_name)
        except (ModuleNotFoundError, ImportError) as exc:
            raise EmbeddingDependencyError(
                "A biblioteca sentence-transformers e necessaria para busca semantica. "
                "Instale-a via: pip install sentence-transformers"
            ) from exc