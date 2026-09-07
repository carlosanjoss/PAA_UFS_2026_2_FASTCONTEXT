from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

FloatMatrix = NDArray[np.float32]
FloatVector = NDArray[np.float32]


class FaissIndexError(RuntimeError):
    """Base exception raised by the FAISS index layer."""


class FaissIndexDependencyError(
    FaissIndexError
):
    """Raised when the FAISS dependency is unavailable."""


@dataclass(frozen=True, slots=True)
class FaissSearchResult:
    """One semantic search result returned by FAISS."""

    chunk_id: str
    score: float
    position: int

    def __post_init__(self) -> None:
        if not self.chunk_id.strip():
            raise ValueError(
                "chunk_id cannot be empty."
            )

        if self.position < 0:
            raise ValueError(
                "position cannot be negative."
            )


class FaissIndex:
    """Manage a flat inner-product FAISS index for chunk embeddings."""

    def __init__(
        self,
        dimension: int,
    ) -> None:
        if dimension <= 0:
            raise ValueError(
                "dimension must be greater than zero."
            )

        self._dimension = dimension
        self._faiss = self._load_faiss()
        self._index = self._create_index()
        self._chunk_ids: list[str] = []

    @property
    def dimension(self) -> int:
        """Return the vector dimension expected by the index."""

        return self._dimension

    @property
    def size(self) -> int:
        """Return the number of indexed vectors."""

        return len(
            self._chunk_ids
        )

    @property
    def is_empty(self) -> bool:
        """Return whether the index contains no vectors."""

        return self.size == 0

    def build(
        self,
        embeddings: FloatMatrix,
        chunk_ids: list[str],
    ) -> None:
        """Build the FAISS index from chunk embeddings."""

        vectors = self._validate_matrix(
            embeddings
        )

        self._validate_chunk_ids(
            chunk_ids,
            expected_count=vectors.shape[0],
        )

        self._index = self._create_index()
        self._chunk_ids = []

        if vectors.shape[0] == 0:
            return

        self._index.add(
            vectors
        )

        self._chunk_ids = list(
            chunk_ids
        )

    def search(
        self,
        query_embedding: FloatVector,
        top_k: int,
    ) -> tuple[FaissSearchResult, ...]:
        """Return the top-k chunk identifiers by inner product."""

        if top_k < 0:
            raise ValueError(
                "top_k cannot be negative."
            )

        if top_k == 0 or self.is_empty:
            return ()

        query = self._validate_query(
            query_embedding
        )

        effective_k = min(
            top_k,
            self.size,
        )

        scores, positions = (
            self._index.search(
                query,
                effective_k,
            )
        )

        results: list[
            FaissSearchResult
        ] = []

        for score, position in zip(
            scores[0],
            positions[0],
            strict=True,
        ):
            index_position = int(
                position
            )

            if index_position < 0:
                continue

            results.append(
                FaissSearchResult(
                    chunk_id=(
                        self._chunk_ids[
                            index_position
                        ]
                    ),
                    score=float(score),
                    position=index_position,
                )
            )

        return tuple(results)

    def save(
        self,
        index_path: str | Path,
        mapping_path: str | Path,
    ) -> None:
        """Persist the FAISS index and chunk identifier mapping."""

        index_file = Path(
            index_path
        )

        mapping_file = Path(
            mapping_path
        )

        index_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        mapping_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._faiss.write_index(
            self._index,
            str(index_file),
        )

        mapping_file.write_text(
            "\n".join(
                self._chunk_ids
            ),
            encoding="utf-8",
        )

    def load(
        self,
        index_path: str | Path,
        mapping_path: str | Path,
    ) -> None:
        """Load a persisted FAISS index and chunk identifier mapping."""

        index_file = Path(
            index_path
        )

        mapping_file = Path(
            mapping_path
        )

        if not index_file.is_file():
            raise FileNotFoundError(
                f"FAISS index file not found: {index_file}"
            )

        if not mapping_file.is_file():
            raise FileNotFoundError(
                f"FAISS mapping file not found: {mapping_file}"
            )

        loaded_index = (
            self._faiss.read_index(
                str(index_file)
            )
        )

        if int(loaded_index.d) != self._dimension:
            raise FaissIndexError(
                "Persisted FAISS index dimension "
                "does not match configured dimension."
            )

        mapping_text = (
            mapping_file.read_text(
                encoding="utf-8"
            )
        )

        chunk_ids = [
            line.strip()
            for line in mapping_text.splitlines()
            if line.strip()
        ]

        vector_count = int(
            loaded_index.ntotal
        )

        if vector_count != len(
            chunk_ids
        ):
            raise FaissIndexError(
                "FAISS vector count does not match "
                "the persisted chunk mapping."
            )

        self._index = loaded_index
        self._chunk_ids = chunk_ids

    def _create_index(
        self,
    ) -> Any:
        """Create the configured flat inner-product index."""

        return self._faiss.IndexFlatIP(
            self._dimension
        )

    def _validate_matrix(
        self,
        embeddings: FloatMatrix,
    ) -> FloatMatrix:
        """Validate an embedding matrix before indexing."""

        vectors = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        if vectors.ndim != 2:
            raise ValueError(
                "embeddings must be a two-dimensional matrix."
            )

        if (
            vectors.shape[1]
            != self._dimension
        ):
            raise ValueError(
                "Embedding dimension does not match "
                "the FAISS index dimension."
            )

        if not np.all(
            np.isfinite(
                vectors
            )
        ):
            raise ValueError(
                "embeddings contain non-finite values."
            )

        return cast(
            FloatMatrix,
            np.ascontiguousarray(
                vectors,
                dtype=np.float32,
            ),
        )

    def _validate_query(
        self,
        query_embedding: FloatVector,
    ) -> FloatMatrix:
        """Validate and reshape one query embedding."""

        query = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        if query.ndim != 1:
            raise ValueError(
                "query_embedding must be one-dimensional."
            )

        if query.shape[0] != self._dimension:
            raise ValueError(
                "Query embedding dimension does not "
                "match the FAISS index dimension."
            )

        if not np.all(
            np.isfinite(
                query
            )
        ):
            raise ValueError(
                "query_embedding contains non-finite values."
            )

        matrix = query.reshape(
            1,
            -1,
        )

        return cast(
            FloatMatrix,
            np.ascontiguousarray(
                matrix,
                dtype=np.float32,
            ),
        )

    @staticmethod
    def _validate_chunk_ids(
        chunk_ids: list[str],
        *,
        expected_count: int,
    ) -> None:
        """Validate the chunk identifier mapping."""

        if len(chunk_ids) != expected_count:
            raise ValueError(
                "chunk_ids count must match "
                "the number of embeddings."
            )

        normalized_ids = [
            chunk_id.strip()
            for chunk_id in chunk_ids
        ]

        if any(
            not chunk_id
            for chunk_id in normalized_ids
        ):
            raise ValueError(
                "chunk_ids cannot contain empty identifiers."
            )

        if (
            len(set(normalized_ids))
            != len(normalized_ids)
        ):
            raise ValueError(
                "chunk_ids must be unique."
            )

    @staticmethod
    def _load_faiss() -> Any:
        """Import FAISS without requiring it at module import time."""

        try:
            return import_module(
                "faiss"
            )
        except ModuleNotFoundError as exc:
            raise FaissIndexDependencyError(
                "faiss-cpu is required "
                "for semantic search."
            ) from exc