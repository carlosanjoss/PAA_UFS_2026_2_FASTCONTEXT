"""Persist and validate semantic index metadata."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

SEMANTIC_INDEX_SCHEMA_VERSION = 1


class SemanticPersistenceError(RuntimeError):
    """Base error for semantic index persistence failures."""


class SemanticIndexCompatibilityError(
    SemanticPersistenceError
):
    """Raised when persisted artifacts do not match the current corpus."""


@dataclass(frozen=True, slots=True)
class SemanticIndexMetadata:
    """Describe one persisted semantic index build."""

    schema_version: int
    corpus_fingerprint: str
    chunk_count: int
    model_name: str
    embedding_dimension: int
    normalize_embeddings: bool
    faiss_index_type: str
    corpus_versions: tuple[str, ...]
    embedding_build_time_ns: int
    faiss_build_time_ns: int

    def __post_init__(self) -> None:
        if self.schema_version <= 0:
            raise ValueError(
                "schema_version must be greater than zero."
            )

        if not self.corpus_fingerprint.strip():
            raise ValueError(
                "corpus_fingerprint cannot be empty."
            )

        if self.chunk_count < 0:
            raise ValueError(
                "chunk_count cannot be negative."
            )

        if not self.model_name.strip():
            raise ValueError(
                "model_name cannot be empty."
            )

        if self.embedding_dimension <= 0:
            raise ValueError(
                "embedding_dimension must be greater than zero."
            )

        if not self.faiss_index_type.strip():
            raise ValueError(
                "faiss_index_type cannot be empty."
            )

        if self.embedding_build_time_ns < 0:
            raise ValueError(
                "embedding_build_time_ns cannot be negative."
            )

        if self.faiss_build_time_ns < 0:
            raise ValueError(
                "faiss_build_time_ns cannot be negative."
            )


def calculate_corpus_fingerprint(
    corpus_chunks: Sequence[Mapping[str, Any]],
) -> str:
    """Return a deterministic fingerprint for ordered corpus chunks."""
    digest = sha256()

    for position, chunk in enumerate(
        corpus_chunks
    ):
        chunk_id = _required_string(
            chunk,
            "chunk_id",
        )

        content = _required_string(
            chunk,
            "content",
        )

        source_path = str(
            chunk.get(
                "source_path",
                "",
            )
        ).strip()

        version = str(
            chunk.get(
                "version",
                "",
            )
        ).strip()

        payload = {
            "position": position,
            "chunk_id": chunk_id,
            "content": content,
            "source_path": source_path,
            "version": version,
        }

        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

        digest.update(
            serialized.encode(
                "utf-8"
            )
        )
        digest.update(b"\n")

    return digest.hexdigest()


def build_semantic_index_metadata(
    corpus_chunks: Sequence[Mapping[str, Any]],
    *,
    model_name: str,
    embedding_dimension: int,
    normalize_embeddings: bool,
    faiss_index_type: str,
    embedding_build_time_ns: int,
    faiss_build_time_ns: int,
) -> SemanticIndexMetadata:
    """Create metadata describing a semantic index build."""
    versions = tuple(
        sorted(
            {
                str(
                    chunk.get(
                        "version",
                        "",
                    )
                ).strip()
                for chunk in corpus_chunks
                if str(
                    chunk.get(
                        "version",
                        "",
                    )
                ).strip()
            }
        )
    )

    return SemanticIndexMetadata(
        schema_version=(
            SEMANTIC_INDEX_SCHEMA_VERSION
        ),
        corpus_fingerprint=(
            calculate_corpus_fingerprint(
                corpus_chunks
            )
        ),
        chunk_count=len(
            corpus_chunks
        ),
        model_name=model_name,
        embedding_dimension=(
            embedding_dimension
        ),
        normalize_embeddings=(
            normalize_embeddings
        ),
        faiss_index_type=(
            faiss_index_type
        ),
        corpus_versions=versions,
        embedding_build_time_ns=(
            embedding_build_time_ns
        ),
        faiss_build_time_ns=(
            faiss_build_time_ns
        ),
    )


def save_semantic_index_metadata(
    metadata: SemanticIndexMetadata,
    path: str | Path,
) -> None:
    """Persist semantic index metadata as JSON."""
    metadata_path = Path(
        path
    )

    metadata_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = asdict(
        metadata
    )

    payload["corpus_versions"] = list(
        metadata.corpus_versions
    )

    metadata_path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def load_semantic_index_metadata(
    path: str | Path,
) -> SemanticIndexMetadata:
    """Load semantic index metadata from JSON."""
    metadata_path = Path(
        path
    )

    if not metadata_path.is_file():
        raise FileNotFoundError(
            "Semantic index metadata file "
            f"not found: {metadata_path}"
        )

    try:
        raw_data = json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )
    except json.JSONDecodeError as error:
        raise SemanticPersistenceError(
            "Semantic index metadata "
            "contains invalid JSON."
        ) from error

    if not isinstance(
        raw_data,
        dict,
    ):
        raise SemanticPersistenceError(
            "Semantic index metadata "
            "must contain a JSON object."
        )

    return _metadata_from_mapping(
        raw_data
    )


def validate_semantic_index_metadata(
    metadata: SemanticIndexMetadata,
    corpus_chunks: Sequence[Mapping[str, Any]],
    *,
    model_name: str,
    embedding_dimension: int,
    normalize_embeddings: bool,
    faiss_index_type: str,
) -> None:
    """Ensure persisted index metadata matches the current runtime."""
    expected_fingerprint = (
        calculate_corpus_fingerprint(
            corpus_chunks
        )
    )

    mismatches: list[str] = []

    if (
        metadata.schema_version
        != SEMANTIC_INDEX_SCHEMA_VERSION
    ):
        mismatches.append(
            "schema version"
        )

    if (
        metadata.corpus_fingerprint
        != expected_fingerprint
    ):
        mismatches.append(
            "corpus fingerprint"
        )

    if (
        metadata.chunk_count
        != len(corpus_chunks)
    ):
        mismatches.append(
            "chunk count"
        )

    if (
        metadata.model_name
        != model_name
    ):
        mismatches.append(
            "embedding model"
        )

    if (
        metadata.embedding_dimension
        != embedding_dimension
    ):
        mismatches.append(
            "embedding dimension"
        )

    if (
        metadata.normalize_embeddings
        != normalize_embeddings
    ):
        mismatches.append(
            "embedding normalization"
        )

    if (
        metadata.faiss_index_type
        != faiss_index_type
    ):
        mismatches.append(
            "FAISS index type"
        )

    if mismatches:
        raise SemanticIndexCompatibilityError(
            "Persisted semantic index is "
            "incompatible with the current runtime: "
            + ", ".join(mismatches)
            + "."
        )


def _metadata_from_mapping(
    data: Mapping[str, Any],
) -> SemanticIndexMetadata:
    """Create validated metadata from decoded JSON."""
    try:
        schema_version = int(
            data["schema_version"]
        )
        corpus_fingerprint = str(
            data["corpus_fingerprint"]
        )
        chunk_count = int(
            data["chunk_count"]
        )
        model_name = str(
            data["model_name"]
        )
        embedding_dimension = int(
            data["embedding_dimension"]
        )
        normalize_embeddings = data[
            "normalize_embeddings"
        ]
        faiss_index_type = str(
            data["faiss_index_type"]
        )
        embedding_build_time_ns = int(
            data["embedding_build_time_ns"]
        )
        faiss_build_time_ns = int(
            data["faiss_build_time_ns"]
        )
    except (
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        raise SemanticPersistenceError(
            "Semantic index metadata "
            "has missing or invalid fields."
        ) from error

    if not isinstance(
        normalize_embeddings,
        bool,
    ):
        raise SemanticPersistenceError(
            "normalize_embeddings must "
            "be a boolean."
        )

    raw_versions = data.get(
        "corpus_versions",
        [],
    )

    if not isinstance(
        raw_versions,
        list,
    ):
        raise SemanticPersistenceError(
            "corpus_versions must be a list."
        )

    corpus_versions = tuple(
        str(version)
        for version in raw_versions
    )

    return SemanticIndexMetadata(
        schema_version=schema_version,
        corpus_fingerprint=(
            corpus_fingerprint
        ),
        chunk_count=chunk_count,
        model_name=model_name,
        embedding_dimension=(
            embedding_dimension
        ),
        normalize_embeddings=(
            normalize_embeddings
        ),
        faiss_index_type=(
            faiss_index_type
        ),
        corpus_versions=corpus_versions,
        embedding_build_time_ns=(
            embedding_build_time_ns
        ),
        faiss_build_time_ns=(
            faiss_build_time_ns
        ),
    )


def _required_string(
    data: Mapping[str, Any],
    field: str,
) -> str:
    """Read a required non-empty string field."""
    value = data.get(
        field
    )

    if not isinstance(
        value,
        str,
    ):
        raise TypeError(
            f"{field} must be a string."
        )

    normalized = value.strip()

    if not normalized:
        raise ValueError(
            f"{field} cannot be empty."
        )

    return normalized