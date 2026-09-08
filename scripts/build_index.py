"""Build and persist the FastContext semantic FAISS index."""

from __future__ import annotations

import argparse
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import yaml

from src.ingestion.chunk_loader import (
    DEFAULT_CHUNKS_PATH,
    load_chunks_jsonl,
)
from src.representations.embeddings import (
    EmbeddingConfig,
    EmbeddingEncoder,
)
from src.semantic.faiss_index import FaissIndex
from src.semantic.persistence import (
    build_semantic_index_metadata,
    save_semantic_index_metadata,
)
from src.utils.config import PROJECT_ROOT

DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "retrieval.yaml"
)

DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "semantic"
)

INDEX_FILENAME = "index.faiss"
MAPPING_FILENAME = "chunk_ids.txt"
METADATA_FILENAME = "metadata.json"


def build_parser() -> argparse.ArgumentParser:
    """Build the semantic index command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Build and persist the FastContext "
            "semantic FAISS index."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to retrieval.yaml.",
    )

    parser.add_argument(
        "--chunks",
        type=Path,
        default=DEFAULT_CHUNKS_PATH,
        help="Path to prepared chunks.jsonl.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory used to persist "
            "semantic index artifacts."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Overwrite existing semantic "
            "index artifacts."
        ),
    )

    return parser


def main(
    argv: Sequence[str] | None = None,
) -> int:
    """Build semantic index artifacts."""
    parser = build_parser()

    args = parser.parse_args(
        argv
    )

    config_path = _resolve_path(
        args.config
    )

    chunks_path = _resolve_path(
        args.chunks
    )

    semantic_config = (
        _load_semantic_config(
            config_path
        )
    )

    persistence_config = (
        semantic_config.get(
            "persistence",
            {},
        )
    )

    if not isinstance(
        persistence_config,
        Mapping,
    ):
        raise TypeError(
            "semantic.persistence "
            "must be a mapping."
        )

    configured_directory = (
        persistence_config.get(
            "directory"
        )
    )

    if args.output_dir is not None:
        output_directory = _resolve_path(
            args.output_dir
        )
    elif isinstance(
        configured_directory,
        str,
    ) and configured_directory.strip():
        output_directory = _resolve_path(
            Path(
                configured_directory
            )
        )
    else:
        output_directory = (
            DEFAULT_OUTPUT_DIRECTORY
        )

    index_path = (
        output_directory
        / INDEX_FILENAME
    )

    mapping_path = (
        output_directory
        / MAPPING_FILENAME
    )

    metadata_path = (
        output_directory
        / METADATA_FILENAME
    )

    artifact_paths = (
        index_path,
        mapping_path,
        metadata_path,
    )

    if (
        not args.force
        and any(
            path.exists()
            for path in artifact_paths
        )
    ):
        parser.error(
            "Semantic index artifacts "
            "already exist. Use --force "
            "to rebuild them."
        )

    model_name = _read_string(
        semantic_config,
        "model",
        "name",
    )

    normalize_embeddings = _read_bool(
        semantic_config,
        "embeddings",
        "normalize",
    )

    configured_dimension = _read_int(
        semantic_config,
        "embeddings",
        "dimension",
    )

    faiss_index_type = _read_string(
        semantic_config,
        "faiss",
        "index_type",
    )

    if (
        faiss_index_type
        != "IndexFlatIP"
    ):
        raise ValueError(
            "Only FAISS IndexFlatIP "
            "is currently supported."
        )

    print(
        "FastContext Semantic Index Build"
    )
    print("=" * 60)

    print(
        f"Chunks: {chunks_path}"
    )

    print(
        f"Configuration: {config_path}"
    )

    print(
        f"Output: {output_directory}"
    )

    print()

    print(
        "[1/5] Loading prepared corpus..."
    )

    corpus_chunks = (
        load_chunks_jsonl(
            chunks_path
        )
    )

    print(
        f"      Loaded {len(corpus_chunks)} chunks."
    )

    print(
        "[2/5] Loading embedding model..."
    )

    encoder = EmbeddingEncoder(
        EmbeddingConfig(
            model_name=model_name,
            normalize_embeddings=(
                normalize_embeddings
            ),
        )
    )

    actual_dimension = (
        encoder.dimension
    )

    if (
        actual_dimension
        != configured_dimension
    ):
        raise ValueError(
            "Configured embedding dimension "
            f"is {configured_dimension}, but "
            f"model '{model_name}' reports "
            f"{actual_dimension}."
        )

    print(
        f"      Model: {model_name}"
    )

    print(
        f"      Dimension: {actual_dimension}"
    )

    print(
        "[3/5] Encoding corpus chunks..."
    )

    document_texts = [
        str(
            chunk["content"]
        )
        for chunk in corpus_chunks
    ]

    embedding_start = (
        time.perf_counter_ns()
    )

    embeddings = (
        encoder.encode_documents(
            document_texts
        )
    )

    embedding_build_time_ns = (
        time.perf_counter_ns()
        - embedding_start
    )

    print(
        "      Embedding build: "
        f"{embedding_build_time_ns / 1_000_000:.2f} ms"
    )

    print(
        "[4/5] Building FAISS index..."
    )

    chunk_ids = [
        str(
            chunk["chunk_id"]
        )
        for chunk in corpus_chunks
    ]

    index = FaissIndex(
        dimension=actual_dimension
    )

    faiss_start = (
        time.perf_counter_ns()
    )

    index.build(
        embeddings,
        chunk_ids,
    )

    faiss_build_time_ns = (
        time.perf_counter_ns()
        - faiss_start
    )

    print(
        "      FAISS build: "
        f"{faiss_build_time_ns / 1_000_000:.2f} ms"
    )

    print(
        "[5/5] Persisting artifacts..."
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    index.save(
        index_path=index_path,
        mapping_path=mapping_path,
    )

    metadata = (
        build_semantic_index_metadata(
            corpus_chunks,
            model_name=model_name,
            embedding_dimension=(
                actual_dimension
            ),
            normalize_embeddings=(
                normalize_embeddings
            ),
            faiss_index_type=(
                faiss_index_type
            ),
            embedding_build_time_ns=(
                embedding_build_time_ns
            ),
            faiss_build_time_ns=(
                faiss_build_time_ns
            ),
        )
    )

    save_semantic_index_metadata(
        metadata,
        metadata_path,
    )

    print(
        f"      Index: {index_path}"
    )

    print(
        f"      Mapping: {mapping_path}"
    )

    print(
        f"      Metadata: {metadata_path}"
    )

    print()

    print(
        "Semantic index ready."
    )

    print(
        f"Vectors: {index.size}"
    )

    print(
        "Corpus fingerprint: "
        f"{metadata.corpus_fingerprint}"
    )

    return 0


def _load_semantic_config(
    path: Path,
) -> Mapping[str, Any]:
    """Load the semantic retrieval configuration."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {path}"
        )

    raw_data = yaml.safe_load(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        raw_data,
        Mapping,
    ):
        raise TypeError(
            "Retrieval configuration "
            "must be a mapping."
        )

    retrieval = raw_data.get(
        "retrieval"
    )

    if not isinstance(
        retrieval,
        Mapping,
    ):
        raise TypeError(
            "retrieval configuration "
            "must be a mapping."
        )

    semantic = retrieval.get(
        "semantic"
    )

    if not isinstance(
        semantic,
        Mapping,
    ):
        raise TypeError(
            "semantic configuration "
            "must be a mapping."
        )

    return cast(
        Mapping[str, Any],
        semantic,
    )


def _read_string(
    data: Mapping[str, Any],
    section: str,
    key: str,
) -> str:
    """Read a required nested string value."""
    section_data = data.get(
        section
    )

    if not isinstance(
        section_data,
        Mapping,
    ):
        raise TypeError(
            f"{section} must be a mapping."
        )

    value = section_data.get(
        key
    )

    if not isinstance(
        value,
        str,
    ):
        raise TypeError(
            f"{section}.{key} must be a string."
        )

    normalized = value.strip()

    if not normalized:
        raise ValueError(
            f"{section}.{key} cannot be empty."
        )

    return normalized


def _read_bool(
    data: Mapping[str, Any],
    section: str,
    key: str,
) -> bool:
    """Read a required nested boolean value."""
    section_data = data.get(
        section
    )

    if not isinstance(
        section_data,
        Mapping,
    ):
        raise TypeError(
            f"{section} must be a mapping."
        )

    value = section_data.get(
        key
    )

    if not isinstance(
        value,
        bool,
    ):
        raise TypeError(
            f"{section}.{key} must be a boolean."
        )

    return value


def _read_int(
    data: Mapping[str, Any],
    section: str,
    key: str,
) -> int:
    """Read a required nested integer value."""
    section_data = data.get(
        section
    )

    if not isinstance(
        section_data,
        Mapping,
    ):
        raise TypeError(
            f"{section} must be a mapping."
        )

    value = section_data.get(
        key
    )

    if (
        not isinstance(
            value,
            int,
        )
        or isinstance(
            value,
            bool,
        )
    ):
        raise TypeError(
            f"{section}.{key} must be an integer."
        )

    return value


def _resolve_path(
    path: Path,
) -> Path:
    """Resolve a project-relative or absolute path."""
    if path.is_absolute():
        return path

    return (
        PROJECT_ROOT
        / path
    ).resolve()


if __name__ == "__main__":
    raise SystemExit(
        main()
    )