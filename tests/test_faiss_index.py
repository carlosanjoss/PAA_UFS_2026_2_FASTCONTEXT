from __future__ import annotations

import numpy as np
import pytest

from src.semantic.faiss_index import (
    FaissIndex,
)


def test_build_tracks_index_size() -> None:
    index = FaissIndex(
        dimension=3
    )

    embeddings = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    index.build(
        embeddings,
        [
            "chunk_001",
            "chunk_002",
        ],
    )

    assert index.size == 2
    assert index.is_empty is False


def test_search_returns_highest_inner_product() -> None:
    index = FaissIndex(
        dimension=3
    )

    embeddings = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    index.build(
        embeddings,
        [
            "dependencies",
            "security",
            "middleware",
        ],
    )

    query = np.asarray(
        [0.9, 0.1, 0.0],
        dtype=np.float32,
    )

    results = index.search(
        query,
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].chunk_id == "dependencies"
    assert results[1].chunk_id == "security"
    assert results[0].score > results[1].score


def test_top_k_larger_than_index_size_is_supported() -> None:
    index = FaissIndex(
        dimension=2
    )

    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    index.build(
        embeddings,
        [
            "a",
            "b",
        ],
    )

    query = np.asarray(
        [1.0, 0.0],
        dtype=np.float32,
    )

    results = index.search(
        query,
        top_k=10,
    )

    assert len(results) == 2


def test_zero_top_k_returns_empty_result() -> None:
    index = FaissIndex(
        dimension=2
    )

    embeddings = np.asarray(
        [
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )

    index.build(
        embeddings,
        ["a"],
    )

    query = np.asarray(
        [1.0, 0.0],
        dtype=np.float32,
    )

    assert index.search(
        query,
        top_k=0,
    ) == ()


def test_empty_index_returns_empty_result() -> None:
    index = FaissIndex(
        dimension=3
    )

    query = np.asarray(
        [1.0, 0.0, 0.0],
        dtype=np.float32,
    )

    assert index.search(
        query,
        top_k=5,
    ) == ()


def test_dimension_mismatch_is_rejected() -> None:
    index = FaissIndex(
        dimension=3
    )

    embeddings = np.asarray(
        [
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="dimension",
    ):
        index.build(
            embeddings,
            ["a"],
        )


def test_duplicate_chunk_ids_are_rejected() -> None:
    index = FaissIndex(
        dimension=2
    )

    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="unique",
    ):
        index.build(
            embeddings,
            [
                "same",
                "same",
            ],
        )


def test_save_and_load_preserve_mapping(
    tmp_path,
) -> None:
    index = FaissIndex(
        dimension=2
    )

    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    index.build(
        embeddings,
        [
            "dependencies",
            "security",
        ],
    )

    index_path = (
        tmp_path
        / "semantic.faiss"
    )

    mapping_path = (
        tmp_path
        / "semantic.mapping"
    )

    index.save(
        index_path,
        mapping_path,
    )

    loaded = FaissIndex(
        dimension=2
    )

    loaded.load(
        index_path,
        mapping_path,
    )

    query = np.asarray(
        [1.0, 0.0],
        dtype=np.float32,
    )

    results = loaded.search(
        query,
        top_k=1,
    )

    assert loaded.size == 2
    assert results[0].chunk_id == "dependencies"