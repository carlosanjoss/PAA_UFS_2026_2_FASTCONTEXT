"""Test prepared JSONL chunk corpus loading and validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.ingestion.chunk_loader import (
    ChunkCorpusFormatError,
    load_chunks_jsonl,
)


def _write_jsonl(
    path: Path,
    records: list[dict],
) -> None:
    path.write_text(
        "\n".join(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            for record in records
        )
        + "\n",
        encoding="utf-8",
    )


def _valid_chunk(
    *,
    chunk_id: str = "tutorial/dependencies-0",
) -> dict:
    return {
        "chunk_id": chunk_id,
        "document": "tutorial/dependencies/index.md",
        "section": "Dependencies",
        "title": "Dependencies",
        "content": (
            "FastAPI provides a dependency injection system "
            "using Depends."
        ),
        "token_count": 12,
        "source_path": "tutorial/dependencies/index.md",
        "version": "0.141.0",
    }


def test_load_chunks_jsonl_loads_valid_corpus(
    tmp_path: Path,
) -> None:
    corpus_path = (
        tmp_path
        / "chunks.jsonl"
    )

    _write_jsonl(
        corpus_path,
        [
            _valid_chunk(),
        ],
    )

    chunks = load_chunks_jsonl(
        corpus_path
    )

    assert len(chunks) == 1

    chunk = chunks[0]

    assert (
        chunk["chunk_id"]
        == "tutorial/dependencies-0"
    )
    assert (
        chunk["section_title"]
        == "Dependencies"
    )
    assert (
        chunk["document_title"]
        == "tutorial/dependencies/index.md"
    )
    assert (
        chunk["source_path"]
        == "tutorial/dependencies/index.md"
    )
    assert chunk["token_count"] == 12

    assert chunk["metadata"] == {
        "document": "tutorial/dependencies/index.md",
        "section": "Dependencies",
        "title": "Dependencies",
        "version": "0.141.0",
    }


def test_load_chunks_jsonl_preserves_original_fields(
    tmp_path: Path,
) -> None:
    corpus_path = (
        tmp_path
        / "chunks.jsonl"
    )

    _write_jsonl(
        corpus_path,
        [
            _valid_chunk(),
        ],
    )

    chunk = load_chunks_jsonl(
        corpus_path
    )[0]

    assert chunk["document"] == (
        "tutorial/dependencies/index.md"
    )
    assert chunk["section"] == "Dependencies"
    assert chunk["title"] == "Dependencies"


def test_load_chunks_jsonl_uses_section_when_title_is_empty(
    tmp_path: Path,
) -> None:
    corpus_path = (
        tmp_path
        / "chunks.jsonl"
    )

    record = _valid_chunk()
    record["title"] = ""

    _write_jsonl(
        corpus_path,
        [record],
    )

    chunk = load_chunks_jsonl(
        corpus_path
    )[0]

    assert (
        chunk["section_title"]
        == "Dependencies"
    )


def test_load_chunks_jsonl_rejects_duplicate_chunk_ids(
    tmp_path: Path,
) -> None:
    corpus_path = (
        tmp_path
        / "chunks.jsonl"
    )

    _write_jsonl(
        corpus_path,
        [
            _valid_chunk(),
            _valid_chunk(),
        ],
    )

    with pytest.raises(
        ChunkCorpusFormatError,
        match="Duplicate chunk_id",
    ):
        load_chunks_jsonl(
            corpus_path
        )


def test_load_chunks_jsonl_rejects_invalid_json(
    tmp_path: Path,
) -> None:
    corpus_path = (
        tmp_path
        / "chunks.jsonl"
    )

    corpus_path.write_text(
        "{invalid json}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ChunkCorpusFormatError,
        match="Invalid JSON",
    ):
        load_chunks_jsonl(
            corpus_path
        )


def test_load_chunks_jsonl_rejects_missing_required_field(
    tmp_path: Path,
) -> None:
    corpus_path = (
        tmp_path
        / "chunks.jsonl"
    )

    record = _valid_chunk()
    del record["content"]

    _write_jsonl(
        corpus_path,
        [record],
    )

    with pytest.raises(
        ChunkCorpusFormatError,
        match="Field 'content'",
    ):
        load_chunks_jsonl(
            corpus_path
        )


def test_load_chunks_jsonl_rejects_invalid_token_count(
    tmp_path: Path,
) -> None:
    corpus_path = (
        tmp_path
        / "chunks.jsonl"
    )

    record = _valid_chunk()
    record["token_count"] = -1

    _write_jsonl(
        corpus_path,
        [record],
    )

    with pytest.raises(
        ChunkCorpusFormatError,
        match="token_count",
    ):
        load_chunks_jsonl(
            corpus_path
        )


def test_load_chunks_jsonl_rejects_empty_corpus(
    tmp_path: Path,
) -> None:
    corpus_path = (
        tmp_path
        / "chunks.jsonl"
    )

    corpus_path.write_text(
        "",
        encoding="utf-8",
    )

    with pytest.raises(
        ChunkCorpusFormatError,
        match="Chunk corpus is empty",
    ):
        load_chunks_jsonl(
            corpus_path
        )


def test_load_chunks_jsonl_missing_file_can_return_empty(
    tmp_path: Path,
) -> None:
    corpus_path = (
        tmp_path
        / "missing.jsonl"
    )

    assert (
        load_chunks_jsonl(
            corpus_path,
            missing_ok=True,
        )
        == []
    )


def test_load_chunks_jsonl_missing_file_is_strict_by_default(
    tmp_path: Path,
) -> None:
    corpus_path = (
        tmp_path
        / "missing.jsonl"
    )

    with pytest.raises(
        FileNotFoundError,
        match="Chunk corpus file not found",
    ):
        load_chunks_jsonl(
            corpus_path
        )