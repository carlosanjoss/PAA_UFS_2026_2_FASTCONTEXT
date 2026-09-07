"""Test corpus preparation orchestration and generated outputs."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.prepare_corpus import (
    _create_chunks,
    _create_record,
    prepare_corpus,
)
from src.preprocessing.normalize import ParsedMarkdown


def test_create_chunks_reads_and_processes_document(
    tmp_path: Path,
    monkeypatch,
) -> None:
    document = tmp_path / "example.md"
    document.write_text(
        "# FastAPI\n\nFastAPI is a web framework.",
        encoding="utf-8",
    )

    markdown = ParsedMarkdown(
        text=document.read_text(encoding="utf-8"),
        tokens=[],
    )

    monkeypatch.setattr(
        "scripts.prepare_corpus.normalize_markdown",
        lambda text: markdown,
    )

    monkeypatch.setattr(
        "scripts.prepare_corpus.chunk_markdown",
        lambda markdown, max_tokens, overlap_tokens: [
            {
                "section": "FastAPI",
                "title": "FastAPI",
                "content": "FastAPI is a web framework.",
                "token_count": 6,
            }
        ],
    )

    chunks = _create_chunks(
        document=document,
        max_tokens=400,
        overlap_tokens=60,
    )

    assert len(chunks) == 1
    assert chunks[0]["section"] == "FastAPI"
    assert chunks[0]["title"] == "FastAPI"
    assert chunks[0]["content"] == "FastAPI is a web framework."
    assert chunks[0]["token_count"] == 6


def test_create_record_contains_required_metadata(
    tmp_path: Path,
) -> None:
    raw_directory = tmp_path / "raw"
    document = raw_directory / "security" / "oauth2.md"
    document.parent.mkdir(parents=True)

    chunk = {
        "section": "Security > OAuth2",
        "title": "OAuth2",
        "content": "OAuth2 content.",
        "token_count": 3,
    }

    record = _create_record(
        document=document,
        raw_directory=raw_directory,
        chunk=chunk,
        index=0,
        version="0.141.0",
    )

    assert record == {
        "chunk_id": "security/oauth2-0",
        "documento": "security/oauth2.md",
        "secao": "Security > OAuth2",
        "titulo": "OAuth2",
        "conteudo": "OAuth2 content.",
        "token_count": 3,
        "source_path": "security/oauth2.md",
        "versao": "0.141.0",
    }


def test_create_record_chunk_id_is_deterministic(
    tmp_path: Path,
) -> None:
    raw_directory = tmp_path / "raw"
    document = raw_directory / "security.md"

    chunk = {
        "section": "Security",
        "title": "Security",
        "content": "Security content.",
        "token_count": 3,
    }

    first = _create_record(
        document=document,
        raw_directory=raw_directory,
        chunk=chunk,
        index=0,
        version="0.141.0",
    )

    second = _create_record(
        document=document,
        raw_directory=raw_directory,
        chunk=chunk,
        index=0,
        version="0.141.0",
    )

    assert first == second
    assert first["chunk_id"] == "security-0"


def test_prepare_corpus_creates_jsonl_and_statistics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    raw_directory = tmp_path / "raw"
    raw_directory.mkdir()

    document = raw_directory / "security.md"
    document.write_text(
        "# Security\n\nSecurity content.",
        encoding="utf-8",
    )

    output_path = tmp_path / "chunks" / "chunks.jsonl"

    monkeypatch.setattr(
        "scripts.prepare_corpus.load_corpus_config",
        lambda: {
            "source": {
                "version": "0.141.0",
            },
            "download": {
                "raw_directory": str(raw_directory),
                "allowed_extensions": [".md"],
            },
            "chunking": {
                "output": str(output_path),
                "max_tokens": 400,
                "overlap_tokens": 60,
            },
        },
    )

    monkeypatch.setattr(
        "scripts.prepare_corpus.get_corpus_documents",
        lambda raw_directory, allowed_extensions, ignored_files: [document],
    )

    monkeypatch.setattr(
        "scripts.prepare_corpus.normalize_markdown",
        lambda text: ParsedMarkdown(
            text=text,
            tokens=[],
        ),
    )

    monkeypatch.setattr(
        "scripts.prepare_corpus.chunk_markdown",
        lambda markdown, max_tokens, overlap_tokens: [
            {
                "section": "Security",
                "title": "Security",
                "content": "Security content.",
                "token_count": 3,
            },
            {
                "section": "Security",
                "title": "Security",
                "content": "More security content.",
                "token_count": 4,
            },
        ],
    )

    prepare_corpus()

    assert output_path.exists()

    records = [
        json.loads(line)
        for line in output_path.read_text(
            encoding="utf-8",
        ).splitlines()
    ]

    assert len(records) == 2

    assert records[0] == {
        "chunk_id": "security-0",
        "documento": "security.md",
        "secao": "Security",
        "titulo": "Security",
        "conteudo": "Security content.",
        "token_count": 3,
        "source_path": "security.md",
        "versao": "0.141.0",
    }

    assert records[1]["chunk_id"] == "security-1"
    assert records[1]["token_count"] == 4

    statistics_path = output_path.with_name(
        "statistics.json"
    )

    assert statistics_path.exists()

    statistics = json.loads(
        statistics_path.read_text(
            encoding="utf-8",
        )
    )

    assert statistics == {
        "chunk_count": 2,
        "mean_tokens": 3.5,
        "median_tokens": 3.5,
        "min_tokens": 3,
        "max_tokens": 4,
        "documents_without_chunks": 0,
    }


def test_prepare_corpus_counts_documents_without_chunks(
    tmp_path: Path,
    monkeypatch,
) -> None:
    raw_directory = tmp_path / "raw"
    raw_directory.mkdir()

    document = raw_directory / "empty.md"
    document.write_text(
        "# Empty",
        encoding="utf-8",
    )

    output_path = tmp_path / "chunks" / "chunks.jsonl"

    monkeypatch.setattr(
        "scripts.prepare_corpus.load_corpus_config",
        lambda: {
            "source": {"version": "0.141.0"},
            "download": {
                "raw_directory": str(raw_directory),
                "allowed_extensions": [".md"],
            },
            "chunking": {
                "output": str(output_path),
                "max_tokens": 400,
                "overlap_tokens": 60,
            },
        },
    )

    monkeypatch.setattr(
        "scripts.prepare_corpus.get_corpus_documents",
        lambda raw_directory, allowed_extensions, ignored_files: [document],
    )

    monkeypatch.setattr(
        "scripts.prepare_corpus.normalize_markdown",
        lambda text: ParsedMarkdown(
            text=text,
            tokens=[],
        ),
    )

    monkeypatch.setattr(
        "scripts.prepare_corpus.chunk_markdown",
        lambda markdown, max_tokens, overlap_tokens: [],
    )

    prepare_corpus()

    statistics_path = output_path.with_name(
        "statistics.json"
    )

    statistics = json.loads(
        statistics_path.read_text(
            encoding="utf-8",
        )
    )

    assert statistics["chunk_count"] == 0
    assert statistics["documents_without_chunks"] == 1