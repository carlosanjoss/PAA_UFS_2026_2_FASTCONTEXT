"""Load and validate prepared FastContext chunks from JSONL."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.utils.config import PROJECT_ROOT

CorpusChunk = dict[str, Any]

DEFAULT_CHUNKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "chunks"
    / "chunks.jsonl"
)


class ChunkCorpusError(ValueError):
    """Base error for prepared chunk corpus failures."""


class ChunkCorpusFormatError(ChunkCorpusError):
    """Raised when a chunk corpus has invalid structure or content."""


def load_chunks_jsonl(
    path: str | Path = DEFAULT_CHUNKS_PATH,
    *,
    missing_ok: bool = False,
) -> list[CorpusChunk]:
    """Load, validate, and normalize a prepared JSONL chunk corpus.

    The prepared corpus produced by ``scripts/prepare_corpus.py`` uses
    ``title`` and ``section`` fields. Retrieval components use the canonical
    ``section_title`` field. This loader preserves the original fields while
    adding the canonical aliases required by the application layer.

    Args:
        path: JSONL file containing prepared corpus chunks.
        missing_ok: Return an empty corpus when the file does not exist.

    Returns:
        Validated chunks ready for retrieval components.

    Raises:
        FileNotFoundError: If the corpus file does not exist and
            ``missing_ok`` is false.
        ChunkCorpusFormatError: If JSON or chunk structure is invalid.
    """
    corpus_path = Path(path)

    if not corpus_path.exists():
        if missing_ok:
            return []

        raise FileNotFoundError(
            f"Chunk corpus file not found: {corpus_path}"
        )

    if not corpus_path.is_file():
        raise ChunkCorpusFormatError(
            f"Chunk corpus path is not a file: {corpus_path}"
        )

    chunks: list[CorpusChunk] = []
    seen_chunk_ids: set[str] = set()

    with corpus_path.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            stripped_line = line.strip()

            if not stripped_line:
                continue

            raw_chunk = _parse_json_line(
                stripped_line,
                corpus_path=corpus_path,
                line_number=line_number,
            )

            chunk = _normalize_chunk(
                raw_chunk,
                corpus_path=corpus_path,
                line_number=line_number,
            )

            chunk_id = chunk["chunk_id"]

            if chunk_id in seen_chunk_ids:
                raise ChunkCorpusFormatError(
                    "Duplicate chunk_id "
                    f"'{chunk_id}' at line {line_number} "
                    f"in {corpus_path}."
                )

            seen_chunk_ids.add(chunk_id)
            chunks.append(chunk)

    if not chunks:
        raise ChunkCorpusFormatError(
            f"Chunk corpus is empty: {corpus_path}"
        )

    return chunks


def _parse_json_line(
    line: str,
    *,
    corpus_path: Path,
    line_number: int,
) -> Mapping[str, Any]:
    """Parse one JSONL record and require a JSON object."""
    try:
        value = json.loads(line)
    except json.JSONDecodeError as error:
        raise ChunkCorpusFormatError(
            "Invalid JSON at line "
            f"{line_number} in {corpus_path}."
        ) from error

    if not isinstance(value, dict):
        raise ChunkCorpusFormatError(
            "Chunk at line "
            f"{line_number} in {corpus_path} "
            "must be a JSON object."
        )

    return value


def _normalize_chunk(
    raw_chunk: Mapping[str, Any],
    *,
    corpus_path: Path,
    line_number: int,
) -> CorpusChunk:
    """Validate one prepared chunk and add canonical retrieval fields."""
    chunk_id = _require_non_empty_string(
        raw_chunk,
        "chunk_id",
        corpus_path=corpus_path,
        line_number=line_number,
    )

    content = _require_non_empty_string(
        raw_chunk,
        "content",
        corpus_path=corpus_path,
        line_number=line_number,
    )

    source_path = _require_non_empty_string(
        raw_chunk,
        "source_path",
        corpus_path=corpus_path,
        line_number=line_number,
    )

    version = _require_non_empty_string(
        raw_chunk,
        "version",
        corpus_path=corpus_path,
        line_number=line_number,
    )

    token_count = raw_chunk.get(
        "token_count"
    )

    if (
        not isinstance(token_count, int)
        or isinstance(token_count, bool)
        or token_count < 0
    ):
        raise ChunkCorpusFormatError(
            "Field 'token_count' at line "
            f"{line_number} in {corpus_path} "
            "must be a non-negative integer."
        )

    document = _optional_string(
        raw_chunk.get("document")
    )

    section = _optional_string(
        raw_chunk.get("section")
    )

    title = _optional_string(
        raw_chunk.get("title")
    )

    section_title = (
        title
        or section
        or Path(source_path).stem
    )

    document_title = (
        document
        or Path(source_path).stem
    )

    raw_metadata = raw_chunk.get(
        "metadata"
    )

    if (
        raw_metadata is not None
        and not isinstance(
            raw_metadata,
            Mapping,
        )
    ):
        raise ChunkCorpusFormatError(
            "Field 'metadata' at line "
            f"{line_number} in {corpus_path} "
            "must be a JSON object when provided."
        )

    metadata: dict[str, Any] = (
        dict(raw_metadata)
        if isinstance(raw_metadata, Mapping)
        else {}
    )

    metadata.setdefault(
        "document",
        document_title,
    )
    metadata.setdefault(
        "section",
        section,
    )
    metadata.setdefault(
        "title",
        title,
    )
    metadata.setdefault(
        "version",
        version,
    )

    normalized_chunk: CorpusChunk = dict(
        raw_chunk
    )

    normalized_chunk.update(
        {
            "chunk_id": chunk_id,
            "content": content,
            "source_path": source_path,
            "version": version,
            "token_count": token_count,
            "document_title": document_title,
            "section_title": section_title,
            "metadata": metadata,
        }
    )

    return normalized_chunk


def _require_non_empty_string(
    data: Mapping[str, Any],
    field: str,
    *,
    corpus_path: Path,
    line_number: int,
) -> str:
    """Read a required non-empty string field."""
    value = data.get(field)

    if not isinstance(value, str):
        raise ChunkCorpusFormatError(
            f"Field '{field}' at line "
            f"{line_number} in {corpus_path} "
            "must be a string."
        )

    normalized = value.strip()

    if not normalized:
        raise ChunkCorpusFormatError(
            f"Field '{field}' at line "
            f"{line_number} in {corpus_path} "
            "cannot be empty."
        )

    return normalized


def _optional_string(
    value: Any,
) -> str:
    """Normalize an optional string value."""
    if not isinstance(value, str):
        return ""

    return value.strip()