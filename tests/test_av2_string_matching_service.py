
"""Tests for the AV2 corpus string matching service."""

import json
from pathlib import Path

import pytest

from src.services.string_matching_service import (
    StringMatchingService,
)


@pytest.fixture
def corpus_path(tmp_path: Path) -> Path:
    """Create a small test corpus without modifying AV1 data."""

    chunks = [
        {
            "chunk_id": "chunk_001",
            "content": "FastAPI supports API development.",
            "source_path": "tutorial/first-steps.md",
            "version": "0.141.0",
            "token_count": 8,
            "section": "First Steps",
        },
        {
            "chunk_id": "chunk_002",
            "content": "API API API",
            "source_path": "tutorial/metadata.md",
            "version": "0.141.0",
            "token_count": 3,
            "section": "Metadata",
        },
        {
            "chunk_id": "chunk_003",
            "content": "Python web framework",
            "source_path": "tutorial/python.md",
            "version": "0.141.0",
            "token_count": 3,
            "section": "Python",
        },
    ]

    path = tmp_path / "chunks.jsonl"

    with path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(
                json.dumps(chunk, ensure_ascii=False) + "\n"
            )

    return path


def test_corpus_size(corpus_path: Path) -> None:
    service = StringMatchingService(corpus_path)

    assert service.corpus_size == 3


def test_matching_chunks(corpus_path: Path) -> None:
    service = StringMatchingService(corpus_path)

    result = service.search("API")

    assert result.algorithm == "naive"
    assert result.total_chunks == 3
    assert result.matching_chunks == 2
    assert result.total_occurrences == 5


def test_match_positions(corpus_path: Path) -> None:
    service = StringMatchingService(corpus_path)

    result = service.search("API")

    assert result.matches[0].chunk_id == "chunk_001"
    assert result.matches[0].positions == (4, 17)

    assert result.matches[1].chunk_id == "chunk_002"
    assert result.matches[1].positions == (0, 4, 8)


def test_no_matches(corpus_path: Path) -> None:
    service = StringMatchingService(corpus_path)

    result = service.search("Django")

    assert result.matching_chunks == 0
    assert result.total_occurrences == 0
    assert result.matches == ()


def test_empty_pattern(corpus_path: Path) -> None:
    service = StringMatchingService(corpus_path)

    with pytest.raises(ValueError):
        service.search("")


def test_unknown_algorithm(corpus_path: Path) -> None:
    service = StringMatchingService(corpus_path)

    with pytest.raises(ValueError):
        service.search("API", algorithm="unknown")


def test_metrics(corpus_path: Path) -> None:
    service = StringMatchingService(corpus_path)

    result = service.search("API")

    assert result.total_comparisons > 0
    assert result.preprocessing_time_ns == 0
    assert result.search_time_ns >= 0
    assert result.total_time_ns >= 0


def test_source_metadata(corpus_path: Path) -> None:
    service = StringMatchingService(corpus_path)

    result = service.search("API")

    assert result.matches[0].source_path == (
        "tutorial/first-steps.md"
    )
    assert result.matches[0].section_title == "First Steps"
