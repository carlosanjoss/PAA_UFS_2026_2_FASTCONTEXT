
"""Tests for the AV2 FastAPI routes."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.services.string_matching_service import (
    StringMatchingService,
)
from src.string_matching.api import (
    get_string_matching_service,
    router,
)


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    """Create an isolated API backed by a temporary corpus."""

    corpus_path = tmp_path / "chunks.jsonl"

    chunks = [
        {
            "chunk_id": "test-001",
            "content": "OAuth2 and OAuth2",
            "source_path": "security/oauth2.md",
            "section": "OAuth2",
            "version": "0.141.0",
            "token_count": 4,
        },
        {
            "chunk_id": "test-002",
            "content": "FastAPI documentation",
            "source_path": "tutorial/index.md",
            "section": "Introduction",
            "version": "0.141.0",
            "token_count": 3,
        },
    ]

    with corpus_path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(chunk) + "\n")

    service = StringMatchingService(corpus_path)

    app = FastAPI()
    app.include_router(router)

    app.dependency_overrides[
        get_string_matching_service
    ] = lambda: service

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def test_available_algorithms(client: TestClient) -> None:
    response = client.get(
        "/api/av2/string-matching/algorithms"
    )

    assert response.status_code == 200
    assert response.json()["algorithms"] == [
        "boyer-moore",
        "kmp",
        "naive",
        "rabin-karp",
    ]
    assert response.json()["corpus_size"] == 2


def test_search_pattern(client: TestClient) -> None:
    response = client.post(
        "/api/av2/string-matching/search",
        json={
            "pattern": "OAuth2",
            "algorithm": "naive",
            "max_results": 5,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["total_chunks"] == 2
    assert data["matching_chunks"] == 1
    assert data["total_occurrences"] == 2
    assert data["matches"][0]["positions"] == [0, 11]


def test_pattern_not_found(client: TestClient) -> None:
    response = client.post(
        "/api/av2/string-matching/search",
        json={"pattern": "Django"},
    )

    assert response.status_code == 200
    assert response.json()["total_occurrences"] == 0


def test_invalid_pattern(client: TestClient) -> None:
    response = client.post(
        "/api/av2/string-matching/search",
        json={"pattern": ""},
    )

    assert response.status_code == 422


def test_unknown_algorithm(client: TestClient) -> None:
    response = client.post(
        "/api/av2/string-matching/search",
        json={
            "pattern": "OAuth2",
            "algorithm": "unknown",
        },
    )

    assert response.status_code == 400
