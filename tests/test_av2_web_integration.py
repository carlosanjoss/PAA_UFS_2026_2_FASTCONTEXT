"""Integration coverage for the AV2 web API and frozen reports."""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from src.services.string_matching_service import StringMatchingService
from src.string_matching import experiments_api
from src.string_matching.api import get_string_matching_service
from src.string_matching.experiments_api import load_av2_experiments
from src.web_api import app


@pytest.fixture
def web_client(tmp_path: Path) -> Iterator[TestClient]:
    """Use the production app with an isolated AV2 corpus dependency."""
    corpus_path = tmp_path / "chunks.jsonl"
    chunks = [
        {
            "chunk_id": "security-1",
            "content": "OAuth2 OAuth2 and oauth2",
            "source_path": "tutorial/security.md",
            "section": "Security",
            "version": "0.141.0",
            "token_count": 4,
        },
        {
            "chunk_id": "unicode-1",
            "content": "😀OAuth2",
            "source_path": "advanced/unicode.md",
            "section": "Unicode",
            "version": "0.141.0",
            "token_count": 2,
        },
    ]
    corpus_path.write_text(
        "".join(json.dumps(chunk, ensure_ascii=False) + "\n" for chunk in chunks),
        encoding="utf-8",
    )
    service = StringMatchingService(corpus_path)
    app.dependency_overrides[get_string_matching_service] = lambda: service
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


def test_av2_routes_are_registered_in_main_application() -> None:
    paths = app.openapi()["paths"]
    assert "/api/av2/string-matching/algorithms" in paths
    assert "/api/av2/string-matching/search" in paths
    assert "/api/av2/experiments" in paths
    assert "/api/health" in paths
    assert "/api/metrics" in paths
    assert "/api/query" in paths


def test_main_application_lists_all_four_matchers(web_client: TestClient) -> None:
    response = web_client.get("/api/av2/string-matching/algorithms")
    assert response.status_code == 200
    assert response.json()["algorithms"] == [
        "boyer-moore",
        "kmp",
        "naive",
        "rabin-karp",
    ]


@pytest.mark.parametrize(
    "algorithm",
    ["naive", "kmp", "boyer-moore", "rabin-karp"],
)
def test_main_application_searches_oauth2_with_real_matcher(
    web_client: TestClient,
    algorithm: str,
) -> None:
    response = web_client.post(
        "/api/av2/string-matching/search",
        json={"pattern": "OAuth2", "algorithm": algorithm, "max_results": 1},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["total_occurrences"] == 3
    assert payload["matching_chunks"] == 2
    assert payload["returned_chunks"] == 1
    assert payload["matches"][0]["positions"] == [0, 7]
    assert payload["matches"][0]["content"] == "OAuth2 OAuth2 and oauth2"


def test_unicode_positions_remain_python_code_point_offsets(
    web_client: TestClient,
) -> None:
    response = web_client.post(
        "/api/av2/string-matching/search",
        json={"pattern": "OAuth2", "algorithm": "kmp", "max_results": 2},
    )
    unicode_match = response.json()["matches"][1]
    assert unicode_match["positions"] == [1]


def test_search_validation_and_unknown_algorithm(web_client: TestClient) -> None:
    empty = web_client.post(
        "/api/av2/string-matching/search",
        json={"pattern": ""},
    )
    unknown = web_client.post(
        "/api/av2/string-matching/search",
        json={"pattern": "OAuth2", "algorithm": "unknown"},
    )
    oversized = web_client.post(
        "/api/av2/string-matching/search",
        json={"pattern": "OAuth2", "max_results": 51},
    )
    blank = web_client.post(
        "/api/av2/string-matching/search",
        json={"pattern": "   "},
    )
    assert empty.status_code == 422
    assert blank.status_code == 422
    assert unknown.status_code == 400
    assert oversized.status_code == 422


def test_frozen_experiments_have_expected_matrix() -> None:
    response = load_av2_experiments()
    assert response.metadata.algorithm_count == 4
    assert response.metadata.pattern_count == 8
    assert response.metadata.scenario_count == 24
    assert response.metadata.timing_measurements == 480
    assert response.metadata.timing_combinations == 96
    assert response.metadata.memory_combinations == 96
    assert len(response.summary) == 96
    assert len(response.memory) == 96
    assert len(response.speedups) == 96


def test_main_experiments_endpoint_returns_frozen_data(
    web_client: TestClient,
) -> None:
    response = web_client.get("/api/av2/experiments")
    assert response.status_code == 200
    assert response.json()["metadata"]["timing_measurements"] == 480
    assert len(response.json()["summary"]) == 96


def test_oauth2_speedups_match_frozen_artifacts() -> None:
    response = load_av2_experiments()
    rows = {
        row.algorithm: row
        for row in response.speedups
        if row.pattern_id == "P-OAUTH2" and row.corpus_fraction == 1.0
    }
    assert rows["boyer-moore"].mean_time_ms == pytest.approx(27.78014)
    assert rows["naive"].mean_time_ms == pytest.approx(148.276)
    assert rows["boyer-moore"].speedup_vs_naive == pytest.approx(5.337482)


def test_missing_av2_artifact_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="raw_results.csv"):
        load_av2_experiments(tmp_path)


def test_experiments_endpoint_translates_missing_file_to_503(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail() -> None:
        raise FileNotFoundError("summary.csv")

    monkeypatch.setattr(experiments_api, "get_av2_experiments", fail)
    with pytest.raises(HTTPException) as raised:
        experiments_api.experiments()
    assert raised.value.status_code == 503
    assert "summary.csv" in str(raised.value.detail)


def test_loading_dashboard_data_does_not_import_benchmark_runner() -> None:
    sys.modules.pop("scripts.av2.run_benchmarks", None)
    load_av2_experiments()
    assert "scripts.av2.run_benchmarks" not in sys.modules
