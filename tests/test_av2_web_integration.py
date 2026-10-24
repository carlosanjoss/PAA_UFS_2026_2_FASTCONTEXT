"""Integration coverage for the AV2 web API and frozen reports."""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from src.rag.pipeline import RAGPipeline
from src.rag.providers.base import (
    GenerationConfig,
    LLMMessage,
    LLMProvider,
    LLMResponse,
)
from src.services.string_matching_service import StringMatchingService
from src.string_matching import api as string_matching_api
from src.string_matching import experiments_api
from src.string_matching.api import get_string_matching_service
from src.string_matching.experiments_api import load_av2_experiments
from src.web_api import app


class FakeAV2Provider(LLMProvider):
    """Deterministic provider for the AV2 grounded-generation contract."""

    def __init__(self, *, available: bool = True) -> None:
        self.available = available
        self.calls = 0

    @property
    def name(self) -> str:
        return "fake-av2"

    def is_available(self) -> bool:
        return self.available

    def generate(
        self,
        messages: Sequence[LLMMessage],
        config: GenerationConfig | None = None,
    ) -> LLMResponse:
        self.calls += 1
        assert messages
        assert config is not None
        return LLMResponse(
            text="OAuth2 aparece no contexto selecionado [security-1].",
            model="fake-model",
            provider=self.name,
        )


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


def test_av2_can_generate_grounded_answer_from_exact_matches(
    web_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = FakeAV2Provider()
    runtime = SimpleNamespace(
        provider=provider,
        rag_pipeline=RAGPipeline(
            provider,
            max_truncation_retries=0,
            max_citation_retries=0,
        ),
    )
    monkeypatch.setattr(string_matching_api, "get_application", lambda: runtime)

    response = web_client.post(
        "/api/av2/string-matching/search",
        json={
            "pattern": "OAuth2",
            "algorithm": "boyer-moore",
            "max_results": 1,
            "use_llm": True,
            "question": "O que o contexto informa sobre OAuth2?",
            "llm_context_limit": 1,
            "temperature": 0.2,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["total_occurrences"] == 3
    assert payload["llm"]["status"] == "generated"
    assert payload["llm"]["answer"].startswith("OAuth2 aparece")
    assert payload["llm"]["citations"] == ["security-1"]
    assert payload["llm"]["citation_valid"] is True
    assert payload["llm"]["context_chunk_ids"] == ["security-1"]
    assert provider.calls == 1


def test_av2_search_survives_unavailable_llm(
    web_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = FakeAV2Provider(available=False)
    runtime = SimpleNamespace(
        provider=provider,
        rag_pipeline=RAGPipeline(provider),
    )
    monkeypatch.setattr(string_matching_api, "get_application", lambda: runtime)

    response = web_client.post(
        "/api/av2/string-matching/search",
        json={
            "pattern": "OAuth2",
            "use_llm": True,
            "question": "Explique OAuth2.",
        },
    )

    assert response.status_code == 200
    assert response.json()["total_occurrences"] == 3
    assert response.json()["llm"]["status"] == "unavailable"
    assert response.json()["llm"]["answer"] is None
    assert provider.calls == 0


def test_av2_does_not_call_llm_without_exact_context(
    web_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_loaded() -> None:
        raise AssertionError("LLM runtime should not load without exact matches")

    monkeypatch.setattr(string_matching_api, "get_application", fail_if_loaded)
    response = web_client.post(
        "/api/av2/string-matching/search",
        json={
            "pattern": "MISSING_PATTERN",
            "use_llm": True,
            "question": "Há informação sobre este padrão?",
        },
    )

    assert response.status_code == 200
    assert response.json()["total_occurrences"] == 0
    assert response.json()["llm"]["status"] == "no_context"


def test_av2_requires_question_when_llm_is_enabled(
    web_client: TestClient,
) -> None:
    response = web_client.post(
        "/api/av2/string-matching/search",
        json={
            "pattern": "OAuth2",
            "use_llm": True,
            "question": "   ",
        },
    )

    assert response.status_code == 422


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
