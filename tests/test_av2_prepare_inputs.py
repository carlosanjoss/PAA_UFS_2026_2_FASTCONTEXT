from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock, call

import pytest

from scripts.av2.prepare_inputs import (
    DEFAULT_PATTERN_CANDIDATES_PATH,
    iter_match_inputs,
    prepare_inputs,
)
from src.ingestion.chunk_loader import load_chunks_jsonl
from src.string_matching.naive import NaiveMatcher


def _write_chunks(path: Path) -> None:
    chunks = [
        {
            "chunk_id": chunk_id,
            "content": (
                "FastAPI dependency OAuth2 WebSocket background tasks " + "a" * 20
            ),
            "source_path": f"{chunk_id}.md",
            "version": "0.141.0",
            "token_count": 10,
            "metadata": {"preserve_this": True},
        }
        for chunk_id in ("b-section-0", "a-section-0", "c-section-0")
    ]
    path.write_text(
        "".join(json.dumps(chunk) + "\n" for chunk in chunks),
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_pattern_candidates(path: Path) -> None:
    path.write_bytes(DEFAULT_PATTERN_CANDIDATES_PATH.read_bytes())


def test_prepare_inputs_is_deterministic_and_preserves_corpus(
    tmp_path: Path,
) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    candidates_path = tmp_path / "pattern_candidates.json"
    patterns_path = tmp_path / "patterns.json"
    scenarios_path = tmp_path / "scenarios.json"
    _write_chunks(chunks_path)
    _copy_pattern_candidates(candidates_path)
    original_candidates = candidates_path.read_bytes()
    original_chunks = chunks_path.read_bytes()

    prepare_inputs(
        chunks_path=chunks_path,
        pattern_candidates_path=candidates_path,
        patterns_path=patterns_path,
        scenarios_path=scenarios_path,
        allow_corpus_mismatch=True,
    )
    first_patterns = patterns_path.read_bytes()
    first_scenarios = scenarios_path.read_bytes()
    prepare_inputs(
        chunks_path=chunks_path,
        pattern_candidates_path=candidates_path,
        patterns_path=patterns_path,
        scenarios_path=scenarios_path,
        allow_corpus_mismatch=True,
    )

    assert patterns_path.read_bytes() == first_patterns
    assert scenarios_path.read_bytes() == first_scenarios
    assert candidates_path.read_bytes() == original_candidates
    assert chunks_path.read_bytes() == original_chunks

    patterns = _read_json(patterns_path)["patterns"]
    scenarios_data = _read_json(scenarios_path)
    corpus_manifest = _read_json(patterns_path)["corpus"]
    patterns_by_id = {item["pattern_id"]: item for item in patterns}
    assert corpus_manifest["origin"] == (
        "https://github.com/fastapi/fastapi/tree/0.141.0/docs/en/docs"
    )
    assert corpus_manifest["document_count"] == 3
    assert corpus_manifest["chunk_count"] == 3
    assert corpus_manifest["matches_known_av1"] is False
    assert patterns_by_id["P-ABSENT-SENTINEL"]["corpus_occurrences"] == 0
    assert patterns_by_id["P-SHORT-COMMON"]["corpus_occurrences"] >= 10
    assert patterns_by_id["P-OAUTH2"]["corpus_occurrences"] > 0
    assert patterns_by_id["P-DEPENDENCY"]["corpus_occurrences"] > 0
    assert patterns_by_id["P-WEBSOCKET"]["corpus_occurrences"] > 0
    assert patterns_by_id["P-LONG-BACKGROUND-TASKS"]["corpus_occurrences"] > 0
    assert len(patterns) == 8
    assert len(scenarios_data["scenarios"]) == 24
    assert patterns_by_id["P-REPETITIVE"]["expected_positions"] == list(range(7))
    assert "candidate_patterns" not in _read_json(patterns_path)
    assert scenarios_data["status"] == "corpus_mismatch_approved"
    assert list(scenarios_data["scenarios"][0])[:5] == [
        "scenario_id",
        "pattern",
        "pattern_type",
        "corpus_fraction",
        "expected_case",
    ]

    fraction_scenarios = [
        scenario
        for scenario in scenarios_data["scenarios"]
        if scenario["pattern_id"] == "P-SHORT-COMMON"
    ]
    assert [item["corpus_fraction"] for item in fraction_scenarios] == [
        0.25,
        0.5,
        1.0,
    ]
    assert [len(item["chunk_ids"]) for item in fraction_scenarios] == [1, 2, 3]
    for index, smaller in enumerate(fraction_scenarios[:-1]):
        larger = fraction_scenarios[index + 1]
        assert set(smaller["chunk_ids"]).issubset(larger["chunk_ids"])
    assert all(
        item["chunk_ids"] == sorted(item["chunk_ids"]) for item in fraction_scenarios
    )

    overlap = next(
        item
        for item in scenarios_data["scenarios"]
        if item["pattern_id"] == "P-OVERLAPPING" and item["corpus_fraction"] == 0.25
    )
    assert patterns_by_id["P-OVERLAPPING"]["expected_positions"] == [0, 2, 4]
    assert overlap["scale"] == "synthetic_microbenchmark"
    assert overlap["pattern_type"] == "overlapping"
    assert overlap["corpus_fraction"] == 0.25
    corpus_scale = next(
        item
        for item in scenarios_data["scenarios"]
        if item["pattern_id"] == "P-OAUTH2" and item["corpus_fraction"] == 0.25
    )
    assert overlap["total_characters"] == corpus_scale["total_characters"]
    assert overlap["reference_characters"] == corpus_scale["total_characters"]
    assert overlap["synthetic_text"].startswith("ABABABA")
    assert overlap["expected_occurrences"] == len(
        NaiveMatcher().search(overlap["synthetic_text"], "ABA").positions
    )

    for pattern in patterns:
        pattern_scenarios = [
            item
            for item in scenarios_data["scenarios"]
            if item["pattern_id"] == pattern["pattern_id"]
        ]
        assert [item["corpus_fraction"] for item in pattern_scenarios] == [
            0.25,
            0.5,
            1.0,
        ]
        expected_scale = (
            "synthetic_microbenchmark" if pattern["scope"] == "synthetic" else "corpus"
        )
        assert all(item["scale"] == expected_scale for item in pattern_scenarios)
        if pattern["scope"] == "synthetic":
            for scenario, fraction in zip(
                pattern_scenarios, (0.25, 0.5, 1.0), strict=True
            ):
                corpus_scale = next(
                    item
                    for item in scenarios_data["scenarios"]
                    if item["pattern_id"] == "P-OAUTH2"
                    and item["corpus_fraction"] == fraction
                )
                assert scenario["total_characters"] == corpus_scale["total_characters"]


def test_shared_iterator_supplies_original_text_and_identical_pattern(
    tmp_path: Path,
) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    candidates_path = tmp_path / "pattern_candidates.json"
    patterns_path = tmp_path / "patterns.json"
    scenarios_path = tmp_path / "scenarios.json"
    _write_chunks(chunks_path)
    _copy_pattern_candidates(candidates_path)
    patterns, scenarios = prepare_inputs(
        chunks_path=chunks_path,
        pattern_candidates_path=candidates_path,
        patterns_path=patterns_path,
        scenarios_path=scenarios_path,
        allow_corpus_mismatch=True,
    )
    chunks = load_chunks_jsonl(chunks_path)

    cases = list(iter_match_inputs(scenarios, patterns, chunks))
    target = next(
        scenario
        for scenario in scenarios
        if scenario["pattern_id"] == "P-DEPENDENCY"
        and scenario["corpus_fraction"] == 1.0
    )
    target_inputs = [
        (chunk_id, text, pattern)
        for scenario_id, chunk_id, text, pattern in cases
        if scenario_id == target["scenario_id"]
    ]

    assert [item[0] for item in target_inputs] == sorted(
        chunk["chunk_id"] for chunk in chunks
    )
    assert {(text, pattern) for _, text, pattern in target_inputs} == {
        (chunk["content"], "dependency") for chunk in chunks
    }

    synthetic_target = next(
        scenario
        for scenario in scenarios
        if scenario["pattern_id"] == "P-OVERLAPPING"
        and scenario["corpus_fraction"] == 0.25
    )
    synthetic_input = next(
        case for case in cases if case[0] == synthetic_target["scenario_id"]
    )
    assert synthetic_input[1] == "synthetic"
    assert len(synthetic_input[2]) == synthetic_target["total_characters"]
    assert synthetic_input[2].startswith("ABABABA")
    assert synthetic_input[3] == "ABA"
    result = NaiveMatcher().search(synthetic_input[2], synthetic_input[3])
    assert result.occurrences == synthetic_target["expected_occurrences"]
    assert result.positions[:3] == (0, 2, 4)

    repetitive_target = next(
        scenario
        for scenario in scenarios
        if scenario["pattern_id"] == "P-REPETITIVE"
        and scenario["corpus_fraction"] == 0.25
    )
    repetitive_input = next(
        case for case in cases if case[0] == repetitive_target["scenario_id"]
    )
    assert len(repetitive_input[2]) == repetitive_target["total_characters"]
    assert repetitive_input[2].startswith("AAAAAAAAAA")
    assert repetitive_input[3] == "AAAA"
    result = NaiveMatcher().search(repetitive_input[2], repetitive_input[3])
    assert result.occurrences == repetitive_target["expected_occurrences"]
    assert result.positions[:7] == tuple(range(7))


def test_frozen_corpus_manifest_matches_av1_reference(tmp_path: Path) -> None:
    patterns_path = tmp_path / "patterns.json"
    scenarios_path = tmp_path / "scenarios.json"

    prepare_inputs(patterns_path=patterns_path, scenarios_path=scenarios_path)

    manifest = _read_json(patterns_path)["corpus"]
    assert manifest["origin"] == (
        "https://github.com/fastapi/fastapi/tree/0.141.0/docs/en/docs"
    )
    assert manifest["version"] == "0.141.0"
    assert manifest["document_count"] == 153
    assert manifest["chunk_count"] == 1305
    assert manifest["fingerprint"] == (
        "e08b47b55d7237f38bcaba48c79a468a40b92e4f3120e968ae9ece999f5e8c2d"
    )
    assert manifest["matches_known_av1"] is True


def test_all_scenarios_load_into_one_deterministic_input_stream(
    tmp_path: Path,
) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    candidates_path = tmp_path / "pattern_candidates.json"
    patterns_path = tmp_path / "patterns.json"
    scenarios_path = tmp_path / "scenarios.json"
    _write_chunks(chunks_path)
    _copy_pattern_candidates(candidates_path)
    prepare_inputs(
        chunks_path=chunks_path,
        pattern_candidates_path=candidates_path,
        patterns_path=patterns_path,
        scenarios_path=scenarios_path,
        allow_corpus_mismatch=True,
    )

    patterns = _read_json(patterns_path)["patterns"]
    scenarios = _read_json(scenarios_path)["scenarios"]
    chunks = load_chunks_jsonl(chunks_path)
    first_run = list(iter_match_inputs(scenarios, patterns, chunks))
    second_run = list(iter_match_inputs(scenarios, patterns, chunks))

    assert first_run == second_run
    assert len(scenarios) == 24
    assert {item[0] for item in first_run} == {
        scenario["scenario_id"] for scenario in scenarios
    }
    assert all(text and pattern for _, _, text, pattern in first_run)


def test_four_algorithm_consumer_slots_receive_identical_inputs(
    tmp_path: Path,
) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    candidates_path = tmp_path / "pattern_candidates.json"
    patterns_path = tmp_path / "patterns.json"
    scenarios_path = tmp_path / "scenarios.json"
    _write_chunks(chunks_path)
    _copy_pattern_candidates(candidates_path)
    prepare_inputs(
        chunks_path=chunks_path,
        pattern_candidates_path=candidates_path,
        patterns_path=patterns_path,
        scenarios_path=scenarios_path,
        allow_corpus_mismatch=True,
    )

    patterns = _read_json(patterns_path)["patterns"]
    scenarios = _read_json(scenarios_path)["scenarios"]
    chunks = load_chunks_jsonl(chunks_path)
    shared_inputs = list(iter_match_inputs(scenarios, patterns, chunks))
    consumers = {
        name: Mock(name=f"{name}_consumer", spec=["search"])
        for name in ("naive", "kmp", "boyer_moore", "rabin_karp")
    }

    for _, _, text, pattern in shared_inputs:
        for consumer in consumers.values():
            consumer.search(text, pattern)

    expected_calls = [
        call.search(text, pattern) for _, _, text, pattern in shared_inputs
    ]
    assert len(consumers) == 4
    assert all(
        consumer.search.call_args_list == expected_calls
        for consumer in consumers.values()
    )


def test_prepared_scenarios_run_through_available_matcher(
    tmp_path: Path,
) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    candidates_path = tmp_path / "pattern_candidates.json"
    patterns_path = tmp_path / "patterns.json"
    scenarios_path = tmp_path / "scenarios.json"
    _write_chunks(chunks_path)
    _copy_pattern_candidates(candidates_path)
    prepare_inputs(
        chunks_path=chunks_path,
        pattern_candidates_path=candidates_path,
        patterns_path=patterns_path,
        scenarios_path=scenarios_path,
        allow_corpus_mismatch=True,
    )

    patterns = _read_json(patterns_path)["patterns"]
    scenarios = _read_json(scenarios_path)["scenarios"]
    chunks = load_chunks_jsonl(chunks_path)
    shared_inputs = list(iter_match_inputs(scenarios, patterns, chunks))
    matcher = NaiveMatcher()
    observed_occurrences = {scenario["scenario_id"]: 0 for scenario in scenarios}
    inputs_by_scenario = {scenario["scenario_id"]: 0 for scenario in scenarios}

    assert shared_inputs
    for scenario_id, _, text, pattern in shared_inputs:
        result = matcher.search(text, pattern)
        expected_positions = tuple(
            position
            for position in range(len(text) - len(pattern) + 1)
            if text.startswith(pattern, position)
        )
        assert result.positions == expected_positions
        observed_occurrences[scenario_id] += result.occurrences
        inputs_by_scenario[scenario_id] += 1

    expected_by_scenario = {
        scenario["scenario_id"]: scenario["expected_occurrences"]
        for scenario in scenarios
    }
    assert observed_occurrences == expected_by_scenario
    assert all(input_count > 0 for input_count in inputs_by_scenario.values())

    # Synthetic scenarios are searched as one generated text, not as corpus chunks.
    for scenario in scenarios:
        if scenario["scale"] == "synthetic_microbenchmark":
            assert inputs_by_scenario[scenario["scenario_id"]] == 1
        else:
            assert inputs_by_scenario[scenario["scenario_id"]] == len(
                scenario["chunk_ids"]
            )


def test_corpus_mismatch_requires_explicit_approval(
    tmp_path: Path,
) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    candidates_path = tmp_path / "pattern_candidates.json"
    patterns_path = tmp_path / "patterns.json"
    scenarios_path = tmp_path / "scenarios.json"
    _write_chunks(chunks_path)
    _copy_pattern_candidates(candidates_path)

    with pytest.raises(ValueError, match="Corpus differs from frozen AV1 reference"):
        prepare_inputs(
            chunks_path=chunks_path,
            pattern_candidates_path=candidates_path,
            patterns_path=patterns_path,
            scenarios_path=scenarios_path,
        )

    assert not scenarios_path.exists()


def test_rejects_duplicate_chunk_ids(tmp_path: Path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    _write_chunks(chunks_path)
    records = [
        json.loads(line)
        for line in chunks_path.read_text(encoding="utf-8").splitlines()
    ]
    records[1]["chunk_id"] = records[0]["chunk_id"]
    chunks_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate chunk_id"):
        load_chunks_jsonl(chunks_path)
