
"""Tests for the FastContext AV2 benchmark runner."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.av2.run_benchmarks import execute_benchmarks


@pytest.fixture
def benchmark_files(tmp_path: Path) -> dict[str, Path]:
    """Build a small, isolated experimental dataset."""

    corpus_path = tmp_path / "chunks.jsonl"
    patterns_path = tmp_path / "patterns.json"
    scenarios_path = tmp_path / "scenarios.json"
    output_dir = tmp_path / "results"

    chunks = [
        {
            "chunk_id": "chunk-001",
            "content": "ABABABA",
            "source_path": "doc1.md",
            "version": "0.141.0",
            "token_count": 3,
        },
        {
            "chunk_id": "chunk-002",
            "content": "BANANA",
            "source_path": "doc2.md",
            "version": "0.141.0",
            "token_count": 3,
        },
    ]

    with corpus_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk) + "\n")

    manifest = {
        "chunk_count": 2,
        "matches_known_av1": True,
        "fingerprint": "test-fixture",
    }

    patterns: dict[str, Any] = {
        "status": "ready",
        "corpus": manifest,
        "patterns": [
            {
                "pattern_id": "P001",
                "pattern": "ABA",
                "category": "overlapping",
                "scope": "corpus",
            },
            {
                "pattern_id": "P002",
                "pattern": "AAA",
                "category": "repetitive",
                "scope": "synthetic",
            },
        ],
    }

    scenarios: dict[str, Any] = {
        "status": "ready",
        "corpus": manifest,
        "scenarios": [
            {
                "scenario_id": "S001",
                "pattern_id": "P001",
                "pattern": "ABA",
                "pattern_category": "overlapping",
                "scale": "corpus",
                "corpus_fraction": 1.0,
                "chunk_ids": [
                    "chunk-001",
                    "chunk-002",
                ],
                "total_characters": 13,
                "expected_occurrences": 3,
            },
            {
                "scenario_id": "S002",
                "pattern_id": "P002",
                "pattern": "AAA",
                "pattern_category": "repetitive",
                "scale": "synthetic_microbenchmark",
                "corpus_fraction": 1.0,
                "chunk_ids": [],
                "synthetic_text": "AAAAAA",
                "total_characters": 6,
                "expected_occurrences": 4,
            },
        ],
    }

    patterns_path.write_text(
        json.dumps(patterns),
        encoding="utf-8",
    )
    scenarios_path.write_text(
        json.dumps(scenarios),
        encoding="utf-8",
    )

    return {
        "chunks_path": corpus_path,
        "patterns_path": patterns_path,
        "scenarios_path": scenarios_path,
        "output_dir": output_dir,
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read benchmark output rows."""

    with path.open(
        encoding="utf-8",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


def test_complete_small_matrix(
    benchmark_files: dict[str, Path],
) -> None:
    """Four algorithms, two scenarios and two repetitions."""

    raw_count, summary_count = execute_benchmarks(
        repetitions=2,
        warmup=1,
        **benchmark_files,
    )

    assert raw_count == 16
    assert summary_count == 8

    output_dir = benchmark_files["output_dir"]

    raw = read_csv(output_dir / "raw_results.csv")
    summary = read_csv(output_dir / "summary.csv")

    assert len(raw) == 16
    assert len(summary) == 8

    assert {row["algorithm"] for row in raw} == {
        "naive",
        "kmp",
        "boyer-moore",
        "rabin-karp",
    }


def test_occurrence_equivalence(
    benchmark_files: dict[str, Path],
) -> None:
    """All algorithms must report the expected occurrences."""

    execute_benchmarks(
        repetitions=1,
        warmup=0,
        **benchmark_files,
    )

    raw = read_csv(
        benchmark_files["output_dir"] / "raw_results.csv"
    )

    for row in raw:
        expected = (
            3 if row["scenario_id"] == "S001" else 4
        )

        assert int(row["occurrences"]) == expected
        assert int(row["total_characters"]) > 0
        assert int(row["comparisons"]) >= 0
        assert int(row["wall_time_ns"]) >= 0


def test_scenario_and_algorithm_filters(
    benchmark_files: dict[str, Path],
) -> None:
    """The runner must support restricted experiment runs."""

    raw_count, summary_count = execute_benchmarks(
        scenarios_filter={"S001"},
        algorithms_filter={"kmp"},
        repetitions=3,
        warmup=0,
        **benchmark_files,
    )

    assert raw_count == 3
    assert summary_count == 1

    rows = read_csv(
        benchmark_files["output_dir"] / "summary.csv"
    )

    assert rows[0]["scenario_id"] == "S001"
    assert rows[0]["algorithm"] == "kmp"
    assert rows[0]["repetitions"] == "3"


def test_memory_output(
    benchmark_files: dict[str, Path],
) -> None:
    """Memory results must be generated when explicitly requested."""

    execute_benchmarks(
        scenarios_filter={"S001"},
        algorithms_filter={"naive"},
        repetitions=1,
        warmup=0,
        measure_memory=True,
        **benchmark_files,
    )

    path = (
        benchmark_files["output_dir"]
        / "memory_results.csv"
    )

    assert path.is_file()

    rows = read_csv(path)

    assert len(rows) == 1
    assert rows[0]["algorithm"] == "naive"
    assert int(rows[0]["peak_memory_bytes"]) > 0


def test_unknown_algorithm_rejected(
    benchmark_files: dict[str, Path],
) -> None:
    """Unknown algorithms must not be silently ignored."""

    with pytest.raises(ValueError, match="Unknown algorithms"):
        execute_benchmarks(
            algorithms_filter={"invalid"},
            repetitions=1,
            warmup=0,
            **benchmark_files,
        )


def test_invalid_expected_occurrences(
    benchmark_files: dict[str, Path],
) -> None:
    """The runner must stop if scenario ground truth is wrong."""

    scenarios_path = benchmark_files["scenarios_path"]

    document = json.loads(
        scenarios_path.read_text(encoding="utf-8")
    )

    document["scenarios"][0]["expected_occurrences"] = 999

    scenarios_path.write_text(
        json.dumps(document),
        encoding="utf-8",
    )

    with pytest.raises(
        AssertionError,
        match="Prepared occurrence count mismatch",
    ):
        execute_benchmarks(
            scenarios_filter={"S001"},
            repetitions=1,
            warmup=0,
            **benchmark_files,
        )


def test_missing_scenarios_file(
    benchmark_files: dict[str, Path],
) -> None:
    """A missing experimental file must cause an explicit error."""

    benchmark_files["scenarios_path"].unlink()

    with pytest.raises(FileNotFoundError):
        execute_benchmarks(
            repetitions=1,
            warmup=0,
            **benchmark_files,
        )
