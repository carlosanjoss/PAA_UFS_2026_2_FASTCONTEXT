"""Read-only API for the frozen AV2 experimental artifacts."""

from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, cast

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.utils.config import PROJECT_ROOT

router = APIRouter(prefix="/api/av2/experiments", tags=["AV2 - Experiments"])

AV2Algorithm = Literal["naive", "kmp", "boyer-moore", "rabin-karp"]


class AV2ExperimentMetadata(BaseModel):
    """Counts derived from the persisted experimental matrix."""

    algorithm_count: int
    pattern_count: int
    scenario_count: int
    timing_measurements: int
    timing_combinations: int
    memory_combinations: int
    corpus_chunks: int
    repetitions: int
    generated_from: list[str]


class AV2Pattern(BaseModel):
    pattern_id: str
    pattern: str
    category: str
    description: str
    scope: Literal["corpus", "synthetic"]
    corpus_occurrences: int


class AV2Scenario(BaseModel):
    scenario_id: str
    pattern_id: str
    pattern: str
    pattern_category: str
    scale: str
    corpus_fraction: float
    total_characters: int
    expected_occurrences: int
    description: str


class AV2SummaryRow(BaseModel):
    scenario_id: str
    pattern_id: str
    pattern_category: str
    scale: str
    corpus_fraction: float
    algorithm: AV2Algorithm
    repetitions: int
    input_count: int
    total_characters: int
    occurrences: int
    mean_time_ms: float
    median_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    mean_comparisons: float
    mean_preprocessing_ms: float
    mean_search_ms: float


class AV2MemoryRow(BaseModel):
    scenario_id: str
    pattern_id: str
    pattern_category: str
    scale: str
    corpus_fraction: float
    algorithm: AV2Algorithm
    total_characters: int
    peak_memory_mib: float


class AV2SpeedupRow(BaseModel):
    scenario_id: str
    pattern_id: str
    pattern_category: str
    scale: str
    corpus_fraction: float
    algorithm: AV2Algorithm
    repetitions: int
    total_characters: int
    occurrences: int
    mean_time_ms: float
    std_time_ms: float
    speedup_vs_naive: float


class AV2ExperimentsResponse(BaseModel):
    metadata: AV2ExperimentMetadata
    patterns: list[AV2Pattern]
    scenarios: list[AV2Scenario]
    summary: list[AV2SummaryRow]
    memory: list[AV2MemoryRow]
    speedups: list[AV2SpeedupRow]
    environment: dict[str, Any]


@router.get("", response_model=AV2ExperimentsResponse)
def experiments() -> AV2ExperimentsResponse:
    """Return frozen AV2 results without executing any benchmark."""
    try:
        return get_av2_experiments()
    except (FileNotFoundError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Artefatos experimentais da AV2 indisponíveis: {exc}",
        ) from exc


@lru_cache(maxsize=1)
def get_av2_experiments() -> AV2ExperimentsResponse:
    """Cache parsed, immutable experiment files for the API process."""
    return load_av2_experiments()


def load_av2_experiments(
    repository_root: Path = PROJECT_ROOT,
) -> AV2ExperimentsResponse:
    """Load and validate all AV2 presentation artifacts from disk."""
    paths = {
        "raw": repository_root / "experiments/av2/final/raw_results.csv",
        "summary": repository_root / "experiments/av2/final/summary.csv",
        "memory_probe": repository_root
        / "experiments/av2/memory_probe/memory_results.csv",
        "speedups": repository_root / "reports/tables/av2/speedups.csv",
        "memory": repository_root / "reports/tables/av2/memory_analysis.csv",
        "environment": repository_root / "reports/tables/av2/environment.json",
        "patterns": repository_root / "data/av2/patterns.json",
        "scenarios": repository_root / "data/av2/scenarios.json",
    }
    missing = [
        str(path.relative_to(repository_root))
        for path in paths.values()
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(", ".join(missing))

    raw_rows = _read_csv(paths["raw"])
    summary_rows = _read_csv(paths["summary"])
    memory_probe_rows = _read_csv(paths["memory_probe"])
    speedup_rows = _read_csv(paths["speedups"])
    memory_rows = _read_csv(paths["memory"])
    pattern_payload = _read_json(paths["patterns"])
    scenario_payload = _read_json(paths["scenarios"])
    environment_payload = _read_json(paths["environment"])

    patterns = [AV2Pattern(**item) for item in pattern_payload["patterns"]]
    scenarios = [
        AV2Scenario(
            scenario_id=item["scenario_id"],
            pattern_id=item["pattern_id"],
            pattern=item["pattern"],
            pattern_category=item["pattern_category"],
            scale=item["scale"],
            corpus_fraction=float(item["corpus_fraction"]),
            total_characters=int(item["total_characters"]),
            expected_occurrences=int(item["expected_occurrences"]),
            description=item["description"],
        )
        for item in scenario_payload["scenarios"]
    ]
    scenario_by_id = {scenario.scenario_id: scenario for scenario in scenarios}

    summary = [AV2SummaryRow(**_typed_summary(row)) for row in summary_rows]
    speedups = [AV2SpeedupRow(**_typed_speedup(row)) for row in speedup_rows]
    memory = [
        AV2MemoryRow(
            scenario_id=row["scenario_id"],
            pattern_id=row["pattern_id"],
            pattern_category=scenario_by_id[row["scenario_id"]].pattern_category,
            scale=row["scale"],
            corpus_fraction=float(row["corpus_fraction"]),
            algorithm=cast(AV2Algorithm, row["algorithm"]),
            total_characters=int(row["total_characters"]),
            peak_memory_mib=float(row["peak_memory_mib"]),
        )
        for row in memory_rows
    ]

    algorithm_names = {row.algorithm for row in summary}
    repetitions = {row.repetitions for row in summary}
    if len(repetitions) != 1:
        raise ValueError("inconsistent repetition counts in summary.csv")
    if len(memory_probe_rows) != len(memory):
        raise ValueError("memory artifact row counts do not agree")

    metadata = AV2ExperimentMetadata(
        algorithm_count=len(algorithm_names),
        pattern_count=len(patterns),
        scenario_count=len(scenarios),
        timing_measurements=len(raw_rows),
        timing_combinations=len(summary),
        memory_combinations=len(memory_probe_rows),
        corpus_chunks=int(pattern_payload["corpus"]["chunk_count"]),
        repetitions=next(iter(repetitions)),
        generated_from=[
            str(path.relative_to(repository_root)).replace("\\", "/")
            for path in paths.values()
        ],
    )

    return AV2ExperimentsResponse(
        metadata=metadata,
        patterns=patterns,
        scenarios=scenarios,
        summary=summary,
        memory=memory,
        speedups=speedups,
        environment=_sanitize_environment(environment_payload),
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain a JSON object")
    return value


def _typed_summary(row: dict[str, str]) -> dict[str, Any]:
    typed: dict[str, Any] = dict(row)
    for key in {"repetitions", "input_count", "total_characters", "occurrences"}:
        typed[key] = int(row[key])
    for key in {
        "corpus_fraction",
        "mean_time_ms",
        "median_time_ms",
        "std_time_ms",
        "min_time_ms",
        "max_time_ms",
        "mean_comparisons",
        "mean_preprocessing_ms",
        "mean_search_ms",
    }:
        typed[key] = float(row[key])
    return typed


def _typed_speedup(row: dict[str, str]) -> dict[str, Any]:
    typed: dict[str, Any] = dict(row)
    for key in {"repetitions", "total_characters", "occurrences"}:
        typed[key] = int(row[key])
    for key in {"corpus_fraction", "mean_time_ms", "std_time_ms", "speedup_vs_naive"}:
        typed[key] = float(row[key])
    return typed


def _sanitize_environment(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "project": payload.get("project"),
        "captured_at": payload.get("captured_at"),
        "python": payload.get("python", {}),
        "system": payload.get("system", {}),
        "packages": payload.get("packages", {}),
        "experimental_matrix": payload.get("experimental_matrix", {}),
        "notes": payload.get("notes", []),
    }
