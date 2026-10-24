
"""Reproducible benchmarks for FastContext AV2 string matching."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import tracemalloc
from collections import defaultdict
from pathlib import Path
from time import perf_counter_ns
from typing import Any

from scripts.av2.prepare_inputs import (
    DEFAULT_PATTERNS_PATH,
    DEFAULT_SCENARIOS_PATH,
    iter_match_inputs,
)
from src.ingestion.chunk_loader import (
    DEFAULT_CHUNKS_PATH,
    load_chunks_jsonl,
)
from src.string_matching.base import StringMatcher
from src.string_matching.models import StringMatchResult
from src.string_matching.naive import NaiveMatcher
from src.string_matching.registry import create_default_registry

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "av2"

RAW_FIELDS = [
    "scenario_id",
    "pattern_id",
    "pattern_category",
    "scale",
    "corpus_fraction",
    "algorithm",
    "repetition",
    "input_count",
    "total_characters",
    "pattern_length",
    "occurrences",
    "comparisons",
    "preprocessing_ns",
    "search_ns",
    "algorithm_total_ns",
    "wall_time_ns",
]

MEMORY_FIELDS = [
    "scenario_id",
    "algorithm",
    "peak_memory_bytes",
    "peak_memory_mb",
]


def load_json(path: Path) -> dict[str, Any]:
    """Read a prepared AV2 JSON document."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Required AV2 file not found: {path}"
        )

    if path.stat().st_size == 0:
        raise ValueError(
            f"AV2 file is empty: {path}. "
            "Run python -m scripts.av2.prepare_inputs."
        )

    with path.open(encoding="utf-8") as handle:
        document = json.load(handle)

    if not isinstance(document, dict):
        raise ValueError(f"Expected a JSON object: {path}")

    return document


def load_experimental_data(
    patterns_path: Path,
    scenarios_path: Path,
    chunks_path: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Load prepared inputs and verify their common corpus."""

    patterns_document = load_json(patterns_path)
    scenarios_document = load_json(scenarios_path)

    if (
        patterns_document.get("status") != "ready"
        or scenarios_document.get("status") != "ready"
    ):
        raise ValueError("AV2 input files are not marked as ready.")

    patterns_manifest = patterns_document.get("corpus", {})
    scenarios_manifest = scenarios_document.get("corpus", {})

    if patterns_manifest != scenarios_manifest:
        raise ValueError("Pattern and scenario corpus manifests differ.")

    if not patterns_manifest.get("matches_known_av1", False):
        raise ValueError("Prepared data does not match frozen AV1.")

    patterns = patterns_document.get("patterns")
    scenarios = scenarios_document.get("scenarios")

    if not isinstance(patterns, list) or not patterns:
        raise ValueError("No prepared AV2 patterns were found.")

    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("No prepared AV2 scenarios were found.")

    chunks = load_chunks_jsonl(chunks_path)

    if len(chunks) != patterns_manifest["chunk_count"]:
        raise ValueError("Corpus size differs from the AV2 manifest.")

    return patterns, scenarios, chunks


def run_matcher(
    matcher: StringMatcher,
    inputs: list[tuple[str, str]],
) -> tuple[list[StringMatchResult], int]:
    """Execute all inputs, measuring wall time without I/O."""

    start_ns = perf_counter_ns()

    results = [
        matcher.search(text, pattern)
        for text, pattern in inputs
    ]

    wall_time_ns = perf_counter_ns() - start_ns

    return results, wall_time_ns


def validate_results(
    results: list[StringMatchResult],
    expected_positions: list[tuple[int, ...]],
    algorithm: str,
    scenario_id: str,
) -> None:
    """Fail immediately on a disagreement with the Naive oracle."""

    if len(results) != len(expected_positions):
        raise AssertionError(
            f"{algorithm}: result count mismatch in {scenario_id}"
        )

    for index, (result, expected) in enumerate(
        zip(results, expected_positions, strict=True)
    ):
        if result.positions != expected:
            raise AssertionError(
                f"CORRECTNESS FAILURE: algorithm={algorithm}, "
                f"scenario={scenario_id}, input={index}, "
                f"expected={expected[:20]}, "
                f"actual={result.positions[:20]}"
            )


def aggregate_results(
    results: list[StringMatchResult],
) -> dict[str, int]:
    """Aggregate per-input algorithm metrics."""

    preprocessing = sum(
        item.preprocessing_time_ns for item in results
    )
    search = sum(item.search_time_ns for item in results)

    return {
        "occurrences": sum(
            item.occurrences for item in results
        ),
        "comparisons": sum(
            item.comparisons for item in results
        ),
        "preprocessing_ns": preprocessing,
        "search_ns": search,
        "algorithm_total_ns": preprocessing + search,
    }


def measure_peak_memory(
    matcher: StringMatcher,
    inputs: list[tuple[str, str]],
) -> int:
    """Measure Python allocation peak in a separate execution."""

    tracemalloc.start()

    try:
        results, _ = run_matcher(matcher, inputs)
        _ = results
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    return peak_bytes


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    """Write a UTF-8 CSV with an explicit column order."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def summarize(
    raw_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize repetitions for each scenario and algorithm."""

    groups: dict[
        tuple[str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in raw_rows:
        key = (
            str(row["scenario_id"]),
            str(row["algorithm"]),
        )
        groups[key].append(row)

    summary: list[dict[str, Any]] = []

    for (scenario_id, algorithm), group in sorted(
        groups.items()
    ):
        times_ms = [
            float(row["wall_time_ns"]) / 1_000_000
            for row in group
        ]
        comparisons = [
            int(row["comparisons"])
            for row in group
        ]

        summary.append(
            {
                "scenario_id": scenario_id,
                "pattern_id": group[0]["pattern_id"],
                "pattern_category": group[0]["pattern_category"],
                "scale": group[0]["scale"],
                "corpus_fraction": group[0]["corpus_fraction"],
                "algorithm": algorithm,
                "repetitions": len(group),
                "input_count": group[0]["input_count"],
                "total_characters": group[0]["total_characters"],
                "occurrences": group[0]["occurrences"],
                "mean_time_ms": round(
                    statistics.fmean(times_ms), 6
                ),
                "median_time_ms": round(
                    statistics.median(times_ms), 6
                ),
                "std_time_ms": round(
                    statistics.stdev(times_ms)
                    if len(times_ms) > 1
                    else 0.0,
                    6,
                ),
                "min_time_ms": round(min(times_ms), 6),
                "max_time_ms": round(max(times_ms), 6),
                "mean_comparisons": round(
                    statistics.fmean(comparisons), 2
                ),
                "mean_preprocessing_ms": round(
                    statistics.fmean(
                        float(row["preprocessing_ns"]) / 1_000_000
                        for row in group
                    ),
                    6,
                ),
                "mean_search_ms": round(
                    statistics.fmean(
                        float(row["search_ns"]) / 1_000_000
                        for row in group
                    ),
                    6,
                ),
            }
        )

    return summary


def execute_benchmarks(
    *,
    scenarios_filter: set[str] | None = None,
    algorithms_filter: set[str] | None = None,
    repetitions: int = 5,
    warmup: int = 1,
    measure_memory: bool = False,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    patterns_path: Path = DEFAULT_PATTERNS_PATH,
    scenarios_path: Path = DEFAULT_SCENARIOS_PATH,
    chunks_path: Path = DEFAULT_CHUNKS_PATH,
) -> tuple[int, int]:
    """Execute the experimental matrix and write result files."""

    if repetitions < 1 or warmup < 0:
        raise ValueError("Invalid repetitions or warmup count.")

    patterns, scenarios, chunks = load_experimental_data(
        patterns_path,
        scenarios_path,
        chunks_path,
    )

    registry = create_default_registry()
    available_algorithms = registry.available_names()

    if algorithms_filter is not None:
        unknown = algorithms_filter - set(available_algorithms)
        if unknown:
            raise ValueError(
                f"Unknown algorithms: {sorted(unknown)}"
            )

    algorithms = [
        name
        for name in available_algorithms
        if algorithms_filter is None
        or name in algorithms_filter
    ]

    selected_scenarios = [
        scenario
        for scenario in scenarios
        if scenarios_filter is None
        or scenario["scenario_id"] in scenarios_filter
    ]

    if not selected_scenarios:
        raise ValueError("No experimental scenarios selected.")

    if not algorithms:
        raise ValueError("No algorithms selected.")

    raw_rows: list[dict[str, Any]] = []
    memory_rows: list[dict[str, Any]] = []

    print(f"Algorithms: {', '.join(algorithms)}")
    print(f"Scenarios: {len(selected_scenarios)}")
    print(f"Repetitions: {repetitions}")
    print(f"Warmup: {warmup}")
    print(f"Memory probe: {measure_memory}")

    for scenario in selected_scenarios:
        scenario_id = str(scenario["scenario_id"])

        prepared_inputs = list(
            iter_match_inputs(
                [scenario],
                patterns,
                chunks,
            )
        )

        inputs = [
            (text, pattern)
            for _, _, text, pattern in prepared_inputs
        ]

        if not inputs:
            raise ValueError(
                f"Scenario {scenario_id} has no inputs."
            )

        total_characters = sum(
            len(text) for text, _ in inputs
        )

        if total_characters != scenario["total_characters"]:
            raise ValueError(
                f"Invalid text size in scenario {scenario_id}."
            )

        # Build correctness oracle outside measured executions.
        oracle = NaiveMatcher()
        expected_positions = [
            oracle.search(text, pattern).positions
            for text, pattern in inputs
        ]

        expected_occurrences = sum(
            len(positions) for positions in expected_positions
        )

        if expected_occurrences != scenario["expected_occurrences"]:
            raise AssertionError(
                f"Prepared occurrence count mismatch: {scenario_id}"
            )

        print(
            f"\n[{scenario_id}] "
            f"{scenario['pattern_id']} | "
            f"scale={scenario['corpus_fraction']} | "
            f"characters={total_characters}"
        )

        # Warmup calls are deliberately excluded from CSV metrics.
        for algorithm in algorithms:
            matcher = registry.create(algorithm)

            for _ in range(warmup):
                warmup_results, _ = run_matcher(
                    matcher, inputs
                )
                validate_results(
                    warmup_results,
                    expected_positions,
                    algorithm,
                    scenario_id,
                )

        for repetition in range(1, repetitions + 1):
            # Rotate the algorithm order between repetitions.
            shift = (repetition - 1) % len(algorithms)
            execution_order = (
                algorithms[shift:] + algorithms[:shift]
            )

            for algorithm in execution_order:
                matcher = registry.create(algorithm)

                results, wall_time_ns = run_matcher(
                    matcher, inputs
                )

                # Verification is outside the measured wall time.
                validate_results(
                    results,
                    expected_positions,
                    algorithm,
                    scenario_id,
                )

                metrics = aggregate_results(results)

                row = {
                    "scenario_id": scenario_id,
                    "pattern_id": scenario["pattern_id"],
                    "pattern_category": scenario[
                        "pattern_category"
                    ],
                    "scale": scenario["scale"],
                    "corpus_fraction": scenario[
                        "corpus_fraction"
                    ],
                    "algorithm": algorithm,
                    "repetition": repetition,
                    "input_count": len(inputs),
                    "total_characters": total_characters,
                    "pattern_length": len(inputs[0][1]),
                    **metrics,
                    "wall_time_ns": wall_time_ns,
                }

                raw_rows.append(row)

                print(
                    f"  rep={repetition} "
                    f"algorithm={algorithm:<12} "
                    f"time={wall_time_ns / 1_000_000:.3f} ms "
                    f"comparisons={metrics['comparisons']}"
                )

        if measure_memory:
            for algorithm in algorithms:
                matcher = registry.create(algorithm)
                peak_bytes = measure_peak_memory(
                    matcher, inputs
                )

                memory_rows.append(
                    {
                        "scenario_id": scenario_id,
                        "algorithm": algorithm,
                        "peak_memory_bytes": peak_bytes,
                        "peak_memory_mb": round(
                            peak_bytes / (1024 * 1024),
                            6,
                        ),
                    }
                )

    summary_rows = summarize(raw_rows)

    write_csv(
        output_dir / "raw_results.csv",
        raw_rows,
        RAW_FIELDS,
    )

    if summary_rows:
        write_csv(
            output_dir / "summary.csv",
            summary_rows,
            list(summary_rows[0]),
        )

    if measure_memory:
        write_csv(
            output_dir / "memory_results.csv",
            memory_rows,
            MEMORY_FIELDS,
        )

    print("\nBENCHMARK COMPLETED")
    print(f"Raw measurements: {len(raw_rows)}")
    print(f"Summary records: {len(summary_rows)}")
    print(f"Output directory: {output_dir}")

    return len(raw_rows), len(summary_rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="FastContext AV2 string matching benchmarks."
    )

    parser.add_argument(
        "--scenario",
        action="append",
        help="Run only the specified scenario ID; repeatable.",
    )
    parser.add_argument(
        "--algorithm",
        action="append",
        help="Run only the specified algorithm; repeatable.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--measure-memory",
        action="store_true",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    args = parser.parse_args()

    try:
        execute_benchmarks(
            scenarios_filter=(
                set(args.scenario)
                if args.scenario
                else None
            ),
            algorithms_filter=(
                set(args.algorithm)
                if args.algorithm
                else None
            ),
            repetitions=args.repetitions,
            warmup=args.warmup,
            measure_memory=args.measure_memory,
            output_dir=args.output_dir,
        )
    except (
        FileNotFoundError,
        ValueError,
        AssertionError,
        OSError,
        json.JSONDecodeError,
    ) as error:
        print(
            f"AV2 benchmark failed: {error}",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
