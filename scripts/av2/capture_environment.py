
"""Audit FastContext AV2 experimental results and capture provenance."""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = PROJECT_ROOT / "reports" / "tables" / "av2"

RAW_RESULTS = Path("experiments/av2/final/raw_results.csv")
SUMMARY_RESULTS = Path("experiments/av2/final/summary.csv")
MEMORY_RESULTS = Path(
    "experiments/av2/memory_probe/memory_results.csv"
)

ALGORITHMS = {
    "naive",
    "kmp",
    "boyer-moore",
    "rabin-karp",
}

ARTIFACTS = (
    "data/av2/patterns.json",
    "data/av2/scenarios.json",
    "experiments/av2/final/raw_results.csv",
    "experiments/av2/final/summary.csv",
    "experiments/av2/memory_probe/memory_results.csv",
    "reports/tables/av2/speedups.csv",
    "reports/tables/av2/analysis.md",
    "reports/tables/av2/memory_analysis.csv",
    "reports/tables/av2/memory_analysis.md",
    "reports/figures/av2/scaling_time.png",
    "reports/figures/av2/character_comparisons.png",
    "reports/figures/av2/full_scale_patterns.png",
    "reports/figures/av2/memory_real_patterns.png",
    "reports/figures/av2/memory_synthetic_patterns.png",
    "docs/av2/EXPERIMENTAL_RESULTS.md",
    "scripts/av2/prepare_inputs.py",
    "scripts/av2/run_benchmarks.py",
    "scripts/av2/generate_plots.py",
    "scripts/av2/analyze_memory.py",
    "scripts/av2/capture_environment.py",
    "requirements.txt",
    "pyproject.toml",
)


def read_csv(relative_path: Path) -> list[dict[str, str]]:
    """Read a required experimental CSV file."""

    path = PROJECT_ROOT / relative_path

    if not path.is_file():
        raise FileNotFoundError(f"Missing experiment: {relative_path}")

    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"Empty experimental file: {relative_path}")

    return rows


def validate_experiments() -> dict[str, Any]:
    """Verify the experimental matrix before publishing results."""

    raw = read_csv(RAW_RESULTS)
    summary = read_csv(SUMMARY_RESULTS)
    memory = read_csv(MEMORY_RESULTS)

    if len(raw) != 480:
        raise ValueError(f"Expected 480 raw rows, found {len(raw)}")

    if len(summary) != 96:
        raise ValueError(
            f"Expected 96 summary rows, found {len(summary)}"
        )

    if len(memory) != 96:
        raise ValueError(
            f"Expected 96 memory rows, found {len(memory)}"
        )

    raw_keys = [
        (
            row["scenario_id"],
            row["algorithm"],
            int(row["repetition"]),
        )
        for row in raw
    ]

    summary_keys = [
        (row["scenario_id"], row["algorithm"])
        for row in summary
    ]

    memory_keys = [
        (row["scenario_id"], row["algorithm"])
        for row in memory
    ]

    if len(set(raw_keys)) != len(raw_keys):
        raise ValueError("Duplicate raw experiment keys")

    if len(set(summary_keys)) != len(summary_keys):
        raise ValueError("Duplicate summary experiment keys")

    if len(set(memory_keys)) != len(memory_keys):
        raise ValueError("Duplicate memory experiment keys")

    if set(summary_keys) != set(memory_keys):
        raise ValueError(
            "Time and memory scenarios do not match"
        )

    counts = Counter(
        (scenario, algorithm)
        for scenario, algorithm, _ in raw_keys
    )

    if set(counts) != set(summary_keys):
        raise ValueError(
            "Raw and summary scenario combinations differ"
        )

    if any(count != 5 for count in counts.values()):
        raise ValueError(
            "Every scenario/algorithm must have five repetitions"
        )

    if any(
        {rep for scenario, name, rep in raw_keys
         if (scenario, name) == key} != {1, 2, 3, 4, 5}
        for key in counts
    ):
        raise ValueError("Invalid repetition numbering")

    scenarios = {
        scenario for scenario, _ in summary_keys
    }

    algorithms = {
        algorithm for _, algorithm in summary_keys
    }

    if len(scenarios) != 24:
        raise ValueError(
            f"Expected 24 scenarios, found {len(scenarios)}"
        )

    if algorithms != ALGORITHMS:
        raise ValueError(
            f"Unexpected algorithms: {sorted(algorithms)}"
        )

    per_algorithm = Counter(
        row["algorithm"] for row in summary
    )

    if any(
        per_algorithm[algorithm] != 24
        for algorithm in ALGORITHMS
    ):
        raise ValueError(
            "Expected 24 scenarios per algorithm"
        )

    return {
        "scenarios": 24,
        "algorithms": sorted(ALGORITHMS),
        "raw_measurements": len(raw),
        "summary_records": len(summary),
        "memory_measurements": len(memory),
        "repetitions": 5,
    }


def cpu_model() -> str:
    """Read CPU information without collecting user identifiers."""

    if sys.platform == "win32":
        try:
            winreg = importlib.import_module("winreg")
            registry_path = (
                "HARDWARE\\DESCRIPTION\\System\\"
                "CentralProcessor\\0"
            )

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                registry_path,
            ) as key:
                value, _ = winreg.QueryValueEx(
                    key, "ProcessorNameString"
                )
                return str(value).strip()
        except (ImportError, OSError):
            pass

    return platform.processor() or "not available"


def ram_gib() -> float | None:
    """Read installed/available physical memory capacity."""

    try:
        psutil = importlib.import_module("psutil")
        total_bytes = float(psutil.virtual_memory().total)
    except (ImportError, AttributeError, OSError):
        return None

    return round(total_bytes / (1024**3), 2)


def package_versions() -> dict[str, str | None]:
    """Capture package versions relevant to the experiment."""

    packages = (
        "matplotlib",
        "numpy",
        "psutil",
        "pytest",
        "ruff",
        "mypy",
        "fastapi",
        "pydantic",
        "python-dotenv",
    )

    results: dict[str, str | None] = {}

    for package in packages:
        try:
            results[package] = version(package)
        except PackageNotFoundError:
            results[package] = None

    return results


def git_output(*args: str) -> str | None:
    """Query Git without changing files or branches."""

    try:
        process = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None

    if process.returncode != 0:
        return None

    return process.stdout.strip()


def sha256_file(path: Path) -> str:
    """Calculate a SHA-256 checksum without loading all bytes."""

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def save_environment(
    audit: dict[str, Any],
) -> Path:
    """Write machine metadata without usernames or absolute paths."""

    output = OUTPUT_DIR / "environment.json"

    document = {
        "project": "FastContext AV2",
        "captured_at": datetime.now().astimezone().isoformat(),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "virtual_environment": (
                sys.prefix != sys.base_prefix
            ),
        },
        "system": {
            "os": platform.system(),
            "release": platform.release(),
            "architecture": platform.machine(),
            "cpu": cpu_model(),
            "logical_processors": os.cpu_count(),
            "ram_gib": ram_gib(),
        },
        "git": {
            "commit": git_output("rev-parse", "HEAD"),
            "branch": git_output(
                "branch", "--show-current"
            ),
        },
        "packages": package_versions(),
        "experimental_matrix": audit,
        "notes": [
            "Benchmarks executed locally in CPython.",
            "AV1 experimental results were not modified.",
            "Timing and memory were measured separately.",
            "Memory reflects Python allocations tracked by tracemalloc.",
            "No personal directories or usernames are recorded.",
        ],
    }

    output.write_text(
        json.dumps(
            document,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    return output


def save_checksums(
    environment_path: Path,
) -> Path:
    """Create a SHA-256 manifest of deliverable artifacts."""

    relative_paths = [
        *ARTIFACTS,
        environment_path.relative_to(PROJECT_ROOT).as_posix(),
    ]

    records: list[dict[str, Any]] = []

    for relative_path in relative_paths:
        path = PROJECT_ROOT / relative_path

        if not path.is_file():
            raise FileNotFoundError(
                f"Missing required artifact: {relative_path}"
            )

        records.append(
            {
                "path": relative_path,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    output = OUTPUT_DIR / "artifact_checksums.csv"

    with output.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "size_bytes", "sha256"],
        )

        writer.writeheader()
        writer.writerows(records)

    return output


def main() -> int:
    """Validate results and create reproducibility metadata."""

    try:
        audit = validate_experiments()

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        environment_path = save_environment(audit)
        checksums_path = save_checksums(environment_path)

    except (
        FileNotFoundError,
        ValueError,
        KeyError,
        OSError,
    ) as error:
        print(f"AV2 AUDIT FAILED: {error}", file=sys.stderr)
        return 1

    print("AV2 EXPERIMENTAL AUDIT PASSED")
    print(f"Raw measurements: {audit['raw_measurements']}")
    print(f"Summary records: {audit['summary_records']}")
    print(f"Memory measurements: {audit['memory_measurements']}")
    print(f"Environment: {environment_path}")
    print(f"Checksums: {checksums_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
