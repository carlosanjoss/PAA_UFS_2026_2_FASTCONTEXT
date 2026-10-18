"""Build deterministic AV2 pattern and scenario files from the frozen AV1 corpus."""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from src.ingestion.chunk_loader import load_chunks_jsonl
from src.semantic.persistence import calculate_corpus_fingerprint
from src.utils.config import (
    get_mapping,
    load_corpus_config,
    load_experiments_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks" / "chunks.jsonl"
DEFAULT_PATTERN_CANDIDATES_PATH = (
    PROJECT_ROOT / "data" / "av2" / "pattern_candidates.json"
)
DEFAULT_PATTERNS_PATH = PROJECT_ROOT / "data" / "av2" / "patterns.json"
DEFAULT_SCENARIOS_PATH = PROJECT_ROOT / "data" / "av2" / "scenarios.json"

CORPUS_FRACTIONS = tuple(load_experiments_config()["av2_corpus_sizes"])
MIN_SHORT_PATTERN_OCCURRENCES = 10
EXPECTED_PATTERN_COUNT = 8


def count_occurrences(text: str, pattern: str) -> int:
    """Count exact, case-sensitive matches, including overlapping matches."""
    count = 0
    start = 0
    while True:
        position = text.find(pattern, start)
        if position < 0:
            return count
        count += 1
        start = position + 1


def _corpus_occurrences(
    chunks: Sequence[Mapping[str, Any]],
    pattern: str,
) -> int:
    return sum(count_occurrences(str(chunk["content"]), pattern) for chunk in chunks)


def _read_candidate_patterns(path: Path) -> list[dict[str, Any]]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Cannot read pattern definitions from {path}: {error}"
        ) from error

    if not isinstance(document, dict):
        raise TypeError(f"Pattern file must contain a JSON object: {path}")

    candidates = document.get("candidate_patterns")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"Pattern file has no candidate_patterns list: {path}")

    _validate_candidate_patterns(candidates)
    return candidates


def _validate_candidate_patterns(candidates: Sequence[Any]) -> None:
    ids: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise TypeError("Each candidate pattern must be an object.")
        pattern_id = _validate_candidate_pattern(candidate, ids)
        ids.add(pattern_id)


def _validate_candidate_pattern(
    candidate: Mapping[str, Any],
    existing_ids: set[str],
) -> str:
    pattern_id = candidate.get("pattern_id")
    if not isinstance(pattern_id, str) or not pattern_id or pattern_id in existing_ids:
        raise ValueError(f"Missing or duplicate candidate pattern_id: {pattern_id!r}.")

    pattern = candidate.get("pattern")
    if not isinstance(pattern, str) or not pattern:
        raise ValueError(f"Pattern {pattern_id!r} must have a non-empty string.")
    description = candidate.get("description")
    if not isinstance(description, str) or not description.strip():
        raise ValueError(f"Pattern {pattern_id!r} must have a description.")
    if candidate.get("category") not in {
        "short",
        "medium",
        "long",
        "absent",
        "repetitive",
        "overlapping",
    }:
        raise ValueError(f"Pattern {pattern_id!r} has an unsupported category.")
    expected_positions = candidate.get("expected_positions")
    if expected_positions is not None and (
        not isinstance(expected_positions, list)
        or any(
            not isinstance(position, int) or isinstance(position, bool) or position < 0
            for position in expected_positions
        )
    ):
        raise ValueError(f"Pattern {pattern_id!r} has invalid expected_positions.")
    return pattern_id


def _make_patterns(
    chunks: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    patterns: list[dict[str, Any]] = []
    seen_texts: set[tuple[str, str]] = set()
    for candidate in candidates:
        pattern = _make_candidate_pattern(chunks, candidate, seen_texts)
        if pattern is not None:
            patterns.append(pattern)

    if len(patterns) != EXPECTED_PATTERN_COUNT:
        raise ValueError(
            f"AV2 requires exactly {EXPECTED_PATTERN_COUNT} verified patterns; "
            f"found {len(patterns)}."
        )
    _validate_required_pattern_categories(patterns)
    return patterns


def _make_candidate_pattern(
    chunks: Sequence[Mapping[str, Any]],
    candidate: Mapping[str, Any],
    seen_texts: set[tuple[str, str]],
) -> dict[str, Any] | None:
    pattern_id = str(candidate["pattern_id"])
    pattern_text = str(candidate["pattern"])
    category = str(candidate["category"])
    synthetic_text = candidate.get("synthetic_text")
    is_synthetic = isinstance(synthetic_text, str)

    _validate_candidate_scope(pattern_id, category, synthetic_text, is_synthetic)
    scope = "synthetic" if is_synthetic else "corpus"
    key = (scope, pattern_text)
    if key in seen_texts:
        raise ValueError(
            f"Duplicate pattern string for the same input scope: {pattern_text!r}."
        )
    seen_texts.add(key)

    occurrences = 0 if is_synthetic else _corpus_occurrences(chunks, pattern_text)
    if _skip_unmatched_candidate(pattern_id, category, occurrences, is_synthetic):
        return None

    pattern = {
        "pattern_id": pattern_id,
        "pattern": pattern_text,
        "category": category,
        "description": str(candidate["description"]),
        "scope": scope,
        "corpus_occurrences": occurrences,
        "verified": True,
    }
    if is_synthetic:
        pattern["synthetic_text"] = str(synthetic_text)
        if "expected_positions" in candidate:
            pattern["expected_positions"] = candidate["expected_positions"]
    return pattern


def _validate_candidate_scope(
    pattern_id: str,
    category: str,
    synthetic_text: Any,
    is_synthetic: bool,
) -> None:
    if category == "absent" and is_synthetic:
        raise ValueError(
            f"Absent pattern {pattern_id!r} must be checked against the corpus."
        )
    if category in {"repetitive", "overlapping"} and not is_synthetic:
        raise ValueError(f"Pattern {pattern_id!r} must declare a synthetic_text.")
    if is_synthetic and (
        not synthetic_text or category not in {"repetitive", "overlapping"}
    ):
        raise ValueError(f"Pattern {pattern_id!r} has an invalid synthetic input.")


def _skip_unmatched_candidate(
    pattern_id: str,
    category: str,
    occurrences: int,
    is_synthetic: bool,
) -> bool:
    if category == "absent" and occurrences:
        raise ValueError(
            f"Absent sentinel {pattern_id!r} occurs {occurrences} times "
            "in the frozen corpus."
        )
    if category == "short" and occurrences < MIN_SHORT_PATTERN_OCCURRENCES:
        raise ValueError(
            f"Short pattern {pattern_id!r} occurs only {occurrences} times; "
            f"at least {MIN_SHORT_PATTERN_OCCURRENCES} are required."
        )
    if not is_synthetic and category not in {"absent", "short"} and not occurrences:
        print(
            f"Skipping unverified candidate {pattern_id}: "
            "no exact occurrence in this corpus.",
            file=sys.stderr,
        )
        return True
    return False


def _validate_required_pattern_categories(
    patterns: Sequence[Mapping[str, Any]],
) -> None:
    has_real_match = any(
        pattern["scope"] == "corpus" and pattern["category"] != "absent"
        for pattern in patterns
    )
    if not has_real_match:
        raise ValueError(
            "No verified present real-text patterns remain in this corpus."
        )
    if not any(pattern["category"] == "medium" for pattern in patterns):
        raise ValueError("No candidate medium-length pattern occurs in this corpus.")

    required = {"short", "medium", "long", "absent", "repetitive", "overlapping"}
    available = {str(pattern["category"]) for pattern in patterns}
    missing = required - available
    if missing:
        raise ValueError(f"Required pattern categories are missing: {sorted(missing)}.")


def _build_scenarios(
    chunks: Sequence[Mapping[str, Any]],
    patterns: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    ordered_chunks = sorted(chunks, key=lambda item: str(item["chunk_id"]))
    scenarios: list[dict[str, Any]] = []

    for pattern in patterns:
        for fraction in CORPUS_FRACTIONS:
            if pattern["scope"] == "synthetic":
                scenarios.append(
                    _make_synthetic_scenario(
                        pattern,
                        fraction,
                        ordered_chunks,
                        len(scenarios) + 1,
                    )
                )
                continue

            scenarios.append(
                _make_corpus_scenario(
                    pattern,
                    fraction,
                    ordered_chunks,
                    len(scenarios) + 1,
                )
            )

    validate_scenarios(scenarios, patterns, chunks)
    return scenarios


def _make_synthetic_scenario(
    pattern: Mapping[str, Any],
    fraction: float,
    ordered_chunks: Sequence[Mapping[str, Any]],
    scenario_number: int,
) -> dict[str, Any]:
    reference_chunk_count = math.ceil(len(ordered_chunks) * fraction)
    reference_chunks = ordered_chunks[:reference_chunk_count]
    reference_characters = sum(len(str(chunk["content"])) for chunk in reference_chunks)
    seed_text = str(pattern["synthetic_text"])
    text = _repeat_to_length(seed_text, reference_characters)
    occurrences = count_occurrences(text, str(pattern["pattern"]))
    return {
        "scenario_id": f"S{scenario_number:03d}",
        "pattern": pattern["pattern"],
        "pattern_type": _scenario_pattern_type(pattern),
        "corpus_fraction": fraction,
        "expected_case": "present" if occurrences else "absent",
        "pattern_id": pattern["pattern_id"],
        "pattern_category": pattern["category"],
        "scale": "synthetic_microbenchmark",
        "chunk_ids": [],
        "reference_chunk_count": reference_chunk_count,
        "reference_characters": reference_characters,
        "synthetic_generation": "repeat_seed_then_truncate",
        "synthetic_seed_text": seed_text,
        "total_characters": len(text),
        "expected_occurrences": occurrences,
        "synthetic_text": text,
        "description": pattern["description"],
    }


def _repeat_to_length(seed_text: str, target_length: int) -> str:
    if target_length <= 0:
        return ""
    repetitions = math.ceil(target_length / len(seed_text))
    return (seed_text * repetitions)[:target_length]


def _scenario_pattern_type(pattern: Mapping[str, Any]) -> str:
    category = str(pattern["category"])
    if category == "absent":
        return "literal_absent"
    if category in {"repetitive", "overlapping"}:
        return category
    return "literal_present"


def _make_corpus_scenario(
    pattern: Mapping[str, Any],
    fraction: float,
    ordered_chunks: Sequence[Mapping[str, Any]],
    scenario_number: int,
) -> dict[str, Any]:
    chunk_count = math.ceil(len(ordered_chunks) * fraction)
    selected_chunks = ordered_chunks[:chunk_count]
    selected_ids = [str(chunk["chunk_id"]) for chunk in selected_chunks]
    pattern_text = str(pattern["pattern"])
    occurrences = sum(
        count_occurrences(str(chunk["content"]), pattern_text)
        for chunk in selected_chunks
    )
    return {
        "scenario_id": f"S{scenario_number:03d}",
        "pattern": pattern_text,
        "pattern_type": _scenario_pattern_type(pattern),
        "corpus_fraction": fraction,
        "expected_case": "present" if occurrences else "absent",
        "pattern_id": pattern["pattern_id"],
        "pattern_category": pattern["category"],
        "scale": "corpus",
        "chunk_ids": selected_ids,
        "total_characters": sum(
            len(str(chunk["content"])) for chunk in selected_chunks
        ),
        "expected_occurrences": occurrences,
        "description": pattern["description"],
    }


def validate_scenarios(
    scenarios: Sequence[Mapping[str, Any]],
    patterns: Sequence[Mapping[str, Any]],
    chunks: Sequence[Mapping[str, Any]],
) -> None:
    _validate_unique_ids(scenarios, patterns)
    patterns_by_id = {str(pattern["pattern_id"]): pattern for pattern in patterns}
    chunks_by_id = {str(chunk["chunk_id"]): chunk for chunk in chunks}
    expected_chunk_order = sorted(chunks_by_id)
    scenarios_by_pattern: dict[str, list[Mapping[str, Any]]] = {}
    prior_ids: list[str] = []
    prior_pattern_id: str | None = None

    for scenario in scenarios:
        pattern_id = str(scenario["pattern_id"])
        scenarios_by_pattern.setdefault(pattern_id, []).append(scenario)
        if pattern_id != prior_pattern_id:
            prior_ids = []
            prior_pattern_id = pattern_id

        pattern = patterns_by_id.get(pattern_id)
        if pattern is None:
            raise ValueError(
                f"Scenario {scenario['scenario_id']} refers to an unknown pattern."
            )
        _validate_scenario_pattern(scenario, pattern)
        prior_ids = _validate_scenario_input(
            scenario,
            pattern,
            chunks_by_id,
            expected_chunk_order,
            prior_ids,
        )

    _validate_pattern_scenario_counts(patterns, scenarios_by_pattern)


def _validate_unique_ids(
    scenarios: Sequence[Mapping[str, Any]],
    patterns: Sequence[Mapping[str, Any]],
) -> None:
    scenario_ids = [scenario["scenario_id"] for scenario in scenarios]
    pattern_ids = [pattern["pattern_id"] for pattern in patterns]
    if len(scenario_ids) != len(set(scenario_ids)):
        raise ValueError("Scenario identifiers are not unique.")
    if len(pattern_ids) != len(set(pattern_ids)):
        raise ValueError("Pattern identifiers are not unique.")


def _validate_scenario_pattern(
    scenario: Mapping[str, Any],
    pattern: Mapping[str, Any],
) -> None:
    if (
        scenario["pattern"] != pattern["pattern"]
        or scenario["pattern_type"] != _scenario_pattern_type(pattern)
        or scenario["pattern_category"] != pattern["category"]
    ):
        raise ValueError(
            f"Scenario {scenario['scenario_id']} does not match its pattern."
        )
    if (
        pattern["scope"] == "corpus"
        and scenario["corpus_fraction"] not in CORPUS_FRACTIONS
    ):
        raise ValueError(
            f"Invalid corpus fraction in scenario {scenario['scenario_id']}."
        )
    if (
        pattern["scope"] == "synthetic"
        and scenario["corpus_fraction"] not in CORPUS_FRACTIONS
    ):
        raise ValueError(
            f"Invalid synthetic scale in scenario {scenario['scenario_id']}."
        )


def _validate_scenario_input(
    scenario: Mapping[str, Any],
    pattern: Mapping[str, Any],
    chunks_by_id: Mapping[str, Mapping[str, Any]],
    expected_chunk_order: Sequence[str],
    prior_ids: Sequence[str],
) -> list[str]:
    scenario_id = str(scenario["scenario_id"])
    chunk_ids = list(scenario["chunk_ids"])
    if any(chunk_id not in chunks_by_id for chunk_id in chunk_ids):
        raise ValueError(f"Scenario {scenario_id} refers to an unknown chunk.")

    if scenario["scale"] == "corpus":
        _validate_corpus_scenario(
            scenario, pattern, chunk_ids, chunks_by_id, expected_chunk_order, prior_ids
        )
        return chunk_ids
    if scenario["scale"] == "synthetic_microbenchmark":
        _validate_synthetic_scenario(
            scenario,
            pattern,
            chunk_ids,
            chunks_by_id,
            expected_chunk_order,
        )
        return []
    raise ValueError(f"Unknown scale in scenario {scenario_id}.")


def _validate_corpus_scenario(
    scenario: Mapping[str, Any],
    pattern: Mapping[str, Any],
    chunk_ids: list[str],
    chunks_by_id: Mapping[str, Mapping[str, Any]],
    expected_chunk_order: Sequence[str],
    prior_ids: Sequence[str],
) -> None:
    scenario_id = str(scenario["scenario_id"])
    if chunk_ids != sorted(chunk_ids):
        raise ValueError(f"Scenario {scenario_id} chunk IDs are not ordered.")

    fraction = scenario["corpus_fraction"]
    if fraction not in CORPUS_FRACTIONS:
        raise ValueError(f"Invalid corpus fraction in scenario {scenario_id}.")
    expected_count = math.ceil(len(chunks_by_id) * fraction)
    if chunk_ids != list(expected_chunk_order[:expected_count]):
        raise ValueError(
            f"Scenario {scenario_id} does not use the deterministic corpus prefix."
        )
    if not set(prior_ids).issubset(chunk_ids):
        raise ValueError("Deterministic corpus subsets are not nested.")

    _validate_corpus_scenario_results(scenario, pattern, chunk_ids, chunks_by_id)


def _validate_corpus_scenario_results(
    scenario: Mapping[str, Any],
    pattern: Mapping[str, Any],
    chunk_ids: Sequence[str],
    chunks_by_id: Mapping[str, Mapping[str, Any]],
) -> None:
    scenario_id = str(scenario["scenario_id"])
    selected_chunks = [chunks_by_id[chunk_id] for chunk_id in chunk_ids]
    total_characters = sum(len(str(chunk["content"])) for chunk in selected_chunks)
    if total_characters != scenario["total_characters"]:
        raise ValueError(f"Incorrect text size in scenario {scenario_id}.")
    if scenario["expected_case"] not in {"present", "absent"}:
        raise ValueError(f"Invalid expected_case in scenario {scenario_id}.")

    occurrences = sum(
        count_occurrences(str(chunk["content"]), str(pattern["pattern"]))
        for chunk in selected_chunks
    )
    expected_case = "present" if occurrences else "absent"
    if (
        occurrences != scenario["expected_occurrences"]
        or scenario["expected_case"] != expected_case
    ):
        raise ValueError(f"Incorrect expected result in scenario {scenario_id}.")


def _validate_synthetic_scenario(
    scenario: Mapping[str, Any],
    pattern: Mapping[str, Any],
    chunk_ids: Sequence[str],
    chunks_by_id: Mapping[str, Mapping[str, Any]],
    expected_chunk_order: Sequence[str],
) -> None:
    scenario_id = str(scenario["scenario_id"])
    fraction = scenario["corpus_fraction"]
    reference_chunk_count = math.ceil(len(chunks_by_id) * fraction)
    reference_chunks = expected_chunk_order[:reference_chunk_count]
    reference_characters = sum(
        len(str(chunks_by_id[chunk_id]["content"])) for chunk_id in reference_chunks
    )
    seed_text = str(pattern["synthetic_text"])
    text = _repeat_to_length(seed_text, reference_characters)
    occurrences = count_occurrences(text, str(pattern["pattern"]))
    if (
        chunk_ids
        or scenario["scale"] != "synthetic_microbenchmark"
        or scenario["synthetic_generation"] != "repeat_seed_then_truncate"
        or scenario["synthetic_seed_text"] != seed_text
        or scenario["reference_chunk_count"] != reference_chunk_count
        or scenario["reference_characters"] != reference_characters
        or scenario["synthetic_text"] != text
        or scenario["total_characters"] != len(text)
        or scenario["expected_occurrences"] != occurrences
    ):
        raise ValueError(f"Invalid synthetic input in scenario {scenario_id}.")
    expected_case = "present" if occurrences else "absent"
    if scenario["expected_case"] != expected_case:
        raise ValueError(f"Incorrect expected result in scenario {scenario_id}.")


def _validate_pattern_scenario_counts(
    patterns: Sequence[Mapping[str, Any]],
    scenarios_by_pattern: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    for pattern in patterns:
        pattern_id = str(pattern["pattern_id"])
        pattern_scenarios = scenarios_by_pattern.get(pattern_id, [])
        fractions = [scenario["corpus_fraction"] for scenario in pattern_scenarios]
        if len(fractions) != len(CORPUS_FRACTIONS) or set(fractions) != set(
            CORPUS_FRACTIONS
        ):
            raise ValueError(f"Pattern {pattern_id} must have one scenario per scale.")
        expected_scale = (
            "synthetic_microbenchmark" if pattern["scope"] == "synthetic" else "corpus"
        )
        if any(scenario["scale"] != expected_scale for scenario in pattern_scenarios):
            raise ValueError(f"Pattern {pattern_id} uses an invalid scale type.")


def iter_match_inputs(
    scenarios: Sequence[Mapping[str, Any]],
    patterns: Sequence[Mapping[str, Any]],
    chunks: Sequence[Mapping[str, Any]],
) -> Iterator[tuple[str, str, str, str]]:
    """Yield shared (scenario_id, chunk_id, text, pattern) inputs."""
    patterns_by_id = {str(item["pattern_id"]): item for item in patterns}
    chunks_by_id = {str(item["chunk_id"]): item for item in chunks}

    for scenario in scenarios:
        pattern = patterns_by_id[str(scenario["pattern_id"])]["pattern"]
        if scenario["scale"] == "synthetic_microbenchmark":
            yield (
                str(scenario["scenario_id"]),
                "synthetic",
                str(scenario["synthetic_text"]),
                str(pattern),
            )
            continue
        for chunk_id in scenario["chunk_ids"]:
            yield (
                str(scenario["scenario_id"]),
                str(chunk_id),
                str(chunks_by_id[str(chunk_id)]["content"]),
                str(pattern),
            )


def _load_corpus_reference() -> dict[str, Any]:
    config = load_corpus_config()
    source = get_mapping(config, "source")
    reference = get_mapping(config, "av1_reference")
    expected = {
        "name": config.get("name"),
        "version": source.get("version"),
        "origin": (
            f"{source.get('repository', '').rstrip('/')}/tree/"
            f"{source.get('version', '')}/"
            f"{source.get('documentation_path', '').strip('/')}"
        ),
        "document_count": reference.get("document_count"),
        "chunk_count": reference.get("chunk_count"),
        "fingerprint": reference.get("fingerprint"),
    }
    if (
        not isinstance(expected["name"], str)
        or not isinstance(expected["version"], str)
        or not isinstance(expected["origin"], str)
        or not isinstance(expected["document_count"], int)
        or not isinstance(expected["chunk_count"], int)
        or not isinstance(expected["fingerprint"], str)
    ):
        raise TypeError(
            "Corpus reference in config/corpus.yaml is incomplete or invalid."
        )
    return expected


def _corpus_manifest(
    chunks: Sequence[Mapping[str, Any]],
    fingerprint: str,
    *,
    fingerprint_matches: bool,
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    versions = sorted({str(chunk.get("version", "")) for chunk in chunks})
    return {
        "name": reference["name"],
        "origin": reference["origin"],
        "version": versions[0] if len(versions) == 1 else versions,
        "document_count": len({str(chunk.get("source_path", "")) for chunk in chunks}),
        "chunk_count": len(chunks),
        "fingerprint": fingerprint,
        "known_av1_fingerprint": reference["fingerprint"],
        "matches_known_av1": fingerprint_matches,
        "expected_document_count": reference["document_count"],
        "expected_chunk_count": reference["chunk_count"],
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _validate_corpus_reference(
    chunks: Sequence[Mapping[str, Any]],
    reference: Mapping[str, Any],
    *,
    allow_mismatch: bool,
) -> tuple[str, bool, list[str]]:
    fingerprint = calculate_corpus_fingerprint(chunks)
    fingerprint_matches = fingerprint == reference["fingerprint"]
    versions = {str(chunk.get("version", "")) for chunk in chunks}
    documents = {str(chunk.get("source_path", "")) for chunk in chunks}
    mismatches = []

    if not fingerprint_matches:
        mismatches.append(f"fingerprint {fingerprint}")
    if len(chunks) != reference["chunk_count"]:
        mismatches.append(f"{len(chunks)} chunks")
    if len(documents) != reference["document_count"]:
        mismatches.append(f"{len(documents)} documents")
    if versions != {reference["version"]}:
        mismatches.append(f"versions {sorted(versions)}")
    if mismatches and not allow_mismatch:
        raise ValueError(
            "Corpus differs from frozen AV1 reference "
            f"(expected {reference['version']}, "
            f"{reference['document_count']} documents, "
            f"{reference['chunk_count']} chunks, fingerprint "
            f"{reference['fingerprint']}; found {', '.join(mismatches)}. "
            "Use the frozen AV1 corpus, or update config/corpus.yaml if this "
            "corpus change is intentional."
        )
    return fingerprint, fingerprint_matches, mismatches


def _write_outputs(
    patterns_path: Path,
    scenarios_path: Path,
    patterns: Sequence[Mapping[str, Any]],
    scenarios: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    status: str,
) -> None:
    _write_json(
        patterns_path,
        {"status": status, "corpus": manifest, "patterns": list(patterns)},
    )
    _write_json(
        scenarios_path,
        {
            "status": status,
            "corpus": manifest,
            "sampling": {
                "ordering": "ascending chunk_id",
                "selection": "first ceil(chunk_count * corpus_fraction) chunks",
                "fractions": list(CORPUS_FRACTIONS),
                "nested": True,
            },
            "scenarios": list(scenarios),
        },
    )


def prepare_inputs(
    *,
    chunks_path: str | Path = DEFAULT_CHUNKS_PATH,
    pattern_candidates_path: str | Path = DEFAULT_PATTERN_CANDIDATES_PATH,
    patterns_path: str | Path = DEFAULT_PATTERNS_PATH,
    scenarios_path: str | Path = DEFAULT_SCENARIOS_PATH,
    allow_corpus_mismatch: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Generate reproducible AV2 JSON files without modifying AV1 chunks."""
    chunks = load_chunks_jsonl(chunks_path)
    reference = _load_corpus_reference()
    fingerprint, fingerprint_matches, mismatches = _validate_corpus_reference(
        chunks,
        reference,
        allow_mismatch=allow_corpus_mismatch,
    )
    candidates = _read_candidate_patterns(Path(pattern_candidates_path))
    patterns = _make_patterns(chunks, candidates)
    scenarios = _build_scenarios(chunks, patterns)
    manifest = _corpus_manifest(
        chunks,
        fingerprint,
        fingerprint_matches=fingerprint_matches,
        reference=reference,
    )
    status = "ready" if not mismatches else "corpus_mismatch_approved"
    _write_outputs(
        Path(patterns_path),
        Path(scenarios_path),
        patterns,
        scenarios,
        manifest,
        status,
    )
    return patterns, scenarios


def main() -> int:
    try:
        patterns, scenarios = prepare_inputs()
    except (FileNotFoundError, TypeError, ValueError) as error:
        print(f"AV2 input preparation failed: {error}", file=sys.stderr)
        return 2

    print("Generated AV2 input files:")
    print(f"  Patterns ({len(patterns)}): {DEFAULT_PATTERNS_PATH}")
    print(f"  Scenarios ({len(scenarios)}): {DEFAULT_SCENARIOS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
