
"""Contract checks for the AV2 × RAG blinded evaluation preparation."""

from __future__ import annotations

import csv
import hashlib
import itertools
from collections import Counter
from pathlib import Path

import pytest

from experiments.av2.prepare_human_evaluation import (
    ANNOTATORS,
    EXPECTED_SHA256,
    INPUT,
    balanced_pairs,
    generate,
    read_frozen,
    sha256,
    without_chunk_citations,
)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def test_frozen_generations_contract() -> None:
    source = INPUT
    rows = read_frozen(source, EXPECTED_SHA256)
    assert len(rows) == 90
    assert len({(r["query_id"], r["condition"]) for r in rows}) == 90


def test_balanced_pairs_determinism() -> None:
    ids = [f"Q-{i:012d}" for i in range(90)]
    a = balanced_pairs(ids, 143)
    assert a == balanced_pairs(ids, 143)
    assert Counter(tuple(v) for v in a.values()) == {
        pair: 9 for pair in itertools.combinations(ANNOTATORS, 2)
    }


def test_strips_only_recognized_chunk_citations() -> None:
    result = without_chunk_citations("A [x/y] B [different]", ["x/y"])
    assert result == "A B [different]"


def test_fails_when_source_tampered(tmp_path: Path) -> None:
    path = tmp_path / "tampered.csv"
    path.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        read_frozen(path, EXPECTED_SHA256)


def test_fails_without_official_codebook(tmp_path: Path) -> None:
    source = INPUT
    rows = read_frozen(source, EXPECTED_SHA256)
    with pytest.raises(FileNotFoundError):
        generate(rows, tmp_path / "out", tmp_path / "not_found.md", sha256(source))


def test_generated_files_are_blinded_and_non_destructive(tmp_path: Path) -> None:
    source = INPUT
    rows = read_frozen(source, EXPECTED_SHA256)
    codebook = tmp_path / "codebook.md"
    codebook.write_text("No-RAG, Optimized ou Semantic", encoding="utf-8")
    destination = tmp_path / "output"
    generate(rows, destination, codebook, sha256(source))
    quality = load_rows(destination / "quality_blinded.csv")
    grounded = load_rows(destination / "groundedness_blinded.csv")
    hidden = load_rows(destination / "hidden_manifest.csv")
    assert len(quality) == len(grounded) == 90
    assert len(hidden) == 180
    assert "condition" not in quality[0] and "condition" not in grounded[0]
    assert set(r["blind_id"] for r in quality).isdisjoint(
        r["blind_id"] for r in grounded
    )
    assert all(not r["correctness"] for r in quality)
    assert all(not r["groundedness"] for r in grounded)
    assert Counter((r["task_type"], r["condition"]) for r in hidden) == {
        (task, cond): 30 for task in ("quality", "groundedness")
        for cond in ("semantic", "literal", "hybrid")
    }
    for name in ANNOTATORS:
        assert len(load_rows(destination / "assignments" / f"{name}_quality.csv")) == 36
        grounded_path = destination / "assignments" / f"{name}_groundedness.csv"
        assert len(load_rows(grounded_path)) == 36
    assert hashlib.sha256(source.read_bytes()).hexdigest() == EXPECTED_SHA256
    with pytest.raises(FileExistsError):
        generate(rows, destination, codebook, sha256(source))
