"""Test corpus validation, manifest generation, and statistics."""

from pathlib import Path

import pytest

from src.ingestion.fastapi_corpus import (
    build_manifest,
    build_statistics,
    find_documents,
    validate_documents,
)


def test_validate_documents_rejects_empty_files(tmp_path: Path) -> None:
    document = tmp_path / "empty.md"
    document.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="Empty document"):
        validate_documents([document], tmp_path, [".md"])


def test_find_documents_ignores_gitkeep_placeholder(
    tmp_path: Path,
) -> None:
    (tmp_path / ".gitkeep").touch()
    document = tmp_path / "index.md"
    document.write_text("# FastAPI", encoding="utf-8")

    assert find_documents(tmp_path) == [document]


def test_find_documents_ignores_configured_files(
    tmp_path: Path,
) -> None:
    ignored = tmp_path / "_llm-test.md"
    ignored.write_text("# test", encoding="utf-8")
    document = tmp_path / "index.md"
    document.write_text("# FastAPI", encoding="utf-8")

    assert find_documents(tmp_path, ["_llm-test.md"]) == [document]


def test_validate_documents_rejects_duplicate_contents(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"
    first.write_text("same", encoding="utf-8")
    second.write_text("same", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="Duplicate document contents",
    ):
        validate_documents([first, second], tmp_path, [".md"])


def test_manifest_and_statistics_include_expected_fields(
    tmp_path: Path,
) -> None:
    document = tmp_path / "index.md"
    document.write_text("# FastAPI", encoding="utf-8")

    manifest = build_manifest(
        [document],
        tmp_path,
        "0.141.0",
    )
    statistics = build_statistics([document])

    assert manifest[0]["path"] == "index.md"
    assert manifest[0]["version"] == "0.141.0"
    assert len(manifest[0]["sha256"]) == 64
    assert statistics["document_count"] == 1
    assert statistics["extensions"] == {".md": 1}
