"""Discover, validate, hash, and catalog downloaded corpus documents."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


def find_documents(
    corpus_directory: Path,
    ignored_files: list[str] | None = None,
) -> list[Path]:
    if not corpus_directory.exists():
        raise FileNotFoundError(
            f"Corpus directory not found: {corpus_directory}"
        )

    ignored = set(ignored_files or [])
    return sorted(
        path
        for path in corpus_directory.rglob("*")
        if (
            path.is_file()
            and path.name != ".gitkeep"
            and path.relative_to(corpus_directory).as_posix()
            not in ignored
        )
    )


def validate_documents(
    documents: list[Path],
    corpus_directory: Path,
    allowed_extensions: list[str],
) -> None:
    if not documents:
        raise ValueError("Corpus is empty.")

    corpus_directory = corpus_directory.resolve()
    resolved_documents = [
        document.resolve()
        for document in documents
    ]

    if len(resolved_documents) != len(set(resolved_documents)):
        raise ValueError("Duplicate documents found.")

    for document in resolved_documents:
        try:
            document.relative_to(corpus_directory)
        except ValueError as error:
            raise ValueError(
                f"Document outside corpus directory: {document}"
            ) from error

    unexpected_extensions = sorted(
        {
            document.suffix
            for document in resolved_documents
            if document.suffix not in allowed_extensions
        }
    )

    if unexpected_extensions:
        raise ValueError(
            "Unexpected file extensions: "
            + ", ".join(unexpected_extensions)
        )

    for document in resolved_documents:
        if (
            document.stat().st_size == 0
            or not document.read_text(
                encoding="utf-8"
            ).strip()
        ):
            raise ValueError(f"Empty document: {document}")

    hashes = [calculate_sha256(document) for document in resolved_documents]
    if len(hashes) != len(set(hashes)):
        raise ValueError("Duplicate document contents found.")


def get_corpus_documents(
    corpus_directory: Path,
    allowed_extensions: list[str],
    ignored_files: list[str] | None = None,
) -> list[Path]:
    documents = find_documents(corpus_directory, ignored_files)

    validate_documents(
        documents=documents,
        corpus_directory=corpus_directory,
        allowed_extensions=allowed_extensions,
    )

    return [
        document
        for document in documents
        if document.suffix in allowed_extensions
    ]


def calculate_sha256(file: Path) -> str:
    return hashlib.sha256(file.read_bytes()).hexdigest()


def build_manifest(
    documents: list[Path],
    corpus_directory: Path,
    version: str,
) -> list[dict]:
    acquisition_date = datetime.now(UTC).isoformat()

    manifest = []

    for document in documents:
        manifest.append(
            {
                "path": str(
                    document.relative_to(corpus_directory)
                ),
                "size": document.stat().st_size,
                "sha256": calculate_sha256(document),
                "version": version,
                "acquisition_date": acquisition_date,
            }
        )

    return manifest


def build_statistics(
    documents: list[Path],
) -> dict:
    extensions: dict[str, int] = {}

    for document in documents:
        extension = document.suffix

        extensions[extension] = (
            extensions.get(extension, 0) + 1
        )

    return {
        "document_count": len(documents),
        "extensions": extensions,
        "total_size": sum(
            document.stat().st_size
            for document in documents
        ),
    }


def save_manifest(
    manifest: list[dict],
    statistics: dict,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = {
        "documents": manifest,
        "statistics": statistics,
    }

    output_path.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )