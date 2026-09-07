"""Download, validate, and catalog the configured FastAPI documentation."""

import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from urllib.request import urlretrieve
from zipfile import ZipFile

from src.ingestion.fastapi_corpus import (
    build_manifest,
    build_statistics,
    get_corpus_documents,
    save_manifest,
)
from src.utils.config import load_corpus_config


def download_corpus() -> None:
    config = load_corpus_config()

    repository = config["source"]["repository"]
    version = config["source"]["version"]
    documentation_path = config["source"]["documentation_path"]
    raw_directory = Path(config["download"]["raw_directory"])
    manifest_path = Path(config["download"]["manifest_path"])
    statistics_path = Path(config["download"]["statistics_path"])
    allowed_extensions = config["files"]["allowed_extensions"]
    ignored_files = config["files"].get("ignored_files", [])

    archive_url = f"{repository}/archive/refs/tags/{version}.zip"

    print(f"[1/5] Downloading FastAPI documentation {version}...")
    with TemporaryDirectory() as temporary_directory:
        temporary_path = Path(temporary_directory)
        archive_path = temporary_path / "fastapi.zip"

        urlretrieve(archive_url, archive_path)

        with ZipFile(archive_path) as archive:
            archive.extractall(temporary_path)
        print("[2/5] Archive extracted in a temporary directory.")

        repository_directory = next(
            path
            for path in temporary_path.iterdir()
            if path.is_dir()
        )

        documentation_directory = (
            repository_directory / documentation_path
        )

        if not documentation_directory.exists():
            raise FileNotFoundError(
                f"Documentation not found: {documentation_path}"
            )

        _clear_raw_directory(raw_directory)
        print(f"[3/5] Copying Markdown files to {raw_directory}...")

        copied_files = 0
        for file in documentation_directory.rglob("*"):
            if not file.is_file() or file.suffix not in allowed_extensions:
                continue
            relative_path = file.relative_to(documentation_directory)
            if relative_path.as_posix() in ignored_files:
                continue
            destination = raw_directory / relative_path

            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(file.read_bytes())
            copied_files += 1

    print(f"[4/5] Validating {copied_files} downloaded documents...")
    documents = get_corpus_documents(
        raw_directory,
        allowed_extensions,
        ignored_files,
    )
    manifest = build_manifest(documents, raw_directory, version)
    statistics = build_statistics(documents)
    save_manifest(manifest, statistics, manifest_path)
    if statistics_path != manifest_path:
        statistics_path.parent.mkdir(parents=True, exist_ok=True)
        statistics_path.write_text(
            json.dumps(
                statistics, indent=2, ensure_ascii=False
            ),
            encoding="utf-8",
        )
    print(f"[5/5] Corpus ready: {len(documents)} documents.")


def _clear_raw_directory(raw_directory: Path) -> None:
    """Remove downloaded files while preserving the Git directory marker."""
    raw_directory.mkdir(parents=True, exist_ok=True)

    for path in raw_directory.iterdir():
        if path.name == ".gitkeep":
            continue

        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


if __name__ == "__main__":
    download_corpus()