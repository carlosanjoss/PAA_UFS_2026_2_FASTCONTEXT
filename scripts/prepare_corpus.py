"""Orchestrate Markdown normalization, chunking, and corpus outputs."""

from __future__ import annotations

import json
import time
from pathlib import Path
from statistics import mean, median

from src.chunking.markdown_chunker import chunk_markdown
from src.ingestion.fastapi_corpus import get_corpus_documents
from src.preprocessing.normalize import normalize_markdown
from src.utils.config import load_corpus_config


def prepare_corpus() -> None:
    """Prepare the FastAPI corpus and generate retrieval chunks.

    The pipeline reads the configured raw documents, normalizes their
    Markdown content, creates chunks using the configured token limits,
    writes the chunks to JSONL, and generates summary statistics.
    """
    config = load_corpus_config()

    raw_directory = Path(config["download"]["raw_directory"])
    output_path = Path(config["chunking"]["output"])

    allowed_extensions = config.get("files", {}).get(
        "allowed_extensions",
        config["download"].get("allowed_extensions", [".md"]),
    )
    ignored_files = config.get("files", {}).get("ignored_files", [])

    documents = get_corpus_documents(
        raw_directory,
        allowed_extensions,
        ignored_files,
    )
    print()
    print("FastContext Corpus Preparation")
    print("=" * 60)
    print(f"Raw corpus: {raw_directory}")
    print(f"Source version: {config['source']['version']}")
    print(f"Documents found: {len(documents)}")
    print()
    print("Chunking configuration")
    print("-" * 60)
    print(
        f"Maximum tokens: {config['chunking']['max_tokens']}"
    )
    print(
        f"Overlap tokens: {config['chunking']['overlap_tokens']}"
    )

    started_at = time.perf_counter()
    print()
    print("Document processing")
    print("-" * 60)
    statistics = _write_chunks(
        documents=documents,
        raw_directory=raw_directory,
        output_path=output_path,
        max_tokens=config["chunking"]["max_tokens"],
        overlap_tokens=config["chunking"]["overlap_tokens"],
        version=config["source"]["version"],
    )
    elapsed_seconds = time.perf_counter() - started_at

    statistics_path = output_path.with_name("statistics.json")
    print()
    print("-" * 60)
    print("Preparation summary")
    print("-" * 60)
    print(f"Documents processed: {len(documents)}")
    print(f"Chunks generated: {statistics['chunk_count']}")
    print(f"Documents without chunks: {statistics['documents_without_chunks']}")
    print(
        "Token counts: "
        f"min={statistics['min_tokens']}, "
        f"mean={statistics['mean_tokens']:.2f}, "
        f"median={statistics['median_tokens']:.2f}, "
        f"max={statistics['max_tokens']}"
    )
    print()
    print("Output files")
    print("-" * 60)
    print(f"Chunks: {output_path}")
    print(f"Statistics: {statistics_path}")
    print()
    print(f"Total preparation time: {elapsed_seconds:.2f}s")


def _write_chunks(
    documents: list[Path],
    raw_directory: Path,
    output_path: Path,
    max_tokens: int,
    overlap_tokens: int,
    version: str,
) -> dict:
    """Process documents, write chunks to JSONL, and collect statistics.

    Args:
        documents: Documents to process.
        raw_directory: Root directory containing the source documents.
        output_path: Destination path for the generated JSONL file.
        max_tokens: Maximum target number of tokens per chunk.
        overlap_tokens: Number of overlapping tokens between chunks.
        version: Version of the source corpus.
    """
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    token_counts: list[int] = []
    documents_without_chunks = 0

    with output_path.open("w", encoding="utf-8") as file:
        for document_number, document in enumerate(documents, start=1):
            relative_document = document.relative_to(raw_directory)
            print(
                f"[{document_number}/{len(documents)}] "
                f"Processing: {relative_document}"
            )
            chunks = _create_chunks(
                document=document,
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
            )

            if not chunks:
                documents_without_chunks += 1
                print("    Result: no chunks generated")
            else:
                document_tokens = [chunk["token_count"] for chunk in chunks]
                print(
                    f"    Result: {len(chunks)} chunks | "
                    f"tokens min={min(document_tokens)}, "
                    f"max={max(document_tokens)}"
                )

            for index, chunk in enumerate(chunks):
                record = _create_record(
                    document=document,
                    raw_directory=raw_directory,
                    chunk=chunk,
                    index=index,
                    version=version,
                )

                _write_record(file, record)
                token_counts.append(chunk["token_count"])

    return _write_statistics(
        output_path=output_path,
        token_counts=token_counts,
        documents_without_chunks=documents_without_chunks,
    )


def _create_chunks(
    document: Path,
    max_tokens: int,
    overlap_tokens: int,
) -> list[dict]:
    """Normalize a document and split it into retrieval chunks.

    Args:
        document: Markdown document to process.
        max_tokens: Maximum target number of tokens per chunk.
        overlap_tokens: Number of overlapping tokens between chunks.

    Returns:
        The chunks generated from the document.
    """
    text = document.read_text(encoding="utf-8")
    if not text.strip():
        return []
    markdown = normalize_markdown(text)

    return chunk_markdown(
        markdown=markdown,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )


def _create_record(
    document: Path,
    raw_directory: Path,
    chunk: dict,
    index: int,
    version: str,
) -> dict:
    """Create the JSON-serializable record for one chunk.

    Args:
        document: Source document.
        raw_directory: Root directory containing the source document.
        chunk: Chunk produced by the chunker.
        index: Deterministic index of the chunk within the document.
        version: Version of the source corpus.

    Returns:
        A dictionary containing the chunk content and required metadata.
    """
    relative_path = document.relative_to(raw_directory)

    return {
        "chunk_id": (
            f"{relative_path.with_suffix('').as_posix()}-{index}"
        ),
        "document": relative_path.as_posix(),
        "section": chunk["section"],
        "title": chunk["title"],
        "content": chunk["content"],
        "token_count": chunk["token_count"],
        "source_path": relative_path.as_posix(),
        "version": version,
    }


def _write_record(
    file,
    record: dict,
) -> None:
    """Write one chunk record as a JSON line."""
    file.write(
        json.dumps(
            record,
            ensure_ascii=False,
        )
        + "\n"
    )


def _write_statistics(
    output_path: Path,
    token_counts: list[int],
    documents_without_chunks: int,
) -> dict:
    """Write chunking statistics to the statistics JSON file.

    Args:
        output_path: Path of the generated chunks JSONL file.
        token_counts: Token counts for all generated chunks.
        documents_without_chunks: Number of documents that produced no chunks.
    """
    statistics = {
        "chunk_count": len(token_counts),
        "mean_tokens": mean(token_counts) if token_counts else 0,
        "median_tokens": median(token_counts) if token_counts else 0,
        "min_tokens": min(token_counts) if token_counts else 0,
        "max_tokens": max(token_counts) if token_counts else 0,
        "documents_without_chunks": documents_without_chunks,
    }

    statistics_path = output_path.with_name(
        "statistics.json"
    )

    statistics_path.write_text(
        json.dumps(
            statistics,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return statistics


if __name__ == "__main__":
    prepare_corpus()