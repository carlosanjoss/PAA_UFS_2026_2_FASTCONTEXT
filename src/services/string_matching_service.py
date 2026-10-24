
"""Exact string matching over the FastContext corpus."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns

from src.ingestion.chunk_loader import (
    DEFAULT_CHUNKS_PATH,
    load_chunks_jsonl,
)
from src.string_matching.registry import (
    StringMatcherRegistry,
    create_default_registry,
)


@dataclass(frozen=True, slots=True)
class ChunkMatch:
    """Matches found within one corpus chunk."""

    chunk_id: str
    source_path: str
    section_title: str
    content: str
    positions: tuple[int, ...]
    comparisons: int

    @property
    def occurrences(self) -> int:
        return len(self.positions)


@dataclass(frozen=True, slots=True)
class CorpusMatchResult:
    """Aggregated matching results for a corpus."""

    algorithm: str
    pattern: str
    total_chunks: int
    matching_chunks: int
    total_occurrences: int
    total_comparisons: int
    preprocessing_time_ns: int
    search_time_ns: int
    total_time_ns: int
    matches: tuple[ChunkMatch, ...]


class StringMatchingService:
    """Execute exact matching over prepared FastContext chunks."""

    def __init__(
        self,
        corpus_path: str | Path = DEFAULT_CHUNKS_PATH,
        registry: StringMatcherRegistry | None = None,
    ) -> None:
        self._chunks = load_chunks_jsonl(corpus_path)

        self._registry = (
            registry
            if registry is not None
            else create_default_registry()
        )

    @property
    def corpus_size(self) -> int:
        """Return the number of loaded chunks."""
        return len(self._chunks)

    @property
    def available_algorithms(self) -> tuple[str, ...]:
        """Return canonical names exposed by the configured registry."""
        return self._registry.available_names()

    def search(
        self,
        pattern: str,
        algorithm: str = "naive",
    ) -> CorpusMatchResult:
        """Search for a pattern in every corpus chunk."""

        matcher = self._registry.create(algorithm)

        matcher.validate_input("", pattern)

        matches: list[ChunkMatch] = []

        total_occurrences = 0
        total_comparisons = 0
        preprocessing_time_ns = 0
        search_time_ns = 0

        start_time = perf_counter_ns()

        for chunk in self._chunks:
            result = matcher.search(
                text=chunk["content"],
                pattern=pattern,
            )

            total_comparisons += result.comparisons
            total_occurrences += result.occurrences

            preprocessing_time_ns += (
                result.preprocessing_time_ns
            )

            search_time_ns += result.search_time_ns

            if not result.positions:
                continue

            matches.append(
                ChunkMatch(
                    chunk_id=chunk["chunk_id"],
                    source_path=chunk["source_path"],
                    section_title=chunk["section_title"],
                    content=chunk["content"],
                    positions=result.positions,
                    comparisons=result.comparisons,
                )
            )

        total_time_ns = perf_counter_ns() - start_time

        return CorpusMatchResult(
            algorithm=matcher.name,
            pattern=pattern,
            total_chunks=len(self._chunks),
            matching_chunks=len(matches),
            total_occurrences=total_occurrences,
            total_comparisons=total_comparisons,
            preprocessing_time_ns=preprocessing_time_ns,
            search_time_ns=search_time_ns,
            total_time_ns=total_time_ns,
            matches=tuple(matches),
        )
