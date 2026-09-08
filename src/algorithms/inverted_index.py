from __future__ import annotations

import re
import time
from typing import Any

from src.algorithms.binary_search import binary_search


class InvertedIndex:
    """Manual inverted index with binary-search vocabulary lookup."""

    def __init__(self) -> None:
        self.index: dict[
            str,
            dict[str, int],
        ] = {}

        self.vocabulary: set[str] = set()
        self.sorted_vocabulary: list[str] = []

        self.build_time_ns: int = 0
        self.total_docs: int = 0

    def build(
        self,
        corpus: list[dict[str, Any]],
    ) -> None:
        """Build postings from corpus chunks."""

        start_time = time.perf_counter_ns()

        self.index.clear()
        self.vocabulary.clear()
        self.sorted_vocabulary.clear()

        self.total_docs = len(corpus)

        for chunk in corpus:
            chunk_id = str(
                chunk.get(
                    "chunk_id",
                    "",
                )
            )

            content = str(
                chunk.get(
                    "content",
                    "",
                )
            )

            tokens = self.tokenize(content)

            for token in tokens:
                if token not in self.index:
                    self.index[token] = {}
                    self.vocabulary.add(token)

                postings = self.index[token]

                postings[chunk_id] = (
                    postings.get(
                        chunk_id,
                        0,
                    )
                    + 1
                )

        self.sorted_vocabulary = sorted(
            self.vocabulary
        )

        self.build_time_ns = (
            time.perf_counter_ns()
            - start_time
        )

    def contains_term(
        self,
        term: str,
    ) -> tuple[bool, int]:
        """Check vocabulary membership using manual binary search."""

        normalized_term = (
            term.strip().lower()
        )

        if not self.sorted_vocabulary:
            return False, 0

        index, comparisons = binary_search(
            self.sorted_vocabulary,
            normalized_term,
        )

        return (
            index is not None,
            comparisons,
        )

    def get_postings(
        self,
        term: str,
    ) -> dict[str, int]:
        """Return postings for a term."""

        postings, _ = (
            self.get_postings_with_comparisons(
                term
            )
        )

        return postings

    def get_postings_with_comparisons(
        self,
        term: str,
    ) -> tuple[dict[str, int], int]:
        """Return postings and binary-search comparison count."""

        normalized_term = (
            term.strip().lower()
        )

        exists, comparisons = (
            self.contains_term(
                normalized_term
            )
        )

        if not exists:
            return {}, comparisons

        return (
            self.index.get(
                normalized_term,
                {},
            ),
            comparisons,
        )

    def get_candidate_chunk_ids(
        self,
        query: str,
    ) -> set[str]:
        """Return chunks containing at least one query term."""

        candidates, _ = (
            self.get_candidate_chunk_ids_with_comparisons(
                query
            )
        )

        return candidates

    def get_candidate_chunk_ids_with_comparisons(
        self,
        query: str,
    ) -> tuple[set[str], int]:
        """Return candidate chunks and binary-search comparisons."""

        query_tokens = list(
            dict.fromkeys(
                self.tokenize(query)
            )
        )

        candidates: set[str] = set()
        comparisons = 0

        for token in query_tokens:
            postings, token_comparisons = (
                self.get_postings_with_comparisons(
                    token
                )
            )

            comparisons += token_comparisons

            candidates.update(
                postings
            )

        return candidates, comparisons

    @staticmethod
    def tokenize(
        text: str,
    ) -> list[str]:
        """Normalize text into lowercase alphanumeric tokens."""

        if not text:
            return []

        return re.findall(
            r"\b\w+\b",
            text.lower(),
        )

    @property
    def vocabulary_size(self) -> int:
        """Return the number of unique indexed terms."""

        return len(
            self.vocabulary
        )