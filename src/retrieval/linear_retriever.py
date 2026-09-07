from __future__ import annotations

import time
from typing import Any

from src.algorithms.merge_sort import merge_sort
from src.representations.tfidf import (
    SparseVector,
    TFIDFVectorizer,
)
from src.retrieval.base import Retriever
from src.retrieval.models import (
    RetrievalMetrics,
    RetrievalResult,
    RetrievedChunk,
)

CorpusChunk = dict[str, Any]
Candidate = tuple[float, CorpusChunk]


class LinearRetriever(Retriever):
    """Exhaustive lexical baseline over every corpus chunk."""

    @property
    def name(self) -> str:
        """Return the canonical retriever identifier."""

        return "linear"

    def __init__(
        self,
        corpus_chunks: list[CorpusChunk],
    ) -> None:
        self._corpus = list(
            corpus_chunks
        )

        self._validate_corpus()

        self._vectorizer = (
            TFIDFVectorizer()
            .fit(
                self._corpus
            )
        )

        self._document_vectors = (
            self._build_document_vectors()
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> RetrievalResult:
        """Score every chunk and rank candidates with manual Merge Sort."""

        normalized_query = (
            self._validate_request(
                query,
                top_k,
            )
        )

        start_time = (
            time.perf_counter_ns()
        )

        if top_k == 0:
            return self._empty_result(
                query=normalized_query,
                top_k=top_k,
                start_time=start_time,
            )

        query_vector = (
            self._vectorizer.transform(
                normalized_query
            )
        )

        candidates: list[
            Candidate
        ] = []

        for chunk in self._corpus:
            chunk_id = str(
                chunk["chunk_id"]
            )

            document_vector = (
                self._document_vectors[
                    chunk_id
                ]
            )

            score = (
                self._vectorizer
                .cosine_similarity(
                    query_vector,
                    document_vector,
                )
            )

            if score > 0.0:
                candidates.append(
                    (
                        score,
                        chunk,
                    )
                )

        sort_start = (
            time.perf_counter_ns()
        )

        ordered_candidates, sort_stats = (
            merge_sort(
                candidates
            )
        )

        sorting_time_ns = (
            time.perf_counter_ns()
            - sort_start
        )

        selected = (
            ordered_candidates[
                :top_k
            ]
        )

        chunks = (
            self._build_retrieved_chunks(
                selected
            )
        )

        retrieval_time_ns = (
            time.perf_counter_ns()
            - start_time
        )

        metrics = RetrievalMetrics(
            retrieval_time_ns=(
                retrieval_time_ns
            ),
            sorting_time_ns=(
                sorting_time_ns
            ),
            comparisons=(
                sort_stats.comparisons
            ),
            chunks_scored=len(
                self._corpus
            ),
            candidates_found=len(
                candidates
            ),
        )

        return RetrievalResult(
            query=normalized_query,
            algorithm=self.name,
            top_k=top_k,
            chunks=chunks,
            metrics=metrics,
            metadata={
                "representation": "tfidf",
                "similarity": "cosine",
                "ranking_strategy": "merge_sort",
                "ranking_comparisons": (
                    sort_stats.comparisons
                ),
            },
        )

    def _empty_result(
        self,
        *,
        query: str,
        top_k: int,
        start_time: int,
    ) -> RetrievalResult:
        """Create an empty result without scoring the corpus."""

        return RetrievalResult(
            query=query,
            algorithm=self.name,
            top_k=top_k,
            chunks=(),
            metrics=RetrievalMetrics(
                retrieval_time_ns=(
                    time.perf_counter_ns()
                    - start_time
                ),
                sorting_time_ns=0,
                comparisons=0,
                chunks_scored=0,
                candidates_found=0,
            ),
            metadata={
                "representation": "tfidf",
                "similarity": "cosine",
                "ranking_strategy": "merge_sort",
            },
        )

    def _build_document_vectors(
        self,
    ) -> dict[str, SparseVector]:
        """Precompute TF-IDF vectors for corpus chunks."""

        vectors: dict[
            str,
            SparseVector,
        ] = {}

        for chunk in self._corpus:
            chunk_id = str(
                chunk["chunk_id"]
            )

            vectors[chunk_id] = (
                self._vectorizer.transform(
                    str(
                        chunk.get(
                            "content",
                            "",
                        )
                    )
                )
            )

        return vectors

    def _build_retrieved_chunks(
        self,
        candidates: list[Candidate],
    ) -> tuple[RetrievedChunk, ...]:
        """Convert ranked candidate tuples into canonical result chunks."""

        retrieved: list[
            RetrievedChunk
        ] = []

        for rank, (
            score,
            chunk,
        ) in enumerate(
            candidates,
            start=1,
        ):
            raw_metadata = (
                chunk.get(
                    "metadata"
                )
            )

            metadata = (
                raw_metadata
                if isinstance(
                    raw_metadata,
                    dict,
                )
                else None
            )

            raw_token_count = (
                chunk.get(
                    "token_count"
                )
            )

            token_count = (
                raw_token_count
                if isinstance(
                    raw_token_count,
                    int,
                )
                else None
            )

            retrieved.append(
                RetrievedChunk(
                    chunk_id=str(
                        chunk[
                            "chunk_id"
                        ]
                    ),
                    content=str(
                        chunk.get(
                            "content",
                            "",
                        )
                    ),
                    source_path=str(
                        chunk.get(
                            "source_path"
                        )
                        or "unknown"
                    ),
                    section_title=str(
                        chunk.get(
                            "section_title"
                        )
                        or "Untitled"
                    ),
                    score=float(
                        score
                    ),
                    rank=rank,
                    token_count=(
                        token_count
                    ),
                    metadata=metadata,
                )
            )

        return tuple(
            retrieved
        )

    def _validate_corpus(self) -> None:
        """Reject duplicate or missing chunk identifiers."""

        chunk_ids: list[str] = []

        for chunk in self._corpus:
            chunk_id = str(
                chunk.get(
                    "chunk_id",
                    "",
                )
            ).strip()

            if not chunk_id:
                raise ValueError(
                    "Every corpus chunk must have a non-empty chunk_id."
                )

            chunk_ids.append(
                chunk_id
            )

        if len(chunk_ids) != len(
            set(chunk_ids)
        ):
            raise ValueError(
                "Corpus chunk identifiers must be unique."
            )