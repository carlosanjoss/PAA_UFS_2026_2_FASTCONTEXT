from __future__ import annotations

import time
from typing import Any

from src.algorithms.inverted_index import InvertedIndex
from src.algorithms.topk_heap import top_k as select_top_k
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


class OptimizedRetriever(Retriever):
    """Use indexed filtering and bounded manual Top-k selection."""

    @property
    def name(self) -> str:
        """Return the canonical retriever identifier."""

        return "optimized"

    def __init__(
        self,
        corpus_chunks: list[CorpusChunk],
    ) -> None:
        self._corpus = list(
            corpus_chunks
        )

        self._corpus_map = (
            self._build_corpus_map()
        )

        self._index = InvertedIndex()
        self._index.build(
            self._corpus
        )

        self._index_build_time_ns = (
            self._index.build_time_ns
        )

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
        """Filter candidates and retain only the best k with a manual heap."""

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

        candidate_ids, binary_comparisons = (
            self._index
            .get_candidate_chunk_ids_with_comparisons(
                normalized_query
            )
        )

        query_vector = (
            self._vectorizer.transform(
                normalized_query
            )
        )

        candidates: list[
            Candidate
        ] = []

        for chunk_id in candidate_ids:
            chunk = self._corpus_map[
                chunk_id
            ]

            score = (
                self._vectorizer
                .cosine_similarity(
                    query_vector,
                    self._document_vectors[
                        chunk_id
                    ],
                )
            )

            if score > 0.0:
                candidates.append(
                    (
                        score,
                        chunk,
                    )
                )

        ranking_start = (
            time.perf_counter_ns()
        )

        selected, heap_stats = (
            select_top_k(
                candidates,
                top_k,
            )
        )

        ranking_time_ns = (
            time.perf_counter_ns()
            - ranking_start
        )

        chunks = (
            self._build_retrieved_chunks(
                selected
            )
        )

        metrics = RetrievalMetrics(
            retrieval_time_ns=(
                time.perf_counter_ns()
                - start_time
            ),
            sorting_time_ns=(
                ranking_time_ns
            ),
            index_build_time_ns=(
                self._index_build_time_ns
            ),
            comparisons=(
                binary_comparisons
                + heap_stats.comparisons
            ),
            chunks_scored=len(
                candidate_ids
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
                "candidate_strategy": (
                    "inverted_index_binary_search"
                ),
                "ranking_strategy": "top_k_heap",
                "binary_search_comparisons": (
                    binary_comparisons
                ),
                "ranking_comparisons": (
                    heap_stats.comparisons
                ),
                "heap_insertions": (
                    heap_stats.insertions
                ),
                "heap_replacements": (
                    heap_stats.replacements
                ),
                "max_heap_size": (
                    heap_stats.max_heap_size
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
        """Create an empty optimized result."""

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
                index_build_time_ns=(
                    self._index_build_time_ns
                ),
                comparisons=0,
                chunks_scored=0,
                candidates_found=0,
            ),
            metadata={
                "representation": "tfidf",
                "similarity": "cosine",
                "candidate_strategy": (
                    "inverted_index_binary_search"
                ),
                "ranking_strategy": "top_k_heap",
            },
        )

    def _build_corpus_map(
        self,
    ) -> dict[str, CorpusChunk]:
        """Build and validate the chunk lookup table."""

        corpus_map: dict[
            str,
            CorpusChunk,
        ] = {}

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

            if chunk_id in corpus_map:
                raise ValueError(
                    "Corpus chunk identifiers must be unique."
                )

            corpus_map[
                chunk_id
            ] = chunk

        return corpus_map

    def _build_document_vectors(
        self,
    ) -> dict[str, SparseVector]:
        """Precompute TF-IDF vectors."""

        return {
            chunk_id: (
                self._vectorizer.transform(
                    str(
                        chunk.get(
                            "content",
                            "",
                        )
                    )
                )
            )
            for chunk_id, chunk
            in self._corpus_map.items()
        }

    @staticmethod
    def _build_retrieved_chunks(
        candidates: list[Candidate],
    ) -> tuple[RetrievedChunk, ...]:
        """Convert candidate tuples to canonical chunks."""

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