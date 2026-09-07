from __future__ import annotations

import time
from typing import Any

from src.representations.embeddings import (
    EmbeddingEncoder,
)
from src.retrieval.base import Retriever
from src.retrieval.models import (
    RetrievalMetrics,
    RetrievalResult,
    RetrievedChunk,
)
from src.semantic.faiss_index import (
    FaissIndex,
    FaissSearchResult,
)

CorpusChunk = dict[str, Any]


class SemanticRetrieverError(RuntimeError):
    """Base exception raised by the semantic retriever."""


class SemanticIndexMappingError(
    SemanticRetrieverError
):
    """Raised when a FAISS result cannot be mapped to a corpus chunk."""


class SemanticRetriever(Retriever):
    """Retrieve chunks with BGE embeddings and FAISS inner-product search."""

    @property
    def name(self) -> str:
        """Return the canonical retriever identifier."""

        return "semantic"

    def __init__(
        self,
        corpus_chunks: list[CorpusChunk],
        *,
        encoder: EmbeddingEncoder | None = None,
        index: FaissIndex | None = None,
    ) -> None:
        self._corpus = list(
            corpus_chunks
        )

        self._corpus_map = (
            self._build_corpus_map()
        )

        self._encoder = (
            encoder
            if encoder is not None
            else EmbeddingEncoder()
        )

        self._index = (
            index
            if index is not None
            else FaissIndex(
                dimension=(
                    self._encoder.dimension
                )
            )
        )

        if (
            self._index.dimension
            != self._encoder.dimension
        ):
            raise ValueError(
                "FAISS index dimension must match "
                "the embedding dimension."
            )

        self._embedding_build_time_ns = 0
        self._faiss_build_time_ns = 0
        self._index_build_time_ns = 0

        self._build_index()

    @property
    def corpus_size(self) -> int:
        """Return the number of corpus chunks."""

        return len(
            self._corpus
        )

    @property
    def embedding_dimension(self) -> int:
        """Return the semantic embedding dimension."""

        return self._encoder.dimension

    @property
    def index_size(self) -> int:
        """Return the number of vectors stored in FAISS."""

        return self._index.size

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> RetrievalResult:
        """Retrieve semantic Top-k chunks using BGE and FAISS."""

        normalized_query = (
            self._validate_request(
                query,
                top_k,
            )
        )

        start_time = (
            time.perf_counter_ns()
        )

        if (
            top_k == 0
            or self._index.is_empty
        ):
            return self._empty_result(
                query=normalized_query,
                top_k=top_k,
                start_time=start_time,
            )

        embedding_start = (
            time.perf_counter_ns()
        )

        query_embedding = (
            self._encoder.encode_query(
                normalized_query
            )
        )

        query_embedding_time_ns = (
            time.perf_counter_ns()
            - embedding_start
        )

        search_start = (
            time.perf_counter_ns()
        )

        faiss_results = (
            self._index.search(
                query_embedding,
                top_k=top_k,
            )
        )

        faiss_search_time_ns = (
            time.perf_counter_ns()
            - search_start
        )

        ranking_start = (
            time.perf_counter_ns()
        )

        ordered_results = sorted(
            faiss_results,
            key=lambda result: (
                -result.score,
                result.chunk_id,
            ),
        )

        ranking_time_ns = (
            time.perf_counter_ns()
            - ranking_start
        )

        chunks = (
            self._build_retrieved_chunks(
                ordered_results
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
            comparisons=None,
            chunks_scored=(
                self._index.size
            ),
            candidates_found=len(
                faiss_results
            ),
        )

        return RetrievalResult(
            query=normalized_query,
            algorithm=self.name,
            top_k=top_k,
            chunks=chunks,
            metrics=metrics,
            metadata={
                "representation": "bge_embeddings",
                "model": (
                    self._encoder.model_name
                ),
                "embedding_dimension": (
                    self._encoder.dimension
                ),
                "normalize_embeddings": (
                    self._encoder
                    .config
                    .normalize_embeddings
                ),
                "similarity": (
                    "inner_product_on_l2_normalized_vectors"
                ),
                "candidate_strategy": (
                    "faiss_index_flat_ip"
                ),
                "ranking_strategy": (
                    "faiss_top_k_with_chunk_id_tie_break"
                ),
                "embedding_build_time_ns": (
                    self._embedding_build_time_ns
                ),
                "faiss_build_time_ns": (
                    self._faiss_build_time_ns
                ),
                "query_embedding_time_ns": (
                    query_embedding_time_ns
                ),
                "faiss_search_time_ns": (
                    faiss_search_time_ns
                ),
            },
        )

    def _build_index(
        self,
    ) -> None:
        """Generate corpus embeddings and build the FAISS index."""

        build_start = (
            time.perf_counter_ns()
        )

        chunk_ids = list(
            self._corpus_map
        )

        contents = [
            str(
                self._corpus_map[
                    chunk_id
                ].get(
                    "content",
                    "",
                )
            )
            for chunk_id in chunk_ids
        ]

        embedding_start = (
            time.perf_counter_ns()
        )

        document_embeddings = (
            self._encoder
            .encode_documents(
                contents
            )
        )

        self._embedding_build_time_ns = (
            time.perf_counter_ns()
            - embedding_start
        )

        faiss_start = (
            time.perf_counter_ns()
        )

        self._index.build(
            document_embeddings,
            chunk_ids,
        )

        self._faiss_build_time_ns = (
            time.perf_counter_ns()
            - faiss_start
        )

        self._index_build_time_ns = (
            time.perf_counter_ns()
            - build_start
        )

    def _build_corpus_map(
        self,
    ) -> dict[str, CorpusChunk]:
        """Build and validate the semantic chunk lookup table."""

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
                    "Every corpus chunk must have "
                    "a non-empty chunk_id."
                )

            if chunk_id in corpus_map:
                raise ValueError(
                    "Corpus chunk identifiers "
                    "must be unique."
                )

            content = str(
                chunk.get(
                    "content",
                    "",
                )
            ).strip()

            if not content:
                raise ValueError(
                    "Every semantic corpus chunk "
                    "must have non-empty content."
                )

            corpus_map[
                chunk_id
            ] = chunk

        return corpus_map

    def _build_retrieved_chunks(
        self,
        results: list[FaissSearchResult]
        | tuple[FaissSearchResult, ...],
    ) -> tuple[RetrievedChunk, ...]:
        """Convert FAISS results into canonical retrieval chunks."""

        retrieved: list[
            RetrievedChunk
        ] = []

        for rank, result in enumerate(
            results,
            start=1,
        ):
            chunk = (
                self._corpus_map.get(
                    result.chunk_id
                )
            )

            if chunk is None:
                raise SemanticIndexMappingError(
                    "FAISS result references an "
                    "unknown corpus chunk."
                )

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
                    chunk_id=(
                        result.chunk_id
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
                        result.score
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

    def _empty_result(
        self,
        *,
        query: str,
        top_k: int,
        start_time: int,
    ) -> RetrievalResult:
        """Create an empty semantic retrieval result."""

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
                comparisons=None,
                chunks_scored=0,
                candidates_found=0,
            ),
            metadata={
                "representation": (
                    "bge_embeddings"
                ),
                "model": (
                    self._encoder.model_name
                ),
                "embedding_dimension": (
                    self._encoder.dimension
                ),
                "normalize_embeddings": (
                    self._encoder
                    .config
                    .normalize_embeddings
                ),
                "similarity": (
                    "inner_product_on_l2_normalized_vectors"
                ),
                "candidate_strategy": (
                    "faiss_index_flat_ip"
                ),
                "ranking_strategy": (
                    "faiss_top_k_with_chunk_id_tie_break"
                ),
                "embedding_build_time_ns": (
                    self._embedding_build_time_ns
                ),
                "faiss_build_time_ns": (
                    self._faiss_build_time_ns
                ),
            },
        )