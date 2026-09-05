from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns

from src.rag.adapters import (
    retrieval_result_to_context_chunks,
)
from src.rag.pipeline import (
    RAGPipeline,
    RAGResult,
)
from src.rag.providers.base import (
    GenerationConfig,
)
from src.retrieval.base import Retriever
from src.retrieval.models import (
    RetrievalResult,
)


class RAGNotConfiguredError(RuntimeError):
    """Raised when RAG generation is requested without a pipeline."""


@dataclass(frozen=True, slots=True)
class FastContextResult:
    """Complete result produced by the FastContext service."""

    query: str
    retrieval: RetrievalResult
    rag: RAGResult
    end_to_end_time_ns: int

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError(
                "query cannot be empty."
            )

        if self.end_to_end_time_ns < 0:
            raise ValueError(
                "end_to_end_time_ns cannot be negative."
            )

        if self.retrieval.query != self.query:
            raise ValueError(
                "Retrieval query does not match "
                "the service query."
            )

        if self.rag.query != self.query:
            raise ValueError(
                "RAG query does not match "
                "the service query."
            )

    @property
    def algorithm(self) -> str:
        """Return the retrieval algorithm used."""

        return self.retrieval.algorithm

    @property
    def retrieval_time_ns(self) -> int:
        """Return retrieval time without LLM generation."""

        return (
            self.retrieval
            .metrics
            .retrieval_time_ns
        )

    @property
    def generation_time_ns(self) -> int:
        """Return LLM generation time."""

        return self.rag.generation_time_ns


class FastContextService:
    """Coordinate retrieval and optional RAG generation."""

    def __init__(
        self,
        retriever: Retriever,
        rag_pipeline: RAGPipeline | None = None,
    ) -> None:
        self._retriever = retriever
        self._rag_pipeline = rag_pipeline

    @property
    def retriever(self) -> Retriever:
        """Return the configured retriever."""

        return self._retriever

    @property
    def rag_pipeline(self) -> RAGPipeline | None:
        """Return the configured RAG pipeline."""

        return self._rag_pipeline

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> RetrievalResult:
        """Run retrieval without invoking an LLM."""

        normalized_query = self._validate_request(
            query=query,
            top_k=top_k,
        )

        return self._retriever.retrieve(
            query=normalized_query,
            top_k=top_k,
        )

    def ask(
        self,
        query: str,
        top_k: int = 5,
        generation_config: (
            GenerationConfig | None
        ) = None,
    ) -> FastContextResult:
        """Run retrieval followed by grounded generation."""

        normalized_query = self._validate_request(
            query=query,
            top_k=top_k,
        )

        if self._rag_pipeline is None:
            raise RAGNotConfiguredError(
                "RAG generation was requested, "
                "but no RAG pipeline is configured."
            )

        start_time = perf_counter_ns()

        retrieval_result = (
            self._retriever.retrieve(
                query=normalized_query,
                top_k=top_k,
            )
        )

        context_chunks = (
            retrieval_result_to_context_chunks(
                retrieval_result
            )
        )

        rag_result = (
            self._rag_pipeline.answer(
                query=normalized_query,
                chunks=context_chunks,
                generation_config=(
                    generation_config
                ),
            )
        )

        end_to_end_time_ns = (
            perf_counter_ns()
            - start_time
        )

        return FastContextResult(
            query=normalized_query,
            retrieval=retrieval_result,
            rag=rag_result,
            end_to_end_time_ns=(
                end_to_end_time_ns
            ),
        )

    @staticmethod
    def _validate_request(
        query: str,
        top_k: int,
    ) -> str:
        """Validate and normalize a service request."""

        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError(
                "Query cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        return normalized_query