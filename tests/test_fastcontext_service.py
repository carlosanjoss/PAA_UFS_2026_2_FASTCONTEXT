from collections.abc import Sequence

import pytest

from src.rag.pipeline import RAGPipeline
from src.rag.providers.base import (
    GenerationConfig,
    LLMMessage,
    LLMProvider,
    LLMResponse,
)
from src.retrieval.base import Retriever
from src.retrieval.models import (
    RetrievalMetrics,
    RetrievalResult,
    RetrievedChunk,
)
from src.services.fastcontext import (
    FastContextService,
    RAGNotConfiguredError,
)


class FakeRetriever(Retriever):
    """Deterministic retriever used for service tests."""

    def __init__(self) -> None:
        self.last_query: str | None = None
        self.last_top_k: int | None = None

    @property
    def name(self) -> str:
        return "fake-linear"

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> RetrievalResult:
        self.last_query = query
        self.last_top_k = top_k

        chunk = RetrievedChunk(
            chunk_id="chunk_001",
            content=(
                "FastAPI has a Dependency "
                "Injection system."
            ),
            source_path=(
                "tutorial/dependencies/index.md"
            ),
            section_title="Dependencies",
            score=0.95,
            rank=1,
            metadata={
                "representation": "tfidf",
            },
        )

        metrics = RetrievalMetrics(
            retrieval_time_ns=1_000,
            sorting_time_ns=200,
            comparisons=10,
            chunks_scored=10,
            candidates_found=5,
        )

        return RetrievalResult(
            query=query,
            algorithm=self.name,
            top_k=top_k,
            chunks=(chunk,),
            metrics=metrics,
        )


class FakeLLMProvider(LLMProvider):
    """Deterministic provider used for service tests."""

    @property
    def name(self) -> str:
        return "fake-llm"

    def is_available(self) -> bool:
        return True

    def generate(
        self,
        messages: Sequence[LLMMessage],
        config: GenerationConfig | None = None,
    ) -> LLMResponse:
        return LLMResponse(
            text=(
                "FastAPI uses a Dependency "
                "Injection system "
                "[chunk_001]."
            ),
            model="fake-model",
            provider=self.name,
            metadata={
                "done_reason": "stop",
                "thinking_enabled": False,
            },
        )


def build_service() -> FastContextService:
    """Create a complete service for tests."""

    retriever = FakeRetriever()

    rag_pipeline = RAGPipeline(
        provider=FakeLLMProvider(),
        max_citation_retries=0,
    )

    return FastContextService(
        retriever=retriever,
        rag_pipeline=rag_pipeline,
    )


def test_service_retrieve_only() -> None:
    retriever = FakeRetriever()

    service = FastContextService(
        retriever=retriever,
    )

    result = service.retrieve(
        query="How do dependencies work?",
        top_k=3,
    )

    assert (
        result.algorithm
        == "fake-linear"
    )

    assert result.top_k == 3
    assert len(result.chunks) == 1

    assert (
        result.chunks[0].chunk_id
        == "chunk_001"
    )

    assert (
        retriever.last_query
        == "How do dependencies work?"
    )

    assert (
        retriever.last_top_k
        == 3
    )


def test_service_ask_runs_retrieval_and_rag() -> None:
    service = build_service()

    result = service.ask(
        query="How do dependencies work?",
        top_k=5,
    )

    assert (
        result.query
        == "How do dependencies work?"
    )

    assert (
        result.algorithm
        == "fake-linear"
    )

    assert (
        result.retrieval.algorithm
        == "fake-linear"
    )

    assert (
        result.rag.provider
        == "fake-llm"
    )

    assert (
        result.rag.model
        == "fake-model"
    )

    assert (
        result.rag.citation_valid
        is True
    )

    assert (
        result.rag.valid_citations
        == ("chunk_001",)
    )

    assert (
        result.end_to_end_time_ns
        >= 0
    )


def test_service_preserves_retrieval_metrics() -> None:
    service = build_service()

    result = service.ask(
        query="How do dependencies work?",
    )

    assert (
        result.retrieval_time_ns
        == 1_000
    )

    assert (
        result.retrieval.metrics.comparisons
        == 10
    )

    assert (
        result.retrieval.metrics.chunks_scored
        == 10
    )

    assert (
        result.retrieval.metrics.candidates_found
        == 5
    )


def test_service_exposes_generation_time() -> None:
    service = build_service()

    result = service.ask(
        query="How do dependencies work?",
    )

    assert (
        result.generation_time_ns
        == result.rag.generation_time_ns
    )

    assert (
        result.generation_time_ns
        >= 0
    )


def test_service_without_rag_can_still_retrieve() -> None:
    service = FastContextService(
        retriever=FakeRetriever(),
    )

    result = service.retrieve(
        query="Test query",
    )

    assert (
        result.query
        == "Test query"
    )


def test_service_without_rag_rejects_ask() -> None:
    service = FastContextService(
        retriever=FakeRetriever(),
    )

    with pytest.raises(
        RAGNotConfiguredError,
        match="no RAG pipeline",
    ):
        service.ask(
            query="Test query",
        )


def test_service_normalizes_query() -> None:
    retriever = FakeRetriever()

    service = FastContextService(
        retriever=retriever,
    )

    result = service.retrieve(
        query="   Test query   ",
    )

    assert (
        result.query
        == "Test query"
    )

    assert (
        retriever.last_query
        == "Test query"
    )


def test_service_rejects_empty_query() -> None:
    service = FastContextService(
        retriever=FakeRetriever(),
    )

    with pytest.raises(
        ValueError,
        match="Query cannot be empty",
    ):
        service.retrieve(
            query="   ",
        )


def test_service_rejects_invalid_top_k() -> None:
    service = FastContextService(
        retriever=FakeRetriever(),
    )

    with pytest.raises(
        ValueError,
        match="top_k must be greater than zero",
    ):
        service.retrieve(
            query="Test query",
            top_k=0,
        )


def test_service_passes_generation_config() -> None:
    service = build_service()

    result = service.ask(
        query="Test query",
        generation_config=GenerationConfig(
            temperature=0.0,
            max_tokens=128,
            think=False,
        ),
    )

    assert result.rag.answer
    assert (
        result.rag.citation_valid
        is True
    )