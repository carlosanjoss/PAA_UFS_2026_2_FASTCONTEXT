from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from src.app.models import ApplicationContainer
from src.rag.factory import create_llm_provider
from src.rag.pipeline import RAGPipeline
from src.rag.providers.base import LLMProvider
from src.rag.settings import RAGSettings, load_rag_settings
from src.retrieval.indexed_retriever import IndexedRetriever
from src.retrieval.linear_retriever import LinearRetriever
from src.retrieval.optimized_retriever import OptimizedRetriever
from src.retrieval.registry import RetrieverFactory, RetrieverRegistry
from src.retrieval.semantic_retriever import SemanticRetriever


def build_retriever_registry(
    registrations: Mapping[str, RetrieverFactory] | None = None,
) -> RetrieverRegistry:
    """Create a retriever registry from optional registrations."""

    registry = RetrieverRegistry()

    if registrations is None:
        return registry

    for name, factory in registrations.items():
        registry.register(
            name=name,
            factory=factory,
        )

    return registry


def build_default_retriever_registry(
    corpus_chunks: Sequence[Mapping[str, Any]] | None = None,
) -> RetrieverRegistry:
    """Create the standard FastContext retrieval registry."""

    chunks = _copy_corpus_chunks(
        corpus_chunks
    )

    registrations: dict[
        str,
        RetrieverFactory,
    ] = {
        "linear": lambda: LinearRetriever(
            list(chunks)
        ),
        "indexed": lambda: IndexedRetriever(
            list(chunks)
        ),
        "optimized": lambda: OptimizedRetriever(
            list(chunks)
        ),
        "semantic": lambda: SemanticRetriever(
            list(chunks)
        ),
    }

    return build_retriever_registry(
        registrations
    )


def create_application(
    *,
    settings: RAGSettings | None = None,
    registry: RetrieverRegistry | None = None,
    provider: LLMProvider | None = None,
    corpus_chunks: Sequence[Mapping[str, Any]] | None = None,
) -> ApplicationContainer:
    """Build the FastContext application dependency container."""

    resolved_settings = (
        settings
        if settings is not None
        else load_rag_settings()
    )

    if registry is not None:
        resolved_registry = registry
        corpus_size: int | None = None
    else:
        normalized_chunks = (
            _copy_corpus_chunks(
                corpus_chunks
            )
        )

        resolved_registry = (
            build_default_retriever_registry(
                normalized_chunks
            )
        )

        corpus_size = len(
            normalized_chunks
        )

    resolved_provider = (
        provider
        if provider is not None
        else create_llm_provider(
            resolved_settings
        )
    )

    rag_pipeline = RAGPipeline(
        provider=resolved_provider,
        max_truncation_retries=1,
        max_retry_tokens=256,
        max_citation_retries=1,
    )

    return ApplicationContainer(
        settings=resolved_settings,
        registry=resolved_registry,
        provider=resolved_provider,
        rag_pipeline=rag_pipeline,
        corpus_size=corpus_size,
    )


def _copy_corpus_chunks(
    corpus_chunks: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Create an isolated mutable copy of corpus chunks."""

    if corpus_chunks is None:
        return []

    return [
        dict(chunk)
        for chunk in corpus_chunks
    ]