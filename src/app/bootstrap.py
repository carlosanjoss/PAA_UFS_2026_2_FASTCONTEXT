from __future__ import annotations

from collections.abc import Mapping

from src.app.models import ApplicationContainer
from src.rag.factory import create_llm_provider
from src.rag.pipeline import RAGPipeline
from src.rag.providers.base import LLMProvider
from src.rag.settings import (
    RAGSettings,
    load_rag_settings,
)
from src.retrieval.registry import (
    RetrieverFactory,
    RetrieverRegistry,
)


def build_retriever_registry(
    registrations: Mapping[
        str,
        RetrieverFactory,
    ]
    | None = None,
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


def create_application(
    *,
    settings: RAGSettings | None = None,
    registry: RetrieverRegistry | None = None,
    provider: LLMProvider | None = None,
) -> ApplicationContainer:
    """Build the FastContext application dependency container."""

    resolved_settings = (
        settings
        if settings is not None
        else load_rag_settings()
    )

    resolved_registry = (
        registry
        if registry is not None
        else build_retriever_registry()
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
    )