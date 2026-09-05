from __future__ import annotations

from src.rag.pipeline import RAGPipeline
from src.retrieval.registry import (
    RetrieverRegistry,
)
from src.services.fastcontext import (
    FastContextService,
)


def create_fastcontext_service(
    algorithm: str,
    registry: RetrieverRegistry,
    *,
    rag_pipeline: RAGPipeline | None = None,
) -> FastContextService:
    """Create a FastContext service for a retrieval algorithm."""

    retriever = registry.create(
        algorithm
    )

    return FastContextService(
        retriever=retriever,
        rag_pipeline=rag_pipeline,
    )