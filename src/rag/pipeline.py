from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns
from typing import Sequence

from src.rag.prompt import ContextChunk, build_rag_prompt
from src.rag.providers.base import (
    GenerationConfig,
    LLMProvider,
    LLMResponse,
)


@dataclass(frozen=True, slots=True)
class RAGResult:
    """Result produced by the RAG pipeline."""

    query: str
    answer: str
    context_chunks: tuple[ContextChunk, ...]
    provider: str
    model: str
    generation_time_ns: int


class RAGPipeline:
    """Generate grounded answers from previously retrieved context."""

    def __init__(
        self,
        provider: LLMProvider,
    ) -> None:
        self._provider = provider

    @property
    def provider(self) -> LLMProvider:
        return self._provider

    def answer(
        self,
        query: str,
        chunks: Sequence[ContextChunk],
        generation_config: GenerationConfig | None = None,
    ) -> RAGResult:
        """Generate an answer grounded in retrieved chunks."""

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        prompt = build_rag_prompt(
            query=query,
            chunks=chunks,
        )

        start_time = perf_counter_ns()

        response: LLMResponse = self._provider.generate(
            prompt=prompt,
            config=generation_config,
        )

        generation_time_ns = perf_counter_ns() - start_time

        return RAGResult(
            query=query.strip(),
            answer=response.text,
            context_chunks=tuple(chunks),
            provider=response.provider,
            model=response.model,
            generation_time_ns=generation_time_ns,
        )