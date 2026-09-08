from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from time import perf_counter_ns
from typing import Any

from src.rag.citations import (
    INSUFFICIENT_CONTEXT_RESPONSE,
    CitationValidationResult,
    validate_citations,
)
from src.rag.prompt import (
    ContextChunk,
    build_rag_prompt,
)
from src.rag.providers.base import (
    GenerationConfig,
    LLMMessage,
    LLMProvider,
    LLMResponse,
    LLMTruncatedResponseError,
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

    citations: tuple[str, ...]
    valid_citations: tuple[str, ...]
    invalid_citations: tuple[str, ...]

    citation_count: int
    citation_valid: bool
    citation_retry_count: int

    metadata: dict[str, Any] | None = None


class RAGPipeline:
    """Generate grounded answers from retrieved context."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        max_truncation_retries: int = 2,
        max_retry_tokens: int = 1024,
        max_citation_retries: int = 1,
    ) -> None:
        if max_truncation_retries < 0:
            raise ValueError(
                "max_truncation_retries cannot be negative."
            )

        if max_retry_tokens <= 0:
            raise ValueError(
                "max_retry_tokens must be greater than zero."
            )

        if max_citation_retries < 0:
            raise ValueError(
                "max_citation_retries cannot be negative."
            )

        self._provider = provider
        self._max_truncation_retries = (
            max_truncation_retries
        )
        self._max_retry_tokens = (
            max_retry_tokens
        )
        self._max_citation_retries = (
            max_citation_retries
        )

    @property
    def provider(self) -> LLMProvider:
        """Return the configured language model provider."""

        return self._provider

    def answer(
        self,
        query: str,
        chunks: Sequence[ContextChunk],
        generation_config: GenerationConfig | None = None,
    ) -> RAGResult:
        """Generate an answer grounded in retrieved chunks."""

        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError(
                "Query cannot be empty."
            )

        prompt = build_rag_prompt(
            query=normalized_query,
            chunks=chunks,
        )

        base_messages = [
            LLMMessage(
                role="system",
                content=prompt.system,
            ),
            LLMMessage(
                role="user",
                content=prompt.user,
            ),
        ]

        resolved_config = (
            generation_config
            or GenerationConfig()
        )

        available_chunk_ids = [
            chunk.chunk_id
            for chunk in chunks
        ]

        start_time = perf_counter_ns()

        response = self._generate_with_retry(
            messages=base_messages,
            config=resolved_config,
        )

        validation = self._validate_response(
            response=response,
            available_chunk_ids=available_chunk_ids,
        )

        citation_retry_count = 0

        while (
            not validation.citation_valid
            and citation_retry_count
            < self._max_citation_retries
        ):
            citation_retry_count += 1

            retry_messages = (
                self._build_citation_retry_messages(
                    base_messages=base_messages,
                    previous_answer=response.text,
                    available_chunk_ids=(
                        available_chunk_ids
                    ),
                )
            )

            response = self._generate_with_retry(
                messages=retry_messages,
                config=resolved_config,
            )

            validation = (
                self._validate_response(
                    response=response,
                    available_chunk_ids=(
                        available_chunk_ids
                    ),
                )
            )

        generation_time_ns = (
            perf_counter_ns()
            - start_time
        )

        metadata = dict(
            response.metadata or {}
        )

        metadata["citation_retry_count"] = (
            citation_retry_count
        )

        metadata["citation_count"] = len(
            validation.citations
        )

        metadata["citation_valid"] = (
            validation.citation_valid
        )

        metadata["valid_citations"] = list(
            validation.valid_citations
        )

        metadata["invalid_citations"] = list(
            validation.invalid_citations
        )

        return RAGResult(
            query=normalized_query,
            answer=response.text,
            context_chunks=tuple(chunks),
            provider=response.provider,
            model=response.model,
            generation_time_ns=(
                generation_time_ns
            ),
            citations=validation.citations,
            valid_citations=(
                validation.valid_citations
            ),
            invalid_citations=(
                validation.invalid_citations
            ),
            citation_count=len(
                validation.citations
            ),
            citation_valid=(
                validation.citation_valid
            ),
            citation_retry_count=(
                citation_retry_count
            ),
            metadata=metadata,
        )

    @staticmethod
    def _validate_response(
        response: LLMResponse,
        available_chunk_ids: Sequence[str],
    ) -> CitationValidationResult:
        """Validate citations contained in an LLM response."""

        return validate_citations(
            answer=response.text,
            available_chunk_ids=(
                available_chunk_ids
            ),
        )

    def _generate_with_retry(
        self,
        messages: Sequence[LLMMessage],
        config: GenerationConfig,
    ) -> LLMResponse:
        """Retry generation when the output is truncated."""

        current_config = config
        retry_count = 0

        while True:
            response = self._provider.generate(
                messages=messages,
                config=current_config,
            )

            metadata = dict(
                response.metadata or {}
            )

            done_reason = metadata.get(
                "done_reason"
            )

            if done_reason != "length":
                metadata["truncation_retries"] = (
                    retry_count
                )

                metadata["effective_max_tokens"] = (
                    current_config.max_tokens
                )

                return LLMResponse(
                    text=response.text,
                    model=response.model,
                    provider=response.provider,
                    metadata=metadata,
                )

            if (
                retry_count
                >= self._max_truncation_retries
            ):
                raise LLMTruncatedResponseError(
                    "Generation remained truncated "
                    f"after {retry_count} retries."
                )

            next_max_tokens = min(
                current_config.max_tokens * 2,
                self._max_retry_tokens,
            )

            if (
                next_max_tokens
                <= current_config.max_tokens
            ):
                raise LLMTruncatedResponseError(
                    "Generation reached the maximum "
                    "retry token limit."
                )

            retry_count += 1

            current_config = replace(
                current_config,
                max_tokens=next_max_tokens,
            )

    @staticmethod
    def _build_citation_retry_messages(
        base_messages: Sequence[LLMMessage],
        previous_answer: str,
        available_chunk_ids: Sequence[str],
    ) -> list[LLMMessage]:
        """Build messages requesting citation correction."""

        messages = list(
            base_messages
        )

        previous_content = (
            previous_answer.strip()
            or "No answer was produced."
        )

        messages.append(
            LLMMessage(
                role="assistant",
                content=previous_content,
            )
        )

        if not available_chunk_ids:
            correction = (
                "Your previous answer failed citation "
                "validation because no documentation "
                "chunks are available.\n\n"
                "Reply exactly with:\n"
                f"{INSUFFICIENT_CONTEXT_RESPONSE}"
            )

        else:
            allowed_citations = ", ".join(
                f"[{chunk_id}]"
                for chunk_id
                in available_chunk_ids
            )

            correction = (
                "Your previous answer failed citation "
                "validation.\n\n"
                "Rewrite the answer using only the "
                "retrieved documentation.\n\n"
                "Allowed citations:\n"
                f"{allowed_citations}\n\n"
                "Requirements:\n"
                "- Include at least one allowed citation.\n"
                "- Do not invent citations.\n"
                "- Keep the answer concise.\n"
                "- Return only the corrected final answer."
            )

        messages.append(
            LLMMessage(
                role="user",
                content=correction,
            )
        )

        return messages