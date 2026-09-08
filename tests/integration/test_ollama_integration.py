from collections.abc import Sequence

import pytest

from src.rag.pipeline import RAGPipeline
from src.rag.prompt import ContextChunk
from src.rag.providers.base import (
    GenerationConfig,
    LLMMessage,
    LLMProvider,
    LLMResponse,
    LLMTruncatedResponseError,
)


class FakeLLMProvider(LLMProvider):
    """Deterministic provider used for tests."""

    def __init__(self) -> None:
        self.received_messages: tuple[
            LLMMessage,
            ...,
        ] = ()

    @property
    def name(self) -> str:
        return "fake"

    def is_available(self) -> bool:
        return True

    def generate(
        self,
        messages: Sequence[LLMMessage],
        config: GenerationConfig | None = None,
    ) -> LLMResponse:
        self.received_messages = tuple(
            messages
        )

        return LLMResponse(
            text=(
                "FastAPI dependencies are "
                "described in [chunk_001]."
            ),
            model="fake-model",
            provider=self.name,
            metadata={
                "done_reason": "stop",
            },
        )


class TruncateOnceProvider(LLMProvider):
    """Return a truncated response once."""

    def __init__(self) -> None:
        self.calls = 0
        self.received_token_limits: list[
            int
        ] = []

    @property
    def name(self) -> str:
        return "truncate-once"

    def is_available(self) -> bool:
        return True

    def generate(
        self,
        messages: Sequence[LLMMessage],
        config: GenerationConfig | None = None,
    ) -> LLMResponse:
        resolved_config = (
            config or GenerationConfig()
        )

        self.calls += 1

        self.received_token_limits.append(
            resolved_config.max_tokens
        )

        if self.calls == 1:
            return LLMResponse(
                text="Incomplete",
                model="fake-model",
                provider=self.name,
                metadata={
                    "done_reason": "length",
                },
            )

        return LLMResponse(
            text=(
                "Complete answer "
                "[chunk_001]."
            ),
            model="fake-model",
            provider=self.name,
            metadata={
                "done_reason": "stop",
            },
        )


class AlwaysTruncatedProvider(LLMProvider):
    """Always return a truncated response."""

    @property
    def name(self) -> str:
        return "always-truncated"

    def is_available(self) -> bool:
        return True

    def generate(
        self,
        messages: Sequence[LLMMessage],
        config: GenerationConfig | None = None,
    ) -> LLMResponse:
        return LLMResponse(
            text="Incomplete",
            model="fake-model",
            provider=self.name,
            metadata={
                "done_reason": "length",
            },
        )


def build_chunk() -> ContextChunk:
    return ContextChunk(
        chunk_id="chunk_001",
        content=(
            "FastAPI supports "
            "dependency injection."
        ),
        source_path=(
            "tutorial/dependencies/index.md"
        ),
        section_title="Dependencies",
        score=0.95,
    )


def test_rag_pipeline_returns_result() -> None:
    provider = FakeLLMProvider()

    pipeline = RAGPipeline(
        provider
    )

    chunk = build_chunk()

    result = pipeline.answer(
        query=(
            "How do FastAPI "
            "dependencies work?"
        ),
        chunks=[chunk],
    )

    assert result.provider == "fake"
    assert result.model == "fake-model"

    assert (
        result.context_chunks
        == (chunk,)
    )

    assert "[chunk_001]" in result.answer

    assert (
        result.generation_time_ns
        >= 0
    )

    assert result.metadata is not None

    assert (
        result.metadata[
            "done_reason"
        ]
        == "stop"
    )

    assert (
        result.metadata[
            "truncation_retries"
        ]
        == 0
    )

    assert len(
        provider.received_messages
    ) == 2

    assert (
        provider.received_messages[
            0
        ].role
        == "system"
    )

    assert (
        provider.received_messages[
            1
        ].role
        == "user"
    )


def test_pipeline_retries_truncation() -> None:
    provider = TruncateOnceProvider()

    pipeline = RAGPipeline(
        provider=provider,
        max_truncation_retries=2,
        max_retry_tokens=512,
    )

    result = pipeline.answer(
        query="How does it work?",
        chunks=[build_chunk()],
        generation_config=GenerationConfig(
            max_tokens=128,
            think=False,
        ),
    )

    assert provider.calls == 2

    assert (
        provider.received_token_limits
        == [128, 256]
    )

    assert result.metadata is not None

    assert (
        result.metadata[
            "truncation_retries"
        ]
        == 1
    )

    assert (
        result.metadata[
            "effective_max_tokens"
        ]
        == 256
    )


def test_pipeline_raises_after_retry_limit() -> None:
    provider = (
        AlwaysTruncatedProvider()
    )

    pipeline = RAGPipeline(
        provider=provider,
        max_truncation_retries=1,
        max_retry_tokens=256,
    )

    with pytest.raises(
        LLMTruncatedResponseError
    ):
        pipeline.answer(
            query="Test query",
            chunks=[build_chunk()],
            generation_config=(
                GenerationConfig(
                    max_tokens=128,
                )
            ),
        )


def test_rag_pipeline_rejects_empty_query() -> None:
    pipeline = RAGPipeline(
        FakeLLMProvider()
    )

    with pytest.raises(
        ValueError,
        match="Query cannot be empty",
    ):
        pipeline.answer(
            query="   ",
            chunks=[],
        )