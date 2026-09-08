from collections.abc import Sequence

import pytest

from src.rag.pipeline import (
    RAGPipeline,
)
from src.rag.prompt import ContextChunk
from src.rag.providers.base import (
    GenerationConfig,
    LLMMessage,
    LLMProvider,
    LLMResponse,
    LLMTruncatedResponseError,
)


class SuccessfulProvider(
    LLMProvider
):
    """Provider that returns a valid cited answer."""

    @property
    def name(self) -> str:
        return "fake"

    def is_available(self) -> bool:
        return True

    def generate(
        self,
        messages: Sequence[
            LLMMessage
        ],
        config: (
            GenerationConfig | None
        ) = None,
    ) -> LLMResponse:
        return LLMResponse(
            text=(
                "FastAPI supports dependency "
                "injection [chunk_001]."
            ),
            model="fake-model",
            provider=self.name,
            metadata={
                "done_reason": "stop",
            },
        )


class MissingCitationThenValidProvider(
    LLMProvider
):
    """Provider that fixes citations on retry."""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def name(self) -> str:
        return "citation-retry"

    def is_available(self) -> bool:
        return True

    def generate(
        self,
        messages: Sequence[
            LLMMessage
        ],
        config: (
            GenerationConfig | None
        ) = None,
    ) -> LLMResponse:
        self.calls += 1

        if self.calls == 1:
            text = (
                "FastAPI supports "
                "dependency injection."
            )
        else:
            text = (
                "FastAPI supports dependency "
                "injection [chunk_001]."
            )

        return LLMResponse(
            text=text,
            model="fake-model",
            provider=self.name,
            metadata={
                "done_reason": "stop",
            },
        )


class AlwaysInvalidCitationProvider(
    LLMProvider
):
    """Provider that always invents a citation."""

    @property
    def name(self) -> str:
        return "invalid-citation"

    def is_available(self) -> bool:
        return True

    def generate(
        self,
        messages: Sequence[
            LLMMessage
        ],
        config: (
            GenerationConfig | None
        ) = None,
    ) -> LLMResponse:
        return LLMResponse(
            text=(
                "Unsupported answer "
                "[chunk_999]."
            ),
            model="fake-model",
            provider=self.name,
            metadata={
                "done_reason": "stop",
            },
        )


class TruncateOnceProvider(
    LLMProvider
):
    """Provider that truncates once."""

    def __init__(self) -> None:
        self.calls = 0
        self.token_limits: list[
            int
        ] = []

    @property
    def name(self) -> str:
        return "truncate-once"

    def is_available(self) -> bool:
        return True

    def generate(
        self,
        messages: Sequence[
            LLMMessage
        ],
        config: (
            GenerationConfig | None
        ) = None,
    ) -> LLMResponse:
        resolved_config = (
            config
            or GenerationConfig()
        )

        self.calls += 1

        self.token_limits.append(
            resolved_config.max_tokens
        )

        if self.calls == 1:
            return LLMResponse(
                text="Incomplete answer",
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


class AlwaysTruncatedProvider(
    LLMProvider
):
    """Provider that always truncates."""

    @property
    def name(self) -> str:
        return "always-truncated"

    def is_available(self) -> bool:
        return True

    def generate(
        self,
        messages: Sequence[
            LLMMessage
        ],
        config: (
            GenerationConfig | None
        ) = None,
    ) -> LLMResponse:
        return LLMResponse(
            text="Incomplete answer",
            model="fake-model",
            provider=self.name,
            metadata={
                "done_reason": "length",
            },
        )


def build_chunk() -> ContextChunk:
    """Create a reusable chunk."""

    return ContextChunk(
        chunk_id="chunk_001",
        content=(
            "FastAPI supports dependency "
            "injection."
        ),
        source_path=(
            "tutorial/dependencies/index.md"
        ),
        section_title="Dependencies",
        score=0.95,
    )


def test_pipeline_returns_valid_citation() -> None:
    pipeline = RAGPipeline(
        SuccessfulProvider()
    )

    result = pipeline.answer(
        query=(
            "How do dependencies work?"
        ),
        chunks=[
            build_chunk()
        ],
    )

    assert result.citation_valid is True

    assert result.citations == (
        "chunk_001",
    )

    assert result.valid_citations == (
        "chunk_001",
    )

    assert (
        result.invalid_citations
        == ()
    )

    assert result.citation_count == 1

    assert (
        result.citation_retry_count
        == 0
    )


def test_pipeline_retries_missing_citation() -> None:
    provider = (
        MissingCitationThenValidProvider()
    )

    pipeline = RAGPipeline(
        provider=provider,
        max_citation_retries=1,
    )

    result = pipeline.answer(
        query=(
            "How do dependencies work?"
        ),
        chunks=[
            build_chunk()
        ],
    )

    assert provider.calls == 2
    assert result.citation_valid is True

    assert (
        result.citation_retry_count
        == 1
    )

    assert result.citations == (
        "chunk_001",
    )


def test_pipeline_marks_invalid_after_retry_limit() -> None:
    pipeline = RAGPipeline(
        provider=(
            AlwaysInvalidCitationProvider()
        ),
        max_citation_retries=1,
    )

    result = pipeline.answer(
        query="Test query",
        chunks=[
            build_chunk()
        ],
    )

    assert result.citation_valid is False

    assert result.invalid_citations == (
        "chunk_999",
    )

    assert (
        result.citation_retry_count
        == 1
    )


def test_pipeline_retries_truncation() -> None:
    provider = (
        TruncateOnceProvider()
    )

    pipeline = RAGPipeline(
        provider=provider,
        max_truncation_retries=2,
        max_retry_tokens=512,
    )

    result = pipeline.answer(
        query="How does it work?",
        chunks=[
            build_chunk()
        ],
        generation_config=(
            GenerationConfig(
                max_tokens=128,
            )
        ),
    )

    assert provider.calls == 2

    assert provider.token_limits == [
        128,
        256,
    ]

    assert result.metadata is not None

    assert (
        result.metadata[
            "truncation_retries"
        ]
        == 1
    )


def test_pipeline_raises_after_truncation_limit() -> None:
    pipeline = RAGPipeline(
        provider=(
            AlwaysTruncatedProvider()
        ),
        max_truncation_retries=1,
        max_retry_tokens=256,
    )

    with pytest.raises(
        LLMTruncatedResponseError
    ):
        pipeline.answer(
            query="Test query",
            chunks=[
                build_chunk()
            ],
            generation_config=(
                GenerationConfig(
                    max_tokens=128,
                )
            ),
        )


def test_pipeline_rejects_empty_query() -> None:
    pipeline = RAGPipeline(
        SuccessfulProvider()
    )

    with pytest.raises(
        ValueError,
        match="Query cannot be empty",
    ):
        pipeline.answer(
            query="   ",
            chunks=[],
        )