from __future__ import annotations

from collections.abc import Sequence

from src.rag.providers.base import (
    GenerationConfig,
    LLMMessage,
    LLMProvider,
    LLMResponse,
    LLMResponseError,
)
from src.rag.providers.fallback import (
    FallbackLLMProvider,
)


class SuccessfulProvider(LLMProvider):
    """Provider that always returns a successful response."""

    def __init__(
        self,
        provider_name: str,
        model: str,
    ) -> None:
        self._provider_name = provider_name
        self._model = model

    @property
    def name(self) -> str:
        return self._provider_name

    def is_available(self) -> bool:
        return True

    def generate(
        self,
        messages: Sequence[LLMMessage],
        config: GenerationConfig | None = None,
    ) -> LLMResponse:
        return LLMResponse(
            text="Successful response.",
            model=self._model,
            provider=self.name,
            metadata={
                "done_reason": "stop",
            },
        )


class FailingProvider(LLMProvider):
    """Provider that always raises a provider error."""

    @property
    def name(self) -> str:
        return "failing"

    def is_available(self) -> bool:
        return False

    def generate(
        self,
        messages: Sequence[LLMMessage],
        config: GenerationConfig | None = None,
    ) -> LLMResponse:
        raise LLMResponseError(
            "Simulated provider failure."
        )


class TruncatedProvider(LLMProvider):
    """Provider that returns a truncated response."""

    @property
    def name(self) -> str:
        return "truncated"

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

        return LLMResponse(
            text="Incomplete response.",
            model="truncated-model",
            provider=self.name,
            metadata={
                "done_reason": "length",
                "requested_max_tokens": (
                    resolved_config.max_tokens
                ),
            },
        )


class TrackingProvider(LLMProvider):
    """Provider that records how many times it is called."""

    def __init__(
        self,
        provider_name: str,
        model: str,
    ) -> None:
        self._provider_name = provider_name
        self._model = model
        self.call_count = 0

    @property
    def name(self) -> str:
        return self._provider_name

    def is_available(self) -> bool:
        return True

    def generate(
        self,
        messages: Sequence[LLMMessage],
        config: GenerationConfig | None = None,
    ) -> LLMResponse:
        self.call_count += 1

        return LLMResponse(
            text="Fallback response.",
            model=self._model,
            provider=self.name,
            metadata={
                "done_reason": "stop",
            },
        )


def build_messages() -> list[LLMMessage]:
    """Create messages used by fallback provider tests."""

    return [
        LLMMessage(
            role="user",
            content="Test prompt.",
        )
    ]


def test_primary_provider_is_used_when_successful() -> None:
    primary = SuccessfulProvider(
        provider_name="primary",
        model="primary-model",
    )

    fallback = SuccessfulProvider(
        provider_name="fallback",
        model="fallback-model",
    )

    provider = FallbackLLMProvider(
        primary=primary,
        fallback=fallback,
    )

    response = provider.generate(
        messages=build_messages()
    )

    assert response.model == "primary-model"
    assert response.provider == "primary"
    assert response.metadata is not None

    assert (
        response.metadata["fallback_used"]
        is False
    )

    assert (
        response.metadata["done_reason"]
        == "stop"
    )


def test_fallback_provider_is_used_on_failure() -> None:
    primary = FailingProvider()

    fallback = SuccessfulProvider(
        provider_name="fallback",
        model="fallback-model",
    )

    provider = FallbackLLMProvider(
        primary=primary,
        fallback=fallback,
    )

    response = provider.generate(
        messages=build_messages()
    )

    assert response.model == "fallback-model"
    assert response.provider == "fallback"
    assert response.metadata is not None

    assert (
        response.metadata["fallback_used"]
        is True
    )

    assert (
        response.metadata["primary_error_type"]
        == "LLMResponseError"
    )

    assert (
        response.metadata["done_reason"]
        == "stop"
    )


def test_truncated_response_does_not_trigger_fallback() -> None:
    primary = TruncatedProvider()

    fallback = TrackingProvider(
        provider_name="fallback",
        model="fallback-model",
    )

    provider = FallbackLLMProvider(
        primary=primary,
        fallback=fallback,
    )

    response = provider.generate(
        messages=build_messages(),
        config=GenerationConfig(
            max_tokens=128,
        ),
    )

    assert response.provider == "truncated"
    assert response.model == "truncated-model"
    assert response.text == "Incomplete response."

    assert response.metadata is not None

    assert (
        response.metadata["done_reason"]
        == "length"
    )

    assert (
        response.metadata["fallback_used"]
        is False
    )

    assert (
        response.metadata[
            "requested_max_tokens"
        ]
        == 128
    )

    assert fallback.call_count == 0


def test_provider_is_available_when_primary_is_available() -> None:
    primary = SuccessfulProvider(
        provider_name="primary",
        model="primary-model",
    )

    fallback = SuccessfulProvider(
        provider_name="fallback",
        model="fallback-model",
    )

    provider = FallbackLLMProvider(
        primary=primary,
        fallback=fallback,
    )

    assert provider.is_available() is True


def test_provider_name_describes_fallback_chain() -> None:
    primary = SuccessfulProvider(
        provider_name="ollama",
        model="qwen3:4b",
    )

    fallback = SuccessfulProvider(
        provider_name="ollama",
        model="qwen3:1.7b",
    )

    provider = FallbackLLMProvider(
        primary=primary,
        fallback=fallback,
    )

    assert (
        provider.name
        == "ollama+fallback:ollama"
    )