from __future__ import annotations

from collections.abc import Sequence

from src.rag.providers.base import (
    GenerationConfig,
    LLMMessage,
    LLMProvider,
    LLMProviderError,
    LLMResponse,
)


class FallbackLLMProvider(LLMProvider):
    """Use a secondary provider when the primary provider fails."""

    def __init__(
        self,
        primary: LLMProvider,
        fallback: LLMProvider,
    ) -> None:
        self._primary = primary
        self._fallback = fallback

    @property
    def name(self) -> str:
        return (
            f"{self._primary.name}"
            f"+fallback:{self._fallback.name}"
        )

    @property
    def primary(self) -> LLMProvider:
        return self._primary

    @property
    def fallback(self) -> LLMProvider:
        return self._fallback

    def is_available(self) -> bool:
        return (
            self._primary.is_available()
            or self._fallback.is_available()
        )

    def generate(
        self,
        messages: Sequence[LLMMessage],
        config: GenerationConfig | None = None,
    ) -> LLMResponse:
        """Generate with primary or fallback provider."""

        try:
            response = self._primary.generate(
                messages=messages,
                config=config,
            )

            return self._with_fallback_metadata(
                response=response,
                fallback_used=False,
            )

        except LLMProviderError as primary_error:
            response = self._fallback.generate(
                messages=messages,
                config=config,
            )

            return self._with_fallback_metadata(
                response=response,
                fallback_used=True,
                primary_error=primary_error,
            )

    @staticmethod
    def _with_fallback_metadata(
        response: LLMResponse,
        fallback_used: bool,
        primary_error: Exception | None = None,
    ) -> LLMResponse:
        """Attach fallback information to the response."""

        metadata = dict(
            response.metadata or {}
        )

        metadata["fallback_used"] = (
            fallback_used
        )

        if primary_error is not None:
            metadata["primary_error_type"] = (
                type(primary_error).__name__
            )

        return LLMResponse(
            text=response.text,
            model=response.model,
            provider=response.provider,
            metadata=metadata,
        )