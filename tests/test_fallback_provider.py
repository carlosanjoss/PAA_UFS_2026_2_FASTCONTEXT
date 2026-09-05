from src.rag.providers.base import (
    GenerationConfig,
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
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> LLMResponse:
        return LLMResponse(
            text="Successful response.",
            model=self._model,
            provider=self.name,
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
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> LLMResponse:
        raise LLMResponseError(
            "Simulated provider failure."
        )


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
        "Test prompt"
    )

    assert response.model == "primary-model"
    assert response.metadata is not None
    assert (
        response.metadata["fallback_used"]
        is False
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
        "Test prompt"
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