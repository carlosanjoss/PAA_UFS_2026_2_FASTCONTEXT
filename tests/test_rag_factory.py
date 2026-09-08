import pytest

from src.rag.factory import create_llm_provider
from src.rag.providers.fallback import (
    FallbackLLMProvider,
)
from src.rag.providers.ollama import OllamaProvider
from src.rag.settings import (
    NvidiaSettings,
    OllamaSettings,
    RAGSettings,
)


def build_settings(
    *,
    provider: str = "ollama",
    fallback_model: str | None = "qwen3:1.7b",
) -> RAGSettings:
    return RAGSettings(
        default_provider=provider,
        ollama=OllamaSettings(
            base_url="http://localhost:11434",
            model="qwen3:4b",
            fallback_model=fallback_model,
        ),
        nvidia=NvidiaSettings(
            enabled=False,
            model="",
            api_key=None,
        ),
    )


def test_factory_creates_fallback_provider() -> None:
    provider = create_llm_provider(
        build_settings()
    )

    assert isinstance(
        provider,
        FallbackLLMProvider,
    )


def test_factory_creates_plain_ollama_without_fallback() -> None:
    provider = create_llm_provider(
        build_settings(
            fallback_model=None
        )
    )

    assert isinstance(
        provider,
        OllamaProvider,
    )


def test_factory_rejects_unknown_provider() -> None:
    settings = build_settings(
        provider="unknown",
    )

    with pytest.raises(
        ValueError,
        match="Unsupported LLM provider",
    ):
        create_llm_provider(settings)