from __future__ import annotations

from src.rag.providers.base import LLMProvider
from src.rag.providers.fallback import FallbackLLMProvider
from src.rag.providers.ollama import OllamaProvider
from src.rag.settings import RAGSettings, load_rag_settings


def create_llm_provider(
    settings: RAGSettings | None = None,
) -> LLMProvider:
    """Create the configured LLM provider."""

    resolved_settings = (
        settings
        if settings is not None
        else load_rag_settings()
    )

    provider_name = (
        resolved_settings.default_provider
        .strip()
        .lower()
    )

    if provider_name == "ollama":
        return _create_ollama_provider(
            resolved_settings
        )

    if provider_name == "nvidia":
        raise NotImplementedError(
            "NVIDIA provider is not implemented yet."
        )

    raise ValueError(
        f"Unsupported LLM provider: {provider_name}"
    )


def _create_ollama_provider(
    settings: RAGSettings,
) -> LLMProvider:
    """Create Ollama provider with optional model fallback."""

    ollama_settings = settings.ollama

    primary = OllamaProvider(
        model=ollama_settings.model,
        base_url=ollama_settings.base_url,
    )

    fallback_model = ollama_settings.fallback_model

    if (
        fallback_model is None
        or fallback_model == ollama_settings.model
    ):
        return primary

    fallback = OllamaProvider(
        model=fallback_model,
        base_url=ollama_settings.base_url,
    )

    return FallbackLLMProvider(
        primary=primary,
        fallback=fallback,
    )