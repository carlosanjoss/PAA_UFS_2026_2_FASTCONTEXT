from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

from src.utils.config import PROJECT_ROOT, load_retrieval_config


@dataclass(frozen=True, slots=True)
class OllamaSettings:
    """Configuration for the local Ollama provider."""

    base_url: str
    model: str
    fallback_model: str | None = None


@dataclass(frozen=True, slots=True)
class NvidiaSettings:
    """Configuration for the NVIDIA provider."""

    enabled: bool
    model: str
    api_key: str | None = None


@dataclass(frozen=True, slots=True)
class RAGSettings:
    """Application settings for retrieval-augmented generation."""

    default_provider: str
    ollama: OllamaSettings
    nvidia: NvidiaSettings


def _get_mapping(
    data: dict[str, Any],
    key: str,
) -> dict[str, Any]:
    """Return a nested configuration mapping."""

    value = data.get(key)

    if not isinstance(value, dict):
        raise TypeError(
            f"Expected '{key}' to be a mapping."
        )

    return value


def _environment_value(
    name: str,
    default: str,
) -> str:
    """Return a non-empty environment value or its default."""

    value = os.getenv(name)

    if value is None or not value.strip():
        return default

    return value.strip()


def load_rag_settings(
    *,
    load_environment: bool = True,
) -> RAGSettings:
    """Load RAG settings from YAML and environment variables.

    Environment variables take precedence over YAML values.
    """

    if load_environment:
        load_dotenv(
            dotenv_path=PROJECT_ROOT / ".env",
            override=False,
        )

    retrieval_config = load_retrieval_config()

    rag_config = _get_mapping(
        retrieval_config,
        "rag",
    )

    ollama_config = _get_mapping(
        rag_config,
        "ollama",
    )

    nvidia_config = _get_mapping(
        rag_config,
        "nvidia",
    )

    default_provider = _environment_value(
        "RAG_DEFAULT_PROVIDER",
        str(rag_config.get("default_provider", "ollama")),
    ).lower()

    base_url = _environment_value(
        "OLLAMA_BASE_URL",
        str(
            ollama_config.get(
                "base_url",
                "http://localhost:11434",
            )
        ),
    )

    model = _environment_value(
        "OLLAMA_MODEL",
        str(
            ollama_config.get(
                "model",
                "qwen3:4b",
            )
        ),
    )

    fallback_model_value = _environment_value(
        "OLLAMA_FALLBACK_MODEL",
        str(
            ollama_config.get(
                "fallback_model",
                "",
            )
        ),
    )

    fallback_model = (
        fallback_model_value
        if fallback_model_value
        else None
    )

    nvidia_model = str(
        nvidia_config.get("model", "")
    ).strip()

    nvidia_api_key = os.getenv("NVIDIA_API_KEY")

    if nvidia_api_key is not None:
        nvidia_api_key = nvidia_api_key.strip() or None

    return RAGSettings(
        default_provider=default_provider,
        ollama=OllamaSettings(
            base_url=base_url,
            model=model,
            fallback_model=fallback_model,
        ),
        nvidia=NvidiaSettings(
            enabled=bool(
                nvidia_config.get("enabled", False)
            ),
            model=nvidia_model,
            api_key=nvidia_api_key,
        ),
    )