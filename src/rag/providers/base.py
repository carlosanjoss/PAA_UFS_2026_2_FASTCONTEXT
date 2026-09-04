from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


class LLMProviderError(RuntimeError):
    """Base exception raised by LLM providers."""


class LLMConnectionError(LLMProviderError):
    """Raised when a provider cannot be reached."""


class LLMResponseError(LLMProviderError):
    """Raised when a provider returns an invalid response."""


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Configuration used during text generation."""

    temperature: float = 0.1
    max_tokens: int = 512


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """Normalized response returned by an LLM provider."""

    text: str
    model: str
    provider: str
    metadata: dict[str, Any] | None = None


class LLMProvider(ABC):
    """Abstract interface implemented by all language model providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider name."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> LLMResponse:
        """Generate text from a prompt."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return whether the provider is currently available."""