from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal


class LLMProviderError(RuntimeError):
    """Base exception raised by LLM providers."""


class LLMConnectionError(LLMProviderError):
    """Raised when a provider cannot be reached."""


class LLMResponseError(LLMProviderError):
    """Raised when a provider returns an invalid response."""


class LLMTruncatedResponseError(LLMResponseError):
    """Raised when generation repeatedly reaches the token limit."""


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Configuration used during text generation."""

    temperature: float = 0.1
    max_tokens: int = 512
    think: bool = False

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError(
                "Temperature cannot be negative."
            )

        if self.max_tokens <= 0:
            raise ValueError(
                "max_tokens must be greater than zero."
            )


@dataclass(frozen=True, slots=True)
class LLMMessage:
    """Message sent to a language model."""

    role: Literal[
        "system",
        "user",
        "assistant",
    ]
    content: str

    def __post_init__(self) -> None:
        if not self.content.strip():
            raise ValueError(
                "Message content cannot be empty."
            )


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """Normalized response returned by an LLM provider."""

    text: str
    model: str
    provider: str
    metadata: dict[str, Any] | None = None


class LLMProvider(ABC):
    """Abstract interface implemented by language model providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider name."""

    @abstractmethod
    def generate(
        self,
        messages: Sequence[LLMMessage],
        config: GenerationConfig | None = None,
    ) -> LLMResponse:
        """Generate text from a sequence of messages."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return whether the provider is currently available."""