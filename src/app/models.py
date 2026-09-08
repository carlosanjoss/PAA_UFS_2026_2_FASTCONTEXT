from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.rag.pipeline import RAGPipeline
from src.rag.providers.base import LLMProvider
from src.rag.settings import RAGSettings
from src.retrieval.registry import RetrieverRegistry

HealthState = Literal[
    "healthy",
    "degraded",
    "unavailable",
]


@dataclass(frozen=True, slots=True)
class ComponentHealth:
    """Health status of an application component."""

    name: str
    status: HealthState
    message: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError(
                "Component name cannot be empty."
            )

        if not self.message.strip():
            raise ValueError(
                "Health message cannot be empty."
            )


@dataclass(frozen=True, slots=True)
class ApplicationHealthReport:
    """Aggregated health report for FastContext."""

    status: HealthState
    components: tuple[
        ComponentHealth,
        ...,
    ]

    @property
    def is_healthy(self) -> bool:
        """Return whether all application components are healthy."""

        return self.status == "healthy"


@dataclass(frozen=True, slots=True)
class ApplicationContainer:
    """Runtime dependencies used by the FastContext application."""

    settings: RAGSettings
    registry: RetrieverRegistry
    provider: LLMProvider
    rag_pipeline: RAGPipeline
    corpus_size: int | None = None

    def __post_init__(self) -> None:
        if (
            self.corpus_size is not None
            and self.corpus_size < 0
        ):
            raise ValueError(
                "corpus_size cannot be negative."
            )