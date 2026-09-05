from __future__ import annotations

from abc import ABC, abstractmethod

from src.retrieval.models import (
    RetrievalResult,
)


class Retriever(ABC):
    """Common interface implemented by all retrieval strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique name of the retrieval strategy."""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> RetrievalResult:
        """Retrieve the highest-ranked chunks for a query."""