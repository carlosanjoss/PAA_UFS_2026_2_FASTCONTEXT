from __future__ import annotations

from abc import ABC, abstractmethod

from src.retrieval.models import RetrievalResult


class Retriever(ABC):
    """Base contract for all FastContext retrievers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the canonical retriever identifier."""

        raise NotImplementedError

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> RetrievalResult:
        """Retrieve the most relevant chunks for a query."""

        raise NotImplementedError