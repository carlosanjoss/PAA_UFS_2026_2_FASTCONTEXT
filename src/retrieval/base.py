from __future__ import annotations

from abc import ABC, abstractmethod

from src.retrieval.models import RetrievalResult


class Retriever(ABC):
    """Base contract for every FastContext retriever."""

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

    def search(
        self,
        query: str,
        k: int = 5,
    ) -> RetrievalResult:
        """Provide backward compatibility with the original search API."""

        return self.retrieve(
            query=query,
            top_k=k,
        )

    @staticmethod
    def _validate_request(
        query: str,
        top_k: int,
    ) -> str:
        """Validate and normalize common retrieval parameters."""

        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError("query cannot be empty.")

        if top_k < 0:
            raise ValueError(
                "top_k must be greater than or equal to zero."
            )

        return normalized_query