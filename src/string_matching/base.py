
from __future__ import annotations

from abc import ABC, abstractmethod

from src.string_matching.models import StringMatchResult


class StringMatcher(ABC):
    """Abstract contract for string matching algorithms."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the canonical algorithm name."""
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        text: str,
        pattern: str,
    ) -> StringMatchResult:
        """Find all pattern occurrences in the text."""
        raise NotImplementedError

    
    @staticmethod
    def validate_input(
        text: str,
        pattern: str,
    ) -> None:
        """Validate common search parameters."""

        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        if not isinstance(pattern, str):
            raise TypeError("pattern must be a string.")

        if not pattern:
            raise ValueError("pattern cannot be empty.")

