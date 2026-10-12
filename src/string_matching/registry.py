
from __future__ import annotations

from collections.abc import Callable

from src.string_matching.base import StringMatcher
from src.string_matching.naive import NaiveMatcher

MatcherFactory = Callable[[], StringMatcher]


class StringMatcherRegistry:
    """Registry for exact string matching algorithms."""

    def __init__(self) -> None:
        self._factories: dict[str, MatcherFactory] = {}

    def register(
        self,
        name: str,
        factory: MatcherFactory,
    ) -> None:
        """Register a matcher factory."""

        normalized_name = name.strip().lower()

        if not normalized_name:
            raise ValueError("Algorithm name cannot be empty.")

        if normalized_name in self._factories:
            raise ValueError(
                f"Algorithm '{normalized_name}' is already registered."
            )

        if not callable(factory):
            raise TypeError("Matcher factory must be callable.")

        self._factories[normalized_name] = factory

    def create(self, name: str) -> StringMatcher:
        """Create a matcher by its registered name."""

        normalized_name = name.strip().lower()

        factory = self._factories.get(normalized_name)

        if factory is None:
            raise ValueError(
                f"Unknown string matching algorithm: '{name}'."
            )

        matcher = factory()

        if not isinstance(matcher, StringMatcher):
            raise TypeError(
                "Registered factory must return a StringMatcher."
            )

        if matcher.name != normalized_name:
            raise ValueError(
                "Registered name does not match the matcher name."
            )

        return matcher

    def available_names(self) -> tuple[str, ...]:
        """Return registered algorithm names."""

        return tuple(sorted(self._factories))

    def contains(self, name: str) -> bool:
        """Check whether an algorithm is registered."""

        return name.strip().lower() in self._factories


def create_default_registry() -> StringMatcherRegistry:
    """Create a registry with available AV2 algorithms."""

    registry = StringMatcherRegistry()

    registry.register(
        name="naive",
        factory=NaiveMatcher,
    )

    return registry
