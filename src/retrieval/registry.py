from __future__ import annotations

from collections.abc import Callable

from src.retrieval.base import Retriever

RetrieverFactory = Callable[[], Retriever]


class RetrieverRegistryError(RuntimeError):
    """Base exception raised by the retriever registry."""


class RetrieverAlreadyRegisteredError(
    RetrieverRegistryError
):
    """Raised when a retriever name is registered twice."""


class UnknownRetrieverError(
    RetrieverRegistryError
):
    """Raised when a requested retriever is not registered."""


class InvalidRetrieverFactoryError(
    RetrieverRegistryError
):
    """Raised when a factory returns an invalid retriever."""


class RetrieverRegistry:
    """Registry responsible for lazy retriever construction."""

    def __init__(self) -> None:
        self._factories: dict[
            str,
            RetrieverFactory,
        ] = {}

    def register(
        self,
        name: str,
        factory: RetrieverFactory,
        *,
        replace: bool = False,
    ) -> None:
        """Register a retriever factory."""

        normalized_name = self._normalize_name(
            name
        )

        if (
            normalized_name
            in self._factories
            and not replace
        ):
            raise (
                RetrieverAlreadyRegisteredError(
                    "Retriever "
                    f"'{normalized_name}' "
                    "is already registered."
                )
            )

        self._factories[
            normalized_name
        ] = factory

    def unregister(
        self,
        name: str,
    ) -> None:
        """Remove a retriever factory."""

        normalized_name = self._normalize_name(
            name
        )

        if (
            normalized_name
            not in self._factories
        ):
            raise UnknownRetrieverError(
                "Retriever "
                f"'{normalized_name}' "
                "is not registered."
            )

        del self._factories[
            normalized_name
        ]

    def create(
        self,
        name: str,
    ) -> Retriever:
        """Create a retriever from a registered factory."""

        normalized_name = self._normalize_name(
            name
        )

        factory = self._factories.get(
            normalized_name
        )

        if factory is None:
            raise UnknownRetrieverError(
                "Retriever "
                f"'{normalized_name}' "
                "is not registered."
            )

        retriever = factory()

        if not isinstance(
            retriever,
            Retriever,
        ):
            raise InvalidRetrieverFactoryError(
                "Factory registered as "
                f"'{normalized_name}' "
                "did not return a Retriever."
            )

        retriever_name = (
            retriever.name
            .strip()
            .lower()
        )

        if (
            retriever_name
            != normalized_name
        ):
            raise InvalidRetrieverFactoryError(
                "Retriever factory name mismatch: "
                f"registered as '{normalized_name}' "
                "but produced retriever named "
                f"'{retriever.name}'."
            )

        return retriever

    def contains(
        self,
        name: str,
    ) -> bool:
        """Return whether a retriever is registered."""

        try:
            normalized_name = (
                self._normalize_name(
                    name
                )
            )
        except ValueError:
            return False

        return (
            normalized_name
            in self._factories
        )

    def available_names(
        self,
    ) -> tuple[str, ...]:
        """Return registered retriever names."""

        return tuple(
            sorted(
                self._factories
            )
        )

    def __len__(self) -> int:
        """Return the number of registered retrievers."""

        return len(
            self._factories
        )

    @staticmethod
    def _normalize_name(
        name: str,
    ) -> str:
        """Normalize a retriever identifier."""

        normalized_name = (
            name.strip().lower()
        )

        if not normalized_name:
            raise ValueError(
                "Retriever name cannot be empty."
            )

        return normalized_name