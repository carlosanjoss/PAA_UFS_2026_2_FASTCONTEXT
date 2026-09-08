import pytest

from src.retrieval.base import Retriever
from src.retrieval.models import (
    RetrievalMetrics,
    RetrievalResult,
    RetrievedChunk,
)
from src.retrieval.registry import (
    InvalidRetrieverFactoryError,
    RetrieverAlreadyRegisteredError,
    RetrieverRegistry,
    UnknownRetrieverError,
)


class FakeLinearRetriever(
    Retriever
):
    """Fake linear retriever used for registry tests."""

    @property
    def name(self) -> str:
        return "linear"

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> RetrievalResult:
        chunk = RetrievedChunk(
            chunk_id="chunk_001",
            content="Example content.",
            source_path="example.md",
            section_title="Example",
            score=1.0,
            rank=1,
        )

        return RetrievalResult(
            query=query,
            algorithm=self.name,
            top_k=top_k,
            chunks=(chunk,),
            metrics=RetrievalMetrics(
                retrieval_time_ns=100,
                comparisons=1,
                chunks_scored=1,
                candidates_found=1,
            ),
        )


class FakeSemanticRetriever(
    Retriever
):
    """Fake semantic retriever used for registry tests."""

    @property
    def name(self) -> str:
        return "semantic"

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> RetrievalResult:
        return RetrievalResult(
            query=query,
            algorithm=self.name,
            top_k=top_k,
            chunks=(),
            metrics=RetrievalMetrics(
                retrieval_time_ns=200,
            ),
        )


def test_registry_starts_empty() -> None:
    registry = RetrieverRegistry()

    assert len(registry) == 0

    assert (
        registry.available_names()
        == ()
    )


def test_registry_registers_factory() -> None:
    registry = RetrieverRegistry()

    registry.register(
        "linear",
        FakeLinearRetriever,
    )

    assert len(registry) == 1

    assert registry.contains(
        "linear"
    )

    assert (
        registry.available_names()
        == ("linear",)
    )


def test_registry_normalizes_name() -> None:
    registry = RetrieverRegistry()

    registry.register(
        "  LINEAR  ",
        FakeLinearRetriever,
    )

    assert registry.contains(
        "linear"
    )

    assert registry.contains(
        "LINEAR"
    )


def test_registry_creates_retriever() -> None:
    registry = RetrieverRegistry()

    registry.register(
        "linear",
        FakeLinearRetriever,
    )

    retriever = registry.create(
        "linear"
    )

    assert isinstance(
        retriever,
        FakeLinearRetriever,
    )

    assert (
        retriever.name
        == "linear"
    )


def test_registry_supports_multiple_retrievers() -> None:
    registry = RetrieverRegistry()

    registry.register(
        "semantic",
        FakeSemanticRetriever,
    )

    registry.register(
        "linear",
        FakeLinearRetriever,
    )

    assert (
        registry.available_names()
        == (
            "linear",
            "semantic",
        )
    )


def test_registry_rejects_duplicate_registration() -> None:
    registry = RetrieverRegistry()

    registry.register(
        "linear",
        FakeLinearRetriever,
    )

    with pytest.raises(
        RetrieverAlreadyRegisteredError,
        match="already registered",
    ):
        registry.register(
            "linear",
            FakeLinearRetriever,
        )


def test_registry_can_replace_registration() -> None:
    registry = RetrieverRegistry()

    registry.register(
        "linear",
        FakeLinearRetriever,
    )

    registry.register(
        "linear",
        FakeLinearRetriever,
        replace=True,
    )

    retriever = registry.create(
        "linear"
    )

    assert (
        retriever.name
        == "linear"
    )


def test_registry_rejects_unknown_retriever() -> None:
    registry = RetrieverRegistry()

    with pytest.raises(
        UnknownRetrieverError,
        match="is not registered",
    ):
        registry.create(
            "unknown"
        )


def test_registry_unregisters_retriever() -> None:
    registry = RetrieverRegistry()

    registry.register(
        "linear",
        FakeLinearRetriever,
    )

    registry.unregister(
        "linear"
    )

    assert not registry.contains(
        "linear"
    )

    assert len(registry) == 0


def test_unregister_unknown_retriever_fails() -> None:
    registry = RetrieverRegistry()

    with pytest.raises(
        UnknownRetrieverError,
        match="is not registered",
    ):
        registry.unregister(
            "linear"
        )


def test_registry_rejects_empty_name() -> None:
    registry = RetrieverRegistry()

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        registry.register(
            "   ",
            FakeLinearRetriever,
        )


def test_registry_detects_factory_name_mismatch() -> None:
    registry = RetrieverRegistry()

    registry.register(
        "linear",
        FakeSemanticRetriever,
    )

    with pytest.raises(
        InvalidRetrieverFactoryError,
        match="name mismatch",
    ):
        registry.create(
            "linear"
        )


def test_registry_detects_invalid_factory_result() -> None:
    registry = RetrieverRegistry()

    def invalid_factory() -> object:
        return object()

    registry.register(
        "linear",
        invalid_factory,  # type: ignore[arg-type]
    )

    with pytest.raises(
        InvalidRetrieverFactoryError,
        match="did not return a Retriever",
    ):
        registry.create(
            "linear"
        )