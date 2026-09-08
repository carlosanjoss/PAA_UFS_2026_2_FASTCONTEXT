from src.retrieval.base import Retriever
from src.retrieval.models import (
    RetrievalMetrics,
    RetrievalResult,
    RetrievedChunk,
)
from src.retrieval.registry import (
    RetrieverRegistry,
)
from src.services.factory import (
    create_fastcontext_service,
)


class LinearRetriever(
    Retriever
):
    """Fake linear retriever used for factory tests."""

    @property
    def name(self) -> str:
        return "linear"

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> RetrievalResult:
        chunk = RetrievedChunk(
            chunk_id="linear_001",
            content="Linear result.",
            source_path="linear.md",
            section_title="Linear",
            score=0.9,
            rank=1,
        )

        return RetrievalResult(
            query=query,
            algorithm=self.name,
            top_k=top_k,
            chunks=(chunk,),
            metrics=RetrievalMetrics(
                retrieval_time_ns=100,
                comparisons=10,
                chunks_scored=10,
                candidates_found=10,
            ),
        )


class SemanticRetriever(
    Retriever
):
    """Fake semantic retriever used for factory tests."""

    @property
    def name(self) -> str:
        return "semantic"

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> RetrievalResult:
        chunk = RetrievedChunk(
            chunk_id="semantic_001",
            content="Semantic result.",
            source_path="semantic.md",
            section_title="Semantic",
            score=0.99,
            rank=1,
        )

        return RetrievalResult(
            query=query,
            algorithm=self.name,
            top_k=top_k,
            chunks=(chunk,),
            metrics=RetrievalMetrics(
                retrieval_time_ns=200,
                chunks_scored=1,
                candidates_found=1,
            ),
        )


def build_registry() -> RetrieverRegistry:
    """Create a registry with multiple algorithms."""

    registry = RetrieverRegistry()

    registry.register(
        "linear",
        LinearRetriever,
    )

    registry.register(
        "semantic",
        SemanticRetriever,
    )

    return registry


def test_factory_creates_linear_service() -> None:
    registry = build_registry()

    service = (
        create_fastcontext_service(
            algorithm="linear",
            registry=registry,
        )
    )

    assert (
        service.retriever.name
        == "linear"
    )

    result = service.retrieve(
        query="Test query",
        top_k=5,
    )

    assert (
        result.algorithm
        == "linear"
    )

    assert (
        result.chunks[0].chunk_id
        == "linear_001"
    )


def test_factory_creates_semantic_service() -> None:
    registry = build_registry()

    service = (
        create_fastcontext_service(
            algorithm="semantic",
            registry=registry,
        )
    )

    assert (
        service.retriever.name
        == "semantic"
    )

    result = service.retrieve(
        query="Test query",
        top_k=5,
    )

    assert (
        result.algorithm
        == "semantic"
    )

    assert (
        result.chunks[0].chunk_id
        == "semantic_001"
    )


def test_algorithm_can_change_without_service_changes() -> None:
    registry = build_registry()

    linear_service = (
        create_fastcontext_service(
            algorithm="linear",
            registry=registry,
        )
    )

    semantic_service = (
        create_fastcontext_service(
            algorithm="semantic",
            registry=registry,
        )
    )

    linear_result = (
        linear_service.retrieve(
            query="Same query",
        )
    )

    semantic_result = (
        semantic_service.retrieve(
            query="Same query",
        )
    )

    assert (
        linear_result.algorithm
        != semantic_result.algorithm
    )

    assert (
        linear_result.query
        == semantic_result.query
    )


def test_factory_preserves_retrieval_interface() -> None:
    registry = build_registry()

    for algorithm in (
        "linear",
        "semantic",
    ):
        service = (
            create_fastcontext_service(
                algorithm=algorithm,
                registry=registry,
            )
        )

        result = service.retrieve(
            query="Common query",
            top_k=3,
        )

        assert (
            result.query
            == "Common query"
        )

        assert (
            result.top_k
            == 3
        )

        assert (
            len(result.chunks)
            <= 3
        )