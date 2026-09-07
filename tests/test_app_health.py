from collections.abc import Sequence

from src.app.health import (
    check_application_health,
)
from src.app.models import (
    ApplicationContainer,
)
from src.rag.pipeline import (
    RAGPipeline,
)
from src.rag.providers.base import (
    GenerationConfig,
    LLMMessage,
    LLMProvider,
    LLMResponse,
)
from src.rag.settings import (
    NvidiaSettings,
    OllamaSettings,
    RAGSettings,
)
from src.retrieval.base import Retriever
from src.retrieval.models import (
    RetrievalMetrics,
    RetrievalResult,
)
from src.retrieval.registry import (
    RetrieverRegistry,
)


class FakeProvider(
    LLMProvider
):
    """Configurable fake provider for health tests."""

    def __init__(
        self,
        available: bool,
    ) -> None:
        self._available = available

    @property
    def name(self) -> str:
        return "fake"

    def is_available(self) -> bool:
        return self._available

    def generate(
        self,
        messages: Sequence[
            LLMMessage
        ],
        config: (
            GenerationConfig | None
        ) = None,
    ) -> LLMResponse:
        return LLMResponse(
            text="Fake",
            model="fake-model",
            provider=self.name,
        )


class FakeRetriever(
    Retriever
):
    """Fake retriever used for health tests."""

    @property
    def name(self) -> str:
        return "linear"

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
                retrieval_time_ns=1,
            ),
        )


def build_settings() -> RAGSettings:
    """Create test RAG settings."""

    return RAGSettings(
        default_provider="ollama",
        ollama=OllamaSettings(
            base_url=(
                "http://localhost:11434"
            ),
            model="qwen2.5:3b",
            fallback_model=None,
        ),
        nvidia=NvidiaSettings(
            enabled=False,
            model="",
            api_key=None,
        ),
    )


def build_application(
    *,
    provider_available: bool,
    with_retriever: bool,
) -> ApplicationContainer:
    """Create a health-test application container."""

    provider = FakeProvider(
        available=provider_available
    )

    registry = RetrieverRegistry()

    if with_retriever:
        registry.register(
            "linear",
            FakeRetriever,
        )

    pipeline = RAGPipeline(
        provider=provider
    )

    return ApplicationContainer(
        settings=build_settings(),
        registry=registry,
        provider=provider,
        rag_pipeline=pipeline,
    )


def test_application_is_healthy_when_all_components_work() -> None:
    application = build_application(
        provider_available=True,
        with_retriever=True,
    )

    report = check_application_health(
        application
    )

    assert (
        report.status
        == "healthy"
    )

    assert report.is_healthy is True


def test_application_is_degraded_without_retrievers() -> None:
    application = build_application(
        provider_available=True,
        with_retriever=False,
    )

    report = check_application_health(
        application
    )

    assert (
        report.status
        == "degraded"
    )


def test_application_is_degraded_without_llm() -> None:
    application = build_application(
        provider_available=False,
        with_retriever=True,
    )

    report = check_application_health(
        application
    )

    assert (
        report.status
        == "degraded"
    )


def test_application_is_degraded_when_everything_is_missing() -> None:
    application = build_application(
        provider_available=False,
        with_retriever=False,
    )

    report = check_application_health(
        application
    )

    assert (
        report.status
        == "degraded"
    )


def test_health_report_contains_all_components() -> None:
    application = build_application(
        provider_available=True,
        with_retriever=True,
    )

    report = check_application_health(
        application
    )

    component_names = {
        component.name
        for component in report.components
    }

    assert component_names == {
        "Retrieval",
        "LLM",
        "RAG",
    }