from collections.abc import Sequence

from src.app.bootstrap import (
    build_retriever_registry,
    create_application,
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


class FakeProvider(
    LLMProvider
):
    """Fake LLM provider used for bootstrap tests."""

    @property
    def name(self) -> str:
        return "fake"

    def is_available(self) -> bool:
        return True

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
            text="Fake response.",
            model="fake-model",
            provider=self.name,
            metadata={
                "done_reason": "stop",
            },
        )


class FakeRetriever(
    Retriever
):
    """Fake retriever used for bootstrap tests."""

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
                retrieval_time_ns=100,
            ),
        )


def build_settings() -> RAGSettings:
    """Create deterministic application settings."""

    return RAGSettings(
        default_provider="ollama",
        ollama=OllamaSettings(
            base_url=(
                "http://localhost:11434"
            ),
            model="qwen2.5:3b",
            fallback_model=(
                "qwen3:1.7b"
            ),
        ),
        nvidia=NvidiaSettings(
            enabled=False,
            model="",
            api_key=None,
        ),
    )


def test_build_registry_without_registrations() -> None:
    registry = (
        build_retriever_registry()
    )

    assert (
        registry.available_names()
        == ()
    )


def test_build_registry_with_registration() -> None:
    registry = (
        build_retriever_registry(
            {
                "linear": FakeRetriever,
            }
        )
    )

    assert (
        registry.available_names()
        == ("linear",)
    )


def test_create_application_uses_injected_dependencies() -> None:
    settings = build_settings()

    registry = (
        build_retriever_registry(
            {
                "linear": FakeRetriever,
            }
        )
    )

    provider = FakeProvider()

    application = (
        create_application(
            settings=settings,
            registry=registry,
            provider=provider,
        )
    )

    assert (
        application.settings
        is settings
    )

    assert (
        application.registry
        is registry
    )

    assert (
        application.provider
        is provider
    )

    assert (
        application.rag_pipeline
        .provider
        is provider
    )


def test_create_application_preserves_registered_algorithms() -> None:
    registry = (
        build_retriever_registry(
            {
                "linear": FakeRetriever,
            }
        )
    )

    application = (
        create_application(
            settings=build_settings(),
            registry=registry,
            provider=FakeProvider(),
        )
    )

    assert (
        application.registry
        .available_names()
        == ("linear",)
    )