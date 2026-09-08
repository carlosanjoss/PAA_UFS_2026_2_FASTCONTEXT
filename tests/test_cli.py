from collections.abc import Sequence

from src.app.models import (
    ApplicationContainer,
)
from src.cli import (
    build_parser,
    print_algorithms,
    print_health,
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
    """Fake provider used for CLI tests."""

    def __init__(
        self,
        available: bool = True,
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
    """Fake retriever used for CLI tests."""

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
    """Create deterministic CLI settings."""

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
    with_retriever: bool,
    provider_available: bool = True,
) -> ApplicationContainer:
    """Create an application container for CLI tests."""

    registry = RetrieverRegistry()

    if with_retriever:
        registry.register(
            "linear",
            FakeRetriever,
        )

    provider = FakeProvider(
        available=provider_available
    )

    return ApplicationContainer(
        settings=build_settings(),
        registry=registry,
        provider=provider,
        rag_pipeline=RAGPipeline(
            provider=provider
        ),
    )


def test_parser_accepts_health_command() -> None:
    parser = build_parser()

    args = parser.parse_args(
        ["health"]
    )

    assert (
        args.command
        == "health"
    )


def test_parser_accepts_algorithms_command() -> None:
    parser = build_parser()

    args = parser.parse_args(
        ["algorithms"]
    )

    assert (
        args.command
        == "algorithms"
    )


def test_parser_accepts_retrieve_arguments() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "retrieve",
            "--algorithm",
            "linear",
            "--query",
            "Test query",
            "--top-k",
            "3",
        ]
    )

    assert (
        args.command
        == "retrieve"
    )

    assert (
        args.algorithm
        == "linear"
    )

    assert (
        args.query
        == "Test query"
    )

    assert args.top_k == 3


def test_parser_accepts_ask_arguments() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "ask",
            "--algorithm",
            "linear",
            "--query",
            "Test query",
            "--top-k",
            "5",
            "--max-tokens",
            "128",
            "--temperature",
            "0.0",
        ]
    )

    assert (
        args.command
        == "ask"
    )

    assert args.top_k == 5

    assert (
        args.max_tokens
        == 128
    )

    assert (
        args.temperature
        == 0.0
    )


def test_print_algorithms_when_empty(
    capsys: object,
) -> None:
    application = build_application(
        with_retriever=False
    )

    exit_code = print_algorithms(
        application
    )

    captured = capsys.readouterr()  # type: ignore[attr-defined]

    assert exit_code == 0

    assert (
        "No retrieval algorithms"
        in captured.out
    )


def test_print_algorithms_lists_registered_retriever(
    capsys: object,
) -> None:
    application = build_application(
        with_retriever=True
    )

    exit_code = print_algorithms(
        application
    )

    captured = capsys.readouterr()  # type: ignore[attr-defined]

    assert exit_code == 0

    assert (
        "linear"
        in captured.out
    )


def test_print_health_returns_zero_when_healthy(
    capsys: object,
) -> None:
    application = build_application(
        with_retriever=True,
        provider_available=True,
    )

    exit_code = print_health(
        application
    )

    captured = capsys.readouterr()  # type: ignore[attr-defined]

    assert exit_code == 0

    assert (
        "FastContext status: healthy"
        in captured.out
    )


def test_print_health_returns_nonzero_when_degraded(
    capsys: object,
) -> None:
    application = build_application(
        with_retriever=False,
        provider_available=True,
    )

    exit_code = print_health(
        application
    )

    capsys.readouterr()  # type: ignore[attr-defined]

    assert exit_code == 1