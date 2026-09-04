from src.rag.pipeline import RAGPipeline
from src.rag.prompt import ContextChunk
from src.rag.providers.base import (
    GenerationConfig,
    LLMProvider,
    LLMResponse,
)


class FakeLLMProvider(LLMProvider):
    """Deterministic provider used only for tests."""

    @property
    def name(self) -> str:
        return "fake"

    def is_available(self) -> bool:
        return True

    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> LLMResponse:
        return LLMResponse(
            text="FastAPI dependencies are described in [chunk_001].",
            model="fake-model",
            provider=self.name,
        )


def test_rag_pipeline_returns_expected_result() -> None:
    provider = FakeLLMProvider()
    pipeline = RAGPipeline(provider)

    chunk = ContextChunk(
        chunk_id="chunk_001",
        content="FastAPI supports dependency injection.",
        source_path="tutorial/dependencies/index.md",
        section_title="Dependencies",
        score=0.95,
    )

    result = pipeline.answer(
        query="How do FastAPI dependencies work?",
        chunks=[chunk],
    )

    assert result.provider == "fake"
    assert result.model == "fake-model"
    assert result.context_chunks == (chunk,)
    assert "[chunk_001]" in result.answer
    assert result.generation_time_ns >= 0