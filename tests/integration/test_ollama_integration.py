import pytest

from src.rag.pipeline import RAGPipeline
from src.rag.prompt import ContextChunk
from src.rag.providers.base import GenerationConfig
from src.rag.providers.ollama import OllamaProvider
from src.rag.settings import load_rag_settings

pytestmark = pytest.mark.integration


def test_real_ollama_generation() -> None:
    settings = load_rag_settings()

    provider = OllamaProvider(
        model=settings.ollama.model,
        base_url=settings.ollama.base_url,
    )

    if not provider.is_available():
        pytest.skip(
            f"Ollama model '{settings.ollama.model}' "
            "is not available locally."
        )

    response = provider.generate(
        prompt=(
            "Answer only with the word OK if you can "
            "understand this message."
        ),
        config=GenerationConfig(
            temperature=0.0,
            max_tokens=64,
            think=False,
        ),
    )

    assert response.text.strip()
    assert response.provider == "ollama"
    assert response.model == settings.ollama.model


def test_real_rag_pipeline_with_fastapi_context() -> None:
    settings = load_rag_settings()

    provider = OllamaProvider(
        model=settings.ollama.model,
        base_url=settings.ollama.base_url,
    )

    if not provider.is_available():
        pytest.skip(
            f"Ollama model '{settings.ollama.model}' "
            "is not available locally."
        )

    pipeline = RAGPipeline(provider)

    chunks = [
        ContextChunk(
            chunk_id="chunk_dependencies_001",
            source_path="tutorial/dependencies/index.md",
            section_title="Dependencies",
            content=(
                "FastAPI has a powerful but intuitive "
                "Dependency Injection system. It allows "
                "a path operation function to declare "
                "things that it requires."
            ),
            score=0.97,
        ),
        ContextChunk(
            chunk_id="chunk_security_001",
            source_path="tutorial/security/index.md",
            section_title="Security",
            content=(
                "FastAPI provides utilities to integrate "
                "security schemes such as OAuth2 into "
                "applications."
            ),
            score=0.81,
        ),
    ]

    result = pipeline.answer(
        query=(
            "What mechanism does FastAPI use to declare "
            "requirements for a path operation?"
        ),
        chunks=chunks,
        generation_config=GenerationConfig(
            temperature=0.0,
            max_tokens=128,
            think=False,
        ),
    )

    assert result.answer.strip()
    assert result.provider == "ollama"
    assert result.model == settings.ollama.model
    assert result.generation_time_ns > 0
    assert len(result.context_chunks) == 2