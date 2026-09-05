from src.rag.factory import create_llm_provider
from src.rag.pipeline import RAGPipeline
from src.rag.prompt import ContextChunk
from src.rag.providers.base import GenerationConfig


def main() -> None:
    """Run a minimal local RAG demonstration."""

    provider = create_llm_provider()

    chunks = [
        ContextChunk(
            chunk_id="chunk_dependencies_001",
            source_path="tutorial/dependencies/index.md",
            section_title="Dependencies",
            content=(
                "FastAPI has a powerful but intuitive "
                "Dependency Injection system. Path operation "
                "functions can declare requirements that "
                "FastAPI resolves automatically."
            ),
            score=0.97,
        ),
        ContextChunk(
            chunk_id="chunk_security_001",
            source_path="tutorial/security/index.md",
            section_title="Security",
            content=(
                "FastAPI provides utilities for implementing "
                "security schemes such as OAuth2."
            ),
            score=0.81,
        ),
    ]

    pipeline = RAGPipeline(
        provider=provider,
    )

    result = pipeline.answer(
        query=(
            "How does FastAPI provide dependencies "
            "to path operation functions?"
        ),
        chunks=chunks,
        generation_config=GenerationConfig(
            temperature=0.0,
            max_tokens=256,
        ),
    )

    print()
    print("FastContext RAG Demo")
    print("=" * 60)
    print(f"Provider: {result.provider}")
    print(f"Model: {result.model}")
    print(
        "Generation time: "
        f"{result.generation_time_ns / 1_000_000:.2f} ms"
    )
    print()
    print("Answer")
    print("-" * 60)
    print(result.answer)
    print()
    print("Retrieved context")
    print("-" * 60)

    for chunk in result.context_chunks:
        print(
            f"{chunk.chunk_id} | "
            f"{chunk.section_title} | "
            f"score={chunk.score}"
        )


if __name__ == "__main__":
    main()