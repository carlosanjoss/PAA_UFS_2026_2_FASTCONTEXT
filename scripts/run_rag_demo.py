from __future__ import annotations

from src.rag.factory import create_llm_provider
from src.rag.pipeline import RAGPipeline, RAGResult
from src.rag.prompt import ContextChunk
from src.rag.providers.base import (
    GenerationConfig,
    LLMProviderError,
    LLMTruncatedResponseError,
)


def nanoseconds_to_milliseconds(
    value: int | None,
) -> float | None:
    """Convert nanoseconds to milliseconds."""

    if value is None:
        return None

    return value / 1_000_000


def print_optional_duration(
    label: str,
    value: object,
) -> None:
    """Print an Ollama duration in milliseconds."""

    if not isinstance(value, int):
        return

    milliseconds = (
        nanoseconds_to_milliseconds(
            value
        )
    )

    if milliseconds is None:
        return

    print(
        f"{label}: {milliseconds:.2f} ms"
    )


def print_generation_metadata(
    result: RAGResult,
) -> None:
    """Print generation metadata."""

    metadata = result.metadata or {}

    fallback_used = metadata.get(
        "fallback_used",
        False,
    )

    thinking_enabled = metadata.get(
        "thinking_enabled",
        False,
    )

    done_reason = metadata.get(
        "done_reason",
        "unknown",
    )

    prompt_tokens = metadata.get(
        "prompt_eval_count",
        "unknown",
    )

    output_tokens = metadata.get(
        "eval_count",
        "unknown",
    )

    truncation_retries = metadata.get(
        "truncation_retries",
        0,
    )

    effective_max_tokens = metadata.get(
        "effective_max_tokens",
        "unknown",
    )

    print()
    print("Generation metadata")
    print("-" * 60)

    print(
        f"Fallback used: {fallback_used}"
    )

    print(
        f"Thinking enabled: {thinking_enabled}"
    )

    print(
        f"Done reason: {done_reason}"
    )

    print(
        f"Prompt tokens: {prompt_tokens}"
    )

    print(
        f"Output tokens: {output_tokens}"
    )

    print(
        "Truncation retries: "
        f"{truncation_retries}"
    )

    print(
        "Effective max tokens: "
        f"{effective_max_tokens}"
    )

    print_optional_duration(
        "Ollama total duration",
        metadata.get(
            "total_duration"
        ),
    )

    print_optional_duration(
        "Model load duration",
        metadata.get(
            "load_duration"
        ),
    )

    print_optional_duration(
        "Prompt evaluation duration",
        metadata.get(
            "prompt_eval_duration"
        ),
    )

    print_optional_duration(
        "Token generation duration",
        metadata.get(
            "eval_duration"
        ),
    )


def print_citation_validation(
    result: RAGResult,
) -> None:
    """Print citation validation information."""

    print()
    print("Citation validation")
    print("-" * 60)

    print(
        f"Citation valid: "
        f"{result.citation_valid}"
    )

    print(
        f"Citation count: "
        f"{result.citation_count}"
    )

    print(
        "Citation retries: "
        f"{result.citation_retry_count}"
    )

    if result.valid_citations:
        valid = ", ".join(
            result.valid_citations
        )
    else:
        valid = "None"

    if result.invalid_citations:
        invalid = ", ".join(
            result.invalid_citations
        )
    else:
        invalid = "None"

    print(
        f"Valid citations: {valid}"
    )

    print(
        f"Invalid citations: {invalid}"
    )


def build_demo_chunks() -> list[
    ContextChunk
]:
    """Create manual chunks for the local RAG demo."""

    return [
        ContextChunk(
            chunk_id=(
                "chunk_dependencies_001"
            ),
            source_path=(
                "tutorial/dependencies/"
                "index.md"
            ),
            section_title="Dependencies",
            content=(
                "FastAPI has a powerful "
                "but intuitive Dependency "
                "Injection system. Path "
                "operation functions can "
                "declare requirements that "
                "FastAPI resolves "
                "automatically."
            ),
            score=0.97,
        ),
        ContextChunk(
            chunk_id=(
                "chunk_security_001"
            ),
            source_path=(
                "tutorial/security/"
                "index.md"
            ),
            section_title="Security",
            content=(
                "FastAPI provides utilities "
                "for implementing security "
                "schemes such as OAuth2."
            ),
            score=0.81,
        ),
    ]


def print_retrieved_context(
    chunks: tuple[
        ContextChunk,
        ...,
    ],
) -> None:
    """Print chunks used by the RAG pipeline."""

    print()
    print("Retrieved context")
    print("-" * 60)

    for chunk in chunks:
        if chunk.score is None:
            score_text = "N/A"
        else:
            score_text = (
                f"{chunk.score:.4f}"
            )

        print(
            f"{chunk.chunk_id} | "
            f"{chunk.section_title} | "
            f"score={score_text}"
        )

        print(
            f"Source: {chunk.source_path}"
        )

        print()


def main() -> None:
    """Run a local FastContext RAG demonstration."""

    print()
    print("FastContext RAG Demo")
    print("=" * 60)

    provider = (
        create_llm_provider()
    )

    pipeline = RAGPipeline(
        provider=provider,
        max_truncation_retries=1,
        max_retry_tokens=256,
        max_citation_retries=1,
    )

    chunks = build_demo_chunks()

    try:
        result = pipeline.answer(
            query=(
                "How does FastAPI provide "
                "dependencies to path "
                "operation functions?"
            ),
            chunks=chunks,
            generation_config=(
                GenerationConfig(
                    temperature=0.0,
                    max_tokens=128,
                    think=False,
                )
            ),
        )

    except (
        LLMTruncatedResponseError
    ) as exc:
        print()
        print("Generation failed")
        print("-" * 60)
        print(str(exc))
        return

    except LLMProviderError as exc:
        print()
        print("Provider error")
        print("-" * 60)
        print(str(exc))
        return

    generation_time_ms = (
        result.generation_time_ns
        / 1_000_000
    )

    print()
    print(
        f"Provider: {result.provider}"
    )

    print(
        f"Model: {result.model}"
    )

    print(
        "Generation time: "
        f"{generation_time_ms:.2f} ms"
    )

    print_generation_metadata(
        result
    )

    print_citation_validation(
        result
    )

    print()
    print("Answer")
    print("-" * 60)
    print(result.answer)

    print_retrieved_context(
        result.context_chunks
    )


if __name__ == "__main__":
    main()