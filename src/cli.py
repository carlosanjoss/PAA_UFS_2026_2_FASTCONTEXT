from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from src.app.bootstrap import create_application
from src.app.health import check_application_health
from src.app.models import ApplicationContainer
from src.rag.providers.base import (
    GenerationConfig,
    LLMProviderError,
    LLMTruncatedResponseError,
)
from src.retrieval.registry import RetrieverRegistryError
from src.services.factory import create_fastcontext_service
from src.services.fastcontext import RAGNotConfiguredError


def build_parser() -> argparse.ArgumentParser:
    """Build the FastContext command-line parser."""

    parser = argparse.ArgumentParser(
        prog="fastcontext",
        description=(
            "Algorithmic context retrieval "
            "for generative AI."
        ),
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    subparsers.add_parser(
        "health",
        help="Check application component health.",
    )

    subparsers.add_parser(
        "algorithms",
        help="List registered retrieval algorithms.",
    )

    retrieve_parser = subparsers.add_parser(
        "retrieve",
        help="Run retrieval without LLM generation.",
    )

    _add_retrieval_arguments(
        retrieve_parser
    )

    ask_parser = subparsers.add_parser(
        "ask",
        help=(
            "Run retrieval followed by "
            "RAG generation."
        ),
    )

    _add_retrieval_arguments(
        ask_parser
    )

    ask_parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help=(
            "Maximum number of generated tokens."
        ),
    )

    ask_parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM generation temperature.",
    )

    return parser


def _add_retrieval_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    """Add common retrieval arguments."""

    parser.add_argument(
        "--algorithm",
        required=True,
        help=(
            "Registered retrieval algorithm "
            "identifier."
        ),
    )

    parser.add_argument(
        "--query",
        required=True,
        help="Question or retrieval query.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help=(
            "Maximum number of chunks "
            "to retrieve."
        ),
    )


def print_health(
    application: ApplicationContainer,
) -> int:
    """Print application health information."""

    report = check_application_health(
        application
    )

    print(
        f"FastContext status: {report.status}"
    )

    print()

    for component in report.components:
        print(
            f"{component.name}: "
            f"{component.status}"
        )

        print(
            f"  {component.message}"
        )

    if report.status == "healthy":
        return 0

    return 1


def print_algorithms(
    application: ApplicationContainer,
) -> int:
    """Print registered retrieval algorithms."""

    algorithms = (
        application.registry
        .available_names()
    )

    if not algorithms:
        print(
            "No retrieval algorithms "
            "are currently registered."
        )

        return 0

    print(
        "Registered retrieval algorithms:"
    )

    for algorithm in algorithms:
        print(
            f"- {algorithm}"
        )

    return 0


def run_retrieve(
    application: ApplicationContainer,
    *,
    algorithm: str,
    query: str,
    top_k: int,
) -> int:
    """Run retrieval and print ranked chunks."""

    service = create_fastcontext_service(
        algorithm=algorithm,
        registry=application.registry,
    )

    result = service.retrieve(
        query=query,
        top_k=top_k,
    )

    comparisons = (
        _format_optional_metric(
            result.metrics.comparisons
        )
    )

    chunks_scored = (
        _format_optional_metric(
            result.metrics.chunks_scored
        )
    )

    candidates_found = (
        _format_optional_metric(
            result.metrics.candidates_found
        )
    )

    print(
        f"Algorithm: {result.algorithm}"
    )

    print(
        f"Query: {result.query}"
    )

    print(
        f"Top-k: {result.top_k}"
    )

    print(
        "Retrieval time: "
        f"{result.metrics.retrieval_time_ns} ns"
    )

    print(
        f"Comparisons: {comparisons}"
    )

    print(
        f"Chunks scored: {chunks_scored}"
    )

    print(
        f"Candidates found: {candidates_found}"
    )

    print()

    if not result.chunks:
        print(
            "No chunks were retrieved."
        )

        return 0

    for chunk in result.chunks:
        print(
            f"[{chunk.rank}] "
            f"{chunk.chunk_id}"
        )

        print(
            f"Score: {chunk.score:.6f}"
        )

        print(
            "Section: "
            f"{chunk.section_title}"
        )

        print(
            "Source: "
            f"{chunk.source_path}"
        )

        print()

        print(
            chunk.content
        )

        print()

        print(
            "-" * 60
        )

    return 0


def run_ask(
    application: ApplicationContainer,
    *,
    algorithm: str,
    query: str,
    top_k: int,
    max_tokens: int,
    temperature: float,
) -> int:
    """Run retrieval followed by grounded generation."""

    if max_tokens <= 0:
        raise ValueError(
            "max_tokens must be greater than zero."
        )

    if temperature < 0:
        raise ValueError(
            "temperature cannot be negative."
        )

    service = create_fastcontext_service(
        algorithm=algorithm,
        registry=application.registry,
        rag_pipeline=application.rag_pipeline,
    )

    result = service.ask(
        query=query,
        top_k=top_k,
        generation_config=GenerationConfig(
            temperature=temperature,
            max_tokens=max_tokens,
            think=False,
        ),
    )

    if result.rag.valid_citations:
        valid_citations = ", ".join(
            result.rag.valid_citations
        )
    else:
        valid_citations = "None"

    if result.rag.invalid_citations:
        invalid_citations = ", ".join(
            result.rag.invalid_citations
        )
    else:
        invalid_citations = "None"

    print(
        f"Algorithm: {result.algorithm}"
    )

    print(
        f"Provider: {result.rag.provider}"
    )

    print(
        f"Model: {result.rag.model}"
    )

    print()

    print(
        "Timing"
    )

    print(
        "-" * 60
    )

    print(
        "Retrieval: "
        f"{result.retrieval_time_ns} ns"
    )

    print(
        "Generation: "
        f"{result.generation_time_ns} ns"
    )

    print(
        "End-to-end: "
        f"{result.end_to_end_time_ns} ns"
    )

    print()

    print(
        "Citation validation"
    )

    print(
        "-" * 60
    )

    print(
        "Valid: "
        f"{result.rag.citation_valid}"
    )

    print(
        "Retries: "
        f"{result.rag.citation_retry_count}"
    )

    print(
        "Valid citations: "
        f"{valid_citations}"
    )

    print(
        "Invalid citations: "
        f"{invalid_citations}"
    )

    print()

    print(
        "Answer"
    )

    print(
        "-" * 60
    )

    print(
        result.rag.answer
    )

    return 0


def _format_optional_metric(
    value: int | float | None,
) -> str:
    """Format an optional CLI metric."""

    if value is None:
        return "N/A"

    return str(value)


def main(
    argv: Sequence[str] | None = None,
) -> int:
    """Run the FastContext CLI."""

    parser = build_parser()

    args = parser.parse_args(
        argv
    )

    try:
        application = (
            create_application()
        )

        if args.command == "health":
            return print_health(
                application
            )

        if args.command == "algorithms":
            return print_algorithms(
                application
            )

        if args.command == "retrieve":
            return run_retrieve(
                application,
                algorithm=args.algorithm,
                query=args.query,
                top_k=args.top_k,
            )

        if args.command == "ask":
            return run_ask(
                application,
                algorithm=args.algorithm,
                query=args.query,
                top_k=args.top_k,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )

        parser.error(
            "Unknown command."
        )

    except (
        ValueError,
        RetrieverRegistryError,
        RAGNotConfiguredError,
        LLMProviderError,
        LLMTruncatedResponseError,
    ) as exc:
        print(
            f"Error: {exc}",
            file=sys.stderr,
        )

        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )