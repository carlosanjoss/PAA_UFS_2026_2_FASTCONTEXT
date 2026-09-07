from __future__ import annotations

from src.app.models import (
    ApplicationContainer,
    ApplicationHealthReport,
    ComponentHealth,
    HealthState,
)


def check_application_health(
    application: ApplicationContainer,
) -> ApplicationHealthReport:
    """Check the health of FastContext runtime components."""

    components = (
        _check_retrieval_health(
            application
        ),
        _check_llm_health(
            application
        ),
        _check_rag_health(
            application
        ),
    )

    status = _calculate_overall_status(
        components
    )

    return ApplicationHealthReport(
        status=status,
        components=components,
    )


def _check_retrieval_health(
    application: ApplicationContainer,
) -> ComponentHealth:
    """Check retrieval registration and corpus availability."""

    algorithms = (
        application.registry
        .available_names()
    )

    if not algorithms:
        return ComponentHealth(
            name="Retrieval",
            status="degraded",
            message=(
                "No retrieval algorithms are "
                "currently registered."
            ),
        )

    algorithm_names = ", ".join(
        algorithms
    )

    if application.corpus_size == 0:
        return ComponentHealth(
            name="Retrieval",
            status="degraded",
            message=(
                "Available algorithms: "
                f"{algorithm_names}. "
                "No corpus chunks are loaded."
            ),
        )

    if application.corpus_size is None:
        return ComponentHealth(
            name="Retrieval",
            status="healthy",
            message=(
                "Available algorithms: "
                f"{algorithm_names}"
            ),
        )

    return ComponentHealth(
        name="Retrieval",
        status="healthy",
        message=(
            "Available algorithms: "
            f"{algorithm_names}. "
            "Corpus chunks: "
            f"{application.corpus_size}"
        ),
    )


def _check_llm_health(
    application: ApplicationContainer,
) -> ComponentHealth:
    """Check whether the configured LLM provider is available."""

    provider = application.provider

    if provider.is_available():
        return ComponentHealth(
            name="LLM",
            status="healthy",
            message=(
                f"Provider '{provider.name}' "
                "is available."
            ),
        )

    return ComponentHealth(
        name="LLM",
        status="unavailable",
        message=(
            f"Provider '{provider.name}' "
            "is not available."
        ),
    )


def _check_rag_health(
    application: ApplicationContainer,
) -> ComponentHealth:
    """Check whether RAG generation can currently run."""

    if application.provider.is_available():
        return ComponentHealth(
            name="RAG",
            status="healthy",
            message=(
                "RAG pipeline is ready "
                "for generation."
            ),
        )

    return ComponentHealth(
        name="RAG",
        status="unavailable",
        message=(
            "RAG pipeline is configured, "
            "but the LLM provider is unavailable."
        ),
    )


def _calculate_overall_status(
    components: tuple[
        ComponentHealth,
        ...,
    ],
) -> HealthState:
    """Calculate the overall application health state."""

    statuses = {
        component.status
        for component in components
    }

    if statuses == {"healthy"}:
        return "healthy"

    healthy_count = sum(
        component.status == "healthy"
        for component in components
    )

    if healthy_count > 0:
        return "degraded"

    if "degraded" in statuses:
        return "degraded"

    return "unavailable"