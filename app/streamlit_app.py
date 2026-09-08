from __future__ import annotations

import streamlit as st

from src.app.bootstrap import create_application
from src.app.health import check_application_health
from src.app.models import ApplicationContainer
from src.app.streamlit_support import (
    build_health_rows,
    build_retrieval_rows,
    format_duration_ns,
    format_optional_number,
)
from src.rag.providers.base import (
    GenerationConfig,
    LLMProviderError,
)
from src.retrieval.models import RetrievalResult
from src.retrieval.registry import (
    RetrieverRegistryError,
)
from src.services.factory import (
    create_fastcontext_service,
)
from src.services.fastcontext import (
    FastContextResult,
)

st.set_page_config(
    page_title="FastContext",
    layout="wide",
)


@st.cache_resource
def get_application() -> ApplicationContainer:
    """Create and cache the FastContext application runtime."""

    return create_application()


def render_sidebar(
    application: ApplicationContainer,
) -> None:
    """Render application health and runtime configuration."""

    st.sidebar.title(
        "FastContext"
    )

    st.sidebar.caption(
        "Algorithmic context retrieval "
        "for generative AI"
    )

    st.sidebar.divider()

    st.sidebar.subheader(
        "Application health"
    )

    report = check_application_health(
        application
    )

    st.sidebar.write(
        f"Overall status: **{report.status}**"
    )

    for component in report.components:
        st.sidebar.write(
            f"**{component.name}**: "
            f"{component.status}"
        )

        st.sidebar.caption(
            component.message
        )

    st.sidebar.divider()

    st.sidebar.subheader(
        "RAG configuration"
    )

    st.sidebar.write(
        "Provider: "
        f"`{application.settings.default_provider}`"
    )

    st.sidebar.write(
        "Model: "
        f"`{application.settings.ollama.model}`"
    )

    fallback_model = (
        application.settings
        .ollama
        .fallback_model
    )

    fallback_text = (
        "None"
        if fallback_model is None
        else fallback_model
    )

    st.sidebar.write(
        "Fallback: "
        f"`{fallback_text}`"
    )


def render_retrieval_metrics(
    result: RetrievalResult,
) -> None:
    """Render retrieval performance metrics."""

    metrics = result.metrics

    primary_columns = st.columns(
        4
    )

    primary_columns[0].metric(
        "Retrieval time",
        format_duration_ns(
            metrics.retrieval_time_ns
        ),
    )

    primary_columns[1].metric(
        "Comparisons",
        format_optional_number(
            metrics.comparisons
        ),
    )

    primary_columns[2].metric(
        "Chunks scored",
        format_optional_number(
            metrics.chunks_scored
        ),
    )

    primary_columns[3].metric(
        "Candidates",
        format_optional_number(
            metrics.candidates_found
        ),
    )

    secondary_columns = st.columns(
        3
    )

    secondary_columns[0].metric(
        "Sorting time",
        format_duration_ns(
            metrics.sorting_time_ns
        ),
    )

    secondary_columns[1].metric(
        "Index build time",
        format_duration_ns(
            metrics.index_build_time_ns
        ),
    )

    memory_text = (
        "N/A"
        if metrics.peak_memory_mb is None
        else f"{metrics.peak_memory_mb:.2f} MB"
    )

    secondary_columns[2].metric(
        "Peak memory",
        memory_text,
    )


def render_retrieval_result(
    result: RetrievalResult,
) -> None:
    """Render retrieval output and ranked chunks."""

    st.subheader(
        "Retrieval result"
    )

    st.write(
        f"Algorithm: **{result.algorithm}**"
    )

    st.write(
        f"Requested top-k: **{result.top_k}**"
    )

    render_retrieval_metrics(
        result
    )

    st.markdown(
        "#### Retrieved chunks"
    )

    rows = build_retrieval_rows(
        result
    )

    if not rows:
        st.info(
            "No chunks were retrieved."
        )

        return

    st.dataframe(
        rows,
        use_container_width=True,
        hide_index=True,
    )


def render_rag_result(
    result: FastContextResult,
) -> None:
    """Render the generated RAG response."""

    rag = result.rag

    st.subheader(
        "RAG answer"
    )

    timing_columns = st.columns(
        3
    )

    timing_columns[0].metric(
        "Retrieval",
        format_duration_ns(
            result.retrieval_time_ns
        ),
    )

    timing_columns[1].metric(
        "Generation",
        format_duration_ns(
            result.generation_time_ns
        ),
    )

    timing_columns[2].metric(
        "End-to-end",
        format_duration_ns(
            result.end_to_end_time_ns
        ),
    )

    st.write(
        rag.answer
    )

    st.markdown(
        "#### Generation metadata"
    )

    generation_columns = st.columns(
        4
    )

    generation_columns[0].metric(
        "Provider",
        rag.provider,
    )

    generation_columns[1].metric(
        "Model",
        rag.model,
    )

    generation_columns[2].metric(
        "Citation valid",
        str(
            rag.citation_valid
        ),
    )

    generation_columns[3].metric(
        "Citation retries",
        str(
            rag.citation_retry_count
        ),
    )

    if rag.valid_citations:
        st.success(
            "Valid citations: "
            + ", ".join(
                rag.valid_citations
            )
        )

    if rag.invalid_citations:
        st.warning(
            "Invalid citations: "
            + ", ".join(
                rag.invalid_citations
            )
        )


def render_health_details(
    application: ApplicationContainer,
) -> None:
    """Render the complete component health report."""

    report = check_application_health(
        application
    )

    with st.expander(
        "Detailed health report"
    ):
        st.dataframe(
            build_health_rows(
                report
            ),
            use_container_width=True,
            hide_index=True,
        )


def render_no_retrievers_message() -> None:
    """Explain why retrieval actions are currently unavailable."""

    st.warning(
        "No retrieval algorithms are registered yet."
    )

    st.info(
        "The application bootstrap, RAG layer, "
        "Ollama integration, health checks, "
        "retrieval contracts, and interface are ready. "
        "The retrieval controls will become available "
        "automatically when the real retrievers are registered."
    )


def main() -> None:
    """Run the FastContext Streamlit application."""

    application = (
        get_application()
    )

    render_sidebar(
        application
    )

    st.title(
        "FastContext"
    )

    st.caption(
        "Compare context retrieval algorithms "
        "and optionally generate grounded "
        "FastAPI answers."
    )

    render_health_details(
        application
    )

    algorithms = (
        application.registry
        .available_names()
    )

    st.markdown(
        "### Query"
    )

    query = st.text_area(
        "FastAPI question",
        placeholder=(
            "How does FastAPI dependency "
            "injection work?"
        ),
        height=100,
    )

    top_k = st.number_input(
        "Top-k",
        min_value=1,
        max_value=50,
        value=5,
        step=1,
    )

    if not algorithms:
        render_no_retrievers_message()
        return

    algorithm = st.selectbox(
        "Retrieval algorithm",
        options=algorithms,
    )

    action_columns = st.columns(
        2
    )

    retrieve_clicked = (
        action_columns[0].button(
            "Retrieve",
            use_container_width=True,
        )
    )

    rag_clicked = (
        action_columns[1].button(
            "Retrieve + Generate",
            use_container_width=True,
        )
    )

    if retrieve_clicked:
        try:
            service = (
                create_fastcontext_service(
                    algorithm=algorithm,
                    registry=(
                        application.registry
                    ),
                )
            )

            retrieval_result = (
                service.retrieve(
                    query=query,
                    top_k=int(
                        top_k
                    ),
                )
            )

            st.session_state[
                "retrieval_result"
            ] = retrieval_result

            st.session_state.pop(
                "fastcontext_result",
                None,
            )

        except (
            ValueError,
            RetrieverRegistryError,
        ) as exc:
            st.error(
                str(exc)
            )

    if rag_clicked:
        try:
            service = (
                create_fastcontext_service(
                    algorithm=algorithm,
                    registry=(
                        application.registry
                    ),
                    rag_pipeline=(
                        application.rag_pipeline
                    ),
                )
            )

            fastcontext_result = (
                service.ask(
                    query=query,
                    top_k=int(
                        top_k
                    ),
                    generation_config=(
                        GenerationConfig(
                            temperature=0.0,
                            max_tokens=128,
                            think=False,
                        )
                    ),
                )
            )

            st.session_state[
                "retrieval_result"
            ] = (
                fastcontext_result
                .retrieval
            )

            st.session_state[
                "fastcontext_result"
            ] = (
                fastcontext_result
            )

        except (
            ValueError,
            RetrieverRegistryError,
            LLMProviderError,
        ) as exc:
            st.error(
                str(exc)
            )

    retrieval_result = (
        st.session_state.get(
            "retrieval_result"
        )
    )

    if isinstance(
        retrieval_result,
        RetrievalResult,
    ):
        st.divider()

        render_retrieval_result(
            retrieval_result
        )

    fastcontext_result = (
        st.session_state.get(
            "fastcontext_result"
        )
    )

    if isinstance(
        fastcontext_result,
        FastContextResult,
    ):
        st.divider()

        render_rag_result(
            fastcontext_result
        )


if __name__ == "__main__":
    main()