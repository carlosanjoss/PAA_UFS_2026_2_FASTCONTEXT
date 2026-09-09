"""Interactive Streamlit interface for the real FastContext runtime."""

from __future__ import annotations

from typing import Any

import streamlit as st
from components.answer import render_answer
from components.comparison import render_experiment_comparison
from components.header import (
    configure_page,
    inject_styles,
    render_header,
)
from components.metrics import render_metrics
from components.results import render_results
from components.sidebar import RunControls, render_controls
from components.status import render_status

from src.app.bootstrap import create_application
from src.app.health import check_application_health
from src.app.models import ApplicationContainer, ApplicationHealthReport
from src.rag.providers.base import GenerationConfig, LLMProviderError
from src.retrieval.models import RetrievalResult
from src.retrieval.registry import RetrieverRegistryError
from src.services.factory import create_fastcontext_service
from src.services.fastcontext import (
    FastContextResult,
    RAGNotConfiguredError,
)

configure_page()


@st.cache_resource
def get_application() -> ApplicationContainer:
    """Create and cache the application dependency container."""

    return create_application()


@st.cache_resource
def get_service(algorithm: str) -> Any:
    """Create one cached service per real retrieval strategy."""

    application = get_application()
    return create_fastcontext_service(
        algorithm=algorithm,
        registry=application.registry,
        rag_pipeline=application.rag_pipeline,
    )


@st.cache_data(ttl=15)
def get_health_report() -> ApplicationHealthReport:
    """Refresh optional provider availability without rebuilding resources."""

    return check_application_health(get_application())


def main() -> None:
    """Run the FastContext Streamlit application."""

    inject_styles()

    try:
        application = get_application()
    except Exception as exc:
        _render_error(
            "FastContext could not initialize the prepared corpus or runtime.",
            exc,
        )
        return

    report = get_health_report()
    algorithms = application.registry.available_names()

    if not algorithms:
        st.error("No retrieval strategies are registered in the application.")
        return

    controls = render_controls(algorithms)
    semantic_metadata = _last_semantic_metadata()
    render_status(application, report, semantic_metadata)
    render_header(application.corpus_size)

    workspace_tab, experiment_tab = st.tabs(("Ask & explain", "Experiment dashboard"))
    with workspace_tab:
        if application.corpus_size == 0:
            st.warning(
                "The prepared corpus is unavailable. Run the documented corpus "
                "preparation workflow before submitting a query."
            )
        else:
            query, submitted = _render_query_form(controls)
            if submitted:
                _execute_request(application, controls, query)
            _render_saved_result()

    with experiment_tab:
        render_experiment_comparison()


def _render_query_form(
    controls: RunControls,
) -> tuple[str, bool]:
    """Render the main query form and resolve an optional suggestion."""

    query_column, guide_column = st.columns((2.25, 1), gap="large")
    with query_column, st.form("query_form"):
        st.markdown("### Ask the documentation")
        st.caption(
            "Ask a question, receive a grounded answer, and inspect exactly "
            "how the context was retrieved."
        )
        entered_query = st.text_area(
            "Ask the FastAPI documentation",
            key="query_input",
            placeholder="How does FastAPI dependency injection work?",
            height=104,
            label_visibility="collapsed",
        )

        if controls.suggestion:
            st.caption(
                "A selected suggested query is used when the text field is empty."
            )

        action_label = "Ask for an answer" if controls.use_rag else "Retrieve context"
        submitted = st.form_submit_button(
            action_label,
            width="stretch",
            type="primary",
        )

    with guide_column, st.container(border=True):
        st.markdown("#### Learning guide")
        st.markdown(
            "1. **Choose** a retrieval strategy in the sidebar.  \n"
            "2. **Ask** a documentation question.  \n"
            "3. **Read** the answer, then inspect the algorithm and its evidence."
        )
        if controls.use_rag:
            st.caption("LLM answer enabled · retrieval evidence stays visible.")
        else:
            st.caption("Retrieval-only mode · no LLM call will be made.")

    resolved_query = entered_query.strip() or (controls.suggestion or "")
    return resolved_query, submitted


def _execute_request(
    application: ApplicationContainer,
    controls: RunControls,
    query: str,
) -> None:
    """Use the service facade for retrieval and optional RAG generation."""

    if not query:
        st.warning("Enter a FastAPI question or select a suggested query.")
        return

    try:
        service = get_service(controls.algorithm)
    except (RetrieverRegistryError, ValueError) as exc:
        _render_error("The selected retriever could not be initialized.", exc)
        return

    if not controls.use_rag:
        with st.spinner("Retrieving and ranking documentation chunks..."):
            try:
                retrieval = service.retrieve(query=query, top_k=controls.top_k)
            except (RetrieverRegistryError, ValueError) as exc:
                _render_error("Retrieval could not be completed.", exc)
                return
            except Exception as exc:
                _render_error("Retrieval failed unexpectedly.", exc)
                return

        _save_retrieval(retrieval)
        st.rerun()
        return

    if not application.provider.is_available():
        st.warning(
            "The LLM provider is offline or the configured model is missing. "
            "Showing retrieval results without generation."
        )
        _retrieve_after_llm_failure(service, query, controls.top_k)
        return

    with st.spinner("Retrieving documentation and generating a grounded answer..."):
        try:
            result = service.ask(
                query=query,
                top_k=controls.top_k,
                generation_config=GenerationConfig(
                    temperature=0.0,
                    max_tokens=256,
                    think=False,
                ),
            )
        except (LLMProviderError, RAGNotConfiguredError) as exc:
            st.warning("Generation was unavailable. Showing retrieval results instead.")
            _retrieve_after_llm_failure(service, query, controls.top_k, exc)
            return
        except (RetrieverRegistryError, ValueError) as exc:
            _render_error("The RAG request could not be completed.", exc)
            return
        except Exception as exc:
            _render_error("The RAG request failed unexpectedly.", exc)
            return

    st.session_state["retrieval_result"] = result.retrieval
    st.session_state["fastcontext_result"] = result
    st.session_state.pop("generation_error", None)
    st.rerun()


def _retrieve_after_llm_failure(
    service: Any,
    query: str,
    top_k: int,
    generation_error: Exception | None = None,
) -> None:
    """Keep retrieval available after an optional LLM failure."""

    try:
        retrieval = service.retrieve(query=query, top_k=top_k)
    except Exception as exc:
        _render_error("Retrieval fallback also failed.", exc)
        return

    _save_retrieval(retrieval, generation_error)
    st.rerun()


def _save_retrieval(
    result: RetrievalResult,
    generation_error: Exception | None = None,
) -> None:
    """Store the latest retrieval and clear stale generated output."""

    st.session_state["retrieval_result"] = result
    st.session_state.pop("fastcontext_result", None)
    if generation_error is None:
        st.session_state.pop("generation_error", None)
    else:
        st.session_state["generation_error"] = (
            str(generation_error) or type(generation_error).__name__
        )


def _render_saved_result() -> None:
    """Render results that survive ordinary Streamlit reruns."""

    result = st.session_state.get("fastcontext_result")
    if isinstance(result, FastContextResult):
        st.divider()
        render_answer(result)

    generation_error = st.session_state.get("generation_error")
    if isinstance(generation_error, str):
        st.warning("An LLM answer was unavailable; retrieved context is shown below.")
        with st.expander("Generation technical details"):
            st.code(generation_error)

    retrieval = st.session_state.get("retrieval_result")
    if isinstance(retrieval, RetrievalResult):
        st.divider()
        render_metrics(retrieval)
        st.divider()
        render_results(retrieval)


def _last_semantic_metadata() -> dict[str, Any] | None:
    """Return metadata from the latest real semantic request, if available."""

    retrieval = st.session_state.get("retrieval_result")
    if not isinstance(retrieval, RetrievalResult):
        return None

    if retrieval.algorithm != "semantic":
        return None

    return retrieval.metadata


def _render_error(message: str, exc: Exception) -> None:
    """Show a useful error without exposing a traceback by default."""

    st.error(message)
    with st.expander("Technical details"):
        st.code(str(exc) or type(exc).__name__)


if __name__ == "__main__":
    main()
