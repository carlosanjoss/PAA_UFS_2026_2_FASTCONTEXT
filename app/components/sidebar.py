"""Sidebar controls for selecting a retrieval run."""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from src.app.streamlit_support import get_strategy_presentation

SUGGESTED_QUERIES = (
    "How do I configure CORS in FastAPI?",
    "How do I use dependencies with Depends?",
    "How can I create a WebSocket endpoint?",
    "How do I test FastAPI endpoints with TestClient?",
)


@dataclass(frozen=True, slots=True)
class RunControls:
    """Selections that determine one retrieval or RAG request."""

    algorithm: str
    top_k: int
    use_rag: bool
    suggestion: str | None


def render_controls(
    algorithms: tuple[str, ...],
) -> RunControls:
    """Render retrieval controls and return their selected values."""

    with st.sidebar:
        st.markdown("<p class='fc-eyebrow'>FASTCONTEXT</p>", unsafe_allow_html=True)
        st.header("Retrieval dashboard")
        st.caption("Configure a run, ask a question, and inspect the evidence.")
        st.markdown(
            "<div class='fc-side-active'>Dashboard</div>", unsafe_allow_html=True
        )
        st.markdown(
            "<div class='fc-side-link'>Experiment charts</div>", unsafe_allow_html=True
        )
        st.markdown(
            "<div class='fc-side-link'>Runtime status</div>", unsafe_allow_html=True
        )
        st.divider()
        st.markdown("#### Run configuration")

        algorithm = st.selectbox(
            "Strategy",
            options=algorithms,
            format_func=lambda item: get_strategy_presentation(item).label,
        )

        presentation = get_strategy_presentation(algorithm)
        st.markdown("**How this strategy works**")
        st.caption(presentation.description)

        top_k = st.select_slider(
            "Context budget (top-k)",
            options=(1, 3, 5, 10),
            value=5,
            help="Maximum number of ranked documentation chunks returned.",
        )

        mode = st.radio(
            "Execution mode",
            options=("Retrieval only", "Retrieval + LLM"),
            index=1,
            help=("Retrieval only never calls the configured LLM provider."),
        )

        suggestion = st.selectbox(
            "Try an example",
            options=("", *SUGGESTED_QUERIES),
            format_func=lambda value: value or "Choose an example",
        )

    return RunControls(
        algorithm=algorithm,
        top_k=top_k,
        use_rag=mode == "Retrieval + LLM",
        suggestion=suggestion or None,
    )
