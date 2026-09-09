"""Compact runtime status presentation."""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.app.models import ApplicationContainer, ApplicationHealthReport


def render_status(
    application: ApplicationContainer,
    report: ApplicationHealthReport,
    semantic_metadata: dict[str, Any] | None,
) -> None:
    """Render non-blocking system health in the sidebar."""

    components = {component.name: component for component in report.components}

    with st.sidebar:
        st.divider()
        st.header("System")
        _render_component("Corpus", _corpus_status(application))
        _render_component("Retrieval", components.get("Retrieval"))
        _render_component(
            "Semantic index",
            _semantic_status(
                application,
                semantic_metadata,
            ),
        )
        _render_component("LLM", components.get("LLM"))
        _render_component("RAG", components.get("RAG"))

        with st.expander("Runtime details"):
            st.caption(f"Provider: {application.settings.default_provider}")
            st.caption(f"Model: {application.settings.ollama.model}")
            fallback = application.settings.ollama.fallback_model or "None"
            st.caption(f"Fallback model: {fallback}")
            for component in report.components:
                st.write(f"**{component.name}:** {component.message}")


def _corpus_status(
    application: ApplicationContainer,
) -> tuple[str, str]:
    """Describe corpus availability without assuming a corpus size."""

    if application.corpus_size == 0:
        return ("Unavailable", "No prepared chunks loaded")

    if application.corpus_size is None:
        return ("Ready", "Externally provided corpus")

    return ("Ready", f"{application.corpus_size:,} chunks")


def _semantic_status(
    application: ApplicationContainer,
    metadata: dict[str, Any] | None,
) -> tuple[str, str]:
    """Describe semantic runtime state from actual retrieval metadata."""

    if not application.registry.contains("semantic"):
        return ("Unavailable", "Semantic retriever is not registered")

    if not metadata:
        return ("Ready", "Initializes on first semantic query")

    source = metadata.get("index_source", "unknown")
    persistence = metadata.get("persistence_status", "unknown")
    return ("Ready", f"{source} index · persistence {persistence}")


def _render_component(
    name: str,
    component: object | tuple[str, str] | None,
) -> None:
    """Render one status line from a health component or tuple."""

    if component is None:
        status, message = "N/A", "Not reported"
    elif isinstance(component, tuple):
        status, message = component
    else:
        status = getattr(component, "status", "unknown").replace("_", " ")
        message = getattr(component, "message", "No details")

    st.markdown(
        f"<div class='fc-status'><strong>{name}</strong> · {status}</div>",
        unsafe_allow_html=True,
    )
    st.caption(message)
