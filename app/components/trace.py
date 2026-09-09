"""Didactic presentation of the real retrieval execution flow."""

from __future__ import annotations

import streamlit as st

from src.app.streamlit_support import build_retrieval_trace
from src.retrieval.models import RetrievalResult


def render_retrieval_trace(
    result: RetrievalResult,
) -> None:
    """Render the representation-to-context path for a completed run."""

    st.markdown("### Retrieval path")
    st.caption("The stages below are derived from this result's actual metadata.")

    steps = build_retrieval_trace(result)
    columns = st.columns(len(steps))

    for index, (column, step) in enumerate(zip(columns, steps, strict=True), start=1):
        with column:
            st.markdown(
                f"<div class='fc-trace-index'>{index:02d}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(f"**{step.title}**")
            st.caption(step.detail)
