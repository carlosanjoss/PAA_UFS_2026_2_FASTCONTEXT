"""Retrieval metrics presentation."""

from __future__ import annotations

import streamlit as st

from components.trace import render_retrieval_trace
from src.app.streamlit_support import (
    build_metric_rows,
    format_duration_ns,
    format_optional_number,
)
from src.retrieval.models import RetrievalResult


def render_metrics(
    result: RetrievalResult,
) -> None:
    """Render available backend measurements without inventing values."""

    metrics = result.metrics
    st.markdown("## Run analysis")
    st.markdown(
        "<div class='fc-analysis-note'>Read the run in three layers: total "
        "latency, computational work, and the detailed instrumentation below.</div>",
        unsafe_allow_html=True,
    )

    render_retrieval_trace(result)
    st.divider()

    latency, work = st.columns(2, gap="large")
    with latency:
        st.markdown("#### Latency")
        timing_columns = st.columns(3)
        timing_columns[0].metric(
            "Retrieval",
            format_duration_ns(metrics.retrieval_time_ns),
        )
        timing_columns[1].metric(
            "Ranking",
            format_duration_ns(metrics.sorting_time_ns),
        )
        timing_columns[2].metric(
            "Index build",
            format_duration_ns(metrics.index_build_time_ns),
        )

    with work:
        st.markdown("#### Computational work")
        work_columns = st.columns(3)
        work_columns[0].metric(
            "Candidates",
            format_optional_number(metrics.candidates_found),
        )
        work_columns[1].metric(
            "Chunks scored",
            format_optional_number(metrics.chunks_scored),
        )
        work_columns[2].metric(
            "Comparisons",
            format_optional_number(metrics.comparisons),
        )

    st.markdown("#### Instrumentation detail")
    st.caption(
        "N/A means the metric was not produced by the selected strategy; "
        "it is not zero."
    )
    st.dataframe(
        build_metric_rows(result),
        width="stretch",
        hide_index=True,
        height="auto",
    )
