"""RAG answer presentation."""

from __future__ import annotations

import streamlit as st

from src.app.streamlit_support import format_duration_ns
from src.services.fastcontext import FastContextResult


def render_answer(
    result: FastContextResult,
) -> None:
    """Render a generated answer, timings, and verified references."""

    rag = result.rag
    st.markdown("## Answer")
    st.caption(
        "A grounded response assembled from the documentation chunks shown below."
    )
    with st.container(border=True):
        st.markdown("### Response")
        st.markdown(rag.answer)
        st.caption(f"Provider: {rag.provider} · Model: {rag.model}")

    timing = st.columns(3)
    timing[0].metric(
        "Retrieval",
        format_duration_ns(result.retrieval_time_ns),
    )
    timing[1].metric(
        "Generation",
        format_duration_ns(result.generation_time_ns),
    )
    timing[2].metric(
        "End-to-end",
        format_duration_ns(result.end_to_end_time_ns),
    )

    quality = st.columns(2)
    quality[0].metric(
        "Citation validation", "Valid" if rag.citation_valid else "Review"
    )
    quality[1].metric("Citation retries", rag.citation_retry_count)

    st.markdown("#### Evidence and validation")
    if rag.valid_citations:
        st.caption("Validated chunk citations used by the response:")
        for citation in rag.valid_citations:
            st.write(f"- `{citation}`")
    else:
        st.info("The response did not contain a validated chunk citation.")

    if rag.invalid_citations:
        st.warning("Unrecognized citations: " + ", ".join(rag.invalid_citations))
