"""Ranked retrieval-result presentation."""

from __future__ import annotations

import streamlit as st

from src.app.streamlit_support import build_chunk_details
from src.retrieval.models import RetrievalResult


def render_results(
    result: RetrievalResult,
) -> None:
    """Render compact ranked cards with content revealed on demand."""

    st.markdown("## Retrieved context")
    result_summary = st.columns((1, 1, 2))
    result_summary[0].metric("Returned", len(result.chunks))
    result_summary[1].metric("Requested top-k", result.top_k)
    result_summary[2].markdown(
        "**Ordering rule**  \nScore descending, then `chunk_id` ascending."
    )

    chunks = build_chunk_details(result)
    if not chunks:
        st.info("No matching chunks were retrieved for this query.")
        return

    st.caption(
        "Open a chunk to inspect its source, token count, and full "
        "documentation context."
    )

    for chunk in chunks:
        title = f"#{chunk['rank']} · score {chunk['score']} · {chunk['section_title']}"
        with st.expander(title, expanded=chunk["rank"] == 1):
            details = st.columns((2, 2, 1))
            details[0].caption("SOURCE")
            details[0].code(chunk["source_path"], language=None)
            details[1].caption("CHUNK ID")
            details[1].code(chunk["chunk_id"], language=None)
            details[2].metric("Tokens", chunk["token_count"])
            st.markdown("##### Documentation excerpt")
            st.markdown(chunk["content"])
