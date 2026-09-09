"""Page framing and dashboard visual styling."""

from __future__ import annotations

import streamlit as st


def configure_page() -> None:
    """Set page metadata before other Streamlit commands."""

    st.set_page_config(
        page_title="FastContext",
        page_icon="FC",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def render_header(
    corpus_size: int | None,
) -> None:
    """Render the product header and corpus summary."""

    st.markdown(
        "<p class='fc-eyebrow'>FASTCONTEXT DASHBOARD</p>",
        unsafe_allow_html=True,
    )
    st.title("Context retrieval analytics")
    st.markdown(
        "<p class='fc-lead'>Ask the FastAPI documentation, inspect the answer's "
        "evidence, and compare the retrieval strategies in one workspace.</p>",
        unsafe_allow_html=True,
    )

    if corpus_size is None:
        corpus_text = "FastAPI docs 0.141.0 · corpus size unavailable"
    else:
        corpus_text = f"FastAPI docs 0.141.0 · {corpus_size:,} prepared chunks"

    summary = st.columns(4, gap="medium")
    summary[0].metric("Prepared corpus", corpus_size or "N/A")
    summary[0].caption(corpus_text)
    summary[1].metric("Retrieval strategies", "4")
    summary[1].caption("Linear · Indexed · Optimized · Semantic")
    summary[2].metric("Default experience", "Answer")
    summary[2].caption("Retrieval + LLM, with safe retrieval fallback")
    summary[3].metric("Ranking rule", "Deterministic")
    summary[3].caption("Score DESC · chunk_id ASC")

    st.divider()


def inject_styles() -> None:
    """Apply a composed scientific-dashboard visual treatment."""

    st.markdown(
        """
        <style>
        .stApp {background: #f7f8fb; color: #151b2c;}
        .block-container {
            max-width: 1540px;
            padding-top: 1.65rem;
            padding-bottom: 3rem;
        }
        [data-testid="stSidebar"] {
            background: #f1f4f9;
            border-right: 1px solid #e1e6ef;
            box-shadow: none;
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
            color: #536176;
        }
        [data-testid="stSidebar"] .fc-eyebrow {color: #e11d48;}
        [data-testid="stSidebar"] [data-baseweb="select"] > div {
            background: #ffffff;
            color: #151b2c;
            border-color: #e1e6ef;
        }
        [data-testid="stSidebar"] .block-container {padding-top: 1.35rem;}
        h1 {
            color: #151b2c;
            font-size: clamp(2rem, 3.2vw, 2.7rem);
            letter-spacing: -0.045em;
            margin-bottom: 0.1rem;
        }
        h2, h3 {color: #151b2c; letter-spacing: -0.026em;}
        .fc-eyebrow {
            color: #e11d48;
            font-size: 0.7rem;
            font-weight: 800;
            letter-spacing: 0.13em;
            margin: 0 0 0.45rem;
        }
        .fc-lead {
            color: #677386;
            font-size: 1rem;
            line-height: 1.6;
            margin: 0 0 1.35rem;
            max-width: 58rem;
        }
        [data-testid="stForm"] {
            background: #ffffff;
            border: 1px solid #e1e6ef;
            border-radius: 14px;
            box-shadow: 0 7px 18px rgba(15, 23, 42, 0.055);
            padding: 1.25rem 1.25rem 0.55rem;
        }
        [data-testid="stMetric"] {
            border: 1px solid #e1e6ef;
            border-radius: 14px;
            padding: 0.95rem 1rem;
            background: #ffffff;
            min-height: 6rem;
            box-shadow: 0 5px 14px rgba(15, 23, 42, 0.035);
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: #ffffff;
            border-color: #e1e6ef;
            border-radius: 14px;
            box-shadow: 0 7px 18px rgba(15, 23, 42, 0.055);
        }
        [data-testid="stMetricLabel"] {color: #69758a; font-size: 0.8rem;}
        [data-testid="stMetricValue"] {color: #151b2c; font-size: 1.35rem;}
        button[data-baseweb="tab"] {
            color: #69758a;
            font-weight: 700;
        }
        button[data-baseweb="tab"][aria-selected="true"] {color: #e11d48;}
        .fc-side-active, .fc-side-link {
            border-radius: 10px;
            font-size: 0.9rem;
            font-weight: 700;
            margin: 0.32rem 0;
            padding: 0.68rem 0.78rem;
        }
        .fc-side-active {
            background: #e11d48;
            color: #ffffff;
        }
        .fc-side-link {color: #667387;}
        .fc-trace-index {
            color: #e11d48;
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            border-top: 2px solid #f9a8bd;
            padding-top: 0.45rem;
            margin-bottom: 0.35rem;
        }
        .fc-analysis-note {
            border-left: 3px solid #e11d48;
            background: #fff5f7;
            color: #525d70;
            padding: 0.75rem 0.9rem;
            margin: 0.25rem 0 1.2rem;
            font-size: 0.9rem;
        }
        .fc-chunk-meta {color: #5b6472; font-size: 0.84rem; padding-bottom: 0.25rem;}
        .fc-status {font-size: 0.85rem; padding: 0.2rem 0;}
        </style>
        """,
        unsafe_allow_html=True,
    )
