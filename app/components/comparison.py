"""Read-only visualization of persisted FastContext experiment reports."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

PERFORMANCE_COLUMNS = {
    "retrieval_time_ms_mean": "Retrieval mean (ms)",
    "total_time_ms_mean": "Total mean (ms)",
    "sorting_time_ms_mean": "Ranking mean (ms)",
    "comparisons_mean": "Comparisons mean",
    "chunks_scored_mean": "Chunks scored mean",
    "candidates_found_mean": "Candidates found mean",
    "peak_memory_mb_mean": "Peak memory mean (MB)",
}

SORTING_COLUMNS = {
    "time_ms_mean": "Mean time (ms)",
    "comparisons_mean": "Mean key comparisons",
}


@st.cache_data
def load_experiment_reports(
    performance_path: str,
    sorting_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load persisted reports without running or modifying experiments."""

    performance = _read_report(Path(performance_path))
    sorting = _read_report(Path(sorting_path))
    return performance, sorting


def render_experiment_comparison() -> None:
    """Render didactic charts from the reports created by experiment scripts."""

    repository_root = Path(__file__).resolve().parents[2]
    performance_path = (
        repository_root / "reports" / "tables" / "performance_by_algorithm.csv"
    )
    sorting_path = repository_root / "reports" / "tables" / "sorting_comparison.csv"
    performance, sorting = load_experiment_reports(
        str(performance_path),
        str(sorting_path),
    )

    st.markdown("## Experimental comparison")
    st.caption(
        "Read-only visualizations of persisted experiment reports. "
        "The interface never executes or rewrites experiments."
    )

    retrieval_tab, sorting_tab = st.tabs(("Retrieval strategies", "Sorting behavior"))
    with retrieval_tab:
        _render_retrieval_comparison(performance)
    with sorting_tab:
        _render_sorting_comparison(sorting)


def _render_retrieval_comparison(performance: pd.DataFrame) -> None:
    """Render comparison charts for the registered retrieval strategies."""

    if performance.empty:
        st.info(
            "No retrieval comparison report is available yet. Run the existing "
            "experiment workflow to generate `performance_by_algorithm.csv`."
        )
        return

    prepared = _prepare_numeric_columns(performance, PERFORMANCE_COLUMNS)
    st.markdown(
        "<div class='fc-analysis-note'>Compare strategies on the same persisted "
        "experimental run. Lower time and fewer comparisons indicate less measured "
        "work; they do not, by themselves, establish retrieval quality.</div>",
        unsafe_allow_html=True,
    )

    time_chart, work_chart = st.columns(2, gap="large")
    with time_chart, st.container(border=True):
        _render_bar_chart(
            prepared,
            metric="total_time_ms_mean",
            title="End-to-end retrieval time",
            unavailable="No total-time readings are available in this report.",
        )
    with work_chart, st.container(border=True):
        _render_bar_chart(
            prepared,
            metric="comparisons_mean",
            title="Measured key comparisons",
            unavailable=(
                "No comparison counts are available. Semantic retrieval is omitted "
                "when it does not instrument comparisons."
            ),
        )

    available_metrics = {
        label: field
        for field, label in PERFORMANCE_COLUMNS.items()
        if field in prepared.columns and prepared[field].notna().any()
    }
    if available_metrics:
        selected_label = st.selectbox(
            "Explore another reported metric",
            options=tuple(available_metrics),
            key="comparison_metric",
        )
        with st.container(border=True):
            _render_bar_chart(
                prepared,
                metric=available_metrics[selected_label],
                title=selected_label,
                unavailable="This metric was not reported for the selected experiment.",
            )

    with st.container(border=True):
        st.markdown("#### Reported means")
        st.caption("N/A means the metric was not measured or does not apply.")
        st.dataframe(
            _build_performance_table(prepared),
            width="stretch",
            hide_index=True,
            height="auto",
        )


def _render_sorting_comparison(sorting: pd.DataFrame) -> None:
    """Render Merge Sort versus Quick Sort curves by input order and size."""

    if sorting.empty:
        st.info(
            "No sorting comparison report is available yet. Run the existing "
            "sorting experiment workflow to generate `sorting_comparison.csv`."
        )
        return

    prepared = _prepare_numeric_columns(sorting, SORTING_COLUMNS)
    scenarios = tuple(sorted(prepared["scenario"].dropna().unique()))
    if not scenarios:
        st.info("The sorting report does not contain an input-order scenario.")
        return

    scenario = st.selectbox(
        "Input-order scenario",
        options=scenarios,
        format_func=lambda value: value.replace("_", " ").title(),
        key="sorting_scenario",
    )
    selected = prepared.loc[prepared["scenario"] == scenario]
    st.markdown(
        "<div class='fc-analysis-note'>This is a separate sorting benchmark. "
        "It compares Merge Sort and Quick Sort across input sizes under one input "
        "order; it is not a retrieval-quality experiment.</div>",
        unsafe_allow_html=True,
    )

    time_chart, comparison_chart = st.columns(2, gap="large")
    with time_chart, st.container(border=True):
        _render_sorting_line_chart(
            selected,
            metric="time_ms_mean",
            title="Mean sorting time by input size",
        )
    with comparison_chart, st.container(border=True):
        _render_sorting_line_chart(
            selected,
            metric="comparisons_mean",
            title="Key comparisons by input size",
        )

    st.caption(
        "Each point is a reported mean. Use the scenario selector to inspect "
        "already sorted, reverse sorted, random, and tie-heavy inputs."
    )


def _read_report(path: Path) -> pd.DataFrame:
    """Return an empty table when an optional report has not been generated."""

    if not path.is_file():
        return pd.DataFrame()

    return pd.read_csv(path)


def _prepare_numeric_columns(
    frame: pd.DataFrame,
    columns: dict[str, str],
) -> pd.DataFrame:
    """Coerce report fields without turning missing values into zero."""

    prepared = frame.copy()
    for field in columns:
        if field in prepared.columns:
            prepared[field] = pd.to_numeric(prepared[field], errors="coerce")
    return prepared


def _render_bar_chart(
    frame: pd.DataFrame,
    metric: str,
    title: str,
    unavailable: str,
) -> None:
    """Render one compact bar chart from a report metric."""

    if metric not in frame.columns:
        st.info(unavailable)
        return

    chart_data = frame.loc[:, ["algorithm", metric]].dropna()
    if chart_data.empty:
        st.info(unavailable)
        return

    st.markdown(f"#### {title}")
    chart_data = chart_data.sort_values(metric)
    st.bar_chart(
        chart_data.rename(columns={"algorithm": "Strategy"}).set_index("Strategy"),
        height=250,
    )


def _render_sorting_line_chart(
    frame: pd.DataFrame,
    metric: str,
    title: str,
) -> None:
    """Render one sorting metric with input size on the horizontal axis."""

    if metric not in frame.columns:
        st.info("This metric was not reported for the selected scenario.")
        return

    chart_data = frame.pivot(
        index="item_count",
        columns="algorithm",
        values=metric,
    ).sort_index()
    if chart_data.empty:
        st.info("This metric was not reported for the selected scenario.")
        return

    st.markdown(f"#### {title}")
    st.line_chart(chart_data, height=250)


def _build_performance_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Select concise comparison columns and retain unavailable values as N/A."""

    displayed = pd.DataFrame({"Strategy": frame["algorithm"]})
    for field, label in PERFORMANCE_COLUMNS.items():
        if field in frame.columns:
            precision = (
                0
                if field.endswith(("comparisons_mean", "scored_mean", "found_mean"))
                else 2
            )
            displayed[label] = frame[field].map(
                lambda value, report_precision=precision: _format_report_value(
                    value,
                    report_precision,
                )
            )

    return displayed


def _format_report_value(value: object, precision: int) -> str:
    """Format a report value without mixing numeric and string table columns."""

    if pd.isna(value):
        return "N/A"

    return f"{float(value):,.{precision}f}"
