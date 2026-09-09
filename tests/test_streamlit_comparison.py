"""Tests for read-only experiment comparison view models."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.components.comparison import (
    _build_performance_table,
    _prepare_numeric_columns,
    _read_report,
)


def test_comparison_table_preserves_missing_measurements() -> None:
    frame = pd.DataFrame(
        {
            "algorithm": ["optimized", "semantic"],
            "total_time_ms_mean": [18.9525, 41.921],
            "comparisons_mean": [1634.0, None],
        }
    )

    table = _build_performance_table(frame)

    assert table.loc[0, "Total mean (ms)"] == "18.95"
    assert table.loc[0, "Comparisons mean"] == "1,634"
    assert table.loc[1, "Comparisons mean"] == "N/A"


def test_numeric_preparation_coerces_missing_report_cells_without_zero() -> None:
    frame = pd.DataFrame(
        {
            "algorithm": ["semantic"],
            "comparisons_mean": [""],
        }
    )

    prepared = _prepare_numeric_columns(
        frame,
        {"comparisons_mean": "Comparisons mean"},
    )

    assert pd.isna(prepared.loc[0, "comparisons_mean"])


def test_missing_optional_report_loads_as_empty_table(tmp_path: Path) -> None:
    report = _read_report(tmp_path / "not-generated.csv")

    assert report.empty
