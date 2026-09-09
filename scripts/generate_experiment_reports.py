"""Generate FastContext experiment tables and figures."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from src.evaluation.reporting import (
    PERFORMANCE_TABLE_FIELDS,
    QUALITY_TABLE_FIELDS,
    aggregate_performance,
    aggregate_quality,
    full_corpus_rows,
    load_results_csv,
    numeric_report_value,
    write_csv_table,
)
from src.utils.config import PROJECT_ROOT

DEFAULT_RESULTS_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "raw_results"
    / "results.csv"
)

DEFAULT_FIGURES_DIRECTORY = (
    PROJECT_ROOT
    / "reports"
    / "figures"
)

DEFAULT_TABLES_DIRECTORY = (
    PROJECT_ROOT
    / "reports"
    / "tables"
)

DEFAULT_MANIFEST_PATH = (
    PROJECT_ROOT
    / "reports"
    / "report_manifest.json"
)

ALGORITHM_ORDER = (
    "linear",
    "indexed",
    "optimized",
    "semantic",
)


def build_parser() -> argparse.ArgumentParser:
    """Create the report-generation CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate FastContext experiment "
            "tables and figures."
        )
    )

    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help=(
            "Path to raw results.csv."
        ),
    )

    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=DEFAULT_FIGURES_DIRECTORY,
        help=(
            "Output directory for figures."
        ),
    )

    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=DEFAULT_TABLES_DIRECTORY,
        help=(
            "Output directory for tables."
        ),
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help=(
            "Output path for report manifest."
        ),
    )

    return parser


def main() -> int:
    """Generate tables, plots, and an artifact manifest."""
    args = (
        build_parser()
        .parse_args()
    )

    results_path = _resolve(
        args.results
    )

    figures_directory = (
        _resolve(
            args.figures_dir
        )
    )

    tables_directory = (
        _resolve(
            args.tables_dir
        )
    )

    manifest_path = (
        _resolve(
            args.manifest
        )
    )

    figures_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    tables_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = load_results_csv(
        results_path
    )

    performance = (
        aggregate_performance(
            rows
        )
    )

    quality = (
        aggregate_quality(
            rows
        )
    )

    generated: list[
        str
    ] = []

    skipped: dict[
        str,
        str,
    ] = {}

    performance_table_path = (
        tables_directory
        / "performance_by_algorithm.csv"
    )

    write_csv_table(
        performance_table_path,
        performance,
        fieldnames=(
            PERFORMANCE_TABLE_FIELDS
        ),
    )

    generated.append(
        str(
            performance_table_path
        )
    )

    quality_table_path = (
        tables_directory
        / "quality_by_algorithm.csv"
    )

    write_csv_table(
        quality_table_path,
        quality,
        fieldnames=(
            QUALITY_TABLE_FIELDS
        ),
    )

    generated.append(
        str(
            quality_table_path
        )
    )

    _generate_performance_figures(
        performance,
        figures_directory,
        generated,
        skipped,
    )

    _generate_quality_figures(
        performance,
        quality,
        figures_directory,
        generated,
        skipped,
    )

    skipped[
        "merge_vs_quick"
    ] = (
        "Quick Sort is not part of the "
        "main retrieval results schema. "
        "It requires a separate sorting experiment."
    )

    manifest = {
        "schema_version": 1,
        "generated_at_utc": (
            datetime.now(
                UTC
            ).isoformat()
        ),
        "source_results": str(
            results_path
        ),
        "raw_result_rows": len(
            rows
        ),
        "performance_table_rows": len(
            performance
        ),
        "quality_table_rows": len(
            quality
        ),
        "quality_available": bool(
            quality
        ),
        "generated_artifacts": (
            generated
        ),
        "skipped_artifacts": (
            skipped
        ),
    }

    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "FastContext Experiment Reports"
    )

    print(
        "=" * 60
    )

    print(
        f"Raw result rows: {len(rows)}"
    )

    print(
        "Performance table rows: "
        f"{len(performance)}"
    )

    print(
        "Quality table rows: "
        f"{len(quality)}"
    )

    print()

    print(
        "Generated artifacts:"
    )

    for artifact in generated:
        print(
            f"  {artifact}"
        )

    if skipped:
        print()

        print(
            "Skipped artifacts:"
        )

        for name, reason in (
            skipped.items()
        ):
            print(
                f"  {name}: {reason}"
            )

    print()

    print(
        f"Manifest: {manifest_path}"
    )

    return 0


def _generate_performance_figures(
    rows: list[
        dict[
            str,
            object,
        ]
    ],
    directory: Path,
    generated: list[str],
    skipped: dict[
        str,
        str,
    ],
) -> None:
    """Generate performance-oriented figures."""
    path = (
        directory
        / "corpus_size_vs_time.png"
    )

    if _plot_metric_by_corpus_size(
        rows,
        metric="retrieval_time_ms_mean",
        ylabel="Mean retrieval time (ms)",
        title="Corpus size vs retrieval time",
        output_path=path,
    ):
        generated.append(
            str(path)
        )
    else:
        skipped[
            "corpus_size_vs_time"
        ] = (
            "No retrieval-time values were available."
        )

    memory_path = (
        directory
        / "corpus_size_vs_memory.png"
    )

    if _plot_metric_by_corpus_size(
        rows,
        metric="peak_memory_mb_mean",
        ylabel="Mean incremental peak memory (MiB)",
        title="Corpus size vs approximate peak memory",
        output_path=memory_path,
    ):
        generated.append(
            str(
                memory_path
            )
        )
    else:
        skipped[
            "corpus_size_vs_memory"
        ] = (
            "Memory profiling values were not available."
        )

    algorithm_path = (
        directory
        / "algorithm_vs_time.png"
    )

    if _plot_full_corpus_bar(
        rows,
        metric="retrieval_time_ms_mean",
        ylabel="Mean retrieval time (ms)",
        title="Algorithm vs retrieval time — full corpus",
        output_path=algorithm_path,
    ):
        generated.append(
            str(
                algorithm_path
            )
        )
    else:
        skipped[
            "algorithm_vs_time"
        ] = (
            "Full-corpus retrieval-time values "
            "were not available."
        )

    comparisons_path = (
        directory
        / "comparisons_vs_corpus_size.png"
    )

    if _plot_metric_by_corpus_size(
        rows,
        metric="comparisons_mean",
        ylabel="Mean number of comparisons",
        title="Comparisons vs corpus size",
        output_path=(
            comparisons_path
        ),
    ):
        generated.append(
            str(
                comparisons_path
            )
        )
    else:
        skipped[
            "comparisons_vs_corpus_size"
        ] = (
            "Comparison counters were not available."
        )


def _generate_quality_figures(
    performance: list[
        dict[
            str,
            object,
        ]
    ],
    quality: list[
        dict[
            str,
            object,
        ]
    ],
    directory: Path,
    generated: list[str],
    skipped: dict[
        str,
        str,
    ],
) -> None:
    """Generate quality-oriented figures."""
    if not quality:
        reason = (
            "Human ground truth is not complete, "
            "so retrieval quality is unavailable."
        )

        skipped[
            "algorithm_vs_precision"
        ] = reason

        skipped[
            "k_vs_precision"
        ] = reason

        skipped[
            "k_vs_recall"
        ] = reason

        skipped[
            "lexical_vs_semantic"
        ] = reason

        return

    max_k = max(
        int(
            row[
                "k"
            ]
        )
        for row in quality
    )

    precision_path = (
        directory
        / f"algorithm_vs_precision_at_{max_k}.png"
    )

    if _plot_quality_full_corpus_bar(
        quality,
        k=max_k,
        metric="precision_mean",
        ylabel=(
            f"Mean Precision@{max_k}"
        ),
        title=(
            f"Algorithm vs Precision@{max_k} "
            "— full corpus"
        ),
        output_path=(
            precision_path
        ),
    ):
        generated.append(
            str(
                precision_path
            )
        )

    k_precision_path = (
        directory
        / "k_vs_precision.png"
    )

    if _plot_quality_by_k(
        quality,
        metric="precision_mean",
        ylabel="Mean Precision@k",
        title="Top-k vs Precision",
        output_path=(
            k_precision_path
        ),
    ):
        generated.append(
            str(
                k_precision_path
            )
        )

    k_recall_path = (
        directory
        / "k_vs_recall.png"
    )

    if _plot_quality_by_k(
        quality,
        metric="recall_mean",
        ylabel="Mean Recall@k",
        title="Top-k vs Recall",
        output_path=(
            k_recall_path
        ),
    ):
        generated.append(
            str(
                k_recall_path
            )
        )

    tradeoff_path = (
        directory
        / "lexical_vs_semantic_tradeoff.png"
    )

    if _plot_time_quality_tradeoff(
        performance,
        quality,
        k=max_k,
        output_path=(
            tradeoff_path
        ),
    ):
        generated.append(
            str(
                tradeoff_path
            )
        )
    else:
        skipped[
            "lexical_vs_semantic"
        ] = (
            "Time and quality values could "
            "not be matched for the full corpus."
        )


def _plot_metric_by_corpus_size(
    rows: list[
        dict[
            str,
            object,
        ]
    ],
    *,
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> bool:
    """Plot one mean performance metric against corpus size."""
    has_data = False

    fig, ax = plt.subplots()

    for algorithm in (
        ALGORITHM_ORDER
    ):
        points: list[
            tuple[
                int,
                float,
            ]
        ] = []

        for row in rows:
            if (
                row.get(
                    "algorithm"
                )
                != algorithm
            ):
                continue

            value = (
                numeric_report_value(
                    row,
                    metric,
                )
            )

            if value is None:
                continue

            corpus_chunks = int(
                row[
                    "corpus_chunks"
                ]
            )

            points.append(
                (
                    corpus_chunks,
                    value,
                )
            )

        if not points:
            continue

        has_data = True

        points.sort(
            key=lambda item: item[
                0
            ]
        )

        ax.plot(
            [
                point[0]
                for point
                in points
            ],
            [
                point[1]
                for point
                in points
            ],
            marker="o",
            label=algorithm,
        )

    if not has_data:
        plt.close(
            fig
        )

        return False

    ax.set_xlabel(
        "Corpus size (chunks)"
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_title(
        title
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=160,
    )

    plt.close(
        fig
    )

    return True


def _plot_full_corpus_bar(
    rows: list[
        dict[
            str,
            object,
        ]
    ],
    *,
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> bool:
    """Plot one full-corpus metric by algorithm."""
    full_rows = (
        full_corpus_rows(
            rows
        )
    )

    labels: list[
        str
    ] = []

    values: list[
        float
    ] = []

    for algorithm in (
        ALGORITHM_ORDER
    ):
        matching = [
            row
            for row in full_rows
            if row.get(
                "algorithm"
            )
            == algorithm
        ]

        if not matching:
            continue

        value = (
            numeric_report_value(
                matching[
                    0
                ],
                metric,
            )
        )

        if value is None:
            continue

        labels.append(
            algorithm
        )

        values.append(
            value
        )

    if not values:
        return False

    fig, ax = plt.subplots()

    ax.bar(
        labels,
        values,
    )

    ax.set_xlabel(
        "Algorithm"
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_title(
        title
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=160,
    )

    plt.close(
        fig
    )

    return True


def _plot_quality_full_corpus_bar(
    rows: list[
        dict[
            str,
            object,
        ]
    ],
    *,
    k: int,
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> bool:
    """Plot one full-corpus quality metric."""
    full_rows = (
        full_corpus_rows(
            rows
        )
    )

    filtered = [
        row
        for row in full_rows
        if int(
            row[
                "k"
            ]
        )
        == k
    ]

    return _plot_full_corpus_bar(
        filtered,
        metric=metric,
        ylabel=ylabel,
        title=title,
        output_path=(
            output_path
        ),
    )


def _plot_quality_by_k(
    rows: list[
        dict[
            str,
            object,
        ]
    ],
    *,
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> bool:
    """Plot a quality metric as k changes on the full corpus."""
    full_rows = (
        full_corpus_rows(
            rows
        )
    )

    has_data = False

    fig, ax = plt.subplots()

    for algorithm in (
        ALGORITHM_ORDER
    ):
        points: list[
            tuple[
                int,
                float,
            ]
        ] = []

        for row in full_rows:
            if (
                row.get(
                    "algorithm"
                )
                != algorithm
            ):
                continue

            value = (
                numeric_report_value(
                    row,
                    metric,
                )
            )

            if value is None:
                continue

            points.append(
                (
                    int(
                        row[
                            "k"
                        ]
                    ),
                    value,
                )
            )

        if not points:
            continue

        has_data = True

        points.sort(
            key=lambda item: item[
                0
            ]
        )

        ax.plot(
            [
                point[0]
                for point
                in points
            ],
            [
                point[1]
                for point
                in points
            ],
            marker="o",
            label=algorithm,
        )

    if not has_data:
        plt.close(
            fig
        )

        return False

    ax.set_xlabel(
        "k"
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_title(
        title
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=160,
    )

    plt.close(
        fig
    )

    return True


def _plot_time_quality_tradeoff(
    performance: list[
        dict[
            str,
            object,
        ]
    ],
    quality: list[
        dict[
            str,
            object,
        ]
    ],
    *,
    k: int,
    output_path: Path,
) -> bool:
    """Plot retrieval-time versus precision trade-off."""
    performance_full = {
        str(
            row[
                "algorithm"
            ]
        ): row
        for row
        in full_corpus_rows(
            performance
        )
    }

    quality_full = {
        str(
            row[
                "algorithm"
            ]
        ): row
        for row
        in full_corpus_rows(
            quality
        )
        if int(
            row[
                "k"
            ]
        )
        == k
    }

    points: list[
        tuple[
            str,
            float,
            float,
        ]
    ] = []

    for algorithm in (
        ALGORITHM_ORDER
    ):
        performance_row = (
            performance_full.get(
                algorithm
            )
        )

        quality_row = (
            quality_full.get(
                algorithm
            )
        )

        if (
            performance_row is None
            or quality_row is None
        ):
            continue

        time_value = (
            numeric_report_value(
                performance_row,
                "retrieval_time_ms_mean",
            )
        )

        precision_value = (
            numeric_report_value(
                quality_row,
                "precision_mean",
            )
        )

        if (
            time_value is None
            or precision_value is None
        ):
            continue

        points.append(
            (
                algorithm,
                time_value,
                precision_value,
            )
        )

    if not points:
        return False

    fig, ax = plt.subplots()

    for (
        algorithm,
        time_value,
        precision_value,
    ) in points:
        ax.scatter(
            [
                time_value
            ],
            [
                precision_value
            ],
        )

        ax.annotate(
            algorithm,
            (
                time_value,
                precision_value,
            ),
        )

    ax.set_xlabel(
        "Mean retrieval time (ms)"
    )

    ax.set_ylabel(
        f"Mean Precision@{k}"
    )

    ax.set_title(
        "Retrieval time vs quality — full corpus"
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=160,
    )

    plt.close(
        fig
    )

    return True


def _resolve(
    path: Path,
) -> Path:
    """Resolve project-relative paths."""
    if path.is_absolute():
        return path

    return (
        PROJECT_ROOT
        / path
    ).resolve()


if __name__ == "__main__":
    raise SystemExit(
        main()
    )