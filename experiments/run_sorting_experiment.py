"""Run the FastContext Merge Sort versus Quick Sort experiment."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import yaml

from src.evaluation.sorting_benchmark import (
    SORTING_ALGORITHMS,
    SORTING_SCENARIOS,
    generate_sort_keys,
    run_sorting_algorithm,
)
from src.evaluation.statistics import calculate_statistics
from src.utils.config import PROJECT_ROOT

CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "sorting_experiment.yaml"
)

RAW_FIELDNAMES = (
    "algorithm",
    "scenario",
    "item_count",
    "repetition",
    "random_seed",
    "input_fingerprint",
    "elapsed_time_ns",
    "elapsed_time_ms",
    "comparisons",
)

SUMMARY_FIELDNAMES = (
    "algorithm",
    "scenario",
    "item_count",
    "sample_count",
    "time_ms_mean",
    "time_ms_median",
    "time_ms_std",
    "time_ms_min",
    "time_ms_max",
    "comparisons_mean",
    "comparisons_median",
    "comparisons_std",
    "comparisons_min",
    "comparisons_max",
)


def build_parser() -> argparse.ArgumentParser:
    """Create the sorting experiment CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare FastContext Merge Sort "
            "and Quick Sort implementations."
        )
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run one size, one scenario, "
            "and one repetition."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Overwrite previous sorting "
            "experiment results."
        ),
    )

    return parser


def main() -> int:
    """Run the sorting comparison experiment."""
    args = (
        build_parser()
        .parse_args()
    )

    config = _load_config()

    experiment = config[
        "sorting_experiment"
    ]

    seed = int(
        experiment[
            "reproducibility"
        ][
            "random_seed"
        ]
    )

    algorithms = tuple(
        str(
            algorithm
        )
        for algorithm
        in experiment[
            "algorithms"
        ]
    )

    input_sizes = tuple(
        int(
            value
        )
        for value
        in experiment[
            "input_sizes"
        ]
    )

    scenarios = tuple(
        str(
            scenario
        )
        for scenario
        in experiment[
            "scenarios"
        ]
    )

    repetitions = int(
        experiment[
            "repetitions"
        ]
    )

    warmup_config = experiment[
        "warmup"
    ]

    warmup_repetitions = (
        int(
            warmup_config[
                "repetitions"
            ]
        )
        if bool(
            warmup_config[
                "enabled"
            ]
        )
        else 0
    )

    output_config = experiment[
        "output"
    ]

    plots_config = experiment[
        "plots"
    ]

    raw_path = _resolve_path(
        output_config[
            "raw_results"
        ]
    )

    summary_path = _resolve_path(
        output_config[
            "summary"
        ]
    )

    table_path = _resolve_path(
        output_config[
            "table"
        ]
    )

    figures_directory = (
        _resolve_path(
            output_config[
                "figures_directory"
            ]
        )
    )

    _validate_configuration(
        algorithms=algorithms,
        input_sizes=input_sizes,
        scenarios=scenarios,
        repetitions=repetitions,
    )

    if args.smoke:
        input_sizes = (
            input_sizes[
                0
            ],
        )

        scenarios = (
            scenarios[
                0
            ],
        )

        repetitions = 1

    output_paths = (
        raw_path,
        summary_path,
        table_path,
    )

    if args.force:
        for path in output_paths:
            path.unlink(
                missing_ok=True
            )

    elif any(
        path.exists()
        for path in output_paths
    ):
        raise FileExistsError(
            "Sorting experiment output already exists. "
            "Use --force to overwrite it."
        )

    raw_rows: list[
        dict[
            str,
            object,
        ]
    ] = []

    measured_runs = (
        len(
            algorithms
        )
        * len(
            input_sizes
        )
        * len(
            scenarios
        )
        * repetitions
    )

    print(
        "FastContext Sorting Experiment"
    )

    print(
        "=" * 60
    )

    print(
        "Algorithms: "
        + ", ".join(
            algorithms
        )
    )

    print(
        f"Input sizes: {input_sizes}"
    )

    print(
        "Scenarios: "
        + ", ".join(
            scenarios
        )
    )

    print(
        f"Repetitions: {repetitions}"
    )

    print(
        f"Warm-up repetitions: {warmup_repetitions}"
    )

    print(
        f"Measured sorting runs: {measured_runs}"
    )

    print()

    completed = 0

    for size_index, item_count in enumerate(
        input_sizes
    ):
        for scenario_index, scenario in enumerate(
            scenarios
        ):
            print(
                f"N={item_count}, "
                f"scenario={scenario}"
            )

            for warmup_index in range(
                warmup_repetitions
            ):
                warmup_seed = (
                    seed
                    + item_count * 10_000
                    + scenario_index * 1_000
                    + warmup_index
                )

                warmup_keys = (
                    generate_sort_keys(
                        item_count,
                        scenario=scenario,
                        seed=warmup_seed,
                    )
                )

                for algorithm in algorithms:
                    run_sorting_algorithm(
                        algorithm,
                        warmup_keys,
                    )

            for repetition in range(
                1,
                repetitions + 1,
            ):
                case_seed = (
                    seed
                    + item_count * 100_000
                    + size_index * 10_000
                    + scenario_index * 1_000
                    + repetition
                )

                keys = generate_sort_keys(
                    item_count,
                    scenario=scenario,
                    seed=case_seed,
                )

                algorithm_order = (
                    algorithms
                    if repetition % 2 == 1
                    else tuple(
                        reversed(
                            algorithms
                        )
                    )
                )

                executions = {}

                for algorithm in algorithm_order:
                    execution = (
                        run_sorting_algorithm(
                            algorithm,
                            keys,
                        )
                    )

                    executions[
                        algorithm
                    ] = execution

                fingerprints = {
                    execution.input_fingerprint
                    for execution
                    in executions.values()
                }

                if len(
                    fingerprints
                ) != 1:
                    raise RuntimeError(
                        "Sorting algorithms did not "
                        "receive identical inputs."
                    )

                for algorithm in algorithms:
                    execution = executions[
                        algorithm
                    ]

                    raw_rows.append(
                        {
                            "algorithm": (
                                algorithm
                            ),
                            "scenario": (
                                scenario
                            ),
                            "item_count": (
                                item_count
                            ),
                            "repetition": (
                                repetition
                            ),
                            "random_seed": (
                                case_seed
                            ),
                            "input_fingerprint": (
                                execution
                                .input_fingerprint
                            ),
                            "elapsed_time_ns": (
                                execution
                                .elapsed_time_ns
                            ),
                            "elapsed_time_ms": (
                                execution
                                .elapsed_time_ns
                                / 1_000_000
                            ),
                            "comparisons": (
                                execution
                                .comparisons
                            ),
                        }
                    )

                    completed += 1

            print(
                f"  progress: "
                f"{completed}/{measured_runs}"
            )

    summary_rows = (
        _summarize(
            raw_rows
        )
    )

    _write_csv(
        raw_path,
        raw_rows,
        RAW_FIELDNAMES,
    )

    _write_csv(
        summary_path,
        summary_rows,
        SUMMARY_FIELDNAMES,
    )

    _write_csv(
        table_path,
        summary_rows,
        SUMMARY_FIELDNAMES,
    )

    figures_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    time_plot = (
        figures_directory
        / str(
            plots_config[
                "merge_vs_quick_time"
            ]
        )
    )

    comparisons_plot = (
        figures_directory
        / str(
            plots_config[
                "merge_vs_quick_comparisons"
            ]
        )
    )

    sensitivity_plot = (
        figures_directory
        / str(
            plots_config[
                "quick_input_sensitivity"
            ]
        )
    )

    _plot_overall_metric(
        summary_rows,
        metric="time_ms_mean",
        ylabel="Mean sorting time (ms)",
        title=(
            "Merge Sort vs Quick Sort "
            "— mean execution time"
        ),
        output_path=time_plot,
    )

    _plot_overall_metric(
        summary_rows,
        metric="comparisons_mean",
        ylabel="Mean comparisons",
        title=(
            "Merge Sort vs Quick Sort "
            "— key comparisons"
        ),
        output_path=(
            comparisons_plot
        ),
    )

    _plot_quick_sensitivity(
        summary_rows,
        output_path=(
            sensitivity_plot
        ),
    )

    print()

    print(
        f"Raw rows: {len(raw_rows)}"
    )

    print(
        f"Summary rows: {len(summary_rows)}"
    )

    print(
        f"Raw results: {raw_path}"
    )

    print(
        f"Summary: {summary_path}"
    )

    print(
        f"Table: {table_path}"
    )

    print(
        f"Time figure: {time_plot}"
    )

    print(
        "Comparisons figure: "
        f"{comparisons_plot}"
    )

    print(
        "Quick Sort sensitivity figure: "
        f"{sensitivity_plot}"
    )

    return 0


def _summarize(
    rows: Sequence[
        Mapping[
            str,
            object,
        ]
    ],
) -> list[
    dict[
        str,
        object,
    ]
]:
    """Aggregate repetitions by algorithm, scenario, and N."""
    groups: dict[
        tuple[
            str,
            str,
            int,
        ],
        list[
            Mapping[
                str,
                object,
            ]
        ],
    ] = defaultdict(
        list
    )

    for row in rows:
        key = (
            str(
                row[
                    "algorithm"
                ]
            ),
            str(
                row[
                    "scenario"
                ]
            ),
            int(
                row[
                    "item_count"
                ]
            ),
        )

        groups[
            key
        ].append(
            row
        )

    summary: list[
        dict[
            str,
            object,
        ]
    ] = []

    for key, group in sorted(
        groups.items(),
        key=lambda item: (
            item[
                0
            ][
                2
            ],
            item[
                0
            ][
                1
            ],
            item[
                0
            ][
                0
            ],
        ),
    ):
        (
            algorithm,
            scenario,
            item_count,
        ) = key

        times = [
            float(
                row[
                    "elapsed_time_ms"
                ]
            )
            for row in group
        ]

        comparisons = [
            float(
                row[
                    "comparisons"
                ]
            )
            for row in group
        ]

        time_stats = (
            calculate_statistics(
                times
            )
        )

        comparison_stats = (
            calculate_statistics(
                comparisons
            )
        )

        summary.append(
            {
                "algorithm": (
                    algorithm
                ),
                "scenario": (
                    scenario
                ),
                "item_count": (
                    item_count
                ),
                "sample_count": (
                    len(
                        group
                    )
                ),
                "time_ms_mean": (
                    time_stats.mean
                ),
                "time_ms_median": (
                    time_stats.median
                ),
                "time_ms_std": (
                    time_stats.std
                ),
                "time_ms_min": (
                    time_stats.minimum
                ),
                "time_ms_max": (
                    time_stats.maximum
                ),
                "comparisons_mean": (
                    comparison_stats.mean
                ),
                "comparisons_median": (
                    comparison_stats.median
                ),
                "comparisons_std": (
                    comparison_stats.std
                ),
                "comparisons_min": (
                    comparison_stats.minimum
                ),
                "comparisons_max": (
                    comparison_stats.maximum
                ),
            }
        )

    return summary


def _plot_overall_metric(
    rows: Sequence[
        Mapping[
            str,
            object,
        ]
    ],
    *,
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    """Plot mean metric across scenarios for each algorithm and N."""
    grouped: dict[
        tuple[
            str,
            int,
        ],
        list[float],
    ] = defaultdict(
        list
    )

    for row in rows:
        grouped[
            (
                str(
                    row[
                        "algorithm"
                    ]
                ),
                int(
                    row[
                        "item_count"
                    ]
                ),
            )
        ].append(
            float(
                row[
                    metric
                ]
            )
        )

    figure, axis = (
        plt.subplots()
    )

    for algorithm in SORTING_ALGORITHMS:
        points: list[
            tuple[
                int,
                float,
            ]
        ] = []

        for (
            grouped_algorithm,
            item_count,
        ), values in grouped.items():
            if (
                grouped_algorithm
                != algorithm
            ):
                continue

            points.append(
                (
                    item_count,
                    sum(
                        values
                    )
                    / len(
                        values
                    ),
                )
            )

        points.sort(
            key=lambda item: item[
                0
            ]
        )

        axis.plot(
            [
                item[
                    0
                ]
                for item in points
            ],
            [
                item[
                    1
                ]
                for item in points
            ],
            marker="o",
            label=algorithm,
        )

    axis.set_xlabel(
        "N"
    )

    axis.set_ylabel(
        ylabel
    )

    axis.set_title(
        title
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=160,
    )

    plt.close(
        figure
    )


def _plot_quick_sensitivity(
    rows: Sequence[
        Mapping[
            str,
            object,
        ]
    ],
    *,
    output_path: Path,
) -> None:
    """Plot Quick Sort time for each input-order scenario."""
    figure, axis = (
        plt.subplots()
    )

    for scenario in SORTING_SCENARIOS:
        points = [
            (
                int(
                    row[
                        "item_count"
                    ]
                ),
                float(
                    row[
                        "time_ms_mean"
                    ]
                ),
            )
            for row in rows
            if (
                row[
                    "algorithm"
                ]
                == "quick"
                and row[
                    "scenario"
                ]
                == scenario
            )
        ]

        points.sort(
            key=lambda item: item[
                0
            ]
        )

        axis.plot(
            [
                point[
                    0
                ]
                for point in points
            ],
            [
                point[
                    1
                ]
                for point in points
            ],
            marker="o",
            label=scenario,
        )

    axis.set_xlabel(
        "N"
    )

    axis.set_ylabel(
        "Mean sorting time (ms)"
    )

    axis.set_title(
        "Quick Sort sensitivity to input order"
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=160,
    )

    plt.close(
        figure
    )


def _write_csv(
    path: Path,
    rows: Sequence[
        Mapping[
            str,
            object,
        ]
    ],
    fieldnames: Sequence[str],
) -> None:
    """Write one experiment CSV."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                fieldnames
            ),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


def _load_config() -> dict[
    str,
    Any,
]:
    """Load the sorting benchmark configuration."""
    data = yaml.safe_load(
        CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        data,
        dict,
    ):
        raise TypeError(
            "Sorting configuration root "
            "must be a mapping."
        )

    experiment = data.get(
        "sorting_experiment"
    )

    if not isinstance(
        experiment,
        dict,
    ):
        raise TypeError(
            "sorting_experiment "
            "must be a mapping."
        )

    return data


def _resolve_path(
    value: object,
) -> Path:
    """Resolve a configured project path."""
    if not isinstance(
        value,
        str,
    ):
        raise TypeError(
            "Configured path must be a string."
        )

    path = Path(
        value
    )

    if path.is_absolute():
        return path

    return (
        PROJECT_ROOT
        / path
    ).resolve()


def _validate_configuration(
    *,
    algorithms: Sequence[str],
    input_sizes: Sequence[int],
    scenarios: Sequence[str],
    repetitions: int,
) -> None:
    """Validate the sorting experiment matrix."""
    if (
        not algorithms
        or set(
            algorithms
        )
        != set(
            SORTING_ALGORITHMS
        )
    ):
        raise ValueError(
            "Sorting algorithms must be "
            "merge and quick."
        )

    if (
        not input_sizes
        or any(
            value <= 0
            for value in input_sizes
        )
    ):
        raise ValueError(
            "Input sizes must be positive."
        )

    if (
        not scenarios
        or (
            set(
                scenarios
            )
            != set(
                SORTING_SCENARIOS
            )
        )
    ):
        raise ValueError(
            "Unexpected sorting scenarios."
        )

    if repetitions <= 0:
        raise ValueError(
            "repetitions must be "
            "greater than zero."
        )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )