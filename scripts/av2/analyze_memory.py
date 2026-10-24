
"""Analyze AV2 memory measurements and join them with timing results."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SUMMARY_PATH = (
    PROJECT_ROOT / "experiments" / "av2" / "final" / "summary.csv"
)

MEMORY_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "av2"
    / "memory_probe"
    / "memory_results.csv"
)

TABLES_DIR = PROJECT_ROOT / "reports" / "tables" / "av2"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures" / "av2"

ALGORITHMS = ("naive", "kmp", "boyer-moore", "rabin-karp")

TIME_REQUIRED = {
    "scenario_id",
    "algorithm",
    "pattern_id",
    "scale",
    "corpus_fraction",
    "total_characters",
    "mean_time_ms",
    "std_time_ms",
    "repetitions",
}

MEMORY_REQUIRED = {
    "scenario_id",
    "algorithm",
    "peak_memory_bytes",
    "peak_memory_mb",
}


def read_csv(
    path: Path,
    required: set[str],
) -> list[dict[str, str]]:
    """Read a CSV file and validate its required columns."""

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open(
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        columns = set(reader.fieldnames or [])
        missing = required - columns

        if missing:
            raise ValueError(
                f"Missing columns in {path.name}: {sorted(missing)}"
            )

        rows = list(reader)

    if not rows:
        raise ValueError(f"Empty CSV file: {path}")

    return rows


def join_results() -> list[dict[str, Any]]:
    """Combine timing and memory by scenario and algorithm."""

    summary = read_csv(SUMMARY_PATH, TIME_REQUIRED)
    memory = read_csv(MEMORY_PATH, MEMORY_REQUIRED)

    memory_index: dict[
        tuple[str, str],
        dict[str, str],
    ] = {}

    for item in memory:
        key = (
            item["scenario_id"],
            item["algorithm"],
        )

        if key in memory_index:
            raise ValueError(
                f"Duplicate memory measurement: {key}"
            )

        memory_index[key] = item

    timing_keys: set[tuple[str, str]] = set()
    results: list[dict[str, Any]] = []

    for item in summary:
        key = (
            item["scenario_id"],
            item["algorithm"],
        )

        if key in timing_keys:
            raise ValueError(
                f"Duplicate timing measurement: {key}"
            )

        timing_keys.add(key)

        if key not in memory_index:
            raise ValueError(
                f"Missing memory measurement for {key}"
            )

        mem = memory_index[key]

        peak_bytes = int(mem["peak_memory_bytes"])
        peak_mib = peak_bytes / (1024 * 1024)

        if peak_bytes < 0:
            raise ValueError(
                f"Invalid memory measurement for {key}"
            )

        if abs(
            peak_mib - float(mem["peak_memory_mb"])
        ) > 0.000002:
            raise ValueError(
                f"Inconsistent memory units for {key}"
            )

        results.append(
            {
                "scenario_id": item["scenario_id"],
                "pattern_id": item["pattern_id"],
                "scale": item["scale"],
                "corpus_fraction": float(
                    item["corpus_fraction"]
                ),
                "algorithm": item["algorithm"],
                "total_characters": int(
                    item["total_characters"]
                ),
                "repetitions": int(item["repetitions"]),
                "mean_time_ms": float(
                    item["mean_time_ms"]
                ),
                "std_time_ms": float(
                    item["std_time_ms"]
                ),
                "peak_memory_bytes": peak_bytes,
                "peak_memory_mib": round(
                    peak_mib, 6
                ),
            }
        )

    if timing_keys != set(memory_index):
        raise ValueError(
            "Timing and memory experiments use different keys."
        )

    if len(results) != 96:
        raise ValueError(
            f"Expected 96 combinations, found {len(results)}."
        )

    return results


def save_csv(
    rows: list[dict[str, Any]],
    path: Path,
) -> None:
    """Export consolidated measurements."""

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_memory(
    rows: list[dict[str, Any]],
    *,
    scale_type: str,
    filename: str,
    title: str,
) -> None:
    """Plot peak allocations for full-scale inputs."""

    selected = [
        row
        for row in rows
        if row["scale"] == scale_type
        and row["corpus_fraction"] == 1.0
    ]

    if not selected:
        raise ValueError(
            f"No 100% results for scale={scale_type}"
        )

    patterns = sorted({
        str(row["pattern_id"])
        for row in selected
    })

    x = list(range(len(patterns)))
    width = 0.2

    fig, ax = plt.subplots(
        figsize=(max(10, len(patterns) * 1.8), 6)
    )

    for index, algorithm in enumerate(ALGORITHMS):
        values: list[float] = []

        for pattern_id in patterns:
            matches = [
                row for row in selected
                if row["algorithm"] == algorithm
                and row["pattern_id"] == pattern_id
            ]

            if len(matches) != 1:
                raise ValueError(
                    f"Expected one result for "
                    f"{pattern_id}/{algorithm}"
                )

            values.append(
                float(matches[0]["peak_memory_mib"])
            )

        positions = [
            number + (index - 1.5) * width
            for number in x
        ]

        ax.bar(
            positions,
            values,
            width=width,
            label=algorithm,
        )

    ax.set_xticks(
        x,
        patterns,
        rotation=30,
        ha="right",
    )

    ax.set_title(title)
    ax.set_xlabel("Padrão")
    ax.set_ylabel(
        "Pico de alocações rastreadas (MiB)"
    )
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    fig.text(
        0.5,
        0.01,
        "tracemalloc: inclui objetos de resultado; "
        "não representa RAM total do processo.",
        ha="center",
        fontsize=9,
    )

    fig.tight_layout(
        rect=(0, 0.04, 1, 1)
    )

    fig.savefig(
        FIGURES_DIR / filename,
        dpi=180,
    )

    plt.close(fig)


def save_report(
    rows: list[dict[str, Any]],
) -> None:
    """Write an evidence-based memory interpretation."""

    oauth = [
        row for row in rows
        if row["pattern_id"] == "P-OAUTH2"
        and row["corpus_fraction"] == 1.0
    ]

    if len(oauth) != 4:
        raise ValueError(
            "Expected four full-scale OAuth2 measurements."
        )

    oauth_by_algorithm = {
        str(row["algorithm"]): row
        for row in oauth
    }

    baseline = float(
        oauth_by_algorithm["naive"]["peak_memory_mib"]
    )

    lines = [
        "# AV2 — Análise de memória",
        "",
        "## Experimento",
        "",
        "- Corpus: FastAPI 0.141.0.",
        "- Cenários: 24.",
        "- Algoritmos: 4.",
        "- Combinações de memória: 96.",
        "- Medição: Python tracemalloc.",
        "- Memória medida separadamente dos tempos.",
        "",
        "## Padrão OAuth2 — 100% do corpus",
        "",
        "| Algoritmo | Pico (MiB) | Diferença vs Naive |",
        "|---|---:|---:|",
    ]

    for algorithm in ALGORITHMS:
        row = oauth_by_algorithm[algorithm]
        peak = float(row["peak_memory_mib"])
        difference = (peak / baseline - 1) * 100

        lines.append(
            f"| {algorithm} | {peak:.6f} | "
            f"{difference:+.2f}% |"
        )

    lines += [
        "",
        "## Interpretação",
        "",
        "Os valores representam picos de alocações rastreadas "
        "pelo Python em execuções específicas.",
        "",
        "As medições incluem objetos produzidos pelo executor, "
        "como estruturas de resultados e listas de posições. "
        "Os textos de entrada já estavam carregados antes "
        "do início do rastreamento.",
        "",
        "Portanto, não se trata de uma medição isolada "
        "da memória auxiliar dos algoritmos.",
        "",
        "O tracemalloc também não mede toda a memória residente "
        "do processo nem, necessariamente, alocações nativas.",
        "",
        "O custo espacial teórico deve ser discutido "
        "separadamente, a partir das estruturas de dados "
        "e das análises de complexidade.",
        "",
        "O executor conserva todos os resultados de cada "
        "cenário durante a medição, de modo que cenários "
        "com muitas ocorrências podem apresentar "
        "picos maiores.",
        "",
        "Os casos sintéticos foram apresentados em gráfico "
        "separado porque têm distribuições e quantidades "
        "de ocorrências distintas do corpus real.",
        "",
        "A memória foi medida uma vez por combinação. "
        "Não foram calculados desvios-padrão ou "
        "intervalos de confiança para essa métrica.",
        "",
        "## Arquivos",
        "",
        "- memory_real_patterns.png",
        "- memory_synthetic_patterns.png",
        "- memory_analysis.csv",
        "",
    ]

    (TABLES_DIR / "memory_analysis.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    """Generate memory figures and documentation."""

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = join_results()

    save_csv(
        rows,
        TABLES_DIR / "memory_analysis.csv",
    )

    plot_memory(
        rows,
        scale_type="corpus",
        filename="memory_real_patterns.png",
        title="AV2 — Memória nos padrões reais (100%)",
    )

    plot_memory(
        rows,
        scale_type="synthetic_microbenchmark",
        filename="memory_synthetic_patterns.png",
        title="AV2 — Memória em entradas sintéticas (100%)",
    )

    save_report(rows)

    print("AV2 MEMORY ANALYSIS COMPLETED")
    print(f"Validated combinations: {len(rows)}")
    print(f"Figures: {FIGURES_DIR}")
    print(f"Tables: {TABLES_DIR}")


if __name__ == "__main__":
    main()
