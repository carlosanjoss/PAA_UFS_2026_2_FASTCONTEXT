
"""Generate reproducible charts for AV2 semantic/literal/hybrid retrieval."""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    ROOT / "experiments" / "av2"
    / "retrieval_comparison" / "final"
)

FIG_DIR = ROOT / "reports" / "figures" / "av2"

REPORT_PATH = (
    ROOT / "reports" / "tables" / "av2"
    / "retrieval_comparison.md"
)

NAMES = {
    "semantic": "Semântica",
    "literal": "Literal",
    "hybrid": "Híbrida",
}

METRICS = {
    "pooled_precision_at_k": "Precision@5 conhecida",
    "pooled_recall_at_k": "Recall@5 conhecido",
    "known_mrr_at_k": "MRR@5 conhecido",
    "known_hit_rate_at_k": "Hit Rate@5 conhecido",
}


def read_csv(
    path: Path,
    required: set[str],
) -> list[dict[str, str]]:
    """Read an existing experiment CSV and validate its schema."""

    with path.open(
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"Colunas ausentes: {path}")

        rows = list(reader)

    if not rows:
        raise ValueError(f"CSV vazio: {path}")

    return rows


def load_results(
) -> tuple[
    dict[str, dict[str, str]],
    list[dict[str, str]],
]:
    """Validate the complete 30-query, 3-condition benchmark."""

    summary_rows = read_csv(
        DATA_DIR / "by_condition.csv",
        {
            "condition",
            "queries",
            "repetitions_per_query",
            "mean_time_ms",
            "mean_precision_lower_bound",
            "mean_precision_upper_bound",
            "unjudged_results",
            *METRICS,
        },
    )

    per_query = read_csv(
        DATA_DIR / "per_query.csv",
        {
            "condition",
            "query_id",
            "pooled_precision_at_k",
            "unjudged_count",
        },
    )

    summary = {
        row["condition"]: row
        for row in summary_rows
    }

    if (
        len(summary_rows) != 3
        or set(summary) != set(NAMES)
    ):
        raise ValueError(
            "Condições esperadas: semantic, literal, hybrid"
        )

    if len(per_query) != 90:
        raise ValueError(
            "Esperados 90 registros por pergunta"
        )

    counts: dict[
        str,
        list[dict[str, str]],
    ] = defaultdict(list)

    keys: set[tuple[str, str]] = set()

    for row in per_query:
        key = (
            row["query_id"],
            row["condition"],
        )

        if key in keys:
            raise ValueError(
                f"Registro duplicado: {key}"
            )

        keys.add(key)
        counts[row["condition"]].append(row)

    questions = {
        row["query_id"]
        for row in per_query
    }

    if len(questions) != 30:
        raise ValueError(
            "Esperadas 30 perguntas distintas"
        )

    for condition in NAMES:
        details = counts[condition]
        item = summary[condition]

        if (
            len(details) != 30
            or int(item["queries"]) != 30
            or int(item["repetitions_per_query"]) != 5
        ):
            raise ValueError(
                f"Matriz incompleta: {condition}"
            )

        total_unjudged = sum(
            int(row["unjudged_count"])
            for row in details
        )

        if total_unjudged != int(item["unjudged_results"]):
            raise ValueError(
                f"Não julgados divergentes: {condition}"
            )

        precision = statistics.mean(
            float(row["pooled_precision_at_k"])
            for row in details
        )

        if (
            abs(
                precision
                - float(item["pooled_precision_at_k"])
            )
            > 1e-9
        ):
            raise ValueError(
                f"Precisão divergente: {condition}"
            )

        for key in METRICS:
            if not 0 <= float(item[key]) <= 1:
                raise ValueError(
                    f"Métrica inválida: {condition}, {key}"
                )

        lower = float(
            item["mean_precision_lower_bound"]
        )
        upper = float(
            item["mean_precision_upper_bound"]
        )

        if not 0 <= lower <= upper <= 1:
            raise ValueError(
                f"Limites de precisão inválidos: {condition}"
            )

    return summary, per_query


def plot_quality(
    summary: dict[str, dict[str, str]],
) -> Path:
    """Plot the four known-label retrieval metrics."""

    fig, ax = plt.subplots(
        figsize=(11, 5.3)
    )

    width = 0.19

    for i, (key, label) in enumerate(
        METRICS.items()
    ):
        ax.bar(
            [
                j + (i - 1.5) * width
                for j in range(3)
            ],
            [
                float(summary[name][key])
                for name in NAMES
            ],
            width=width,
            label=label,
        )

    ax.set_xticks(
        range(3),
        [NAMES[key] for key in NAMES],
    )

    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Média das 30 perguntas")
    ax.set_title(
        "AV2 × RAG — qualidade conhecida no Top-5"
    )
    ax.legend(ncol=2)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)

    fig.tight_layout()

    path = (
        FIG_DIR
        / "retrieval_quality_comparison.png"
    )

    fig.savefig(path, dpi=180)
    plt.close(fig)

    return path


def plot_bounds(
    summary: dict[str, dict[str, str]],
) -> Path:
    """Plot conservative and optimistic Precision@5 bounds."""

    fig, ax = plt.subplots(
        figsize=(9, 5.2)
    )

    for i, name in enumerate(NAMES):
        row = summary[name]

        low = float(
            row["mean_precision_lower_bound"]
        )

        high = float(
            row["mean_precision_upper_bound"]
        )

        ax.vlines(
            i,
            low,
            high,
            linewidth=6,
        )

        ax.scatter(
            [i, i],
            [low, high],
            s=55,
            zorder=3,
        )

        ax.annotate(
            (
                f"{low:.3f}–{high:.3f}\n"
                f"{row['unjudged_results']} não julgados"
            ),
            (i, high),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    ax.set_xticks(
        range(3),
        [NAMES[key] for key in NAMES],
    )

    ax.set_ylim(0, 0.96)
    ax.set_xlim(-0.6, 2.6)

    ax.set_ylabel(
        "Precision@5 possível"
    )
    ax.set_title(
        "Impacto dos resultados ainda não julgados"
    )

    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)

    fig.tight_layout()

    path = (
        FIG_DIR
        / "retrieval_precision_bounds.png"
    )

    fig.savefig(path, dpi=180)
    plt.close(fig)

    return path


def plot_latency(
    summary: dict[str, dict[str, str]],
) -> Path:
    """Plot measured mean retrieval latency."""

    values = [
        float(summary[name]["mean_time_ms"])
        for name in NAMES
    ]

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    bars = ax.bar(
        list(NAMES.values()),
        values,
    )

    ax.bar_label(
        bars,
        labels=[
            f"{value:.2f} ms"
            for value in values
        ],
    )

    ax.set_ylim(
        0,
        max(values) * 1.2,
    )

    ax.set_ylabel(
        "Tempo médio de recuperação (ms)"
    )
    ax.set_title(
        "Latência de recuperação — AV2 × RAG"
    )

    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)

    fig.tight_layout()

    path = (
        FIG_DIR
        / "retrieval_latency_comparison.png"
    )

    fig.savefig(path, dpi=180)
    plt.close(fig)

    return path


def plot_deltas(
    rows: list[dict[str, str]],
) -> Path:
    """Plot per-question differences from the semantic baseline."""

    indexed = {
        (
            row["query_id"],
            row["condition"],
        ): row
        for row in rows
    }

    questions = sorted({
        row["query_id"]
        for row in rows
    })

    fig, ax = plt.subplots(
        figsize=(13, 5.5)
    )

    for name in (
        "literal",
        "hybrid",
    ):
        delta = [
            (
                float(
                    indexed[
                        (question, name)
                    ]["pooled_precision_at_k"]
                )
                - float(
                    indexed[
                        (question, "semantic")
                    ]["pooled_precision_at_k"]
                )
            )
            for question in questions
        ]

        ax.plot(
            range(30),
            delta,
            marker="o",
            linewidth=1,
            label=NAMES[name],
        )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1,
    )

    ax.set_xticks(
        range(30),
        questions,
        rotation=60,
    )

    ax.set_ylabel(
        "Δ Precision@5 conhecida contra semântica"
    )
    ax.set_title(
        "Variação da precisão por pergunta"
    )

    ax.legend()
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()

    path = (
        FIG_DIR
        / "retrieval_per_query_differences.png"
    )

    fig.savefig(path, dpi=180)
    plt.close(fig)

    return path


def write_analysis(
    summary: dict[str, dict[str, str]],
) -> Path:
    """Record the measured outcome and methodological caveats."""

    lines = [
        "# AV2 × RAG — comparação de recuperação",
        "",
        (
            "Protocolo: 30 perguntas, Top-5, cinco repetições "
            "medidas e um aquecimento por condição."
        ),
        "",
        (
            "| Condição | P@5 conhecida | P@5 máxima | "
            "Recall@5 | MRR@5 | Hit Rate@5 | "
            "Tempo (ms) | Não julgados |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for condition in NAMES:
        row = summary[condition]

        lines.append(
            f"| {NAMES[condition]} | "
            f"{float(row['pooled_precision_at_k']):.4f} | "
            f"{float(row['mean_precision_upper_bound']):.4f} | "
            f"{float(row['pooled_recall_at_k']):.4f} | "
            f"{float(row['known_mrr_at_k']):.4f} | "
            f"{float(row['known_hit_rate_at_k']):.4f} | "
            f"{float(row['mean_time_ms']):.2f} | "
            f"{row['unjudged_results']} |"
        )

    semantic_precision = float(
        summary["semantic"]["pooled_precision_at_k"]
    )

    literal_upper = float(
        summary["literal"]["mean_precision_upper_bound"]
    )

    hybrid_upper = float(
        summary["hybrid"]["mean_precision_upper_bound"]
    )

    ratio = (
        float(
            summary["hybrid"]["mean_time_ms"]
        )
        / float(
            summary["semantic"]["mean_time_ms"]
        )
    )

    lines.extend([
        "",
        "## Conclusões descritivas",
        "",
        (
            "- Precisão conhecida da semântica: "
            f"{semantic_precision:.4f}."
        ),
        (
            "- Melhor cenário possível da literal: "
            f"{literal_upper:.4f}."
        ),
        (
            "- Melhor cenário possível da híbrida: "
            f"{hybrid_upper:.4f}."
        ),
        (
            f"- O híbrido foi {ratio:.2f} vezes mais lento "
            "que a semântica."
        ),
        "",
        "## Limitações",
        "",
        (
            "- As métricas usam relevância conhecida "
            "no pool original da AV1."
        ),
        (
            "- Resultados ainda não julgados não são "
            "negativos confirmados."
        ),
        (
            "- O limite superior considera todos os "
            "não julgados relevantes."
        ),
        (
            "- As cinco repetições não equivalem a "
            "perguntas independentes."
        ),
        (
            "- Não foi realizado teste de "
            "significância estatística."
        ),
        (
            "- Não se mediu qualidade de respostas "
            "geradas pelo LLM nesta etapa."
        ),
        (
            "- O desempenho não prova que toda "
            "fusão híbrida seja inferior."
        ),
        "",
    ])

    REPORT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return REPORT_PATH


def main() -> None:
    """Read persisted results without rerunning experiments."""

    summary, rows = load_results()

    FIG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    generated = [
        plot_quality(summary),
        plot_bounds(summary),
        plot_latency(summary),
        plot_deltas(rows),
        write_analysis(summary),
    ]

    print(
        "AV2 × RAG — RETRIEVAL ANALYSIS COMPLETED"
    )

    for path in generated:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
