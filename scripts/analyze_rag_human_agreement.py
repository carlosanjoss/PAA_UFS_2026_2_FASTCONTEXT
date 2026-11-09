"""Auditoria descritiva da concordancia na avaliacao humana cega FastContext.

Usar depois de --finalize do validador de devolutivas:
    python -m scripts.analyze_rag_human_agreement

Nao realiza descegamento; nao calcula testes inferenciais ou kappa/alpha.
Nenhum arquivo de entrada e modificado.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from scripts import validate_rag_model_comparison_reviews as validator

ROOT = validator.ROOT
ADMIN = ROOT / "administrativo"
RESULT = ROOT / "analise_concordancia_cega"
JUDGMENTS = ADMIN / "julgamentos_validados_cegos.csv"
AUDIT = ADMIN / "auditoria_devolutivas.json"
DIST_AUDIT = ADMIN / "auditoria_distribuicao.json"

CRITERIA = {
    "qualidade": (
        "correctness",
        "completeness",
        "clarity",
        "hallucination_present",
    ),
    "fundamentacao": ("groundedness",),
}

ORDINAL = {
    "correctness",
    "completeness",
    "clarity",
    "groundedness",
}

PAIR_FIELDS = (
    "dimension",
    "blind_id",
    "criterion",
    "reviewer_1",
    "reviewer_2",
    "score_1",
    "score_2",
    "exact_agreement",
    "absolute_difference",
)

SUMMARY_FIELDS = (
    "dimension",
    "criterion",
    "item_pairs",
    "exact_agreement_n",
    "exact_agreement_pct",
    "mean_absolute_difference",
    "difference_0",
    "difference_1",
    "difference_2",
    "difference_3",
    "difference_4",
)

BY_REVIEWER_FIELDS = (
    "reviewer_id",
    "dimension",
    "criterion",
    "judgments",
    "mean_score",
    "count_0",
    "count_1",
    "count_2",
    "count_3",
    "count_4",
    "count_5",
)

BY_PAIR_FIELDS = (
    "dimension",
    "criterion",
    "reviewer_1",
    "reviewer_2",
    "item_pairs",
    "exact_agreement_n",
    "exact_agreement_pct",
    "mean_absolute_difference",
)


def require(ok: bool, detail: str) -> None:
    if not ok:
        raise RuntimeError(detail)


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def load_csv(
    path: Path,
    fields: tuple[str, ...],
) -> list[dict[str, str]]:
    require(path.is_file(), f"Arquivo ausente: {path}")

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        reader = csv.DictReader(stream, strict=True)

        require(
            tuple(reader.fieldnames or ()) == fields,
            f"Cabecalho inesperado em {path.name}",
        )

        rows = list(reader)

    require(
        all(
            None not in row and all(v is not None for v in row.values()) for row in rows
        ),
        f"Linhas malformadas em {path.name}",
    )

    return rows


def encode_csv(
    rows: list[dict],
    fields: tuple[str, ...],
) -> bytes:
    output = io.StringIO(newline="")

    writer = csv.DictWriter(
        output,
        fieldnames=fields,
        lineterminator="\n",
        extrasaction="raise",
    )

    writer.writeheader()
    writer.writerows(rows)

    return output.getvalue().encode("utf-8-sig")


def write_once(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        require(
            path.read_bytes() == data,
            f"Saida existente diverge: {path}. Nao sobrescrevi.",
        )
        print(f"VERIFICADO: {path.name}")
        return

    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()

    print(f"CRIADO: {path.name}")


def pct(numerator: int, denominator: int) -> str:
    return f"{100 * numerator / denominator:.4f}" if denominator else ""


def avg(values: list[int]) -> str:
    return f"{mean(values):.6f}" if values else ""


def summarize_pairs(
    rows: list[dict],
    *,
    by_reviewer_pair: bool,
) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)

    for row in rows:
        group = (
            row["dimension"],
            row["criterion"],
        )

        if by_reviewer_pair:
            group += (
                row["reviewer_1"],
                row["reviewer_2"],
            )

        groups[group].append(row)

    result = []

    for group, items in sorted(groups.items()):
        diffs = [int(row["absolute_difference"]) for row in items]

        n = len(items)
        exact = diffs.count(0)

        entry = {
            "dimension": group[0],
            "criterion": group[1],
            "item_pairs": n,
            "exact_agreement_n": exact,
            "exact_agreement_pct": pct(exact, n),
            "mean_absolute_difference": avg(diffs),
        }

        if by_reviewer_pair:
            entry["reviewer_1"], entry["reviewer_2"] = group[2:]
        else:
            entry.update({f"difference_{k}": diffs.count(k) for k in range(5)})

        result.append(entry)

    return result


def main() -> None:
    # Valida a distribuicao original e as atribuicoes
    # sem revelar o modelo.
    distribution, templates, original_pairs = validator.catalog()

    require(
        AUDIT.is_file(),
        "Execute primeiro --finalize no validador.",
    )

    audit = json.loads(AUDIT.read_text(encoding="utf-8"))

    require(
        audit.get("status") == "devolutivas_formalmente_completas",
        "Auditoria de devolutivas nao esta finalizada.",
    )

    require(
        audit.get("evaluation_records") == 596,
        "Auditoria nao registra 596 julgamentos.",
    )

    require(
        audit.get("unique_items") == 298,
        "Auditoria nao registra 298 itens.",
    )

    require(
        audit.get("exact_duplicate_judgment_flags") == 0,
        "Existem flags de duplicacao que exigem revisao antes desta etapa.",
    )

    require(
        audit.get("distribution_audit_sha256") == sha256(DIST_AUDIT.read_bytes()),
        "Auditoria de distribuicao mudou desde a consolidacao.",
    )

    require(
        audit.get("consolidated_sha256") == sha256(JUDGMENTS.read_bytes()),
        "Consolidado cego difere do hash aprovado.",
    )

    # Os dez CSVs recebidos devem permanecer
    # identicos aos arquivos validados.
    input_hashes = audit.get("inputs_sha256", {})

    require(
        len(input_hashes) == 10,
        "Inventario das devolutivas incompleto.",
    )

    for filename, expected in input_hashes.items():
        path = validator.RECEIVED / filename

        require(
            path.is_file() and sha256(path.read_bytes()) == expected,
            f"Devolutiva foi alterada apos a validacao: {filename}",
        )

    judgments = load_csv(
        JUDGMENTS,
        validator.COMBINED_COLUMNS,
    )

    require(
        len(judgments) == 596,
        "Consolidado deve conter 596 linhas.",
    )

    expected_keys = set()

    for reviewer in validator.REVIEWERS:
        for dimension in CRITERIA:
            filename = f"{reviewer}_{dimension}.csv"

            for template in templates[filename]:
                expected_keys.add(
                    (
                        dimension,
                        template["blind_id"],
                        reviewer,
                    )
                )

    require(
        len(expected_keys) == 596,
        "Alocacao esperada nao e unica.",
    )

    grouped: dict[
        tuple[str, str],
        list[dict],
    ] = defaultdict(list)

    observed_keys = set()
    review_count = Counter()

    for row in judgments:
        dim = row["dimension"]

        require(
            dim in CRITERIA,
            f"Dimensao invalida: {dim}",
        )

        key = (
            dim,
            row["blind_id"],
            row["reviewer_id"],
        )

        require(
            key not in observed_keys,
            f"Julgamento duplicado: {key}",
        )

        observed_keys.add(key)

        require(
            validator.assess(
                row,
                dim,
                "julgamentos_validados_cegos.csv",
            ),
            f"Registro incompleto: {key}",
        )

        # Impede notas preenchidas em colunas
        # que pertencem a outra dimensao.
        for other in set(validator.SCORE_FIELDS) - {dim}:
            for field in validator.SCORE_FIELDS[other]:
                require(
                    row[field] == "",
                    f"Nota em dimensao incorreta: {key}/{field}",
                )

        grouped[(dim, row["blind_id"])].append(row)

        review_count[row["reviewer_id"]] += 1

    require(
        observed_keys == expected_keys,
        "Julgamentos nao correspondem exatamente aos IDs atribuidos.",
    )

    require(
        len(grouped) == 298,
        "Esperados 298 pares de julgamento.",
    )

    require(
        sorted(review_count.values()) == [119, 119, 119, 119, 120],
        "Cargas de avaliadores divergentes.",
    )

    require(
        all(
            len(rows) == 2 and rows[0]["reviewer_id"] != rows[1]["reviewer_id"]
            for rows in grouped.values()
        ),
        "Algum item nao possui dois avaliadores distintos.",
    )

    require(
        set(grouped) == set(original_pairs),
        "Conjunto de itens cegos divergente.",
    )

    # Uma linha por item e criterio, com as
    # duas notas preservadas separadamente.
    pair_rows: list[dict] = []

    for (dimension, blind_id), entries in sorted(grouped.items()):
        entries = sorted(
            entries,
            key=lambda x: x["reviewer_id"],
        )

        for criterion in CRITERIA[dimension]:
            x, y = (int(entry[criterion]) for entry in entries)

            pair_rows.append(
                {
                    "dimension": dimension,
                    "blind_id": blind_id,
                    "criterion": criterion,
                    "reviewer_1": entries[0]["reviewer_id"],
                    "reviewer_2": entries[1]["reviewer_id"],
                    "score_1": x,
                    "score_2": y,
                    "exact_agreement": int(x == y),
                    "absolute_difference": abs(x - y),
                }
            )

    require(
        len(pair_rows) == 835,
        f"Esperados 835 pares criterio-item; encontrados {len(pair_rows)}.",
    )

    overall = summarize_pairs(
        pair_rows,
        by_reviewer_pair=False,
    )

    by_pair = summarize_pairs(
        pair_rows,
        by_reviewer_pair=True,
    )

    require(
        len(overall) == 5,
        "Esperados cinco criterios.",
    )

    require(
        len(by_pair) == 50,
        "Esperados dez pares de avaliadores por cinco criterios.",
    )

    reviewer_groups = defaultdict(list)

    for row in judgments:
        for criterion in CRITERIA[row["dimension"]]:
            reviewer_groups[
                (
                    row["reviewer_id"],
                    row["dimension"],
                    criterion,
                )
            ].append(int(row[criterion]))

    by_reviewer = []

    for (
        reviewer,
        dimension,
        criterion,
    ), values in sorted(reviewer_groups.items()):
        counts = Counter(values)

        by_reviewer.append(
            {
                "reviewer_id": reviewer,
                "dimension": dimension,
                "criterion": criterion,
                "judgments": len(values),
                "mean_score": avg(values),
                **{f"count_{n}": counts[n] for n in range(6)},
            }
        )

    require(
        len(by_reviewer) == 25,
        "Esperados cinco avaliadores por cinco criterios.",
    )

    outputs = {
        "concordancia_por_criterio.csv": encode_csv(
            overall,
            SUMMARY_FIELDS,
        ),
        "discordancias_por_item_cego.csv": encode_csv(
            pair_rows,
            PAIR_FIELDS,
        ),
        "concordancia_por_par_de_avaliadores.csv": encode_csv(
            by_pair,
            BY_PAIR_FIELDS,
        ),
        "distribuicao_notas_por_avaliador.csv": encode_csv(
            by_reviewer,
            BY_REVIEWER_FIELDS,
        ),
    }

    summary = {
        "status": "auditoria_descritiva_cega",
        "records": 596,
        "unique_dimension_items": 298,
        "criterion_item_pairs": 835,
        "reviewers": 5,
        "exact_identical_notes_and_scores_flags_in_prior_audit": 0,
        "method": (
            "Concordancia exata e diferenca absoluta "
            "entre dois avaliadores por item e criterio; "
            "sem descegamento, testes inferenciais "
            "ou kappa/alpha."
        ),
        "interpretation": (
            "Concordancia descritiva nao comprova "
            "independencia entre avaliadores nem "
            "qualidade factual das respostas."
        ),
        "inputs_sha256": {
            "julgamentos_validados_cegos.csv": sha256(JUDGMENTS.read_bytes()),
            "auditoria_devolutivas.json": sha256(AUDIT.read_bytes()),
            "auditoria_distribuicao.json": sha256(DIST_AUDIT.read_bytes()),
        },
        "outputs_sha256": {name: sha256(content) for name, content in outputs.items()},
    }

    outputs["auditoria_concordancia.json"] = (
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")

    for name, raw in outputs.items():
        write_once(
            RESULT / name,
            raw,
        )

    print("=" * 70)
    print("CONCORDANCIA DESCRITIVA CEGA CONCLUIDA")
    print("Julgamentos validados: 596/596")
    print("Itens cegos: 298")
    print("Comparacoes item-criterio: 835")

    for row in overall:
        print(
            f"{row['criterion']:22s} "
            f"n={row['item_pairs']:3d} "
            f"acordo={row['exact_agreement_pct']:>8s}% "
            f"dif_abs_media="
            f"{row['mean_absolute_difference']}"
        )

    print(f"Saidas: {RESULT}")

    print("Nenhum modelo foi identificado; nenhuma avaliacao foi alterada.")


if __name__ == "__main__":
    main()
