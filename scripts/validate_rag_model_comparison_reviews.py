"""Valida devolutivas independentes da avaliacao cega FastContext.

Nao cria notas nem altera arquivos originais.

Coloque os dez CSVs preenchidos em:
  experiments/rag_model_comparison_results/avaliacao_humana_cega/
  distribuicao_v1/devolutivas/

Execute na raiz do projeto:
  python -m scripts.validate_rag_model_comparison_reviews --status
  python -m scripts.validate_rag_model_comparison_reviews --finalize

--status: permite devolutivas incompletas.
--finalize: exige 596 julgamentos completos e cria consolidado cego.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "rag_model_comparison_results"
    / "avaliacao_humana_cega"
    / "distribuicao_v1"
)

TEMPLATES = ROOT / "avaliadores"
RECEIVED = ROOT / "devolutivas"
ADMIN = ROOT / "administrativo"
AUDIT_PATH = ADMIN / "auditoria_distribuicao.json"

REVIEWERS = tuple(f"avaliador_{i:02d}" for i in range(1, 6))

FORMATS = {
    "qualidade": (
        "blind_id",
        "question",
        "answer",
        "correctness",
        "completeness",
        "clarity",
        "hallucination_present",
        "notes",
    ),
    "fundamentacao": (
        "blind_id",
        "question",
        "answer",
        "retrieved_context",
        "groundedness",
        "notes",
    ),
}

STATIC_FIELDS = {
    "qualidade": (
        "blind_id",
        "question",
        "answer",
    ),
    "fundamentacao": (
        "blind_id",
        "question",
        "answer",
        "retrieved_context",
    ),
}

SCORE_FIELDS = {
    "qualidade": (
        "correctness",
        "completeness",
        "clarity",
        "hallucination_present",
    ),
    "fundamentacao": ("groundedness",),
}

COMBINED_COLUMNS = (
    "reviewer_id",
    "dimension",
    "blind_id",
    "correctness",
    "completeness",
    "clarity",
    "hallucination_present",
    "groundedness",
    "notes",
)


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def parse_csv(
    path: Path,
    columns: tuple[str, ...],
) -> list[dict[str, str]]:
    try:
        with path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            reader = csv.DictReader(
                file,
                strict=True,
            )
            require(
                tuple(reader.fieldnames or ()) == columns,
                f"Cabecalho divergente: {path.name}",
            )
            rows = list(reader)

    except UnicodeDecodeError as exc:
        raise RuntimeError(f"CSV nao esta em UTF-8: {path}") from exc

    require(
        all(
            None not in row and all(value is not None for value in row.values())
            for row in rows
        ),
        f"Linha CSV malformada: {path}",
    )

    ids = [row["blind_id"] for row in rows]

    require(
        len(ids) == len(set(ids)),
        f"blind_id duplicado: {path}",
    )

    require(
        all(ids),
        f"blind_id vazio: {path}",
    )

    return rows


def csv_payload(
    rows: list[dict],
    columns: tuple[str, ...],
) -> bytes:
    buf = io.StringIO(newline="")

    writer = csv.DictWriter(
        buf,
        fieldnames=columns,
        lineterminator="\n",
        extrasaction="raise",
    )

    writer.writeheader()
    writer.writerows(rows)

    return buf.getvalue().encode("utf-8-sig")


def immutable_save(
    path: Path,
    raw: bytes,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if path.exists():
        require(
            path.read_bytes() == raw,
            f"Arquivo consolidado existente difere. Nao sobrescrevi: {path}",
        )

        print(f"VALIDADO: {path.name}")
        return

    with path.open("xb") as file:
        file.write(raw)

    print(f"CRIADO: {path.name}")


def catalog() -> tuple[dict, dict, dict]:
    """Confere a distribuicao original."""

    require(
        AUDIT_PATH.is_file(),
        f"Auditoria ausente: {AUDIT_PATH}",
    )

    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))

    require(
        audit.get("assignments") == 596,
        "Auditoria nao registra 596 atribuicoes.",
    )

    require(
        audit.get("quality_items") == 179,
        "Esperados 179 itens de qualidade.",
    )

    require(
        audit.get("grounding_items") == 119,
        "Esperados 119 itens de fundamentacao.",
    )

    hashes = audit["file_sha256"]
    templates = {}

    all_pairs = defaultdict(set)
    totals = Counter()

    for reviewer in REVIEWERS:
        for dimension, columns in FORMATS.items():
            name = f"{reviewer}_{dimension}.csv"
            relative = f"avaliadores/{name}"
            path = TEMPLATES / name

            require(
                path.is_file(),
                f"Modelo original ausente: {path}",
            )

            require(
                digest(path.read_bytes()) == hashes.get(relative),
                f"Distribuicao modificada: {name}",
            )

            rows = parse_csv(path, columns)

            templates[name] = rows
            totals[dimension] += len(rows)

            for row in rows:
                all_pairs[
                    (
                        dimension,
                        row["blind_id"],
                    )
                ].add(reviewer)

    require(
        totals
        == Counter(
            {
                "qualidade": 358,
                "fundamentacao": 238,
            }
        ),
        f"Distribuicao inesperada: {totals}",
    )

    require(
        len(all_pairs) == 298,
        "Esperados 298 itens cegos.",
    )

    require(
        all(len(reviewers) == 2 for reviewers in all_pairs.values()),
        "Algum item nao tem dois avaliadores distintos.",
    )

    require(
        sum(dimension == "qualidade" for dimension, _ in all_pairs) == 179
        and sum(dimension == "fundamentacao" for dimension, _ in all_pairs) == 119,
        "Composicao dos itens divergente.",
    )

    for reviewer in REVIEWERS:
        observed = sum(
            len(templates[f"{reviewer}_{dimension}.csv"]) for dimension in FORMATS
        )

        require(
            observed == audit["workloads"][reviewer]["total"],
            f"Carga divergente: {reviewer}",
        )

    allocation = ADMIN / "alocacao_com_chave.csv"

    require(
        allocation.is_file()
        and digest(allocation.read_bytes())
        == hashes.get("administrativo/alocacao_com_chave.csv"),
        "Chave administrativa ausente ou modificada.",
    )

    return audit, templates, all_pairs


def assess(
    row: dict[str, str],
    dimension: str,
    name: str,
) -> bool:
    scores = SCORE_FIELDS[dimension]

    for field in scores:
        value = row[field].strip()

        if not value:
            continue

        if field == "hallucination_present":
            valid = {"0", "1"}
        else:
            valid = {"1", "2", "3", "4", "5"}

        require(
            value in valid,
            f"Nota invalida {field}={value!r}, em {name} / {row['blind_id']}",
        )

    return all(row[field].strip() for field in (*scores, "notes"))


def inspect_returns(
    templates: dict,
    *,
    allow_incomplete: bool,
):
    records = []
    counts = Counter()
    input_hashes = {}

    incomplete_examples = []
    expected_names = set(templates)

    if RECEIVED.is_dir():
        actual_names = {path.name for path in RECEIVED.glob("*.csv")}
        unexpected = actual_names - expected_names
    else:
        unexpected = set()

    require(
        not unexpected,
        f"Arquivos CSV inesperados: {sorted(unexpected)}",
    )

    for reviewer in REVIEWERS:
        for dimension in FORMATS:
            name = f"{reviewer}_{dimension}.csv"

            template_rows = templates[name]
            path = RECEIVED / name

            if not path.is_file():
                counts["arquivos_ausentes"] += 1
                counts["julgamentos_pendentes"] += len(template_rows)

                print(f"AUSENTE: {name} ({len(template_rows)} itens)")
                continue

            counts["arquivos_recebidos"] += 1
            input_hashes[name] = digest(path.read_bytes())

            rows = parse_csv(
                path,
                FORMATS[dimension],
            )

            original = {row["blind_id"]: row for row in template_rows}

            received_ids = {row["blind_id"] for row in rows}

            require(
                received_ids == set(original),
                f"Itens faltantes ou extras: {name}",
            )

            for row in rows:
                item = original[row["blind_id"]]

                for field in STATIC_FIELDS[dimension]:
                    given = row[field].replace(
                        "\r\n",
                        "\n",
                    )
                    expected = item[field].replace(
                        "\r\n",
                        "\n",
                    )

                    require(
                        given == expected,
                        "Campo original alterado: "
                        f"{name} / "
                        f"{row['blind_id']} / "
                        f"{field}",
                    )

                complete = assess(
                    row,
                    dimension,
                    name,
                )

                if complete:
                    counts["julgamentos_completos"] += 1
                else:
                    counts["julgamentos_pendentes"] += 1
                    incomplete_examples.append(f"{name}:{row['blind_id']}")

                records.append(
                    {
                        "reviewer_id": reviewer,
                        "dimension": dimension,
                        "blind_id": row["blind_id"],
                        "correctness": row.get("correctness", "").strip(),
                        "completeness": row.get("completeness", "").strip(),
                        "clarity": row.get("clarity", "").strip(),
                        "hallucination_present": row.get(
                            "hallucination_present",
                            "",
                        ).strip(),
                        "groundedness": row.get(
                            "groundedness",
                            "",
                        ).strip(),
                        "notes": row["notes"],
                    }
                )

            print(f"VALIDADO: {name} ({len(rows)} itens)")

    require(
        counts["julgamentos_completos"] + counts["julgamentos_pendentes"] == 596,
        "Contagem de 596 julgamentos inconsistente.",
    )

    if not allow_incomplete:
        require(
            counts["arquivos_recebidos"] == 10,
            f"Faltam {10 - counts['arquivos_recebidos']} devolutivas.",
        )

        require(
            counts["julgamentos_pendentes"] == 0,
            f"Existem julgamentos incompletos: {incomplete_examples[:10]}",
        )

    return records, counts, input_hashes


def repeated_judgments(
    rows: list[dict],
) -> list[dict]:
    """Sinaliza avaliações integralmente iguais."""

    grouped = defaultdict(list)

    for row in rows:
        grouped[
            (
                row["dimension"],
                row["blind_id"],
            )
        ].append(row)

    flags = []

    for (dimension, blind_id), entries in grouped.items():
        if len(entries) != 2:
            continue

        fields = (
            *SCORE_FIELDS[dimension],
            "notes",
        )

        identical = all(
            entries[0][field].strip() == entries[1][field].strip() for field in fields
        )

        if identical:
            flags.append(
                {
                    "dimension": dimension,
                    "blind_id": blind_id,
                    "reviewer_1": entries[0]["reviewer_id"],
                    "reviewer_2": entries[1]["reviewer_id"],
                    "reason": "Notas e justificativa integralmente identicas",
                }
            )

    return sorted(
        flags,
        key=lambda row: (
            row["dimension"],
            row["blind_id"],
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--status",
        action="store_true",
        help="Conferir devolutivas parciais",
    )

    group.add_argument(
        "--finalize",
        action="store_true",
        help="Consolidar 596 julgamentos completos",
    )

    args = parser.parse_args()

    audit, templates, _ = catalog()

    records, counts, hashes = inspect_returns(
        templates,
        allow_incomplete=args.status,
    )

    print("=" * 67)
    print("VALIDACAO DAS DEVOLUTIVAS - FASTCONTEXT AV2")

    print(f"Arquivos recebidos: {counts['arquivos_recebidos']}/10")

    print(f"Julgamentos completos: {counts['julgamentos_completos']}/596")

    print(f"Julgamentos pendentes: {counts['julgamentos_pendentes']}/596")

    if args.status:
        print("Nenhum resultado foi consolidado e nenhuma nota foi criada.")
        return

    require(
        len(records) == 596,
        "Esperados 596 julgamentos.",
    )

    pairs = defaultdict(set)

    for record in records:
        pairs[
            (
                record["dimension"],
                record["blind_id"],
            )
        ].add(record["reviewer_id"])

    require(
        len(pairs) == 298 and all(len(reviewers) == 2 for reviewers in pairs.values()),
        "Consolidacao nao contem dois avaliadores distintos por item.",
    )

    flags = repeated_judgments(records)

    ordered = sorted(
        records,
        key=lambda row: (
            row["dimension"],
            row["blind_id"],
            row["reviewer_id"],
        ),
    )

    consolidated = csv_payload(
        ordered,
        COMBINED_COLUMNS,
    )

    flag_columns = (
        "dimension",
        "blind_id",
        "reviewer_1",
        "reviewer_2",
        "reason",
    )

    alerts = csv_payload(
        flags,
        flag_columns,
    )

    report = {
        "status": "devolutivas_formalmente_completas",
        "evaluation_records": 596,
        "unique_items": 298,
        "reviewers": 5,
        "independent_assignment_per_item": 2,
        "inputs_sha256": dict(sorted(hashes.items())),
        "distribution_audit_sha256": digest(AUDIT_PATH.read_bytes()),
        "consolidated_sha256": digest(consolidated),
        "exact_duplicate_judgment_flags": len(flags),
        "warnings": (
            "Atribuicao a dois avaliadores distintos "
            "nao prova independencia efetiva. "
            "Julgamentos identicos exigem revisao humana. "
            "Nenhuma nota foi inferida."
        ),
        "model_identification": (
            "Nao incluida no consolidado; chave administrativa separada"
        ),
    }

    immutable_save(
        ADMIN / "julgamentos_validados_cegos.csv",
        consolidated,
    )

    immutable_save(
        ADMIN / "sinais_duplicacao_para_revisao.csv",
        alerts,
    )

    immutable_save(
        ADMIN / "auditoria_devolutivas.json",
        (
            json.dumps(
                report,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8"),
    )

    print(f"Pares com notas e justificativas identicas (revisar): {len(flags)}")

    print(
        "Validacao formal concluida. "
        "Ainda nao houve descegamento "
        "nem analise estatistica."
    )


if __name__ == "__main__":
    main()
