"""Recupera formularios preenchidos sem descartar julgamentos.

Uso:
    python -m scripts.recover_rag_evaluation_forms
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from scripts import (
    distribute_rag_model_comparison_evaluation as dist,
)
from scripts import (
    validate_rag_model_comparison_reviews as valid,
)

BASE = dist.DEST
TEMPLATES = BASE / "avaliadores"
RETURNS = BASE / "devolutivas"
BACKUP = BASE / "recuperacao_backup"

AUDIT = BASE / "administrativo" / "auditoria_distribuicao.json"

PREP_AUDIT = dist.ROOT / "administrativo" / "auditoria_preparacao.json"

ALLOC = BASE / "administrativo" / "alocacao_com_chave.csv"

ALLOCATION_COLUMNS = (
    "blind_id",
    "dimension",
    "reviewer_id",
    "judgment_slot",
    "run_key",
    "model_role",
    "model",
    "condition",
    "query_id",
)


def require(test: bool, message: str) -> None:
    if not test:
        raise RuntimeError(message)


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_csv(
    path: Path,
    fields: tuple[str, ...],
) -> list[dict[str, str]]:

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        reader = csv.DictReader(
            stream,
            strict=True,
        )

        require(
            tuple(reader.fieldnames or ()) == fields,
            f"Cabecalho inesperado: {path}",
        )

        rows = list(reader)

    require(
        all(
            None not in row and all(value is not None for value in row.values())
            for row in rows
        ),
        f"CSV malformado: {path}",
    )

    return rows


def restore_atomic(
    path: Path,
    raw: bytes,
) -> None:

    fd, temporary = tempfile.mkstemp(
        prefix=".fastcontext_restore_",
        dir=path.parent,
    )

    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())

        os.replace(temporary, path)

    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def immutable_copy(
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
            f"Arquivo existente possui outro conteudo; operacao cancelada: {path}",
        )

    else:
        with path.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())

    require(
        path.read_bytes() == raw,
        f"Falha ao verificar copia: {path}",
    )


def normalize_newlines(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n")


def main() -> None:

    require(
        AUDIT.is_file() and PREP_AUDIT.is_file(),
        "Auditorias de distribuicao e preparacao sao obrigatorias.",
    )

    distribution = json.loads(AUDIT.read_text(encoding="utf-8"))

    preparation = json.loads(PREP_AUDIT.read_text(encoding="utf-8"))

    expected_hashes = distribution["file_sha256"]

    masters = {}

    for dimension, fields, expected_count, name in (
        (
            "qualidade",
            dist.QUALITY,
            179,
            "qualidade_cega.csv",
        ),
        (
            "fundamentacao",
            dist.GROUNDING,
            119,
            "fundamentacao_cega.csv",
        ),
    ):
        source = dist.ROOT / "avaliadores" / name

        relative = f"avaliadores/{name}"

        require(
            digest(source.read_bytes()) == preparation["output_sha256"][relative],
            f"Formulario mestre alterado: {relative}",
        )

        require(
            digest(source.read_bytes()) == distribution["input_sha256"][relative],
            f"Mestre nao corresponde a distribuicao: {relative}",
        )

        rows = read_csv(source, fields)

        require(
            len(rows) == expected_count,
            f"Quantidade inesperada no mestre: {name}",
        )

        require(
            len({row["blind_id"] for row in rows}) == expected_count,
            f"IDs duplicados no mestre: {name}",
        )

        for row in rows:
            for field in valid.SCORE_FIELDS[dimension] + ("notes",):
                require(
                    row[field] == "",
                    f"Mestre ja preenchido: {name}",
                )

        masters[dimension] = {row["blind_id"]: row for row in rows}

    require(
        digest(ALLOC.read_bytes())
        == expected_hashes["administrativo/alocacao_com_chave.csv"],
        "Mapa de alocacao administrativo alterado.",
    )

    allocation = read_csv(
        ALLOC,
        ALLOCATION_COLUMNS,
    )

    require(
        len(allocation) == 596,
        "Alocacao nao possui 596 entradas.",
    )

    by_reviewer = defaultdict(
        lambda: {
            "qualidade": [],
            "fundamentacao": [],
        }
    )

    assigned = defaultdict(set)

    for item in allocation:
        dimension = item["dimension"]
        blind_id = item["blind_id"]
        reviewer = item["reviewer_id"]

        require(
            dimension in masters and reviewer in dist.REVIEWERS,
            f"Alocacao invalida: {item}",
        )

        require(
            blind_id in masters[dimension],
            f"Item ausente do mestre: {blind_id}",
        )

        require(
            reviewer not in assigned[(dimension, blind_id)],
            f"Alocacao repetida: {dimension}, {blind_id}, {reviewer}",
        )

        assigned[(dimension, blind_id)].add(reviewer)

        new_row = dict(masters[dimension][blind_id])

        if dimension == "qualidade":
            new_row["answer"], _ = dist.mask_citations(new_row["answer"])

        by_reviewer[reviewer][dimension].append(new_row)

    require(
        len(assigned) == 298 and all(len(people) == 2 for people in assigned.values()),
        "Mapa nao contem exatamente dois avaliadores por item.",
    )

    originals = {}
    filled = {}
    filled_counts = Counter()

    for reviewer in dist.REVIEWERS:
        for dimension, fields in (
            ("qualidade", dist.QUALITY),
            ("fundamentacao", dist.GROUNDING),
        ):
            filename = f"{reviewer}_{dimension}.csv"

            relative = f"avaliadores/{filename}"

            expected_rows = sorted(
                by_reviewer[reviewer][dimension],
                key=lambda row: row["blind_id"],
            )

            original_bytes = dist.encode_csv(
                expected_rows,
                fields,
            )

            require(
                digest(original_bytes) == expected_hashes[relative],
                f"Nao foi possivel reproduzir o formulario original: {filename}",
            )

            originals[filename] = original_bytes

            current_path = TEMPLATES / filename

            require(
                current_path.is_file(),
                f"Formulario ausente: {current_path}",
            )

            current_bytes = current_path.read_bytes()

            if current_bytes == original_bytes:
                preserved = RETURNS / filename

                require(
                    preserved.is_file(),
                    f"Formulario em branco sem devolutiva preservada: {filename}",
                )

                current_bytes = preserved.read_bytes()

                current_rows = read_csv(
                    preserved,
                    fields,
                )

            else:
                current_rows = read_csv(
                    current_path,
                    fields,
                )

            require(
                len(current_rows) == len(expected_rows),
                f"Quantidade de linhas modificada: {filename}",
            )

            current_by_id = {row["blind_id"]: row for row in current_rows}

            require(
                len(current_by_id) == len(current_rows)
                and set(current_by_id) == {row["blind_id"] for row in expected_rows},
                f"IDs adicionados ou removidos: {filename}",
            )

            for original in expected_rows:
                received = current_by_id[original["blind_id"]]

                for field in valid.STATIC_FIELDS[dimension]:
                    require(
                        normalize_newlines(received[field])
                        == normalize_newlines(original[field]),
                        "Campo protegido alterado: "
                        f"{filename}, "
                        f"{original['blind_id']}, "
                        f"{field}",
                    )

                if valid.assess(
                    received,
                    dimension,
                    filename,
                ):
                    filled_counts["completos"] += 1

                else:
                    filled_counts["incompletos"] += 1

            filled[filename] = current_bytes

            print(f"CONFERIDO: {filename} | {len(expected_rows)} linhas")

    require(
        len(originals) == 10 and sum(filled_counts.values()) == 596,
        "A matriz de respostas nao contem 596 atribuicoes.",
    )

    print("PRE-VALIDACAO CONCLUIDA")

    print(
        "Julgamentos completos:",
        filled_counts["completos"],
    )

    print(
        "Julgamentos incompletos:",
        filled_counts["incompletos"],
    )

    print("Nenhum campo protegido foi alterado.")

    # Confere se as pastas de destino ja contem
    # algum arquivo diferente, sem sobrescreve-lo.

    for name, data in filled.items():
        for root in (RETURNS, BACKUP):
            target = root / name

            if target.exists():
                require(
                    target.read_bytes() == data,
                    f"Ja existe copia diferente: {target}",
                )

    # Primeiro salva duas copias completas de
    # todos os formularios preenchidos.

    for name, data in filled.items():
        immutable_copy(
            RETURNS / name,
            data,
        )

        immutable_copy(
            BACKUP / name,
            data,
        )

    print("20 COPIAS VERIFICADAS: devolutivas/ e recuperacao_backup/")

    # Somente depois restaura os originais.

    for name, raw in originals.items():
        path = TEMPLATES / name

        if path.read_bytes() != raw:
            restore_atomic(
                path,
                raw,
            )

        require(
            digest(path.read_bytes()) == expected_hashes[f"avaliadores/{name}"],
            f"Falhou restauracao de {name}",
        )

        print(f"ORIGINAL EM BRANCO RESTAURADO: {name}")

    # Reexecuta a auditoria de integridade
    # estabelecida no validador original.

    valid.catalog()

    print("=" * 70)
    print("RECUPERACAO CONCLUIDA")

    print(
        "Devolutivas preservadas:",
        RETURNS,
    )

    print(
        "Copias de seguranca:",
        BACKUP,
    )

    print("Formularios originais restaurados e hashes verificados: 10/10")

    print("Nenhuma nota foi criada, apagada ou reescrita nas devolutivas.")

    print("Execute: python -m scripts.validate_rag_model_comparison_reviews --status")


if __name__ == "__main__":
    main()
