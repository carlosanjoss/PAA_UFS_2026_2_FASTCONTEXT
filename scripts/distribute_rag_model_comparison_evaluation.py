"""Distribuicao cega AV2: 5 avaliadores, 2 julgamentos por item.

Execute da raiz do FastContext:
    python -m scripts.distribute_rag_model_comparison_evaluation

Entrada e resultados de geracao sao somente leitura. Nenhuma nota e atribuida.
"""

from __future__ import annotations

import csv
import hashlib
import io
import itertools
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from src.utils.config import PROJECT_ROOT

ROOT = (
    PROJECT_ROOT
    / "experiments"
    / "rag_model_comparison_results"
    / "avaliacao_humana_cega"
)
DEST = ROOT / "distribuicao_v1"
QUALITY = (
    "blind_id",
    "question",
    "answer",
    "correctness",
    "completeness",
    "clarity",
    "hallucination_present",
    "notes",
)
GROUNDING = (
    "blind_id",
    "question",
    "answer",
    "retrieved_context",
    "groundedness",
    "notes",
)
KEY = (
    "blind_id",
    "dimension",
    "run_key",
    "run_id",
    "condition",
    "query_id",
    "model_role",
    "model",
    "prompt_sha256",
    "record_sha256",
)
REVIEWERS = tuple(f"avaliador_{n:02d}" for n in range(1, 6))

# Formato [chunk_id] dos documentos FastContext, incluindo IDs invalidos
# emitidos pelo gerador. Nao remove citacoes em codigo Markdown.
CITE = re.compile(r"\[(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+-\d+\](?!\()")
CODE = re.compile(r"(```[\s\S]*?```|`[^`\n]*`)")


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_csv(path: Path, columns: tuple[str, ...], expected: int) -> list[dict]:
    require(path.is_file(), f"Entrada ausente: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        require(
            tuple(reader.fieldnames or ()) == columns, f"Colunas inesperadas: {path}"
        )
        rows = list(reader)
    require(
        len(rows) == expected,
        f"Contagem incorreta em {path}: {len(rows)} != {expected}",
    )
    require(
        all(None not in row for row in rows),
        f"CSV com colunas extras ou linhas invalidas: {path}",
    )
    return rows


def encode_csv(rows: list[dict], columns: tuple[str, ...]) -> bytes:
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(
        buf, fieldnames=columns, lineterminator="\n", extrasaction="raise"
    )
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8-sig")


def immutable_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(
            path.read_bytes() == content,
            f"Arquivo existente difere; nao sobrescrevi: {path}",
        )
        print(f"VALIDADO: {path.relative_to(ROOT)}")
    else:
        with path.open("xb") as stream:
            stream.write(content)
            stream.flush()
        print(f"CRIADO: {path.relative_to(ROOT)}")


def rng_for(fingerprint: str, purpose: str) -> random.Random:
    seed = int(sha(f"{fingerprint}|{purpose}|v1".encode()), 16)
    return random.Random(seed)


def mask_citations(answer: str) -> tuple[str, int]:
    """Remove somente [identificador-de-chunk-N] fora de codigo."""
    pieces = CODE.split(answer)
    total = 0
    for index in range(0, len(pieces), 2):
        pieces[index], count = CITE.subn("", pieces[index])
        total += count
    return "".join(pieces), total


def make_allocations(
    quality: list[dict], grounding: list[dict], fingerprint: str
) -> tuple[list[dict], dict]:
    """Distribui de forma balanceada por pares distintos de avaliadores.

    Para 179 itens: nove pares recebem 18 itens e um recebe 17.
    Para 119 itens: nove pares recebem 12 itens e um recebe 11.
    Os dois pares de menor frequencia sao disjuntos, dando distribuicao
    total 119,119,119,119,120 entre as cinco pessoas.
    """
    pairs = list(itertools.combinations(REVIEWERS, 2))
    pick = rng_for(fingerprint, "pares-minoritarios")
    short_quality = pick.choice(pairs)
    short_grounding = pick.choice(
        [pair for pair in pairs if set(pair).isdisjoint(short_quality)]
    )
    assignments = []
    pair_counts = {"qualidade": Counter(), "fundamentacao": Counter()}

    for dimension, rows, regular, short in (
        ("qualidade", quality, 18, short_quality),
        ("fundamentacao", grounding, 12, short_grounding),
    ):
        pair_list = [
            pair for pair in pairs for _ in range(regular - int(pair == short))
        ]
        require(
            len(pair_list) == len(rows), f"Falha de tamanho na alocacao: {dimension}"
        )
        rng_for(fingerprint, f"pares-{dimension}").shuffle(pair_list)
        ordered = sorted(rows, key=lambda row: row["blind_id"])
        rng_for(fingerprint, f"itens-{dimension}").shuffle(ordered)
        for row, pair in zip(ordered, pair_list, strict=True):
            pair_counts[dimension][pair] += 1
            for slot, reviewer in enumerate(pair, 1):
                assignments.append(
                    {
                        "blind_id": row["blind_id"],
                        "dimension": dimension,
                        "reviewer_id": reviewer,
                        "judgment_slot": str(slot),
                    }
                )

    require(len(assignments) == 596, "Numero incorreto de atribuicoes: esperadas 596.")
    counts = Counter(x["reviewer_id"] for x in assignments)
    require(
        sorted(counts.values()) == [119, 119, 119, 119, 120],
        f"Distribuicao desequilibrada: {counts}",
    )
    require(
        all(
            len(
                {
                    x["reviewer_id"]
                    for x in assignments
                    if x["blind_id"] == row["blind_id"] and x["dimension"] == dim
                }
            )
            == 2
            for dim, rows in (("qualidade", quality), ("fundamentacao", grounding))
            for row in rows
        ),
        "Item sem dois avaliadores diferentes.",
    )
    require(
        sorted(pair_counts["qualidade"].values()) == [17] + [18] * 9,
        "Pares de qualidade desequilibrados.",
    )
    require(
        sorted(pair_counts["fundamentacao"].values()) == [11] + [12] * 9,
        "Pares de fundamentacao desequilibrados.",
    )
    return assignments, {
        "short_pair_quality": list(short_quality),
        "short_pair_grounding": list(short_grounding),
        "workload_by_reviewer": dict(sorted(counts.items())),
        "pair_counts": {
            d: {"+".join(pair): count for pair, count in sorted(counter.items())}
            for d, counter in pair_counts.items()
        },
    }


def main() -> None:
    path_quality = ROOT / "avaliadores" / "qualidade_cega.csv"
    path_grounding = ROOT / "avaliadores" / "fundamentacao_cega.csv"
    path_key = ROOT / "administrativo" / "chave_cego_admin.csv"
    path_audit = ROOT / "administrativo" / "auditoria_preparacao.json"

    require(path_audit.is_file(), "Auditoria da preparacao cega ausente.")
    audit = json.loads(path_audit.read_text(encoding="utf-8"))
    inputs = {
        "avaliadores/qualidade_cega.csv": path_quality,
        "avaliadores/fundamentacao_cega.csv": path_grounding,
        "administrativo/chave_cego_admin.csv": path_key,
    }
    for name, path in inputs.items():
        require(path.is_file(), f"Entrada ausente: {path}")
        require(
            sha(path.read_bytes()) == audit["output_sha256"][name],
            f"Arquivo de entrada alterado desde a preparacao: {name}",
        )

    quality = read_csv(path_quality, QUALITY, 179)
    grounding = read_csv(path_grounding, GROUNDING, 119)
    key_rows = read_csv(path_key, KEY, 298)
    require(
        all(
            row["blind_id"] and row["question"] and row["answer"]
            for row in quality + grounding
        ),
        "Formulario com texto ausente.",
    )
    for row in quality:
        require(
            all(row[field] == "" for field in QUALITY[3:]),
            f"Qualidade ja preenchida: {row['blind_id']}",
        )
    for row in grounding:
        require(
            row["groundedness"] == "" and row["notes"] == "",
            f"Groundedness ja preenchida: {row['blind_id']}",
        )
    q_ids = [row["blind_id"] for row in quality]
    g_ids = [row["blind_id"] for row in grounding]
    require(
        len(set(q_ids)) == 179 and all(re.fullmatch(r"Q\d{4}", x) for x in q_ids),
        "IDs de qualidade repetidos ou invalidos.",
    )
    require(
        len(set(g_ids)) == 119 and all(re.fullmatch(r"G\d{4}", x) for x in g_ids),
        "IDs de fundamentacao repetidos ou invalidos.",
    )
    key_ids = [row["blind_id"] for row in key_rows]
    require(
        len(set(key_ids)) == 298 and set(key_ids) == set(q_ids + g_ids),
        "Chave administrativa nao coincide com os itens cegos.",
    )
    require(
        all(
            row["dimension"]
            == ("qualidade" if row["blind_id"].startswith("Q") else "fundamentacao")
            for row in key_rows
        ),
        "Dimensao inconsistente na chave administrativa.",
    )
    require(
        audit["quality_form_rows"] == 179 and audit["grounding_form_rows"] == 119,
        "Auditoria de preparacao inesperada.",
    )
    require(audit["blind_key_rows"] == 298, "Chave administrativa incompleta.")

    fingerprint = audit["source_frozen_prompts_sha256"]
    assignments, allocation_audit = make_allocations(quality, grounding, fingerprint)
    item_map = {("qualidade", row["blind_id"]): row for row in quality} | {
        ("fundamentacao", row["blind_id"]): row for row in grounding
    }
    key_map = {row["blind_id"]: row for row in key_rows}

    reviewer_rows = defaultdict(lambda: {"qualidade": [], "fundamentacao": []})
    administrative = []
    masked_by_id = {}
    for allocation in assignments:
        blind_id = allocation["blind_id"]
        dimension = allocation["dimension"]
        item = dict(item_map[(dimension, blind_id)])
        if dimension == "qualidade":
            item["answer"], removed = mask_citations(item["answer"])
            require(item["answer"].strip(), f"Resposta vazia apos mascara: {blind_id}")
            masked_by_id.setdefault(blind_id, removed)
        reviewer_rows[allocation["reviewer_id"]][dimension].append(item)
        administrative.append(
            {
                **allocation,
                "run_key": key_map[blind_id]["run_key"],
                "model_role": key_map[blind_id]["model_role"],
                "model": key_map[blind_id]["model"],
                "condition": key_map[blind_id]["condition"],
                "query_id": key_map[blind_id]["query_id"],
            }
        )

    masked_markers = sum(masked_by_id.values())
    output = {}
    workloads = {}
    for reviewer in REVIEWERS:
        qr = sorted(reviewer_rows[reviewer]["qualidade"], key=lambda x: x["blind_id"])
        gr = sorted(
            reviewer_rows[reviewer]["fundamentacao"], key=lambda x: x["blind_id"]
        )
        workloads[reviewer] = {
            "qualidade": len(qr),
            "fundamentacao": len(gr),
            "total": len(qr) + len(gr),
        }
        output[f"avaliadores/{reviewer}_qualidade.csv"] = encode_csv(qr, QUALITY)
        output[f"avaliadores/{reviewer}_fundamentacao.csv"] = encode_csv(gr, GROUNDING)

    require(
        sorted(x["total"] for x in workloads.values()) == [119, 119, 119, 119, 120],
        "Carga nao balanceada entre os avaliadores.",
    )
    require(
        sum(x["qualidade"] for x in workloads.values()) == 358
        and sum(x["fundamentacao"] for x in workloads.values()) == 238,
        "Contagens de avaliacoes nao batem.",
    )

    admin_cols = (
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
    output["administrativo/alocacao_com_chave.csv"] = encode_csv(
        sorted(
            administrative,
            key=lambda x: (x["dimension"], x["blind_id"], x["reviewer_id"]),
        ),
        admin_cols,
    )
    guide = (
        "# Instrucoes aos avaliadores — FastContext\n\n"
        "Cada conjunto e individual. Nao compartilhe planilhas, notas ou "
        "justificativas "
        "com outros avaliadores antes de encerrar sua avaliacao.\n\n"
        "Qualidade: correctness, completeness e clarity de 1 a 5; "
        "hallucination_present igual a 0 ou 1; notes com justificativa breve. "
        "Marcadores de citacao [chunk_id] foram ocultados da resposta apresentada.\n\n"
        "Fundamentacao: groundedness de 1 a 5, avaliando suporte nos "
        "trechos de retrieved_context; notes com justificativa breve. "
        "Os textos e os contextos dessa dimensao nao foram mascarados.\n\n"
        "Preencha apenas as colunas de notas e justificativas. Nao mude blind_id, "
        "question, answer ou retrieved_context. Nao tente identificar o modelo. "
        "Envie os arquivos originais preenchidos, sem consolidar respostas "
        "de diferentes avaliadores.\n"
    )
    output["avaliadores/LEIA-ME_DISTRIBUICAO.md"] = guide.encode("utf-8")
    output["administrativo/auditoria_distribuicao.json"] = (
        json.dumps(
            {
                "protocol": (
                    "5 avaliadores; 2 julgamentos distintos por item; AV2 1-5 e 0-1"
                ),
                "quality_items": 179,
                "grounding_items": 119,
                "assignments": 596,
                "masked_citation_markers_in_unique_quality_items": masked_markers,
                "mask_pattern": CITE.pattern,
                "quality_preserves_other_text": True,
                "grounding_answers_and_context_unmodified": True,
                "input_sha256": {
                    name: sha(path.read_bytes()) for name, path in inputs.items()
                },
                "preparation_audit_sha256": sha(path_audit.read_bytes()),
                "workloads": workloads,
                **allocation_audit,
                "file_sha256": {name: sha(data) for name, data in output.items()},
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")

    for relative, content in output.items():
        immutable_write(DEST / relative, content)

    print("=" * 68)
    print("DISTRIBUICAO CEGA CONCLUIDA")
    print("Itens unicos: 179 qualidade; 119 fundamentacao")
    print("Atribuicoes: 358 qualidade; 238 fundamentacao; 596 no total")
    print("Marcadores ocultados na qualidade:", masked_markers)
    for evaluator, values in workloads.items():
        print(
            f"{evaluator}: {values['qualidade']} qualidade + "
            f"{values['fundamentacao']} fundamentacao = {values['total']}"
        )
    print("Saidas:", DEST)
    print("Chaves e mapas administrativos NAO devem ser enviados aos avaliadores.")
    print("Nenhuma nota foi atribuida; os arquivos originais foram preservados.")


if __name__ == "__main__":
    main()
