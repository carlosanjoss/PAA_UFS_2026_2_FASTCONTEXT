"""Prepare blinded human evaluation files for FastContext RAG outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import random
import re
import shutil
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.utils.config import PROJECT_ROOT

CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "human_evaluation.yaml"
)

QUALITY_TASK = "quality"
GROUNDEDNESS_TASK = "groundedness"

QUALITY_SCORE_FIELDS = (
    "correctness",
    "completeness",
    "clarity",
    "hallucination_present",
)

GROUNDEDNESS_SCORE_FIELDS = (
    "groundedness",
)

QUALITY_MASTER_FIELDS = (
    "blind_id",
    "question",
    "answer",
    *QUALITY_SCORE_FIELDS,
    "notes",
)

GROUNDEDNESS_MASTER_FIELDS = (
    "blind_id",
    "question",
    "answer",
    "retrieved_context",
    *GROUNDEDNESS_SCORE_FIELDS,
    "notes",
)

MANIFEST_FIELDS = (
    "task_type",
    "blind_id",
    "run_id",
    "query_id",
    "condition",
    "annotator_1",
    "annotator_2",
    "citation_valid",
    "citation_retry_count",
    "valid_citations",
    "generation_outcome",
)

SUMMARY_FIELDS = (
    "annotator",
    "quality_items",
    "groundedness_items",
    "total_items",
)


@dataclass(frozen=True, slots=True)
class EvaluationItem:
    """One blinded human-evaluation item."""

    task_type: str
    blind_id: str
    run_id: str
    query_id: str
    condition: str
    question: str
    answer: str
    retrieved_context: str
    citation_valid: str
    citation_retry_count: str
    valid_citations: tuple[str, ...]
    generation_outcome: str


def build_parser() -> argparse.ArgumentParser:
    """Build the human-evaluation preparation CLI."""

    parser = argparse.ArgumentParser(
        description=(
            "Prepare blinded FastContext human-evaluation files."
        )
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Replace only generated human-evaluation files."
        ),
    )

    return parser


def main() -> int:
    """Generate blinded evaluation masters and balanced assignments."""

    args = (
        build_parser()
        .parse_args()
    )

    config = _load_config()
    evaluation = config[
        "human_evaluation"
    ]

    seed = int(
        evaluation[
            "seed"
        ]
    )

    annotators = tuple(
        str(value).strip()
        for value in evaluation[
            "annotators"
        ]
    )

    judgments_per_item = int(
        evaluation[
            "judgments_per_item"
        ]
    )

    if judgments_per_item != 2:
        raise ValueError(
            "This balanced assignment protocol "
            "requires exactly two judgments per item."
        )

    _validate_annotators(
        annotators
    )

    inputs = evaluation[
        "inputs"
    ]

    generations_path = (
        _project_path(
            inputs[
                "generations_csv"
            ]
        )
    )

    chunks_path = _project_path(
        inputs[
            "chunks_jsonl"
        ]
    )

    expected_hash = str(
        inputs[
            "expected_generations_sha256"
        ]
    ).upper()

    actual_hash = _sha256_file(
        generations_path
    ).upper()

    if actual_hash != expected_hash:
        raise ValueError(
            "Frozen generations.csv SHA-256 mismatch."
        )

    rows = _load_csv(
        generations_path
    )

    _validate_generation_rows(
        rows=rows,
        expected_rows=int(
            inputs[
                "expected_rows"
            ]
        ),
        expected_model=str(
            inputs[
                "expected_model"
            ]
        ),
        expected_seed=str(
            inputs[
                "expected_seed"
            ]
        ),
        expected_corpus_fingerprint=str(
            inputs[
                "expected_corpus_fingerprint"
            ]
        ),
        expected_condition_counts={
            str(key): int(value)
            for key, value
            in inputs[
                "expected_condition_counts"
            ].items()
        },
    )

    chunks = _load_chunks(
        chunks_path
    )

    quality_conditions = tuple(
        str(value)
        for value in evaluation[
            "quality"
        ][
            "conditions"
        ]
    )

    groundedness_conditions = tuple(
        str(value)
        for value in evaluation[
            "groundedness"
        ][
            "conditions"
        ]
    )

    strip_chunk_citations = bool(
        evaluation[
            "quality"
        ][
            "strip_chunk_citations"
        ]
    )

    quality_items = _build_quality_items(
        rows=rows,
        seed=seed,
        conditions=quality_conditions,
        strip_chunk_citations=(
            strip_chunk_citations
        ),
    )

    groundedness_items = (
        _build_groundedness_items(
            rows=rows,
            chunks=chunks,
            seed=seed,
            conditions=(
                groundedness_conditions
            ),
        )
    )

    quality_assignments = (
        _assign_balanced_pairs(
            items=quality_items,
            annotators=annotators,
            seed=seed + 101,
        )
    )

    groundedness_assignments = (
        _assign_balanced_pairs(
            items=groundedness_items,
            annotators=annotators,
            seed=seed + 202,
        )
    )

    output_directory = (
        _project_path(
            evaluation[
                "output"
            ][
                "directory"
            ]
        )
    )

    _prepare_output_directory(
        output_directory,
        force=args.force,
    )

    _write_quality_master(
        output_directory
        / "quality_blinded.csv",
        quality_items,
    )

    _write_groundedness_master(
        output_directory
        / "groundedness_blinded.csv",
        groundedness_items,
    )

    _write_manifest(
        output_directory
        / "hidden_manifest.csv",
        quality_items=quality_items,
        groundedness_items=(
            groundedness_items
        ),
        quality_assignments=(
            quality_assignments
        ),
        groundedness_assignments=(
            groundedness_assignments
        ),
    )

    _write_assignments(
        output_directory=(
            output_directory
        ),
        annotators=annotators,
        quality_items=quality_items,
        groundedness_items=(
            groundedness_items
        ),
        quality_assignments=(
            quality_assignments
        ),
        groundedness_assignments=(
            groundedness_assignments
        ),
        seed=seed,
    )

    _write_assignment_summary(
        output_directory
        / "assignment_summary.csv",
        annotators=annotators,
        quality_assignments=(
            quality_assignments
        ),
        groundedness_assignments=(
            groundedness_assignments
        ),
    )

    _write_codebook(
        output_directory
        / "evaluation_codebook.md"
    )

    _write_admin_readme(
        output_directory
        / "README_ADMIN.md",
        generations_sha256=(
            actual_hash
        ),
        annotators=annotators,
        seed=seed,
    )

    _audit_assignments(
        quality_items=quality_items,
        groundedness_items=(
            groundedness_items
        ),
        quality_assignments=(
            quality_assignments
        ),
        groundedness_assignments=(
            groundedness_assignments
        ),
        annotators=annotators,
    )

    print(
        "FastContext Human Evaluation Prepared"
    )
    print(
        "=" * 60
    )
    print(
        f"Quality items: {len(quality_items)}"
    )
    print(
        "Groundedness items: "
        f"{len(groundedness_items)}"
    )
    print(
        "Total unique evaluation items: "
        f"{len(quality_items) + len(groundedness_items)}"
    )
    print(
        "Total human judgments: "
        f"{2 * (len(quality_items) + len(groundedness_items))}"
    )
    print(
        f"Annotators: {len(annotators)}"
    )
    print(
        "Expected load per annotator: "
        "36 quality + 24 groundedness = 60"
    )
    print(
        f"Output: {output_directory}"
    )
    print()
    print(
        "IMPORTANT: Do not share hidden_manifest.csv "
        "with annotators."
    )

    return 0


def _build_quality_items(
    *,
    rows: Sequence[Mapping[str, str]],
    seed: int,
    conditions: Sequence[str],
    strip_chunk_citations: bool,
) -> tuple[EvaluationItem, ...]:
    """Build blinded general-quality items."""

    items: list[
        EvaluationItem
    ] = []

    for row in rows:
        condition = row[
            "condition"
        ]

        if condition not in conditions:
            continue

        retrieved_ids = tuple(
            _json_list(
                row.get(
                    "retrieved_chunk_ids",
                    "",
                )
            )
        )

        answer = row[
            "answer"
        ]

        if strip_chunk_citations:
            answer = _strip_chunk_citations(
                answer,
                retrieved_ids,
            )

        items.append(
            EvaluationItem(
                task_type=QUALITY_TASK,
                blind_id=_blind_id(
                    task_type=QUALITY_TASK,
                    run_id=row[
                        "run_id"
                    ],
                    seed=seed,
                ),
                run_id=row[
                    "run_id"
                ],
                query_id=row[
                    "query_id"
                ],
                condition=condition,
                question=row[
                    "question"
                ],
                answer=answer,
                retrieved_context="",
                citation_valid=row.get(
                    "citation_valid",
                    "",
                ),
                citation_retry_count=(
                    row.get(
                        "citation_retry_count",
                        "",
                    )
                ),
                valid_citations=tuple(
                    _json_list(
                        row.get(
                            "valid_citations",
                            "",
                        )
                    )
                ),
                generation_outcome=(
                    _generation_outcome(
                        row
                    )
                ),
            )
        )

    _validate_unique_blind_ids(
        items
    )

    return tuple(
        items
    )


def _build_groundedness_items(
    *,
    rows: Sequence[Mapping[str, str]],
    chunks: Mapping[str, Mapping[str, Any]],
    seed: int,
    conditions: Sequence[str],
) -> tuple[EvaluationItem, ...]:
    """Build blinded groundedness items with retrieved evidence."""

    items: list[
        EvaluationItem
    ] = []

    for row in rows:
        condition = row[
            "condition"
        ]

        if condition not in conditions:
            continue

        retrieved_ids = tuple(
            _json_list(
                row.get(
                    "retrieved_chunk_ids",
                    "",
                )
            )
        )

        context = _build_retrieved_context(
            retrieved_ids,
            chunks,
        )

        items.append(
            EvaluationItem(
                task_type=(
                    GROUNDEDNESS_TASK
                ),
                blind_id=_blind_id(
                    task_type=(
                        GROUNDEDNESS_TASK
                    ),
                    run_id=row[
                        "run_id"
                    ],
                    seed=seed,
                ),
                run_id=row[
                    "run_id"
                ],
                query_id=row[
                    "query_id"
                ],
                condition=condition,
                question=row[
                    "question"
                ],
                answer=row[
                    "answer"
                ],
                retrieved_context=(
                    context
                ),
                citation_valid=row.get(
                    "citation_valid",
                    "",
                ),
                citation_retry_count=(
                    row.get(
                        "citation_retry_count",
                        "",
                    )
                ),
                valid_citations=tuple(
                    _json_list(
                        row.get(
                            "valid_citations",
                            "",
                        )
                    )
                ),
                generation_outcome=(
                    _generation_outcome(
                        row
                    )
                ),
            )
        )

    _validate_unique_blind_ids(
        items
    )

    return tuple(
        items
    )


def _assign_balanced_pairs(
    *,
    items: Sequence[EvaluationItem],
    annotators: Sequence[str],
    seed: int,
) -> dict[str, tuple[str, str]]:
    """Assign each item to one annotator pair with exact balance."""

    pairs = tuple(
        itertools.combinations(
            annotators,
            2,
        )
    )

    if not pairs:
        raise ValueError(
            "At least two annotators are required."
        )

    if len(items) % len(pairs) != 0:
        raise ValueError(
            "Item count must be divisible by the "
            "number of annotator pairs for exact balance."
        )

    repeat_count = (
        len(items)
        // len(pairs)
    )

    pair_schedule = list(
        pairs
        * repeat_count
    )

    shuffled_items = list(
        items
    )

    random.Random(
        seed
    ).shuffle(
        shuffled_items
    )

    random.Random(
        seed + 1
    ).shuffle(
        pair_schedule
    )

    return {
        item.blind_id: pair
        for item, pair
        in zip(
            shuffled_items,
            pair_schedule,
            strict=True,
        )
    }


def _strip_chunk_citations(
    answer: str,
    chunk_ids: Sequence[str],
) -> str:
    """Remove only exact retrieved chunk citation tokens."""

    result = answer

    for chunk_id in sorted(
        set(
            chunk_ids
        ),
        key=len,
        reverse=True,
    ):
        token_pattern = re.escape(
            f"[{chunk_id}]"
        )

        result = re.sub(
            token_pattern,
            "",
            result,
        )

    result = re.sub(
        r"[ \t]+([.,;:!?])",
        r"\1",
        result,
    )

    result = re.sub(
        r"[ \t]{2,}",
        " ",
        result,
    )

    result = re.sub(
        r"\n[ \t]+",
        "\n",
        result,
    )

    return result.strip()


def _build_retrieved_context(
    chunk_ids: Sequence[str],
    chunks: Mapping[str, Mapping[str, Any]],
) -> str:
    """Render the actual retrieved Top-k context for human review."""

    blocks: list[str] = []

    for rank, chunk_id in enumerate(
        chunk_ids,
        start=1,
    ):
        chunk = chunks.get(
            chunk_id
        )

        if chunk is None:
            raise ValueError(
                "Retrieved chunk is missing from "
                f"the frozen corpus: {chunk_id}"
            )

        source_path = str(
            chunk.get(
                "source_path"
            )
            or "unknown"
        )

        section_title = str(
            chunk.get(
                "section_title"
            )
            or "Untitled"
        )

        content = str(
            chunk.get(
                "content"
            )
            or ""
        ).strip()

        blocks.append(
            f"Rank {rank}\n"
            f"Chunk ID: {chunk_id}\n"
            f"Source: {source_path}\n"
            f"Section: {section_title}\n"
            f"{content}"
        )

    return (
        "\n\n---\n\n".join(
            blocks
        )
    )


def _generation_outcome(
    row: Mapping[str, str],
) -> str:
    """Classify the automatic generation outcome for admin analysis."""

    condition = row[
        "condition"
    ]

    if condition == "no_rag":
        return "no_rag"

    citation_valid = (
        row.get(
            "citation_valid",
            ""
        ).strip().lower()
        == "true"
    )

    valid_citations = _json_list(
        row.get(
            "valid_citations",
            "",
        )
    )

    answer = row.get(
        "answer",
        "",
    ).strip().lower()

    abstention = (
        "provided context is insufficient"
        in answer
    )

    if citation_valid and valid_citations:
        return "cited_answer"

    if citation_valid and abstention:
        return "valid_abstention"

    if (
        not citation_valid
        and abstention
    ):
        return "invalid_abstention"

    if (
        not citation_valid
        and not valid_citations
    ):
        return "invalid_uncited_answer"

    return "other"


def _blind_id(
    *,
    task_type: str,
    run_id: str,
    seed: int,
) -> str:
    """Build an opaque deterministic evaluation identifier."""

    prefix = (
        "Q"
        if task_type == QUALITY_TASK
        else "G"
    )

    payload = (
        f"{seed}|{task_type}|{run_id}"
    ).encode()

    digest = hashlib.sha256(
        payload
    ).hexdigest()[
        :12
    ].upper()

    return (
        f"{prefix}-{digest}"
    )


def _write_quality_master(
    path: Path,
    items: Sequence[EvaluationItem],
) -> None:
    """Write the blinded quality master file."""

    rows = [
        {
            "blind_id": item.blind_id,
            "question": item.question,
            "answer": item.answer,
            "correctness": "",
            "completeness": "",
            "clarity": "",
            "hallucination_present": "",
            "notes": "",
        }
        for item in sorted(
            items,
            key=lambda value: (
                value.blind_id
            ),
        )
    ]

    _write_csv(
        path,
        QUALITY_MASTER_FIELDS,
        rows,
    )


def _write_groundedness_master(
    path: Path,
    items: Sequence[EvaluationItem],
) -> None:
    """Write the blinded groundedness master file."""

    rows = [
        {
            "blind_id": item.blind_id,
            "question": item.question,
            "answer": item.answer,
            "retrieved_context": (
                item.retrieved_context
            ),
            "groundedness": "",
            "notes": "",
        }
        for item in sorted(
            items,
            key=lambda value: (
                value.blind_id
            ),
        )
    ]

    _write_csv(
        path,
        GROUNDEDNESS_MASTER_FIELDS,
        rows,
    )


def _write_manifest(
    path: Path,
    *,
    quality_items: Sequence[EvaluationItem],
    groundedness_items: Sequence[EvaluationItem],
    quality_assignments: Mapping[str, tuple[str, str]],
    groundedness_assignments: Mapping[str, tuple[str, str]],
) -> None:
    """Write the admin-only mapping from blind IDs to conditions."""

    rows: list[
        dict[
            str,
            str,
        ]
    ] = []

    for item in (
        list(
            quality_items
        )
        + list(
            groundedness_items
        )
    ):
        assignments = (
            quality_assignments
            if item.task_type
            == QUALITY_TASK
            else groundedness_assignments
        )

        annotator_1, annotator_2 = (
            assignments[
                item.blind_id
            ]
        )

        rows.append(
            {
                "task_type": item.task_type,
                "blind_id": item.blind_id,
                "run_id": item.run_id,
                "query_id": item.query_id,
                "condition": item.condition,
                "annotator_1": annotator_1,
                "annotator_2": annotator_2,
                "citation_valid": (
                    item.citation_valid
                ),
                "citation_retry_count": (
                    item.citation_retry_count
                ),
                "valid_citations": json.dumps(
                    list(
                        item.valid_citations
                    ),
                    ensure_ascii=False,
                ),
                "generation_outcome": (
                    item.generation_outcome
                ),
            }
        )

    _write_csv(
        path,
        MANIFEST_FIELDS,
        sorted(
            rows,
            key=lambda row: (
                row[
                    "task_type"
                ],
                row[
                    "blind_id"
                ],
            ),
        ),
    )


def _write_assignments(
    *,
    output_directory: Path,
    annotators: Sequence[str],
    quality_items: Sequence[EvaluationItem],
    groundedness_items: Sequence[EvaluationItem],
    quality_assignments: Mapping[str, tuple[str, str]],
    groundedness_assignments: Mapping[str, tuple[str, str]],
    seed: int,
) -> None:
    """Write separate quality and groundedness files per annotator."""

    assignment_directory = (
        output_directory
        / "assignments"
    )

    assignment_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    quality_by_id = {
        item.blind_id: item
        for item in quality_items
    }

    groundedness_by_id = {
        item.blind_id: item
        for item in groundedness_items
    }

    for index, annotator in enumerate(
        annotators
    ):
        quality_ids = [
            blind_id
            for blind_id, pair
            in quality_assignments.items()
            if annotator in pair
        ]

        groundedness_ids = [
            blind_id
            for blind_id, pair
            in groundedness_assignments.items()
            if annotator in pair
        ]

        random.Random(
            seed
            + 1000
            + index
        ).shuffle(
            quality_ids
        )

        random.Random(
            seed
            + 2000
            + index
        ).shuffle(
            groundedness_ids
        )

        quality_rows = [
            {
                "blind_id": blind_id,
                "question": (
                    quality_by_id[
                        blind_id
                    ].question
                ),
                "answer": (
                    quality_by_id[
                        blind_id
                    ].answer
                ),
                "correctness": "",
                "completeness": "",
                "clarity": "",
                "hallucination_present": "",
                "notes": "",
            }
            for blind_id in quality_ids
        ]

        groundedness_rows = [
            {
                "blind_id": blind_id,
                "question": (
                    groundedness_by_id[
                        blind_id
                    ].question
                ),
                "answer": (
                    groundedness_by_id[
                        blind_id
                    ].answer
                ),
                "retrieved_context": (
                    groundedness_by_id[
                        blind_id
                    ].retrieved_context
                ),
                "groundedness": "",
                "notes": "",
            }
            for blind_id
            in groundedness_ids
        ]

        _write_csv(
            assignment_directory
            / f"{annotator}_quality.csv",
            QUALITY_MASTER_FIELDS,
            quality_rows,
        )

        _write_csv(
            assignment_directory
            / f"{annotator}_groundedness.csv",
            GROUNDEDNESS_MASTER_FIELDS,
            groundedness_rows,
        )


def _write_assignment_summary(
    path: Path,
    *,
    annotators: Sequence[str],
    quality_assignments: Mapping[str, tuple[str, str]],
    groundedness_assignments: Mapping[str, tuple[str, str]],
) -> None:
    """Write admin workload counts."""

    quality_counts = Counter(
        annotator
        for pair
        in quality_assignments.values()
        for annotator in pair
    )

    groundedness_counts = Counter(
        annotator
        for pair
        in groundedness_assignments.values()
        for annotator in pair
    )

    rows = [
        {
            "annotator": annotator,
            "quality_items": (
                quality_counts[
                    annotator
                ]
            ),
            "groundedness_items": (
                groundedness_counts[
                    annotator
                ]
            ),
            "total_items": (
                quality_counts[
                    annotator
                ]
                + groundedness_counts[
                    annotator
                ]
            ),
        }
        for annotator in annotators
    ]

    _write_csv(
        path,
        SUMMARY_FIELDS,
        rows,
    )


def _write_codebook(
    path: Path,
) -> None:
    """Write evaluator instructions in Portuguese."""

    content = """# FastContext — Protocolo de avaliação humana cega

## Regras gerais

- Avalie cada item de forma independente.
- Não tente descobrir se a resposta veio de No-RAG, Optimized ou Semantic.
- Não consulte `hidden_manifest.csv`.
- Não compare respostas com outros avaliadores antes de concluir sua própria
  planilha.
- Preencha apenas números permitidos e, quando necessário, use `notes`.
- Não altere `blind_id`, `question`, `answer` ou `retrieved_context`.

## Avaliação de qualidade geral

Nesta planilha, as citações de chunks foram ocultadas para reduzir
vazamento da condição experimental.

### correctness

- `0`: resposta incorreta, contraditória ou essencialmente errada.
- `1`: parcialmente correta, mas contém erro, imprecisão ou limitação
  relevante.
- `2`: tecnicamente correta para a pergunta.

### completeness

- `0`: não responde o que foi pedido ou omite quase tudo que seria
  necessário.
- `1`: responde parcialmente, mas falta informação importante.
- `2`: resposta suficientemente completa para a pergunta proposta.

### clarity

- `0`: confusa, ambígua, incoerente ou difícil de usar.
- `1`: compreensível, mas com problemas de organização, precisão ou
  concisão.
- `2`: clara, direta e adequadamente formulada.

### hallucination_present

- `0`: não foi identificada afirmação factual inventada ou claramente não
  sustentada.
- `1`: existe pelo menos uma afirmação factual inventada, incorreta ou
  apresentada sem base plausível.

Se houver dúvida técnica relevante, registre a dúvida em `notes`;
não invente uma justificativa.

## Avaliação de groundedness

Nesta planilha, você recebe exatamente o Top-5 recuperado que foi enviado
ao modelo naquela execução.

### groundedness

- `0`: a resposta não é sustentada pelo contexto recuperado, contradiz o
  contexto ou faz afirmações substanciais ausentes das evidências.
- `1`: parte importante da resposta é sustentada, mas há afirmações
  relevantes não apoiadas, extrapolações ou cobertura incompleta.
- `2`: as afirmações substantivas da resposta são sustentadas pelo contexto
  recuperado ou a abstenção é apropriada diante do contexto.

A avaliação de groundedness não deve premiar uma resposta apenas porque
possui uma citação sintaticamente válida. O critério é sustentação factual
pelo conteúdo recuperado.

## Abstenções

Respostas que afirmam que o contexto é insuficiente devem ser avaliadas
pelo comportamento observado:

- uma abstenção pode receber boa avaliação se realmente não houver
  evidência suficiente;
- uma abstenção pode receber avaliação baixa se o contexto disponível
  responder claramente à pergunta.

## Independência

Cada item é avaliado por duas pessoas. Divergências serão analisadas
somente após o encerramento das avaliações independentes.
"""

    path.write_text(
        content,
        encoding="utf-8",
    )


def _write_admin_readme(
    path: Path,
    *,
    generations_sha256: str,
    annotators: Sequence[str],
    seed: int,
) -> None:
    """Write administrative instructions."""

    content = f"""# FastContext — Administração da avaliação humana

## Fonte congelada

`generations.csv` SHA-256:

`{generations_sha256}`

Seed de preparação e cegamento: `{seed}`.

## Carga

- 90 itens de qualidade geral × 2 avaliadores = 180 julgamentos.
- 60 itens de groundedness × 2 avaliadores = 120 julgamentos.
- Total = 300 julgamentos.
- 5 avaliadores.
- Cada avaliador recebe 36 itens de qualidade + 24 de groundedness = 60 julgamentos.

Avaliadores:

{chr(10).join(f"- {name}" for name in annotators)}

## Arquivos que podem ser entregues

Entregue a cada avaliador apenas:

- `evaluation_codebook.md`
- `assignments/<nome>_quality.csv`
- `assignments/<nome>_groundedness.csv`

## Arquivos administrativos — não compartilhar

Não compartilhe durante a fase cega:

- `hidden_manifest.csv`
- `assignment_summary.csv`
- os arquivos originais de `experiments/rag_results/`

## Após a coleta

Reúna as planilhas preenchidas sem alterar `blind_id`. A próxima etapa será:

1. validação dos valores;
2. junção pelo `blind_id`;
3. cálculo de concordância;
4. identificação das divergências;
5. adjudicação por terceiro avaliador;
6. abertura do `hidden_manifest.csv` somente para a análise final por condição.
"""

    path.write_text(
        content,
        encoding="utf-8",
    )


def _audit_assignments(
    *,
    quality_items: Sequence[EvaluationItem],
    groundedness_items: Sequence[EvaluationItem],
    quality_assignments: Mapping[str, tuple[str, str]],
    groundedness_assignments: Mapping[str, tuple[str, str]],
    annotators: Sequence[str],
) -> None:
    """Require exact two-rater coverage and balanced workload."""

    _audit_task_assignments(
        items=quality_items,
        assignments=(
            quality_assignments
        ),
        annotators=annotators,
        expected_per_annotator=36,
    )

    _audit_task_assignments(
        items=groundedness_items,
        assignments=(
            groundedness_assignments
        ),
        annotators=annotators,
        expected_per_annotator=24,
    )


def _audit_task_assignments(
    *,
    items: Sequence[EvaluationItem],
    assignments: Mapping[str, tuple[str, str]],
    annotators: Sequence[str],
    expected_per_annotator: int,
) -> None:
    """Validate one task's assignment matrix."""

    expected_ids = {
        item.blind_id
        for item in items
    }

    if set(
        assignments
    ) != expected_ids:
        raise RuntimeError(
            "Assignment coverage does not match evaluation items."
        )

    counts = Counter()

    for blind_id, pair in assignments.items():
        if len(
            pair
        ) != 2:
            raise RuntimeError(
                "Each item must have exactly two annotators."
            )

        if pair[
            0
        ] == pair[
            1
        ]:
            raise RuntimeError(
                f"Duplicate annotator for {blind_id}."
            )

        counts.update(
            pair
        )

    expected_annotators = set(
        annotators
    )

    if set(
        counts
    ) != expected_annotators:
        raise RuntimeError(
            "Assignment annotators do not match configuration."
        )

    if any(
        counts[
            annotator
        ]
        != expected_per_annotator
        for annotator in annotators
    ):
        raise RuntimeError(
            "Annotator workload is not exactly balanced."
        )


def _validate_generation_rows(
    *,
    rows: Sequence[Mapping[str, str]],
    expected_rows: int,
    expected_model: str,
    expected_seed: str,
    expected_corpus_fingerprint: str,
    expected_condition_counts: Mapping[str, int],
) -> None:
    """Validate the frozen 90-generation experiment."""

    if len(
        rows
    ) != expected_rows:
        raise ValueError(
            "Unexpected number of generation rows."
        )

    run_ids = [
        row[
            "run_id"
        ]
        for row in rows
    ]

    if len(
        set(
            run_ids
        )
    ) != expected_rows:
        raise ValueError(
            "Generation run IDs are not unique."
        )

    conditions = Counter(
        row[
            "condition"
        ]
        for row in rows
    )

    if dict(
        conditions
    ) != dict(
        expected_condition_counts
    ):
        raise ValueError(
            "Generation condition counts do not match "
            "the frozen experiment."
        )

    if {
        row[
            "actual_model"
        ]
        for row in rows
    } != {
        expected_model
    }:
        raise ValueError(
            "Unexpected generation model."
        )

    if {
        row[
            "generation_seed"
        ]
        for row in rows
    } != {
        expected_seed
    }:
        raise ValueError(
            "Unexpected generation seed."
        )

    if {
        row[
            "corpus_fingerprint"
        ]
        for row in rows
    } != {
        expected_corpus_fingerprint
    }:
        raise ValueError(
            "Unexpected corpus fingerprint."
        )

    if len(
        {
            row[
                "experiment_signature"
            ]
            for row in rows
        }
    ) != 1:
        raise ValueError(
            "Multiple experiment signatures detected."
        )

    if any(
        not row[
            "answer"
        ].strip()
        for row in rows
    ):
        raise ValueError(
            "Empty generated answer detected."
        )


def _validate_annotators(
    annotators: Sequence[str],
) -> None:
    """Validate annotator identifiers."""

    if len(
        annotators
    ) != 5:
        raise ValueError(
            "This exact-balance protocol requires five annotators."
        )

    if any(
        not value
        for value in annotators
    ):
        raise ValueError(
            "Annotator identifiers cannot be empty."
        )

    if len(
        set(
            annotators
        )
    ) != len(
        annotators
    ):
        raise ValueError(
            "Annotator identifiers must be unique."
        )


def _validate_unique_blind_ids(
    items: Sequence[EvaluationItem],
) -> None:
    """Require unique blind IDs within a task."""

    blind_ids = [
        item.blind_id
        for item in items
    ]

    if len(
        blind_ids
    ) != len(
        set(
            blind_ids
        )
    ):
        raise RuntimeError(
            "Blind ID collision detected."
        )


def _load_chunks(
    path: Path,
) -> dict[str, dict[str, Any]]:
    """Load the frozen chunk corpus into a lookup table."""

    result: dict[
        str,
        dict[
            str,
            Any,
        ],
    ] = {}

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            stripped = line.strip()

            if not stripped:
                continue

            item = json.loads(
                stripped
            )

            if not isinstance(
                item,
                dict,
            ):
                raise TypeError(
                    "Chunk JSONL entries must be objects."
                )

            chunk_id = str(
                item.get(
                    "chunk_id",
                    "",
                )
            ).strip()

            if not chunk_id:
                raise ValueError(
                    "Missing chunk_id at line "
                    f"{line_number}."
                )

            if chunk_id in result:
                raise ValueError(
                    "Duplicate corpus chunk_id: "
                    f"{chunk_id}"
                )

            result[
                chunk_id
            ] = item

    return result


def _load_csv(
    path: Path,
) -> list[dict[str, str]]:
    """Read a UTF-8 CSV into dictionaries."""

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        return list(
            csv.DictReader(
                file
            )
        )


def _json_list(
    value: str,
) -> list[str]:
    """Parse one JSON-encoded list field."""

    if not value.strip():
        return []

    parsed = json.loads(
        value
    )

    if not isinstance(
        parsed,
        list,
    ):
        raise TypeError(
            "Expected a JSON list."
        )

    return [
        str(
            item
        )
        for item in parsed
    ]


def _load_config() -> dict[str, Any]:
    """Load the human-evaluation configuration."""

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
            "Human-evaluation config must be a mapping."
        )

    return data


def _prepare_output_directory(
    path: Path,
    *,
    force: bool,
) -> None:
    """Create a clean output directory without touching RAG raw data."""

    if path.exists():
        if not force:
            raise FileExistsError(
                "Human-evaluation output already exists. "
                "Use --force to rebuild only this derived dataset."
            )

        shutil.rmtree(
            path
        )

    path.mkdir(
        parents=True,
        exist_ok=True,
    )


def _write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    """Write one deterministic UTF-8 CSV."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    field: row.get(
                        field,
                        "",
                    )
                    for field in fieldnames
                }
            )


def _project_path(
    value: object,
) -> Path:
    """Resolve a project-relative path."""

    path = Path(
        str(
            value
        )
    )

    if path.is_absolute():
        return path

    return (
        PROJECT_ROOT
        / path
    )


def _sha256_file(
    path: Path,
) -> str:
    """Calculate SHA-256 for one file."""

    hasher = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:
        for block in iter(
            lambda: file.read(
                1024
                * 1024
            ),
            b"",
        ):
            hasher.update(
                block
            )

    return hasher.hexdigest()


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
