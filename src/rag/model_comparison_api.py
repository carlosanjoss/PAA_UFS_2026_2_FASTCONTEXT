"""Read-only API for the audited Qwen 2.5 versus Llama 3.2 experiment."""

from __future__ import annotations

import csv
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.utils.config import PROJECT_ROOT

router = APIRouter(
    prefix="/api/rag/model-comparison",
    tags=["RAG - Model comparison"],
)

ModelRole = Literal["baseline", "candidate"]
Condition = Literal["no_rag", "optimized", "semantic"]
HumanCriterion = Literal[
    "correctness",
    "completeness",
    "clarity",
    "groundedness",
    "hallucination_present",
]


class ModelDescriptor(BaseModel):
    role: ModelRole
    model: str
    parameter_size: str
    quantization: str
    context_length: int


class ComparisonMetadata(BaseModel):
    attempts: int
    complete_answers: int
    truncated_answers: int
    frozen_prompts: int
    human_judgments: int
    valid_criterion_comparisons: int
    planned_criterion_comparisons: int
    missing_criterion_comparisons: int
    reviewers: int
    descriptive_only: bool
    generated_from: list[str]


class RetrievalDescriptor(BaseModel):
    model: str
    embedding_dimension: int
    normalized: bool
    index_type: str
    top_k: int
    reference: str
    reference_url: str
    faiss_reference_url: str


class GenerationConfiguration(BaseModel):
    provider: str
    temperature: float
    initial_max_tokens: int
    max_retry_tokens: int
    max_truncation_retries: int
    seed: int
    think: bool


class AutomaticMetricRow(BaseModel):
    role: ModelRole
    model: str
    condition: Condition
    planned: int
    success: int
    truncated: int
    success_rate: float
    generation_mean_ms: float
    generation_median_ms: float
    citation_accepted_n: int | None
    citation_compliance_rate: float | None
    with_valid_citation_n: int | None
    valid_citation_presence_rate: float | None
    citation_retry_mean: float | None


class HumanMetricRow(BaseModel):
    role: ModelRole
    model: str
    condition: Condition
    criterion: HumanCriterion
    responses: int
    mean_score: float
    exact_agreement_n: int
    exact_agreement_rate: float
    mean_absolute_disagreement: float
    discordant_items: int


class PairedHumanMetricRow(BaseModel):
    condition: Condition
    criterion: HumanCriterion
    planned_pairs: int
    both_complete: int
    missing_pair: int
    mean_candidate_minus_baseline: float
    candidate_better: int
    baseline_better: int
    ties: int


class AgreementMetricRow(BaseModel):
    dimension: Literal["qualidade", "fundamentacao"]
    criterion: HumanCriterion
    item_pairs: int
    exact_agreement_n: int
    exact_agreement_rate: float
    mean_absolute_difference: float


class RAGModelComparisonResponse(BaseModel):
    metadata: ComparisonMetadata
    models: list[ModelDescriptor]
    conditions: list[Condition]
    retrieval: RetrievalDescriptor
    generation: GenerationConfiguration
    automatic_metrics: list[AutomaticMetricRow]
    human_metrics: list[HumanMetricRow]
    paired_human_metrics: list[PairedHumanMetricRow]
    agreement_metrics: list[AgreementMetricRow]
    automatic_methodology_note: str
    human_methodology_note: str


@router.get("", response_model=RAGModelComparisonResponse)
def model_comparison() -> RAGModelComparisonResponse:
    """Return aggregate experimental results without exposing reviewer records."""
    try:
        return get_rag_model_comparison()
    except (
        FileNotFoundError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Comparação de modelos RAG indisponível: {exc}",
        ) from exc


@lru_cache(maxsize=1)
def get_rag_model_comparison() -> RAGModelComparisonResponse:
    """Cache validated immutable artifacts once per API process."""
    return load_rag_model_comparison()


def load_rag_model_comparison(
    repository_root: Path = PROJECT_ROOT,
) -> RAGModelComparisonResponse:
    """Load only audited aggregate artifacts from the controlled experiment."""
    results = repository_root / "experiments/rag_model_comparison_results"
    human_root = results / "avaliacao_humana_cega/distribuicao_v1"
    paths = {
        "generation_summary": results / "generation_status_summary.json",
        "generation_status": results / "generation_status.csv",
        "prompt_manifest": results / "frozen_prompts.manifest.json",
        "prompts": results / "frozen_prompts.jsonl",
        "provenance": results / "model_provenance.json",
        "automatic_audit": results / "analysis/auditoria_180.json",
        "automatic_summary": results / "analysis/resumo_por_modelo_condicao.csv",
        "human_audit": human_root
        / "analise_modelos_descegada/auditoria_descegamento.json",
        "human_summary": human_root
        / "analise_modelos_descegada/resumo_por_modelo_condicao_criterio.csv",
        "human_paired": human_root
        / "analise_modelos_descegada/resumo_pareado_criterio.csv",
        "agreement_audit": human_root
        / "analise_concordancia_cega/auditoria_concordancia.json",
        "agreement_summary": human_root
        / "analise_concordancia_cega/concordancia_por_criterio.csv",
        "judgments": human_root / "administrativo/julgamentos_validados_cegos.csv",
        "retrieval_config": repository_root / "config/retrieval.yaml",
        "comparison_config": repository_root / "config/rag_model_comparison.yaml",
    }
    missing = [
        str(path.relative_to(repository_root))
        for path in paths.values()
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(", ".join(missing))

    generation_summary = _read_json(paths["generation_summary"])
    manifest = _read_json(paths["prompt_manifest"])
    provenance = _read_json(paths["provenance"])
    automatic_audit = _read_json(paths["automatic_audit"])
    human_audit = _read_json(paths["human_audit"])
    agreement_audit = _read_json(paths["agreement_audit"])
    retrieval_config = _read_yaml(paths["retrieval_config"])["retrieval"]
    comparison_config = _read_yaml(paths["comparison_config"])["model_comparison"]

    status_rows = _read_csv(paths["generation_status"])
    prompt_count = _count_nonempty_lines(paths["prompts"])
    judgment_count = len(_read_csv(paths["judgments"]))
    automatic_rows = _load_automatic_rows(paths["automatic_summary"])
    human_rows = _load_human_rows(paths["human_summary"])
    paired_rows = _load_paired_rows(paths["human_paired"])
    agreement_rows = _load_agreement_rows(paths["agreement_summary"])

    status_counts = Counter(row["status"] for row in status_rows)
    attempts = int(generation_summary["expected"])
    complete_answers = int(generation_summary["counts"]["success"])
    truncated_answers = int(generation_summary["counts"]["truncated"])
    valid_comparisons = int(human_audit["complete_criterion_comparisons"])
    if len(status_rows) != attempts or attempts != 180:
        raise ValueError("generation status must contain 180 attempts")
    if status_counts["success"] != complete_answers or complete_answers != 179:
        raise ValueError("complete answer count does not match generation status")
    if status_counts["truncated"] != truncated_answers or truncated_answers != 1:
        raise ValueError("truncation count does not match generation status")
    if prompt_count != int(manifest["rows"]) or prompt_count != 90:
        raise ValueError("frozen prompt count must be 90")
    if judgment_count != int(human_audit["human_judgments"]) or judgment_count != 596:
        raise ValueError("human judgment count must be 596")
    if sum(row.both_complete for row in paired_rows) != valid_comparisons:
        raise ValueError(
            "valid criterion comparison total does not match paired summary"
        )
    if valid_comparisons != 415:
        raise ValueError("valid criterion comparison count must be 415")
    if len(automatic_rows) != 6 or len(agreement_rows) != 5:
        raise ValueError("aggregate model comparison dimensions are incomplete")

    semantic = retrieval_config["semantic"]
    generation = comparison_config["generation"]
    model_descriptors = []
    for role in ("baseline", "candidate"):
        typed_role = cast(ModelRole, role)
        model = provenance["models"][role]
        model_descriptors.append(
            ModelDescriptor(
                role=typed_role,
                model=str(model["name"]),
                parameter_size=str(model["details"]["parameter_size"]),
                quantization=str(model["details"]["quantization_level"]),
                context_length=int(model["details"]["context_length"]),
            )
        )

    return RAGModelComparisonResponse(
        metadata=ComparisonMetadata(
            attempts=attempts,
            complete_answers=complete_answers,
            truncated_answers=truncated_answers,
            frozen_prompts=prompt_count,
            human_judgments=judgment_count,
            valid_criterion_comparisons=valid_comparisons,
            planned_criterion_comparisons=int(
                human_audit["planned_criterion_comparisons"]
            ),
            missing_criterion_comparisons=int(
                human_audit["missing_criterion_comparisons"]
            ),
            reviewers=int(agreement_audit["reviewers"]),
            descriptive_only=True,
            generated_from=[
                str(path.relative_to(repository_root)).replace("\\", "/")
                for path in paths.values()
                if path not in {paths["judgments"]}
            ],
        ),
        models=model_descriptors,
        conditions=["no_rag", "optimized", "semantic"],
        retrieval=RetrievalDescriptor(
            model=str(semantic["model"]["name"]),
            embedding_dimension=int(semantic["embeddings"]["dimension"]),
            normalized=bool(semantic["embeddings"]["normalize"]),
            index_type=str(semantic["faiss"]["index_type"]),
            top_k=int(comparison_config["retrieval"]["top_k"]),
            reference=(
                "Xiao, Shitao et al. C-Pack: Packed Resources For General "
                "Chinese Embeddings. SIGIR 2024. arXiv:2309.07597."
            ),
            reference_url="https://arxiv.org/abs/2309.07597",
            faiss_reference_url="https://faiss.ai/cpp_api/struct/structfaiss_1_1IndexFlatIP.html",
        ),
        generation=GenerationConfiguration(
            provider=str(comparison_config["baseline"]["provider"]),
            temperature=float(generation["temperature"]),
            initial_max_tokens=int(generation["max_tokens"]),
            max_retry_tokens=int(generation["max_retry_tokens"]),
            max_truncation_retries=int(generation["max_truncation_retries"]),
            seed=int(generation["seed"]),
            think=bool(generation["think"]),
        ),
        automatic_metrics=automatic_rows,
        human_metrics=human_rows,
        paired_human_metrics=paired_rows,
        agreement_metrics=agreement_rows,
        automatic_methodology_note=str(automatic_audit["analysis_note"]),
        human_methodology_note=str(human_audit["limitations"]),
    )


def _load_automatic_rows(path: Path) -> list[AutomaticMetricRow]:
    result = []
    for row in _read_csv(path):
        success = int(row["success"])
        accepted = _nullable_int(row["citation_accepted_n"])
        with_valid = _nullable_int(row["with_valid_citation_n"])
        result.append(
            AutomaticMetricRow(
                role=cast(ModelRole, row["role"]),
                model=row["model"],
                condition=cast(Condition, row["condition"]),
                planned=int(row["planned"]),
                success=success,
                truncated=int(row["truncated"]),
                success_rate=float(row["success_rate"]),
                generation_mean_ms=float(row["generation_mean_ms_success_only"]),
                generation_median_ms=float(row["generation_median_ms_success_only"]),
                citation_accepted_n=accepted,
                citation_compliance_rate=(
                    accepted / success if accepted is not None else None
                ),
                with_valid_citation_n=with_valid,
                valid_citation_presence_rate=(
                    with_valid / success if with_valid is not None else None
                ),
                citation_retry_mean=_nullable_float(
                    row["citation_retry_mean_success_only"]
                ),
            )
        )
    return result


def _load_human_rows(path: Path) -> list[HumanMetricRow]:
    return [
        HumanMetricRow(
            role=cast(ModelRole, row["model_role"]),
            model=row["model"],
            condition=cast(Condition, row["condition"]),
            criterion=cast(HumanCriterion, row["criterion"]),
            responses=int(row["n_responses"]),
            mean_score=float(row["mean_of_item_means"]),
            exact_agreement_n=int(row["exact_agreement_n"]),
            exact_agreement_rate=float(row["exact_agreement_pct"]) / 100,
            mean_absolute_disagreement=float(row["mean_absolute_disagreement"]),
            discordant_items=int(row["discordant_items"]),
        )
        for row in _read_csv(path)
    ]


def _load_paired_rows(path: Path) -> list[PairedHumanMetricRow]:
    return [
        PairedHumanMetricRow(
            condition=cast(Condition, row["condition"]),
            criterion=cast(HumanCriterion, row["criterion"]),
            planned_pairs=int(row["planned_pairs"]),
            both_complete=int(row["both_complete"]),
            missing_pair=int(row["missing_pair"]),
            mean_candidate_minus_baseline=float(row["mean_candidate_minus_baseline"]),
            candidate_better=int(row["candidate_better"]),
            baseline_better=int(row["baseline_better"]),
            ties=int(row["ties"]),
        )
        for row in _read_csv(path)
    ]


def _load_agreement_rows(path: Path) -> list[AgreementMetricRow]:
    return [
        AgreementMetricRow(
            dimension=cast(Literal["qualidade", "fundamentacao"], row["dimension"]),
            criterion=cast(HumanCriterion, row["criterion"]),
            item_pairs=int(row["item_pairs"]),
            exact_agreement_n=int(row["exact_agreement_n"]),
            exact_agreement_rate=float(row["exact_agreement_pct"]) / 100,
            mean_absolute_difference=float(row["mean_absolute_difference"]),
        )
        for row in _read_csv(path)
    ]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain a JSON object")
    return value


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain a YAML mapping")
    return value


def _count_nonempty_lines(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)


def _nullable_int(value: str) -> int | None:
    return int(value) if value.strip() else None


def _nullable_float(value: str) -> float | None:
    return float(value) if value.strip() else None
