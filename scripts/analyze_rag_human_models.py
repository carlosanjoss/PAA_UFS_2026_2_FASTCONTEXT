"""Descegamento administrativo e comparacao descritiva FastContext.

Execute da raiz do projeto:
    python -m scripts.analyze_rag_human_models

Nao modifica notas, respostas, chaves nem configuracoes experimentais.
As medias dos dois julgamentos sao somente resumos descritivos.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

PROJECT = Path(__file__).resolve().parents[1]
BASE = PROJECT / "experiments" / "rag_model_comparison_results"
BLIND = BASE / "avaliacao_humana_cega"
DIST = BLIND / "distribuicao_v1"
ADMIN = DIST / "administrativo"
AGREE = DIST / "analise_concordancia_cega"
OUTPUT = DIST / "analise_modelos_descegada"

CRITERIA = {
    "qualidade": ("correctness", "completeness", "clarity", "hallucination_present"),
    "fundamentacao": ("groundedness",),
}
CONDITIONS = ("no_rag", "optimized", "semantic")
ROLES = ("baseline", "candidate")
QUALITY_COLUMNS = (
    "dimension",
    "blind_id",
    "run_key",
    "run_id",
    "query_id",
    "condition",
    "model_role",
    "model",
    "criterion",
    "reviewer_1",
    "reviewer_2",
    "score_1",
    "score_2",
    "item_mean",
    "exact_agreement",
    "absolute_difference",
)
GROUP_COLUMNS = (
    "model_role",
    "model",
    "condition",
    "criterion",
    "n_responses",
    "mean_of_item_means",
    "exact_agreement_n",
    "exact_agreement_pct",
    "mean_absolute_disagreement",
    "discordant_items",
)
PAIR_COLUMNS = (
    "condition",
    "query_id",
    "criterion",
    "baseline_status",
    "candidate_status",
    "baseline_mean",
    "candidate_mean",
    "candidate_minus_baseline",
    "favorable_to",
)
PAIRED_SUMMARY_COLUMNS = (
    "condition",
    "criterion",
    "planned_pairs",
    "both_complete",
    "missing_pair",
    "mean_candidate_minus_baseline",
    "candidate_better",
    "baseline_better",
    "ties",
)


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise RuntimeError(reason)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_json(path: Path) -> dict:
    require(path.is_file(), f"Arquivo ausente: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def checked_csv(path: Path, expected_sha: str | None = None) -> list[dict]:
    require(path.is_file(), f"Arquivo ausente: {path}")
    if expected_sha is not None:
        require(
            sha(path.read_bytes()) == expected_sha,
            f"SHA-256 diverge; arquivo nao pode ser usado: {path}",
        )
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, strict=True)
        require(bool(reader.fieldnames), f"Cabecalho CSV ausente: {path}")
        rows = list(reader)
    require(
        all(None not in r and all(v is not None for v in r.values()) for r in rows),
        f"CSV malformado: {path}",
    )
    return rows


def encode(rows: list[dict], columns: tuple[str, ...]) -> bytes:
    f = io.StringIO(newline="")
    writer = csv.DictWriter(
        f, fieldnames=columns, extrasaction="raise", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    return f.getvalue().encode("utf-8-sig")


def write_new_or_verify(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(
            path.read_bytes() == data,
            f"Saida existente e diferente; nao foi sobrescrita: {path}",
        )
        print("VERIFICADO:", path.name)
    else:
        with path.open("xb") as f:
            f.write(data)
            f.flush()
        print("CRIADO:", path.name)


def main() -> None:
    preparation = read_json(BLIND / "administrativo" / "auditoria_preparacao.json")
    distribution = read_json(ADMIN / "auditoria_distribuicao.json")
    returns_audit_path = ADMIN / "auditoria_devolutivas.json"
    returns = read_json(returns_audit_path)
    agreement_audit_path = AGREE / "auditoria_concordancia.json"
    agreement = read_json(agreement_audit_path)
    generation_audit = read_json(BASE / "analysis" / "auditoria_180.json")
    status_path = BASE / "generation_status.csv"
    status_summary_path = BASE / "generation_status_summary.json"

    require(
        returns.get("status") == "devolutivas_formalmente_completas"
        and returns.get("evaluation_records") == 596
        and returns.get("unique_items") == 298,
        "Avaliacao humana ainda nao esta formalmente consolidada.",
    )
    require(
        returns.get("exact_duplicate_judgment_flags") == 0,
        "Ha duplicacoes integrais sinalizadas para revisao.",
    )
    require(
        agreement.get("status") == "auditoria_descritiva_cega"
        and agreement.get("records") == 596
        and agreement.get("criterion_item_pairs") == 835,
        "Auditoria de concordancia nao esta completa.",
    )
    require(
        agreement["inputs_sha256"]["auditoria_devolutivas.json"]
        == sha(returns_audit_path.read_bytes()),
        "Auditoria de devolutivas mudou apos analise cega.",
    )
    require(
        returns["distribution_audit_sha256"]
        == sha((ADMIN / "auditoria_distribuicao.json").read_bytes()),
        "Auditoria de distribuicao alterada.",
    )
    require(
        generation_audit.get("expected_attempts") == 180
        and generation_audit.get("successful_answers") == 179
        and generation_audit.get("truncation_failures") == 1
        and generation_audit.get("pending") == 0,
        "Auditoria das geracoes diverge do experimento validado.",
    )
    require(
        generation_audit["generation_status_sha256"] == sha(status_path.read_bytes())
        and generation_audit["generation_status_summary_sha256"]
        == sha(status_summary_path.read_bytes()),
        "Status experimental alterado apos auditoria.",
    )
    status_summary = read_json(status_summary_path)
    require(
        status_summary["counts"] == {"success": 179, "truncated": 1, "pending": 0},
        "Contagem experimental inesperada.",
    )

    key_path = BLIND / "administrativo" / "chave_cego_admin.csv"
    judges_path = ADMIN / "julgamentos_validados_cegos.csv"
    allocations_path = ADMIN / "alocacao_com_chave.csv"
    disagreement_path = AGREE / "discordancias_por_item_cego.csv"
    key = checked_csv(
        key_path, preparation["output_sha256"]["administrativo/chave_cego_admin.csv"]
    )
    judges = checked_csv(judges_path, returns["consolidated_sha256"])
    assignments = checked_csv(
        allocations_path,
        distribution["file_sha256"]["administrativo/alocacao_com_chave.csv"],
    )
    disagreement = checked_csv(
        disagreement_path,
        agreement["outputs_sha256"]["discordancias_por_item_cego.csv"],
    )
    status = checked_csv(status_path, generation_audit["generation_status_sha256"])

    require(
        len(key) == 298
        and len(judges) == 596
        and len(assignments) == 596
        and len(disagreement) == 835
        and len(status) == 180,
        "Quantidade de linhas de entrada inesperada.",
    )
    keys_by_blind = {r["blind_id"]: r for r in key}
    status_by_run = {r["run_key"]: r for r in status}
    require(
        len(keys_by_blind) == 298 and len(status_by_run) == 180,
        "Identificadores duplicados na chave ou no status.",
    )
    require(
        Counter(r["status"] for r in status)
        == Counter({"success": 179, "truncated": 1}),
        "Existem execucoes pendentes ou estados inesperados.",
    )
    require(
        set(r["model_role"] for r in status) == set(ROLES)
        and set(r["condition"] for r in status) == set(CONDITIONS),
        "Modelos ou condicoes inesperados.",
    )
    role_models: dict[str, str] = {}
    status_ids: dict[tuple[str, str, str], dict] = {}
    for row in status:
        role, condition, query = row["model_role"], row["condition"], row["query_id"]
        require(
            role in ROLES and condition in CONDITIONS,
            "Condicao/papel experimental desconhecido.",
        )
        role_models.setdefault(role, row["model"])
        require(role_models[role] == row["model"], "Modelo mudou de identificador.")
        unique = (role, condition, query)
        require(unique not in status_ids, f"Execucao repetida: {unique}")
        status_ids[unique] = row
    require(len(status_ids) == 180, "Matriz experimental nao possui 180 chaves unicas.")
    condition_queries = {}
    for cond in CONDITIONS:
        queries = {r["query_id"] for r in status if r["condition"] == cond}
        require(
            len(queries) == 30
            and all(
                (role, cond, query) in status_ids for role in ROLES for query in queries
            ),
            f"Matriz 30x2 incompleta: {cond}",
        )
        condition_queries[cond] = sorted(queries)

    quality_runs = set()
    grounding_runs = set()
    for blind_id, meta in keys_by_blind.items():
        require(meta["dimension"] in CRITERIA, f"Dimensao desconhecida: {blind_id}")
        require(
            meta["model_role"] in ROLES and meta["condition"] in CONDITIONS,
            f"Mapeamento de modelo/condicao invalido: {blind_id}",
        )
        require(
            meta["model"] == role_models[meta["model_role"]],
            f"Modelo da chave nao coincide com os runs: {blind_id}",
        )
        original = status_by_run.get(meta["run_key"])
        require(
            original is not None and original["status"] == "success",
            f"Item cego aponta para run sem resposta completa: {blind_id}",
        )
        require(
            all(
                original[field] == meta[field]
                for field in (
                    "model_role",
                    "model",
                    "condition",
                    "query_id",
                    "prompt_sha256",
                )
            ),
            f"Mapeamento divergente: {blind_id}",
        )
        require(
            meta["run_key"] == f"{meta['model_role']}|{meta['run_id']}",
            f"Run key invalida: {blind_id}",
        )
        if meta["dimension"] == "qualidade":
            quality_runs.add(meta["run_key"])
        else:
            require(meta["condition"] != "no_rag", "Groundedness nao pode ser No-RAG.")
            grounding_runs.add(meta["run_key"])
    require(
        len(quality_runs) == 179 and len(grounding_runs) == 119,
        "Itens avaliados repetidos ou ausentes.",
    )
    success_runs = {r["run_key"] for r in status if r["status"] == "success"}
    require(
        quality_runs == success_runs
        and grounding_runs
        == {
            r["run_key"]
            for r in status
            if r["status"] == "success" and r["condition"] != "no_rag"
        },
        "Formularios nao representam exatamente as respostas completas.",
    )

    allocation_keys = {
        (r["dimension"], r["blind_id"], r["reviewer_id"]) for r in assignments
    }
    judge_keys = {(r["dimension"], r["blind_id"], r["reviewer_id"]) for r in judges}
    require(
        len(allocation_keys) == 596
        and len(judge_keys) == 596
        and allocation_keys == judge_keys,
        "Julgamentos nao coincidem com a atribuicao cega original.",
    )
    for r in assignments:
        meta = keys_by_blind.get(r["blind_id"])
        require(
            meta is not None
            and r["dimension"] == meta["dimension"]
            and all(
                r[f] == meta[f]
                for f in ("run_key", "model_role", "model", "condition", "query_id")
            ),
            f"Atribuicao administrativa divergente: {r['blind_id']}",
        )
    review_map: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in judges:
        require(
            r["blind_id"] in keys_by_blind
            and r["dimension"] == keys_by_blind[r["blind_id"]]["dimension"],
            "Julgamento aponta para identificador cego invalido.",
        )
        review_map[(r["dimension"], r["blind_id"])].append(r)
    require(
        len(review_map) == 298
        and all(
            len(v) == 2 and v[0]["reviewer_id"] != v[1]["reviewer_id"]
            for v in review_map.values()
        ),
        "Nem todo item tem dois avaliadores distintos.",
    )

    values = []
    seen_criteria = set()
    for d in disagreement:
        dimension, blind_id, criterion = d["dimension"], d["blind_id"], d["criterion"]
        entry = (dimension, blind_id, criterion)
        require(
            entry not in seen_criteria and criterion in CRITERIA.get(dimension, ()),
            f"Criterio repetido ou desconhecido: {entry}",
        )
        seen_criteria.add(entry)
        reviewers = sorted(
            review_map[(dimension, blind_id)], key=lambda r: r["reviewer_id"]
        )
        require(
            tuple(r["reviewer_id"] for r in reviewers)
            == (d["reviewer_1"], d["reviewer_2"]),
            f"Pareamento de avaliadores divergente: {entry}",
        )
        x, y = int(reviewers[0][criterion]), int(reviewers[1][criterion])
        require(
            str(x) == d["score_1"]
            and str(y) == d["score_2"]
            and abs(x - y) == int(d["absolute_difference"])
            and int(x == y) == int(d["exact_agreement"]),
            f"Discordancia nao coincide com notas originais: {entry}",
        )
        require(
            (x in (0, 1) and y in (0, 1))
            if criterion == "hallucination_present"
            else (1 <= x <= 5 and 1 <= y <= 5),
            f"Nota fora da escala: {entry}",
        )
        metadata = keys_by_blind[blind_id]
        values.append(
            {
                "dimension": dimension,
                "blind_id": blind_id,
                "run_key": metadata["run_key"],
                "run_id": metadata["run_id"],
                "query_id": metadata["query_id"],
                "condition": metadata["condition"],
                "model_role": metadata["model_role"],
                "model": metadata["model"],
                "criterion": criterion,
                "reviewer_1": d["reviewer_1"],
                "reviewer_2": d["reviewer_2"],
                "score_1": x,
                "score_2": y,
                "item_mean": (x + y) / 2,
                "exact_agreement": int(x == y),
                "absolute_difference": abs(x - y),
            }
        )
    require(
        len(values) == 835 and len(seen_criteria) == 835,
        "Concordancia ou consolidacao dos criterios esta incompleta.",
    )

    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in values:
        grouped[(r["model_role"], r["condition"], r["criterion"])].append(r)
    summary = []
    for (role, cond, criterion), items in sorted(grouped.items()):
        n = len(items)
        exact = sum(r["exact_agreement"] for r in items)
        summary.append(
            {
                "model_role": role,
                "model": role_models[role],
                "condition": cond,
                "criterion": criterion,
                "n_responses": n,
                "mean_of_item_means": f"{mean(r['item_mean'] for r in items):.6f}",
                "exact_agreement_n": exact,
                "exact_agreement_pct": f"{100 * exact / n:.4f}",
                "mean_absolute_disagreement": (
                    f"{mean(r['absolute_difference'] for r in items):.6f}"
                ),
                "discordant_items": n - exact,
            }
        )
    require(len(summary) == 28, "Esperados 28 grupos modelo-condicao-criterio.")
    indexed = {
        (r["model_role"], r["condition"], r["query_id"], r["criterion"]): r
        for r in values
    }
    require(len(indexed) == 835, "Itens-criterio de modelo repetidos.")
    pairs = []
    for cond in CONDITIONS:
        crit = (
            *CRITERIA["qualidade"],
            *(CRITERIA["fundamentacao"] if cond != "no_rag" else ()),
        )
        for query in condition_queries[cond]:
            for criterion in crit:
                a, b = (indexed.get((role, cond, query, criterion)) for role in ROLES)
                status_a = status_ids[("baseline", cond, query)]["status"]
                status_b = status_ids[("candidate", cond, query)]["status"]
                require(
                    (a is not None) == (status_a == "success")
                    and (b is not None) == (status_b == "success"),
                    f"Falta nota em resposta completa: {cond}|{query}|{criterion}",
                )
                delta = (
                    (b["item_mean"] - a["item_mean"])
                    if a is not None and b is not None
                    else None
                )
                if delta is None:
                    favorable = "nao_comparavel"
                elif delta == 0:
                    favorable = "empate"
                elif delta < 0 if criterion == "hallucination_present" else delta > 0:
                    favorable = "candidate"
                else:
                    favorable = "baseline"
                pairs.append(
                    {
                        "condition": cond,
                        "query_id": query,
                        "criterion": criterion,
                        "baseline_status": status_a,
                        "candidate_status": status_b,
                        "baseline_mean": a["item_mean"] if a else "",
                        "candidate_mean": b["item_mean"] if b else "",
                        "candidate_minus_baseline": delta if delta is not None else "",
                        "favorable_to": favorable,
                    }
                )
    require(
        len(pairs) == 420
        and sum(r["favorable_to"] == "nao_comparavel" for r in pairs) == 5,
        "Matriz de pares inesperada; verificar resposta truncada.",
    )
    paired_summary = []
    for cond in CONDITIONS:
        crit = (
            *CRITERIA["qualidade"],
            *(CRITERIA["fundamentacao"] if cond != "no_rag" else ()),
        )
        for criterion in crit:
            entries = [
                r
                for r in pairs
                if r["condition"] == cond and r["criterion"] == criterion
            ]
            comparable = [r for r in entries if r["favorable_to"] != "nao_comparavel"]
            mean_delta = mean(
                float(row["candidate_minus_baseline"]) for row in comparable
            )
            paired_summary.append(
                {
                    "condition": cond,
                    "criterion": criterion,
                    "planned_pairs": len(entries),
                    "both_complete": len(comparable),
                    "missing_pair": len(entries) - len(comparable),
                    "mean_candidate_minus_baseline": f"{mean_delta:.6f}",
                    "candidate_better": sum(
                        r["favorable_to"] == "candidate" for r in comparable
                    ),
                    "baseline_better": sum(
                        r["favorable_to"] == "baseline" for r in comparable
                    ),
                    "ties": sum(r["favorable_to"] == "empate" for r in comparable),
                }
            )
    require(len(paired_summary) == 14, "Esperados 14 resumos de comparacao pareada.")
    output_data = {
        "notas_por_resposta_criterio.csv": encode(values, QUALITY_COLUMNS),
        "resumo_por_modelo_condicao_criterio.csv": encode(summary, GROUP_COLUMNS),
        "comparacao_pareada_criterio.csv": encode(pairs, PAIR_COLUMNS),
        "resumo_pareado_criterio.csv": encode(paired_summary, PAIRED_SUMMARY_COLUMNS),
    }
    report = {
        "status": "descegamento_administrativo_descritivo",
        "n_generation_attempts": 180,
        "complete_answers": 179,
        "truncated_responses": 1,
        "human_judgments": 596,
        "dimension_items": 298,
        "criterion_item_records": 835,
        "planned_criterion_comparisons": 420,
        "complete_criterion_comparisons": 415,
        "missing_criterion_comparisons": 5,
        "aggregation": (
            "Media simples dos dois julgamentos por resposta e criterio, "
            "sem substituir notas originais"
        ),
        "hallucination_interpretation": (
            "Media dos dois indicadores 0/1; 0.5 indica discordancia, "
            "nao rotulo definitivo"
        ),
        "limitations": (
            "Somente analise descritiva. Valores medios de escalas ordinais nao "
            "comprovam significancia ou validade factual; ausencia de adjudicacao. "
            "Comparacao pareada exclui o Qwen optimized q29 truncado. "
            "As duas anotacoes nao representam duas respostas independentes."
        ),
        "source_sha256": {
            "key": sha(key_path.read_bytes()),
            "judgments": sha(judges_path.read_bytes()),
            "allocation": sha(allocations_path.read_bytes()),
            "disagreement": sha(disagreement_path.read_bytes()),
            "generation_status": sha(status_path.read_bytes()),
            "agreement_audit": sha(agreement_audit_path.read_bytes()),
            "reviews_audit": sha(returns_audit_path.read_bytes()),
        },
        "output_sha256": {name: sha(raw) for name, raw in output_data.items()},
    }
    output_data["auditoria_descegamento.json"] = (
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    for name, data in output_data.items():
        write_new_or_verify(OUTPUT / name, data)
    print("=" * 74)
    print("DESCEGAMENTO ADMINISTRATIVO E ANALISE DESCRITIVA CONCLUIDOS")
    print("Julgamentos originais: 596 | respostas completas: 179")
    print("Itens-dimensao: 298 | itens-criterio: 835")
    print("Comparacoes criterio-pergunta: 415/420 (5 nao comparaveis)")
    print("Criterio 'hallucination_present': menor media e melhor.")
    print("\nResumo das diferencas pareadas (Llama - Qwen):")
    for r in paired_summary:
        print(
            f"  {r['condition']:10s} {r['criterion']:22s} "
            f"n={r['both_complete']:2d} "
            f"delta={r['mean_candidate_minus_baseline']:>9s} "
            f"Llama={r['candidate_better']:2d} "
            f"Qwen={r['baseline_better']:2d} empates={r['ties']:2d}"
        )
    print("\nSaidas administrativas:", OUTPUT)
    print("Nenhuma nota original, resposta ou configuracao foi alterada.")


if __name__ == "__main__":
    main()
