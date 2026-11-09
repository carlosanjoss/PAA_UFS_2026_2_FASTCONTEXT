"""Localiza a revisao cuja assinatura reproduz o baseline RAG FastContext.

Somente leitura: nunca altera os dados nem a arvore Git.
Execute na raiz do projeto com:
    python -m scripts.find_fastcontext_baseline_revision
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from functools import lru_cache
from pathlib import Path

import yaml

from experiments.run_rag_experiments import (
    CONFIG_PATH,
    PIPELINE_PATH,
    PROMPT_PATH,
    RUNNER_PATH,
    SIGNATURE_SOURCE_PATHS,
    _calculate_signature,
)
from src.utils.config import PROJECT_ROOT

ROOT = Path(PROJECT_ROOT)
BASELINE = ROOT / "experiments/rag_results/generations.csv"
GT_PATH = ROOT / "data/ground_truth/ground_truth.final.json"


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


CORE = {
    "config_sha256": relative(CONFIG_PATH),
    "ground_truth_sha256": relative(GT_PATH),
    "prompt_sha256": relative(PROMPT_PATH),
    "pipeline_sha256": relative(PIPELINE_PATH),
    "runner_sha256": relative(RUNNER_PATH),
}
SOURCES = [relative(path) for path in SIGNATURE_SOURCE_PATHS]
ALL_PATHS = list(dict.fromkeys([*CORE.values(), *SOURCES]))


def git(*args: str, required: bool = True) -> bytes | None:
    result = subprocess.run(
        ["git", "-C", str(ROOT), *args],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        if not required:
            return None
        raise RuntimeError(
            f"Falha git {' '.join(args)}: " + result.stderr.decode("utf-8", "replace")
        )
    return result.stdout


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


@lru_cache(maxsize=25000)
def blob(revision: str, path: str) -> bytes | None:
    return git("show", f"{revision}:{path}", required=False)


def calculate(revision: str, style: str, fingerprint: str) -> tuple[str, dict] | None:
    """Replica exatamente o payload de _calculate_signature de um commit."""
    blobs = {}
    for path in ALL_PATHS:
        value = blob(revision, path)
        if value is None:
            return None
        blobs[path] = value

    try:
        config = yaml.safe_load(blobs[CORE["config_sha256"]])["rag_experiment"]
        gt = json.loads(blobs[CORE["ground_truth_sha256"]].decode("utf-8-sig"))
        gen = config["generation"]
        query_ids = [q["query_id"] for q in gt["queries"]]
    except (KeyError, TypeError, ValueError, UnicodeError):
        return None

    def source_key(path: str) -> str:
        return path.replace("/", "\\") if style == "windows" else path

    payload = {
        "config_sha256": sha(blobs[CORE["config_sha256"]]),
        "ground_truth_sha256": sha(blobs[CORE["ground_truth_sha256"]]),
        "prompt_sha256": sha(blobs[CORE["prompt_sha256"]]),
        "pipeline_sha256": sha(blobs[CORE["pipeline_sha256"]]),
        "runner_sha256": sha(blobs[CORE["runner_sha256"]]),
        "source_sha256": {source_key(path): sha(blobs[path]) for path in SOURCES},
        "corpus_fingerprint": fingerprint,
        "query_ids": query_ids,
        "conditions": list(config["conditions"]),
        "top_k": int(config["retrieval"]["top_k"]),
        "provider": str(gen["provider"]),
        "model": str(gen["model"]),
        "temperature": float(gen["temperature"]),
        "max_tokens": int(gen["max_tokens"]),
        "think": bool(gen["think"]),
        "generation_seed": int(gen["seed"]),
        "smoke": False,
    }
    digest = sha(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return digest, config


def main() -> None:
    with BASELINE.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))

    signatures = {row["experiment_signature"] for row in rows}
    fingerprints = {row["corpus_fingerprint"] for row in rows}
    if len(signatures) != 1 or len(fingerprints) != 1 or len(rows) != 90:
        raise RuntimeError("Baseline incompleto ou metadados inconsistentes.")
    target = next(iter(signatures))
    fingerprint = next(iter(fingerprints))

    head = git("rev-parse", "HEAD").decode().strip()
    local_gt = json.loads(GT_PATH.read_text(encoding="utf-8-sig"))
    current = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["rag_experiment"]
    gen = current["generation"]
    real_current = _calculate_signature(
        config_path=CONFIG_PATH,
        ground_truth_path=GT_PATH,
        prompt_path=PROMPT_PATH,
        pipeline_path=PIPELINE_PATH,
        runner_path=RUNNER_PATH,
        corpus_fingerprint=fingerprint,
        query_ids=tuple(q["query_id"] for q in local_gt["queries"]),
        conditions=tuple(current["conditions"]),
        top_k=int(current["retrieval"]["top_k"]),
        expected_provider=str(gen["provider"]),
        expected_model=str(gen["model"]),
        temperature=float(gen["temperature"]),
        max_tokens=int(gen["max_tokens"]),
        think=bool(gen["think"]),
        generation_seed=int(gen["seed"]),
        smoke=False,
    )

    print("=" * 72)
    print("LOCALIZACAO DA ASSINATURA HISTORICA FASTCONTEXT")
    print("=" * 72)
    print("Baseline: ", target)
    print("Atual:    ", real_current)
    print("HEAD:     ", head)
    print()

    # O formato das chaves no hash depende do SO (Windows vs POSIX).
    current_style = "windows" if __import__("os").name == "nt" else "posix"
    head_native = calculate(head, current_style, fingerprint)
    if head_native is None or head_native[0] != real_current:
        raise RuntimeError(
            "Nao foi possivel reproduzir a assinatura atual a partir do "
            "Git HEAD. Verifique o codigo da assinatura antes de prosseguir."
        )

    for style in ("windows", "posix"):
        candidate = calculate(head, style, fingerprint)
        if candidate is not None:
            print(f"Assinatura HEAD ({style}): {candidate[0]}")
            if candidate[0] == target:
                print("\nCORRESPONDENCIA: mesmo commit, estilo de caminhos distinto.")
                print("Nenhum arquivo foi alterado.")
                return

    raw_revisions = git("rev-list", "--all", "--", *ALL_PATHS)
    revisions = list(dict.fromkeys([head, *raw_revisions.decode().splitlines()]))
    print(f"\nRevisoes candidatas: {len(revisions)}")

    for index, revision in enumerate(revisions, 1):
        if index % 25 == 0:
            print(f"Avaliadas {index}/{len(revisions)} revisoes...")
        for style in ("windows", "posix"):
            candidate = calculate(revision, style, fingerprint)
            if candidate is None or candidate[0] != target:
                continue
            cfg = candidate[1]
            g = cfg["generation"]
            print("\nASSINATURA HISTORICA ENCONTRADA")
            print("Revisao:  ", revision)
            print("Estilo:   ", style)
            print("Assinatura:", candidate[0])
            print("Modelo:   ", g["model"])
            print("Temperatura:", g["temperature"])
            print("Max tokens:", g["max_tokens"])
            print("Truncation retries:", g["max_truncation_retries"])
            print("Retry token limit: ", g["max_retry_tokens"])
            print("\nArquivos de assinatura diferentes do HEAD:")
            changed = [
                path for path in ALL_PATHS if blob(revision, path) != blob(head, path)
            ]
            for path in changed:
                print(" -", path)
            if not changed:
                print(" (nenhum)")
            print("\nNenhum arquivo foi alterado.")
            return

    print("\nSEM CORRESPONDENCIA NO HISTORICO LOCAL ALCANCAVEL")
    print(
        "A assinatura pode ter vindo de revisao nao presente neste clone, "
        "ou de arquivos sem commit no momento da geracao."
    )
    print(
        "Nao remova a validacao nem congele prompts como se fossem "
        "historicamente identicos."
    )
    print("Nenhum arquivo foi alterado.")


if __name__ == "__main__":
    main()
