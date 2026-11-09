"""Testes offline do executor controlado; sem download nem chamada ao Ollama.

No repositorio: pytest -q tests/test_rag_model_comparison.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from experiments.run_rag_model_comparison import (
    execute_one,
    save_run,
    sha256,
    stable_json,
    validate_record,
    write_outputs,
)
from src.rag.prompt import ContextChunk, build_rag_prompt
from src.rag.providers.base import LLMResponse, LLMTruncatedResponseError


class FakeOllama:
    def __init__(self, model: str) -> None:
        self.model = model
        self.calls = 0

    def generate(self, messages, config):
        self.calls += 1
        assert len(messages) >= 2
        assert config.temperature == 0.0
        assert config.max_tokens == 256
        assert config.seed == 42
        return LLMResponse(
            text=(
                "OK"
                if "Reply only with OK" in messages[-1].content
                else "Documented answer [chunk-1]"
            ),
            provider="ollama",
            model=self.model,
            metadata={"done_reason": "stop", "eval_count": 5},
        )


def config() -> dict:
    return {
        "baseline": {"model": "qwen2.5:3b"},
        "candidate": {"model": "llama3.2:3b"},
        "generation": {
            "temperature": 0.0,
            "max_tokens": 256,
            "think": False,
            "seed": 42,
            "max_truncation_retries": 2,
            "max_retry_tokens": 1024,
            "citation_retries": 1,
        },
        "output": {
            "generations_csv": "generations_llama32_3b.csv",
            "generations_jsonl": "generations_llama32_3b.jsonl",
            "checkpoint": "checkpoint_llama32_3b.json",
        },
    }


def manifest() -> dict:
    return {
        "frozen_prompts_sha256": "frozen-hash",
        "current_source_signature": "source-hash",
        "corpus_fingerprint": "corpus-hash",
    }


def item(condition: str) -> dict:
    question = "How to use FastAPI?"
    if condition == "no_rag":
        messages = [
            {"role": "system", "content": "Answer from memory."},
            {"role": "user", "content": question},
        ]
        context_ids = []
    else:
        chunk = ContextChunk(
            chunk_id="chunk-1",
            content="FastAPI is a web framework.",
            source_path="guide.md",
            section_title="Guide",
        )
        prompt = build_rag_prompt(query=question, chunks=[chunk])
        messages = [
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": prompt.user},
        ]
        context_ids = ["chunk-1"]
    return {
        "run_id": f"{condition}|q01",
        "condition": condition,
        "query_id": "q01",
        "category": "tutorial",
        "question": question,
        "context_chunk_ids": context_ids,
        "messages": messages,
        "prompt_sha256": sha256(stable_json(messages)),
    }


@pytest.mark.parametrize("condition", ["no_rag", "semantic"])
def test_frozen_generation_uses_config_and_real_pipeline(condition: str) -> None:
    source = item(condition)
    fake = FakeOllama("qwen2.5:3b")
    chunk_map = {
        "chunk-1": {
            "content": "FastAPI is a web framework.",
            "source_path": "guide.md",
            "section_title": "Guide",
        }
    }
    result = execute_one(
        source,
        "baseline",
        "qwen2.5:3b",
        fake,
        config(),
        manifest(),
        chunk_map,
    )
    validate_record(result, source, "baseline", "qwen2.5:3b", config(), manifest())
    assert fake.calls == 1
    assert result["model"] == "qwen2.5:3b"
    assert result["prompt_sha256"] == source["prompt_sha256"]
    if condition == "no_rag":
        assert result["citation_valid"] is None
    else:
        assert result["citation_valid"] is True
        assert result["valid_citations"] == ["chunk-1"]


def test_corrupted_result_is_rejected() -> None:
    source = item("semantic")
    fake = FakeOllama("qwen2.5:3b")
    record = execute_one(
        source,
        "baseline",
        "qwen2.5:3b",
        fake,
        config(),
        manifest(),
        {
            "chunk-1": {
                "content": "FastAPI is a web framework.",
                "source_path": "guide.md",
                "section_title": "Guide",
            }
        },
    )
    record["answer"] = "tampered"
    with pytest.raises(RuntimeError, match="Checksum invalido"):
        validate_record(record, source, "baseline", "qwen2.5:3b", config(), manifest())


def test_checkpoint_and_output_files_are_separated_by_model(tmp_path: Path) -> None:
    source = item("no_rag")
    for role, model in (("baseline", "qwen2.5:3b"), ("candidate", "llama3.2:3b")):
        record = execute_one(
            source, role, model, FakeOllama(model), config(), manifest(), {}
        )
        p = tmp_path / "runs" / role / "no_rag" / "q01.json"
        save_run(p, record)
        assert p.is_file()

    completed, expected = write_outputs(
        tmp_path, config(), manifest(), [source], smoke=False
    )
    assert (completed, expected) == (2, 2)
    assert (tmp_path / "generations_qwen25_3b.csv").is_file()
    assert (tmp_path / "generations_llama32_3b.csv").is_file()
    assert (tmp_path / "generations_both_models.csv").is_file()
    with (tmp_path / "generations_both_models.csv").open(
        encoding="utf-8", newline=""
    ) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert {r["model"] for r in rows} == {"qwen2.5:3b", "llama3.2:3b"}
    checkpoint = json.loads(
        (tmp_path / "checkpoint_both_models.json").read_text(encoding="utf-8")
    )
    assert checkpoint["status"] == "complete"
    assert checkpoint["completed_rows"] == 2
    assert write_outputs(tmp_path, config(), manifest(), [source], smoke=False) == (
        2,
        2,
    )


def test_refuse_to_overwrite_existing_run(tmp_path: Path) -> None:
    record = execute_one(
        item("no_rag"),
        "baseline",
        "qwen2.5:3b",
        FakeOllama("qwen2.5:3b"),
        config(),
        manifest(),
        {},
    )
    path = tmp_path / "runs" / "baseline" / "no_rag" / "q01.json"
    save_run(path, record)
    with pytest.raises(RuntimeError, match="recusei sobrescrever"):
        save_run(path, record)


class TruncatingFakeOllama:
    """Simula truncamento ate um limite de tokens conhecido."""

    def __init__(self, model: str, required_tokens: int) -> None:
        self.model = model
        self.required_tokens = required_tokens
        self.requested: list[int] = []

    def generate(self, messages, config):
        self.requested.append(config.max_tokens)
        assert config.temperature == 0.0
        assert config.seed == 42
        assert config.think is False
        return LLMResponse(
            text="resposta completa"
            if config.max_tokens >= self.required_tokens
            else "resposta truncada",
            model=self.model,
            provider="ollama",
            metadata={
                "done_reason": "stop"
                if config.max_tokens >= self.required_tokens
                else "length",
                "requested_max_tokens": config.max_tokens,
            },
        )


@pytest.mark.parametrize(
    "role,model",
    [
        ("baseline", "qwen2.5:3b"),
        ("candidate", "llama3.2:3b"),
    ],
)
def test_no_rag_truncation_retry_256_to_512(role: str, model: str) -> None:
    source = item("no_rag")
    fake = TruncatingFakeOllama(model, required_tokens=512)
    record = execute_one(source, role, model, fake, config(), manifest(), {})
    validate_record(record, source, role, model, config(), manifest())
    assert fake.requested == [256, 512]
    assert record["answer"] == "resposta completa"
    assert record["generation_metadata"]["truncation_retries"] == 1
    assert record["generation_metadata"]["effective_max_tokens"] == 512
    assert record["max_tokens"] == 256  # Limite inicial congelado


def test_no_rag_truncation_retry_256_to_512_to_1024() -> None:
    source = item("no_rag")
    fake = TruncatingFakeOllama("llama3.2:3b", required_tokens=1024)
    record = execute_one(
        source,
        "candidate",
        "llama3.2:3b",
        fake,
        config(),
        manifest(),
        {},
    )
    assert fake.requested == [256, 512, 1024]
    assert record["generation_metadata"]["truncation_retries"] == 2
    assert record["generation_metadata"]["effective_max_tokens"] == 1024
    validate_record(
        record,
        source,
        "candidate",
        "llama3.2:3b",
        config(),
        manifest(),
    )


def test_no_rag_aborts_after_two_truncated_retries() -> None:
    source = item("no_rag")
    fake = TruncatingFakeOllama("llama3.2:3b", required_tokens=2048)
    with pytest.raises(LLMTruncatedResponseError, match="permaneceu truncada"):
        execute_one(
            source,
            "candidate",
            "llama3.2:3b",
            fake,
            config(),
            manifest(),
            {},
        )
    assert fake.requested == [256, 512, 1024]
