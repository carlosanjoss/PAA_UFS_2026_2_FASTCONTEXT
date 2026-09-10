from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.app.bootstrap import create_application
from src.rag.prompt import ContextChunk, build_rag_prompt
from src.services.factory import create_fastcontext_service


ground_truth_path = Path(
    "data/ground_truth/ground_truth.final.json"
)

ground_truth = json.loads(
    ground_truth_path.read_text(
        encoding="utf-8"
    )
)

query = ground_truth["queries"][0]["question"]

application = create_application()

algorithms = (
    "linear",
    "indexed",
    "optimized",
)

hashes = {}
retrieved_ids = {}

for algorithm in algorithms:
    service = create_fastcontext_service(
        algorithm=algorithm,
        registry=application.registry,
    )

    result = service.retrieve(
        query=query,
        top_k=5,
    )

    chunks = [
        ContextChunk(
            chunk_id=chunk.chunk_id,
            content=chunk.content,
            source_path=chunk.source_path,
            section_title=chunk.section_title,
            score=chunk.score,
        )
        for chunk in result.chunks
    ]

    prompt = build_rag_prompt(
        query=query,
        chunks=chunks,
    )

    serialized = (
        "SYSTEM\n"
        + prompt.system
        + "\nUSER\n"
        + prompt.user
    ).encode("utf-8")

    digest = hashlib.sha256(
        serialized
    ).hexdigest()

    hashes[algorithm] = digest
    retrieved_ids[algorithm] = tuple(
        chunk.chunk_id
        for chunk in result.chunks
    )

    print("=" * 70)
    print("Algorithm:", algorithm)
    print("Prompt SHA-256:", digest)
    print("Prompt bytes:", len(serialized))
    print("Chunk IDs:")
    for chunk_id in retrieved_ids[algorithm]:
        print(" -", chunk_id)

print()
print("=" * 70)
print("FINAL AUDIT")
print("=" * 70)

print(
    "Distinct Top-5:",
    len(
        set(
            retrieved_ids.values()
        )
    ),
)

print(
    "Distinct prompt hashes:",
    len(
        set(
            hashes.values()
        )
    ),
)

print(
    "PROMPTS IDENTICAL:",
    len(
        set(
            hashes.values()
        )
    )
    == 1,
)
