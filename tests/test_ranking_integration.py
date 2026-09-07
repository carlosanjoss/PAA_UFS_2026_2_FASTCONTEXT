"""
tests/test_ranking_integration.py
Testes de integração de RANKING (nível de algoritmo).

Objetivo: garantir que os algoritmos clássicos de Mateus (Merge Sort, Quick Sort
e Top-k) produzem exatamente a MESMA ordenação que os retrievers do Wilson
assumem hoje, isto é, o critério:

    key = (-score, chunk_id)   # score DESC, chunk_id ASC

Estes testes NÃO importam nem alteram os retrievers/contratos dos colegas. Eles
usam candidatos no mesmo formato interno dos retrievers — tuplas
`(score, chunk_dict)` com `chunk_dict["chunk_id"]` — para atuar como testes de
regressão do contrato de ordenação, protegendo a integração futura.
"""

import random

from src.algorithms.merge_sort import merge_sort
from src.algorithms.quick_sort import quick_sort
from src.algorithms.topk_heap import top_k
from src.algorithms.ordering import default_key


def make_candidate(score, chunk_id, content="conteudo"):
    """Candidato no formato usado internamente pelos retrievers do Wilson."""
    return (score, {"chunk_id": chunk_id, "content": content})


def retriever_style_order(candidates):
    """
    Replica a ordenação que os retrievers do Wilson aplicam hoje:
    `candidates.sort(key=lambda item: (-item[0], item[1]["chunk_id"]))`.
    Usada APENAS como oráculo de referência no teste.
    """
    ordered = sorted(candidates, key=lambda item: (-item[0], item[1]["chunk_id"]))
    return [c[1]["chunk_id"] for c in ordered]


def test_merge_sort_matches_retriever_ordering():
    rng = random.Random(2026)
    for _ in range(30):
        n = rng.randint(0, 40)
        candidates = [
            make_candidate(round(rng.uniform(0, 3), 2), f"chunk_{rng.randint(0, 99):02d}_{i}")
            for i in range(n)
        ]
        ordered, _ = merge_sort(candidates)
        got = [c[1]["chunk_id"] for c in ordered]
        assert got == retriever_style_order(candidates)


def test_quick_sort_matches_retriever_ordering():
    rng = random.Random(4096)
    for _ in range(30):
        n = rng.randint(0, 40)
        candidates = [
            make_candidate(round(rng.uniform(0, 3), 2), f"chunk_{rng.randint(0, 99):02d}_{i}")
            for i in range(n)
        ]
        ordered, _ = quick_sort(candidates)
        got = [c[1]["chunk_id"] for c in ordered]
        assert got == retriever_style_order(candidates)


def test_topk_matches_retriever_slice():
    """Top-k deve equivaler a ordenar (estilo retriever) e fatiar os k primeiros."""
    rng = random.Random(777)
    for _ in range(30):
        n = rng.randint(0, 40)
        k = rng.randint(0, n + 3)
        candidates = [
            make_candidate(round(rng.uniform(0, 3), 2), f"chunk_{rng.randint(0, 99):02d}_{i}")
            for i in range(n)
        ]
        result, _ = top_k(candidates, k)
        got = [c[1]["chunk_id"] for c in result]
        expected = retriever_style_order(candidates)[: max(k, 0)]
        assert got == expected


def test_ranking_example_from_context_doc():
    """Exemplo canônico da seção 8 do documento de contexto."""
    candidates = [
        make_candidate(0.72, "chunk_10"),
        make_candidate(0.91, "chunk_02"),
        make_candidate(0.72, "chunk_04"),
    ]
    expected = ["chunk_02", "chunk_04", "chunk_10"]

    merged, _ = merge_sort(candidates)
    quicked, _ = quick_sort(candidates)
    topped, _ = top_k(candidates, 3)

    assert [c[1]["chunk_id"] for c in merged] == expected
    assert [c[1]["chunk_id"] for c in quicked] == expected
    assert [c[1]["chunk_id"] for c in topped] == expected


def test_three_algorithms_agree_on_full_ranking():
    """Merge, Quick e Top-k(k=N) devem concordar na ordenação completa."""
    rng = random.Random(31337)
    for _ in range(20):
        n = rng.randint(0, 50)
        candidates = [
            make_candidate(round(rng.uniform(-2, 2), 2), f"chunk_{rng.randint(0, 999):03d}_{i}")
            for i in range(n)
        ]
        merged, _ = merge_sort(candidates)
        quicked, _ = quick_sort(candidates)
        topped, _ = top_k(candidates, n if n > 0 else 1)

        merged_keys = [default_key(c) for c in merged]
        quicked_keys = [default_key(c) for c in quicked]
        topped_keys = [default_key(c) for c in topped]

        assert merged_keys == quicked_keys
        if n > 0:
            assert topped_keys == merged_keys
