"""
tests/test_topk_heap.py
Testes da seleção Top-k com min-heap manual (score DESC, chunk_id ASC).

O oráculo de referência usa ``sorted()`` APENAS no teste; a implementação
avaliada (src/algorithms/topk_heap.py) não usa ordenação pronta nem heapq.
"""

import random

from src.algorithms.ordering import default_key
from src.algorithms.topk_heap import top_k


def reference_topk(items, k):
    """Top-k de referência via sorted() + fatiamento (apenas no teste)."""
    if k <= 0:
        return []
    ordered = sorted(items, key=lambda it: (-default_key(it)[0], default_key(it)[1]))
    return ordered[:k]


def make_item(score, chunk_id):
    """Candidato no formato dos retrievers do Wilson: (score, chunk_dict)."""
    return (score, {"chunk_id": chunk_id})


def ids(items):
    return [it[1]["chunk_id"] for it in items]


# ---------------------------------------------------------------------------
# Casos de borda de k
# ---------------------------------------------------------------------------
def test_k_zero():
    items = [make_item(0.9, "a"), make_item(0.5, "b")]
    result, stats = top_k(items, 0)
    assert result == []
    assert stats.max_heap_size == 0


def test_k_negative():
    items = [make_item(0.9, "a")]
    result, _ = top_k(items, -3)
    assert result == []


def test_k_one():
    items = [make_item(0.5, "b"), make_item(0.9, "a"), make_item(0.1, "c")]
    result, _ = top_k(items, 1)
    assert ids(result) == ["a"]


def test_k_equals_n():
    items = [make_item(0.5, "b"), make_item(0.9, "a"), make_item(0.1, "c")]
    result, _ = top_k(items, 3)
    assert ids(result) == ["a", "b", "c"]


def test_k_greater_than_n():
    items = [make_item(0.5, "b"), make_item(0.9, "a")]
    result, _ = top_k(items, 10)
    assert ids(result) == ["a", "b"]
    assert len(result) == 2


def test_empty_list():
    result, stats = top_k([], 5)
    assert result == []
    assert stats.max_heap_size == 0


def test_empty_list_and_k_zero():
    result, _ = top_k([], 0)
    assert result == []


# ---------------------------------------------------------------------------
# Ordenação e desempate
# ---------------------------------------------------------------------------
def test_output_ordered_desc():
    items = [make_item(0.1, "c"), make_item(0.9, "a"), make_item(0.5, "b")]
    result, _ = top_k(items, 3)
    assert ids(result) == ["a", "b", "c"]


def test_tie_breaks_by_chunk_id_asc():
    items = [
        make_item(0.72, "chunk_10"),
        make_item(0.91, "chunk_02"),
        make_item(0.72, "chunk_04"),
    ]
    result, _ = top_k(items, 3)
    assert ids(result) == ["chunk_02", "chunk_04", "chunk_10"]


def test_ties_at_cutoff():
    # Empate exatamente na fronteira do corte: chunk_id ASC decide quem entra.
    items = [
        make_item(0.9, "a"),
        make_item(0.5, "b1"),
        make_item(0.5, "b2"),
        make_item(0.5, "b3"),
    ]
    result, _ = top_k(items, 2)
    # top-2: score 0.9 (a) e, entre os 0.5, o menor chunk_id (b1).
    assert ids(result) == ["a", "b1"]


def test_many_ties():
    items = [make_item(1.0, f"c{i:02d}") for i in (5, 1, 9, 3, 7, 2)]
    result, _ = top_k(items, 4)
    assert ids(result) == sorted(ids(items))[:4]


def test_negative_and_zero_scores():
    items = [make_item(-1.0, "a"), make_item(0.0, "b"), make_item(-0.5, "c")]
    result, _ = top_k(items, 2)
    assert ids(result) == ["b", "c"]


# ---------------------------------------------------------------------------
# Propriedades: os k são realmente os melhores; equivalência com oráculo
# ---------------------------------------------------------------------------
def test_selects_actual_best_random():
    rng = random.Random(99)
    for _ in range(60):
        n = rng.randint(0, 80)
        k = rng.randint(0, n + 5)
        items = [
            make_item(round(rng.uniform(-3, 3), 2), f"c{rng.randint(0, 999):03d}_{i}")
            for i in range(n)
        ]
        result, _ = top_k(items, k)
        expected = reference_topk(items, k)
        assert [default_key(it) for it in result] == [
            default_key(it) for it in expected
        ]


def test_result_size_capped_at_k():
    items = [make_item(random.random(), f"c{i:03d}") for i in range(50)]
    result, _ = top_k(items, 7)
    assert len(result) == 7


def test_determinism():
    items = [make_item(0.5, "c03"), make_item(0.9, "c01"), make_item(0.5, "c02")]
    first, s1 = top_k(items, 2)
    second, s2 = top_k(items, 2)
    assert ids(first) == ids(second)
    assert s1.comparisons == s2.comparisons


def test_input_not_mutated():
    items = [make_item(0.1, "b"), make_item(0.9, "a")]
    snapshot = list(items)
    top_k(items, 1)
    assert items == snapshot


# ---------------------------------------------------------------------------
# Instrumentação e complexidade
# ---------------------------------------------------------------------------
def test_max_heap_size_bounded_by_k():
    items = [make_item(random.random(), f"c{i:04d}") for i in range(200)]
    _, stats = top_k(items, 5)
    assert stats.max_heap_size == 5  # nunca excede k


def test_instrumentation_counts():
    items = [make_item(random.random(), f"c{i:03d}") for i in range(30)]
    _, stats = top_k(items, 5)
    assert stats.insertions == 5  # preenche a heap até k
    assert stats.replacements >= 0
    assert stats.comparisons > 0
