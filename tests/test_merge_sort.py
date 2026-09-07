"""
tests/test_merge_sort.py
Testes do Merge Sort manual (score DESC, chunk_id ASC).

O oráculo de referência usa ``sorted()`` APENAS no teste; a implementação
avaliada (src/algorithms/merge_sort.py) não usa ordenação pronta.
"""

import random

import pytest

from src.algorithms.merge_sort import merge_sort, merge_sort_keys
from src.algorithms.ordering import default_key


# ---------------------------------------------------------------------------
# Oráculo de referência (permitido APENAS no teste)
# ---------------------------------------------------------------------------
def reference_sort(items):
    """Ordena por score DESC, chunk_id ASC usando sorted() como referência."""
    return sorted(items, key=lambda it: (-default_key(it)[0], default_key(it)[1]))


def make_item(score, chunk_id):
    """Candidato no formato dos retrievers do Wilson: (score, chunk_dict)."""
    return (score, {"chunk_id": chunk_id})


# ---------------------------------------------------------------------------
# Casos de borda de tamanho
# ---------------------------------------------------------------------------
def test_empty_list():
    ordered, stats = merge_sort([])
    assert ordered == []
    assert stats.comparisons == 0


def test_single_element():
    items = [make_item(0.5, "c1")]
    ordered, stats = merge_sort(items)
    assert ordered == items
    assert stats.comparisons == 0


def test_two_elements_need_swap():
    items = [make_item(0.2, "c1"), make_item(0.9, "c2")]
    ordered, _ = merge_sort(items)
    assert [it[1]["chunk_id"] for it in ordered] == ["c2", "c1"]


# ---------------------------------------------------------------------------
# Ordenações principais
# ---------------------------------------------------------------------------
def test_already_sorted():
    items = [make_item(0.9, "a"), make_item(0.5, "b"), make_item(0.1, "c")]
    ordered, _ = merge_sort(items)
    assert ordered == items


def test_reverse_order():
    items = [make_item(0.1, "c"), make_item(0.5, "b"), make_item(0.9, "a")]
    ordered, _ = merge_sort(items)
    assert [it[1]["chunk_id"] for it in ordered] == ["a", "b", "c"]


def test_tie_breaks_by_chunk_id_asc():
    # exemplo da seção 8 do documento de contexto
    items = [
        make_item(0.72, "chunk_10"),
        make_item(0.91, "chunk_02"),
        make_item(0.72, "chunk_04"),
    ]
    ordered, _ = merge_sort(items)
    assert [it[1]["chunk_id"] for it in ordered] == ["chunk_02", "chunk_04", "chunk_10"]


def test_many_ties():
    items = [make_item(1.0, f"c{i:02d}") for i in (5, 1, 9, 3, 7, 2)]
    ordered, _ = merge_sort(items)
    ids = [it[1]["chunk_id"] for it in ordered]
    assert ids == sorted(ids)  # todos empatados -> chunk_id ASC


def test_negative_and_zero_scores():
    items = [make_item(-1.0, "a"), make_item(0.0, "b"), make_item(-0.5, "c")]
    ordered, _ = merge_sort(items)
    assert [it[1]["chunk_id"] for it in ordered] == ["b", "c", "a"]


# ---------------------------------------------------------------------------
# Propriedades: equivalência com oráculo, preservação e determinismo
# ---------------------------------------------------------------------------
def test_equivalence_with_reference_random():
    rng = random.Random(42)
    for _ in range(50):
        n = rng.randint(0, 60)
        items = [
            make_item(round(rng.uniform(-2, 2), 2), f"c{rng.randint(0, 999):03d}_{i}")
            for i in range(n)
        ]
        ordered, _ = merge_sort(items)
        expected = reference_sort(items)
        assert [default_key(it) for it in ordered] == [default_key(it) for it in expected]


def test_preserves_all_elements():
    items = [make_item(0.5, f"c{i}") for i in range(20)]
    ordered, _ = merge_sort(items)
    assert len(ordered) == len(items)
    assert {id(it) for it in ordered} == {id(it) for it in items}


def test_determinism():
    items = [
        make_item(0.5, "c03"),
        make_item(0.9, "c01"),
        make_item(0.5, "c02"),
    ]
    first, _ = merge_sort(items)
    second, _ = merge_sort(items)
    assert [default_key(it) for it in first] == [default_key(it) for it in second]


def test_input_not_mutated():
    items = [make_item(0.1, "b"), make_item(0.9, "a")]
    snapshot = list(items)
    merge_sort(items)
    assert items == snapshot


# ---------------------------------------------------------------------------
# Instrumentação
# ---------------------------------------------------------------------------
def test_comparison_count_upper_bound():
    # Merge Sort faz no máximo N*ceil(log2 N) comparações; verificamos coerência.
    items = [make_item(random.random(), f"c{i:03d}") for i in range(64)]
    _, stats = merge_sort(items)
    assert stats.comparisons > 0
    assert stats.comparisons <= 64 * 6  # 64 * log2(64)


def test_keys_variant():
    keys = [(0.72, "chunk_10"), (0.91, "chunk_02"), (0.72, "chunk_04")]
    ordered, _ = merge_sort_keys(keys)
    assert ordered == [(0.91, "chunk_02"), (0.72, "chunk_04"), (0.72, "chunk_10")]
