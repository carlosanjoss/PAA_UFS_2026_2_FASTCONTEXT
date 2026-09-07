"""
src/algorithms/merge_sort.py
Implementação MANUAL do Merge Sort com contagem de comparações para PAA.

Ordena candidatos segundo o critério canônico do projeto
(score decrescente; em empate, chunk_id crescente), definido em
``src/algorithms/ordering.py``.

Requisitos acadêmicos atendidos:
    - divisão explícita da lista;
    - chamadas recursivas;
    - rotina de merge escrita à mão;
    - contagem da operação elementar (comparações de chave);
    - NENHUM uso de ``sorted()`` ou ``list.sort()``.

O algoritmo é puro e agnóstico: não importa contratos de retrieval nem de RAG.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from src.algorithms.ordering import KeyFunc, SortKey, default_key, precedes


@dataclass
class MergeSortStats:
    """Instrumentação de uma execução do Merge Sort."""

    comparisons: int = 0  # comparações de chave (operação elementar)
    merges: int = 0       # número de operações de merge realizadas
    moves: int = 0        # atribuições de elementos durante os merges


def merge_sort(
    items: List[Any],
    key: KeyFunc = default_key,
) -> tuple[List[Any], MergeSortStats]:
    """
    Ordena ``items`` de forma estável usando Merge Sort manual.

    Args:
        items: lista de candidatos a ordenar. Cada item é convertido em sua
            chave ``(score, chunk_id)`` pelo extrator ``key``.
        key: função que extrai a chave de ordenação de um item.
            Por padrão, aceita ``(score, payload)`` ou objetos/dicts com
            ``score``/``chunk_id`` (ver ``ordering.default_key``).

    Returns:
        (ordenados, stats):
            - nova lista ordenada por score DESC e chunk_id ASC;
            - ``MergeSortStats`` com comparações, merges e movimentações.

    A lista de entrada não é modificada.
    """
    stats = MergeSortStats()

    # Pré-computa as chaves uma única vez para não recalcular durante a recursão
    # e para manter a contagem de comparações restrita à comparação de chaves.
    indexed: List[tuple[SortKey, Any]] = [(key(item), item) for item in items]

    ordered_pairs = _merge_sort_recursive(indexed, stats)

    return [pair[1] for pair in ordered_pairs], stats


def _merge_sort_recursive(
    pairs: List[tuple[SortKey, Any]],
    stats: MergeSortStats,
) -> List[tuple[SortKey, Any]]:
    """Recursão principal: divide, ordena metades e faz o merge."""
    n = len(pairs)

    # Caso base: 0 ou 1 elemento já está ordenado.
    if n <= 1:
        return pairs

    # Divisão explícita em duas metades.
    mid = n // 2
    left = _merge_sort_recursive(pairs[:mid], stats)
    right = _merge_sort_recursive(pairs[mid:], stats)

    # Combinação das metades ordenadas.
    return _merge(left, right, stats)


def _merge(
    left: List[tuple[SortKey, Any]],
    right: List[tuple[SortKey, Any]],
    stats: MergeSortStats,
) -> List[tuple[SortKey, Any]]:
    """
    Intercala duas listas já ordenadas preservando o critério canônico.

    Estabilidade: em empate de chave, o elemento da metade esquerda (que
    ocorre antes na lista original) é escolhido primeiro. Como ``chunk_id``
    é único no projeto, empates de chave não devem ocorrer entre itens
    distintos, mas a estabilidade é mantida por segurança.
    """
    stats.merges += 1

    merged: List[tuple[SortKey, Any]] = []
    i = 0
    j = 0

    while i < len(left) and j < len(right):
        # Operação elementar: uma comparação de chave.
        stats.comparisons += 1
        key_left = left[i][0]
        key_right = right[j][0]

        # Se right precede estritamente left, consome right; caso contrário
        # consome left (mantendo estabilidade em empates).
        if precedes(key_right, key_left):
            merged.append(right[j])
            j += 1
        else:
            merged.append(left[i])
            i += 1
        stats.moves += 1

    # Elementos restantes de uma das metades (a outra se esgotou).
    while i < len(left):
        merged.append(left[i])
        i += 1
        stats.moves += 1

    while j < len(right):
        merged.append(right[j])
        j += 1
        stats.moves += 1

    return merged


def merge_sort_keys(
    keys: List[SortKey],
) -> tuple[List[SortKey], MergeSortStats]:
    """
    Variante utilitária que ordena diretamente uma lista de chaves
    ``(score, chunk_id)``. Útil em testes e para uso isolado do algoritmo.
    """
    return merge_sort(keys, key=lambda k: k)
