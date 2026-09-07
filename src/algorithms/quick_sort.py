"""
src/algorithms/quick_sort.py
Implementação MANUAL do Quick Sort com contagem de comparações para PAA.

Serve como algoritmo clássico de comparação com o Merge Sort. Ordena candidatos
segundo o critério canônico do projeto (score decrescente; em empate, chunk_id
crescente), definido em ``src/algorithms/ordering.py``.

Requisitos acadêmicos atendidos:
    - escolha de pivô determinística e documentada;
    - particionamento escrito à mão;
    - chamadas recursivas;
    - contagem da operação elementar (comparações de chave);
    - comportamento determinístico (importante para testes);
    - NENHUM uso de ``sorted()`` ou ``list.sort()``.

Estratégia de pivô: MEDIANA DE TRÊS (primeiro, meio, último).
    - É determinística: para a mesma entrada produz sempre o mesmo pivô.
    - Evita o pior caso O(N^2) nas entradas já ordenadas / em ordem inversa,
      que são justamente os casos degenerados do pivô "último elemento".
    - O pior caso quadrático ainda existe para entradas adversárias
      construídas contra a mediana de três (ver relatório).

O algoritmo é puro e agnóstico: não importa contratos de retrieval nem de RAG.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from src.algorithms.ordering import KeyFunc, SortKey, default_key, precedes


@dataclass
class QuickSortStats:
    """Instrumentação de uma execução do Quick Sort."""

    comparisons: int = 0     # comparações de chave (operação elementar)
    partitions: int = 0      # número de operações de particionamento
    swaps: int = 0           # trocas de elementos
    max_depth: int = 0       # profundidade máxima da recursão


def quick_sort(
    items: List[Any],
    key: KeyFunc = default_key,
) -> tuple[List[Any], QuickSortStats]:
    """
    Ordena ``items`` usando Quick Sort manual (mediana de três).

    Args:
        items: lista de candidatos a ordenar.
        key: extrator da chave de ordenação ``(score, chunk_id)``.

    Returns:
        (ordenados, stats):
            - nova lista ordenada por score DESC e chunk_id ASC;
            - ``QuickSortStats`` com comparações, partições, trocas e
              profundidade máxima.

    A lista de entrada não é modificada (trabalha-se sobre uma cópia).
    """
    stats = QuickSortStats()

    # Pré-computa chaves emparelhadas com os itens; ordena a cópia in-place.
    pairs: List[tuple[SortKey, Any]] = [(key(item), item) for item in items]

    _quick_sort_recursive(pairs, 0, len(pairs) - 1, stats, depth=1)

    return [pair[1] for pair in pairs], stats


def _quick_sort_recursive(
    pairs: List[tuple[SortKey, Any]],
    low: int,
    high: int,
    stats: QuickSortStats,
    depth: int,
) -> None:
    """Ordena in-place o subintervalo ``pairs[low..high]``."""
    if low >= high:
        return

    if depth > stats.max_depth:
        stats.max_depth = depth

    pivot_index = _partition(pairs, low, high, stats)

    # Recursão nas duas partições (excluindo o pivô, já posicionado).
    _quick_sort_recursive(pairs, low, pivot_index - 1, stats, depth + 1)
    _quick_sort_recursive(pairs, pivot_index + 1, high, stats, depth + 1)


def _partition(
    pairs: List[tuple[SortKey, Any]],
    low: int,
    high: int,
    stats: QuickSortStats,
) -> int:
    """
    Particiona ``pairs[low..high]`` em torno de um pivô (mediana de três).

    Esquema de Lomuto adaptado ao comparador canônico: coloca o pivô na posição
    ``high`` e move para a esquerda todos os elementos que precedem o pivô.

    Returns:
        índice final do pivô após o particionamento.
    """
    stats.partitions += 1

    # Escolha determinística do pivô: mediana de três (low, mid, high).
    pivot_index = _median_of_three(pairs, low, high, stats)

    # Move o pivô escolhido para o fim do intervalo.
    _swap(pairs, pivot_index, high, stats)
    pivot_key = pairs[high][0]

    # i marca a fronteira dos elementos que precedem o pivô.
    i = low
    for j in range(low, high):
        # Operação elementar: uma comparação de chave.
        stats.comparisons += 1
        if precedes(pairs[j][0], pivot_key):
            _swap(pairs, i, j, stats)
            i += 1

    # Posiciona o pivô logo após os elementos que o precedem.
    _swap(pairs, i, high, stats)
    return i


def _median_of_three(
    pairs: List[tuple[SortKey, Any]],
    low: int,
    high: int,
    stats: QuickSortStats,
) -> int:
    """
    Retorna o índice do elemento mediano entre ``low``, ``mid`` e ``high``
    segundo o critério canônico. Determinístico.

    As comparações realizadas aqui também contam como operação elementar,
    pois participam da escolha do pivô.
    """
    mid = (low + high) // 2

    a_key = pairs[low][0]
    b_key = pairs[mid][0]
    c_key = pairs[high][0]

    # Ordena logicamente as três chaves para achar a do meio.
    stats.comparisons += 1
    a_before_b = precedes(a_key, b_key)

    stats.comparisons += 1
    b_before_c = precedes(b_key, c_key)

    if a_before_b:
        if b_before_c:
            return mid  # a < b < c
        stats.comparisons += 1
        return high if precedes(a_key, c_key) else low  # a < b, c <= b
    else:
        if not b_before_c:
            return mid  # b <= a, c <= b  -> b é a mediana
        stats.comparisons += 1
        return low if precedes(a_key, c_key) else high  # b <= a, b < c


def _swap(
    pairs: List[tuple[SortKey, Any]],
    i: int,
    j: int,
    stats: QuickSortStats,
) -> None:
    """Troca dois elementos in-place, contabilizando a movimentação."""
    if i == j:
        return
    pairs[i], pairs[j] = pairs[j], pairs[i]
    stats.swaps += 1


def quick_sort_keys(
    keys: List[SortKey],
) -> tuple[List[SortKey], QuickSortStats]:
    """
    Variante utilitária que ordena diretamente uma lista de chaves
    ``(score, chunk_id)``. Útil em testes e para uso isolado do algoritmo.
    """
    return quick_sort(keys, key=lambda k: k)
