"""
src/algorithms/topk_heap.py
Seleção Top-k com MIN-HEAP implementada manualmente para PAA.

Seleciona os ``k`` melhores candidatos sem ordenar todos os ``N``, segundo o
critério canônico do projeto (score decrescente; em empate, chunk_id crescente),
definido em ``src/algorithms/ordering.py``.

Ideia central:
    - Mantém-se uma heap de tamanho no máximo ``k``.
    - A "raiz" da heap é o PIOR item (o menos preferido) entre os k melhores
      vistos até agora. Assim, um novo item só entra se for melhor que a raiz,
      substituindo-a. Isso mantém, a todo momento, os k melhores candidatos.
    - Como o critério de ranking prefere score maior, o "pior" é o de menor
      score (desempate: chunk_id maior). Definimos ``_worse_root`` para a
      comparação de heap.

Complexidade:
    - Tempo: O(N log k)  (N inserções/comparações, cada operação de heap O(log k)).
    - Espaço: O(min(N, k)) efetivo, com O(k) como limite superior
      (a heap nunca contém mais que min(N, k) elementos).

Requisitos acadêmicos:
    - lógica de min-heap (sift-up / sift-down) escrita à mão; NÃO se usa ``heapq``;
    - instrumentação de comparações, inserções, substituições e tamanho máximo;
    - saída final ordenada por score DESC / chunk_id ASC.

O algoritmo é puro e agnóstico: não importa contratos de retrieval nem de RAG.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.algorithms.ordering import KeyFunc, SortKey, default_key, precedes


@dataclass
class TopKStats:
    """Instrumentação de uma execução da seleção Top-k."""

    comparisons: int = 0     # comparações de chave (operação elementar)
    insertions: int = 0      # itens inseridos na heap (heap ainda com espaço)
    replacements: int = 0    # substituições da raiz (heap cheia, item melhor)
    max_heap_size: int = 0   # maior tamanho atingido pela heap


def top_k(
    items: list[Any],
    k: int,
    key: KeyFunc = default_key,
) -> tuple[list[Any], TopKStats]:
    """
    Retorna os ``k`` melhores itens de ``items`` em ordem final de ranking.

    Args:
        items: candidatos a selecionar.
        k: número de melhores itens desejado.
        key: extrator da chave ``(score, chunk_id)``.

    Returns:
        (melhores, stats):
            - lista com no máximo ``k`` itens, ordenada por score DESC e
              chunk_id ASC;
            - ``TopKStats`` com a instrumentação da execução.

    Casos de borda:
        - ``k <= 0``  -> retorna lista vazia;
        - ``items`` vazio -> retorna lista vazia;
        - ``k >= len(items)`` -> retorna todos os itens ordenados.

    A lista de entrada não é modificada.
    """
    stats = TopKStats()

    if k <= 0 or not items:
        return [], stats

    # Heap de pares (chave, item). A raiz (índice 0) é o PIOR dos k melhores.
    heap: list[tuple[SortKey, Any]] = []

    for item in items:
        entry = (key(item), item)

        if len(heap) < k:
            # Ainda há espaço: insere e sobe (sift-up).
            heap.append(entry)
            _sift_up(heap, len(heap) - 1, stats)
            stats.insertions += 1
        else:
            # Heap cheia: substitui a raiz apenas se o novo item for MELHOR
            # que o pior atual (a raiz). "Melhor" = precede a chave da raiz.
            stats.comparisons += 1
            if precedes(entry[0], heap[0][0]):
                heap[0] = entry
                _sift_down(heap, 0, stats)
                stats.replacements += 1

        if len(heap) > stats.max_heap_size:
            stats.max_heap_size = len(heap)

    # A heap contém os k melhores, mas não em ordem de ranking.
    # Ordena a saída final por score DESC / chunk_id ASC.
    result_pairs = _heap_to_sorted(heap, stats)
    return [pair[1] for pair in result_pairs], stats


# ---------------------------------------------------------------------------
# Operações de heap (min-heap segundo "_worse_root")
# ---------------------------------------------------------------------------
def _worse(key_a: SortKey, key_b: SortKey, stats: TopKStats) -> bool:
    """
    Retorna ``True`` se ``key_a`` é PIOR que ``key_b`` no ranking, isto é, se
    ``key_b`` precede ``key_a``. Esta é a relação de ordem da min-heap: a raiz
    é o elemento "mínimo" = pior candidato.

    Conta uma comparação de chave.
    """
    stats.comparisons += 1
    return precedes(key_b, key_a)


def _sift_up(heap: list[tuple[SortKey, Any]], idx: int, stats: TopKStats) -> None:
    """Sobe o elemento em ``idx`` enquanto for pior que seu pai."""
    while idx > 0:
        parent = (idx - 1) // 2
        # Min-heap: o pai deve ser o pior (menor). Se o filho é pior que o pai,
        # eles estão fora de ordem e trocamos.
        if _worse(heap[idx][0], heap[parent][0], stats):
            heap[idx], heap[parent] = heap[parent], heap[idx]
            idx = parent
        else:
            break


def _sift_down(heap: list[tuple[SortKey, Any]], idx: int, stats: TopKStats) -> None:
    """Desce a raiz/elemento em ``idx`` restaurando a propriedade de min-heap."""
    n = len(heap)
    while True:
        left = 2 * idx + 1
        right = 2 * idx + 2
        worst = idx  # queremos o PIOR (mínimo) no topo

        if left < n and _worse(heap[left][0], heap[worst][0], stats):
            worst = left
        if right < n and _worse(heap[right][0], heap[worst][0], stats):
            worst = right

        if worst == idx:
            break

        heap[idx], heap[worst] = heap[worst], heap[idx]
        idx = worst


def _heap_to_sorted(
    heap: list[tuple[SortKey, Any]],
    stats: TopKStats,
) -> list[tuple[SortKey, Any]]:
    """
    Converte a heap (não ordenada) na ordem final de ranking:
    score DESC / chunk_id ASC.

    Extrai repetidamente a raiz (o PIOR restante) e a coloca no fim do resultado,
    equivalente a um heapsort ascendente-em-pior, produzindo a ordem desejada.
    """
    working = list(heap)
    ordered_worst_first: list[tuple[SortKey, Any]] = []

    while working:
        # A raiz é o pior; move-a para o fim removendo-a da heap.
        last = len(working) - 1
        working[0], working[last] = working[last], working[0]
        ordered_worst_first.append(working.pop())
        if working:
            _sift_down(working, 0, stats)

    # ordered_worst_first está do pior para o melhor; invertendo obtemos
    # melhor -> pior (score DESC / chunk_id ASC).
    ordered_worst_first.reverse()
    return ordered_worst_first
