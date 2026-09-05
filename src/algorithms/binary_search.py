"""
src/algorithms/binary_search.py
Implementacao manual da Busca Binaria com contagem de comparacoes para PAA.
"""

from typing import List, Any, Tuple, Optional


def binary_search(elements: List[Any], target: Any) -> Tuple[Optional[int], int]:
    """
    Executa busca binaria manual sobre uma lista ordenada.

    Args:
        elements: Lista de elementos ordenados em ordem nao decrescente.
        target: Elemento a ser localizado.

    Returns:
        Tuple[Optional[int], int]:
            - Indice onde o elemento foi encontrado (ou None se ausente).
            - Quantidade de comparacoes de chave realizadas.
    """
    comparisons = 0
    left = 0
    right = len(elements) - 1

    while left <= right:
        mid = (left + right) // 2
        comparisons += 1

        if elements[mid] == target:
            return mid, comparisons
        elif elements[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return None, comparisons