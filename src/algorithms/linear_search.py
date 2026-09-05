"""
src/algorithms/linear_search.py
Implementacao manual da Busca Linear com contagem de comparacoes para PAA.
"""

from typing import List, Any, Tuple, Optional


def linear_search(elements: List[Any], target: Any) -> Tuple[Optional[int], int]:
    """
    Executa busca linear manual sobre uma sequencia de elementos.

    Args:
        elements: Lista de elementos (nao precisa estar ordenada).
        target: Elemento a ser localizado.

    Returns:
        Tuple[Optional[int], int]:
            - Indice onde o elemento foi encontrado pela primeira vez (ou None se ausente).
            - Quantidade de comparacoes de chave realizadas.
    """
    comparisons = 0

    for idx, item in enumerate(elements):
        comparisons += 1
        if item == target:
            return idx, comparisons

    return None, comparisons