"""
tests/test_linear_search.py
Testes unitarios para a busca linear manual e contagem de operacoes elementares.
"""

from src.algorithms.linear_search import linear_search


def test_linear_search_empty_list():
    """Valida busca sobre lista vazia."""
    index, comparisons = linear_search([], "termo")
    assert index is None
    assert comparisons == 0


def test_linear_search_best_case():
    """Valida melhor caso: elemento na primeira posicao (1 comparacao)."""
    elements = ["auth", "cors", "fastapi", "jwt"]
    index, comparisons = linear_search(elements, "auth")
    assert index == 0
    assert comparisons == 1


def test_linear_search_worst_case_last_element():
    """Valida elemento na ultima posicao (N comparacoes)."""
    elements = ["auth", "cors", "fastapi", "jwt"]
    index, comparisons = linear_search(elements, "jwt")
    assert index == 3
    assert comparisons == 4


def test_linear_search_not_found():
    """Valida elemento ausente na lista (N comparacoes)."""
    elements = ["auth", "cors", "fastapi", "jwt"]
    index, comparisons = linear_search(elements, "inexistente")
    assert index is None
    assert comparisons == 4