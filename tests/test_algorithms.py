"""
tests/test_algorithms.py
Testes unitarios para a busca binaria manual e contagem de operacoes elementares.
"""

from src.algorithms.binary_search import binary_search


def test_binary_search_empty_list():
    """Valida o caso de busca em vetor vazio."""
    index, comparisons = binary_search([], "termo")
    assert index is None
    assert comparisons == 0


def test_binary_search_found_middle():
    """Valida encontro no centro exato na primeira comparacao (melhor caso)."""
    elements = ["auth", "cors", "fastapi", "jwt", "security"]
    index, comparisons = binary_search(elements, "fastapi")
    assert index == 2
    assert comparisons == 1


def test_binary_search_found_edges():
    """Valida encontro nos extremos do vetor (indices 0 e n-1)."""
    elements = ["auth", "cors", "fastapi", "jwt", "security"]

    # Primeiro elemento
    index_first, comp_first = binary_search(elements, "auth")
    assert index_first == 0
    assert comp_first > 0

    # Ultimo elemento
    index_last, comp_last = binary_search(elements, "security")
    assert index_last == 4
    assert comp_last > 0


def test_binary_search_not_found():
    """Valida busca de elemento ausente no vetor."""
    elements = ["auth", "cors", "fastapi", "jwt", "security"]
    index, comparisons = binary_search(elements, "database")
    assert index is None
    assert comparisons > 0