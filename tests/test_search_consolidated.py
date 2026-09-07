"""
tests/test_search_consolidated.py
Consolidação de testes de busca linear e binária.

IMPORTANTE (autoria): as implementações `binary_search` e `linear_search` são de
autoria do Wilson (`feature/retrieval`). Estes testes exercitam as ASSINATURAS
REAIS dessas funções:

    binary_search(elements, target) -> (index | None, comparisons)
    linear_search(elements, target) -> (index | None, comparisons)

onde `elements` é uma lista simples de valores comparáveis e, para a busca
binária, ordenada em ordem não decrescente.

Os testes cobrem os casos exigidos pelo escopo: termo no início, meio, fim,
ausente, coleção vazia e coleção de um elemento.
"""

from src.algorithms.binary_search import binary_search
from src.algorithms.linear_search import linear_search

VOCAB = ["auth", "cors", "fastapi", "jwt", "security"]  # ordenado


# ---------------------------------------------------------------------------
# Busca binária
# ---------------------------------------------------------------------------
def test_binary_empty_collection():
    index, comparisons = binary_search([], "fastapi")
    assert index is None
    assert comparisons == 0


def test_binary_single_element_found():
    index, comparisons = binary_search(["fastapi"], "fastapi")
    assert index == 0
    assert comparisons == 1


def test_binary_single_element_absent():
    index, comparisons = binary_search(["fastapi"], "auth")
    assert index is None
    assert comparisons >= 1


def test_binary_found_start():
    index, comparisons = binary_search(VOCAB, "auth")
    assert index == 0
    assert comparisons > 0


def test_binary_found_middle():
    index, comparisons = binary_search(VOCAB, "fastapi")
    assert index == 2
    assert comparisons == 1  # meio exato na primeira comparação


def test_binary_found_end():
    index, comparisons = binary_search(VOCAB, "security")
    assert index == 4
    assert comparisons > 0


def test_binary_absent():
    index, comparisons = binary_search(VOCAB, "database")
    assert index is None
    assert comparisons > 0


def test_binary_comparisons_logarithmic():
    # Para 5 elementos, o número de comparações no pior caso é pequeno (<= 3).
    _, comparisons = binary_search(VOCAB, "database")
    assert comparisons <= 3


# ---------------------------------------------------------------------------
# Busca linear
# ---------------------------------------------------------------------------
def test_linear_empty_collection():
    index, comparisons = linear_search([], "fastapi")
    assert index is None
    assert comparisons == 0


def test_linear_single_element_found():
    index, comparisons = linear_search(["fastapi"], "fastapi")
    assert index == 0
    assert comparisons == 1


def test_linear_found_start():
    index, comparisons = linear_search(VOCAB, "auth")
    assert index == 0
    assert comparisons == 1  # encontra na primeira posição


def test_linear_found_middle():
    index, comparisons = linear_search(VOCAB, "fastapi")
    assert index == 2
    assert comparisons == 3  # percorre até a 3ª posição


def test_linear_found_end():
    index, comparisons = linear_search(VOCAB, "security")
    assert index == 4
    assert comparisons == 5  # percorre toda a lista


def test_linear_absent():
    index, comparisons = linear_search(VOCAB, "database")
    assert index is None
    assert comparisons == len(VOCAB)  # verifica todos os elementos


def test_linear_first_occurrence():
    # Retorna o primeiro índice em caso de duplicatas.
    index, comparisons = linear_search(["a", "b", "b", "c"], "b")
    assert index == 1
    assert comparisons == 2
