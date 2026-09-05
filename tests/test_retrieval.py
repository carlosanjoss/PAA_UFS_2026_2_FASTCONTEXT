"""
tests/test_retrieval.py
Testes unitarios para os contratos de retrieval e a baseline LinearRetriever.
"""

from src.retrieval.linear_retriever import LinearRetriever


def test_linear_retriever_name():
    """Valida se o identificador canonico e exatamente 'linear'."""
    retriever = LinearRetriever([])
    assert retriever.name == "linear"


def test_linear_retriever_empty_query():
    """Valida o tratamento de borda para consulta vazia."""
    corpus = [{"chunk_id": "c1", "content": "FastAPI security"}]
    retriever = LinearRetriever(corpus)
    result = retriever.search("", k=3)
    assert result.is_empty()
    assert result.retriever_name == "linear"


def test_linear_retriever_ranking_and_deterministic_tiebreak():
    """Valida ordenacao por score decrescente e desempate deterministico por chunk_id crescente."""
    corpus = [
        {"chunk_id": "chunk_b", "content": "fastapi dependencies auth"},
        {"chunk_id": "chunk_a", "content": "fastapi dependencies tutorial"},
    ]
    retriever = LinearRetriever(corpus)
    result = retriever.search("dependencies", k=2)

    assert len(result.chunks) == 2
    # Ambos empatam em score; chunk_a deve vir antes de chunk_b alfabeticamente
    assert result.chunks[0].chunk_id == "chunk_a"
    assert result.chunks[0].rank == 1
    assert result.chunks[1].chunk_id == "chunk_b"
    assert result.chunks[1].rank == 2