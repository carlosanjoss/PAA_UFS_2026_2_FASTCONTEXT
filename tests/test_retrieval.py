"""
tests/test_retrieval.py
Testes unitarios para LinearRetriever e IndexedRetriever.
Valida nomes canonicos, casos de borda e ordenacao com desempate deterministico.
"""

from src.retrieval.linear_retriever import LinearRetriever
from src.retrieval.indexed_retriever import IndexedRetriever


# --- Testes do LinearRetriever ---

def test_linear_retriever_name():
    """Valida se o identificador canonico e exatamente 'linear'."""
    retriever = LinearRetriever([])
    assert retriever.name == "linear"


def test_linear_retriever_empty_query():
    """Valida o tratamento de borda quando a busca recebe texto em branco."""
    corpus = [{"chunk_id": "c1", "content": "FastAPI security"}]
    retriever = LinearRetriever(corpus)
    result = retriever.search("", k=3)
    assert result.is_empty()
    assert result.retriever_name == "linear"


def test_linear_retriever_ranking_and_deterministic_tiebreak():
    """Valida ordenacao por score decrescente e desempate por chunk_id crescente."""
    corpus = [
        {"chunk_id": "chunk_b", "content": "fastapi dependencies auth"},
        {"chunk_id": "chunk_a", "content": "fastapi dependencies tutorial"},
    ]
    retriever = LinearRetriever(corpus)
    result = retriever.search("dependencies", k=2)

    assert len(result.chunks) == 2
    assert result.chunks[0].chunk_id == "chunk_a"
    assert result.chunks[0].rank == 1
    assert result.chunks[1].chunk_id == "chunk_b"
    assert result.chunks[1].rank == 2


# --- Testes do IndexedRetriever ---

def test_indexed_retriever_name():
    """Valida se o identificador canonico e exatamente 'indexed'."""
    retriever = IndexedRetriever([])
    assert retriever.name == "indexed"


def test_indexed_retriever_empty_query_and_corpus():
    """Valida comportamento de borda com busca vazia e corpus vazio."""
    retriever_empty = IndexedRetriever([])
    assert retriever_empty.search("termo").is_empty()

    corpus = [{"chunk_id": "c1", "content": "FastAPI security"}]
    retriever = IndexedRetriever(corpus)
    result = retriever.search("", k=2)
    assert result.is_empty()
    assert result.retriever_name == "indexed"


def test_indexed_retriever_search_and_ranking():
    """Valida busca filtrada por indice e desempate deterministico."""
    corpus = [
        {"chunk_id": "chunk_b", "content": "fastapi dependencies auth"},
        {"chunk_id": "chunk_a", "content": "fastapi dependencies tutorial"},
        {"chunk_id": "chunk_c", "content": "unrelated database schema"},
    ]
    retriever = IndexedRetriever(corpus)
    result = retriever.search("dependencies", k=2)

    assert len(result.chunks) == 2
    assert result.chunks[0].chunk_id == "chunk_a"
    assert result.chunks[1].chunk_id == "chunk_b"
    # chunk_c nao deve ser pontuado nem retornado
    assert result.metrics.candidates_found == 2
    assert result.metrics.chunks_scored == 2


from src.retrieval.optimized_retriever import OptimizedRetriever


# --- Testes do OptimizedRetriever ---

def test_optimized_retriever_name():
    """Valida se o identificador canonico e exatamente 'optimized'."""
    retriever = OptimizedRetriever([])
    assert retriever.name == "optimized"


def test_optimized_retriever_empty_query_and_corpus():
    """Valida bordas com query vazia e corpus vazio."""
    retriever_empty = OptimizedRetriever([])
    assert retriever_empty.search("termo").is_empty()

    corpus = [{"chunk_id": "c1", "content": "FastAPI security"}]
    retriever = OptimizedRetriever(corpus)
    result = retriever.search("", k=2)
    assert result.is_empty()
    assert result.retriever_name == "optimized"


def test_optimized_retriever_search_and_ranking():
    """Valida ordenacao correta e desempate deterministico no modo otimizado."""
    corpus = [
        {"chunk_id": "chunk_b", "content": "fastapi dependencies auth"},
        {"chunk_id": "chunk_a", "content": "fastapi dependencies tutorial"},
        {"chunk_id": "chunk_c", "content": "unrelated content"},
    ]
    retriever = OptimizedRetriever(corpus)
    result = retriever.search("dependencies auth", k=2)

    assert len(result.chunks) == 2
    # chunk_b possui 2 termos em comum ("dependencies", "auth"), score = 2.0
    assert result.chunks[0].chunk_id == "chunk_b"
    assert result.chunks[0].score == 2.0
    assert result.chunks[0].rank == 1
    # chunk_a possui 1 termo ("dependencies"), score = 1.0
    assert result.chunks[1].chunk_id == "chunk_a"
    assert result.chunks[1].score == 1.0
    assert result.chunks[1].rank == 2


# --- Validacao de Casos de Borda para k (Exigencia PAA) ---

def test_retrievers_edge_cases_k_zero():
    """Valida que nenhum chunk e retornado quando k = 0."""
    corpus = [{"chunk_id": "c1", "content": "FastAPI security"}]

    linear = LinearRetriever(corpus)
    indexed = IndexedRetriever(corpus)
    optimized = OptimizedRetriever(corpus)

    assert linear.search("security", k=0).chunks == []
    assert indexed.search("security", k=0).chunks == []
    assert optimized.search("security", k=0).chunks == []


def test_retrievers_edge_cases_k_greater_than_n():
    """Valida retorno seguro quando k e maior que o total de chunks (k > N)."""
    corpus = [
        {"chunk_id": "c1", "content": "FastAPI middleware"},
        {"chunk_id": "c2", "content": "FastAPI routing"},
    ]

    linear = LinearRetriever(corpus)
    indexed = IndexedRetriever(corpus)
    optimized = OptimizedRetriever(corpus)

    # N = 2, k = 10
    res_linear = linear.search("FastAPI", k=10)
    res_indexed = indexed.search("FastAPI", k=10)
    res_optimized = optimized.search("FastAPI", k=10)

    assert len(res_linear.chunks) == 2
    assert len(res_indexed.chunks) == 2
    assert len(res_optimized.chunks) == 2


