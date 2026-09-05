"""
tests/test_inverted_index.py
Testes unitarios para a estrutura de dados de Indice Invertido.
"""

from src.algorithms.inverted_index import InvertedIndex


def test_empty_corpus_build():
    """Valida construcao sobre corpus vazio."""
    index = InvertedIndex()
    index.build([])
    assert index.vocabulary_size == 0
    assert index.get_candidate_chunk_ids("qualquer") == set()


def test_inverted_index_postings():
    """Valida mapeamento correto das listas de postagens e frequencias."""
    corpus = [
        {"chunk_id": "doc_1", "content": "FastAPI authentication security"},
        {"chunk_id": "doc_2", "content": "FastAPI documentation tutorial"},
        {"chunk_id": "doc_3", "content": "Security policies in microservices"},
    ]
    index = InvertedIndex()
    index.build(corpus)

    # O termo 'fastapi' aparece em doc_1 e doc_2
    fastapi_postings = index.get_postings("fastapi")
    assert set(fastapi_postings.keys()) == {"doc_1", "doc_2"}
    assert fastapi_postings["doc_1"] == 1

    # O termo 'security' aparece em doc_1 e doc_3
    security_postings = index.get_postings("security")
    assert set(security_postings.keys()) == {"doc_1", "doc_3"}


def test_query_candidate_retrieval():
    """Valida recuperacao da uniao dos chunks relevantes para consultas compostas."""
    corpus = [
        {"chunk_id": "c1", "content": "OAuth2 authentication"},
        {"chunk_id": "c2", "content": "Database migrations with Alembic"},
        {"chunk_id": "c3", "content": "Docker compose setup"},
    ]
    index = InvertedIndex()
    index.build(corpus)

    candidates = index.get_candidate_chunk_ids("database authentication")
    assert candidates == {"c1", "c2"}


def test_term_not_in_vocabulary():
    """Valida busca por termo ausente no indice."""
    corpus = [{"chunk_id": "c1", "content": "Python async await"}]
    index = InvertedIndex()
    index.build(corpus)

    assert index.get_postings("inexistente") == {}
    assert index.get_candidate_chunk_ids("inexistente") == set()