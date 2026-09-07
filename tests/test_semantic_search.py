"""
tests/test_semantic_search.py
Testes unitarios para o pipeline de busca semantica (FAISS e SemanticRetriever).
Utiliza dados sinteticos e mocks seguros para testes deterministas e rapidos.
"""

from unittest.mock import patch
import numpy as np
import pytest

from src.retrieval.semantic_retriever import SemanticRetriever
from src.semantic.faiss_index import FaissIndex


# --- Fixtures e Dados Sinteticos ---

@pytest.fixture
def synthetic_corpus():
    return [
        {
            "chunk_id": "chunk_auth",
            "content": "FastAPI OAuth2 authentication with JWT tokens",
            "source_path": "docs/security.md",
            "section_title": "Security",
            "token_count": 7,
        },
        {
            "chunk_id": "chunk_db",
            "content": "Database migrations and SQLModel connection handling",
            "source_path": "docs/database.md",
            "section_title": "Database",
            "token_count": 6,
        },
        {
            "chunk_id": "chunk_tutorial",
            "content": "First steps tutorial and tutorial configuration",
            "source_path": "docs/tutorial.md",
            "section_title": "Tutorial",
            "token_count": 6,
        },
    ]


# --- Testes do Indice FAISS ---

def test_faiss_index_build_and_search():
    dimension = 4
    index = FaissIndex(dimension=dimension)
    assert index.is_empty
    assert index.dimension == dimension

    # Vetores sinteticos ortogonais normalizados
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    chunk_ids = ["doc_a", "doc_b", "doc_c"]

    index.build(embeddings, chunk_ids)
    assert index.size == 3
    assert not index.is_empty

    # Consulta alinhada com doc_b
    query_vector = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    results = index.search(query_vector, top_k=2)

    assert len(results) == 2
    assert results[0].chunk_id == "doc_b"
    assert pytest.approx(results[0].score, rel=1e-4) == 1.0
    assert results[0].position == 1


def test_faiss_index_edge_cases():
    index = FaissIndex(dimension=2)

    # Busca em indice vazio
    query = np.array([1.0, 0.0], dtype=np.float32)
    assert index.search(query, top_k=5) == ()

    # Busca com k = 0
    embeddings = np.array([[1.0, 0.0]], dtype=np.float32)
    index.build(embeddings, ["d1"])
    assert index.search(query, top_k=0) == ()


# --- Testes do SemanticRetriever ---

def test_semantic_retriever_name():
    with patch("src.retrieval.semantic_retriever.EmbeddingGenerator") as MockGen:
        MockGen.return_value.dimension = 4
        MockGen.return_value.encode_corpus.return_value = np.zeros((0, 4), dtype=np.float32)
        retriever = SemanticRetriever(corpus_chunks=[])
        assert retriever.name == "semantic"


def test_semantic_retriever_empty_cases():
    with patch("src.retrieval.semantic_retriever.EmbeddingGenerator") as MockGen:
        MockGen.return_value.dimension = 4
        MockGen.return_value.encode_corpus.return_value = np.zeros((0, 4), dtype=np.float32)
        
        # Corpus vazio
        retriever = SemanticRetriever(corpus_chunks=[])
        res_empty_corpus = retriever.search("fastapi", k=3)
        assert res_empty_corpus.is_empty()
        assert res_empty_corpus.retriever_name == "semantic"

        # Consulta vazia
        corpus = [{"chunk_id": "c1", "content": "FastAPI router"}]
        MockGen.return_value.encode_corpus.return_value = np.zeros((1, 4), dtype=np.float32)
        retriever_with_doc = SemanticRetriever(corpus_chunks=corpus)
        res_empty_query = retriever_with_doc.search("", k=3)
        assert res_empty_query.is_empty()


def test_semantic_retriever_search_ranking(synthetic_corpus):
    dim = 4
    mock_embeddings = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],  # chunk_auth
            [0.0, 1.0, 0.0, 0.0],  # chunk_db
            [0.0, 0.0, 1.0, 0.0],  # chunk_tutorial
        ],
        dtype=np.float32,
    )

    with patch("src.retrieval.semantic_retriever.EmbeddingGenerator") as MockGen:
        instance = MockGen.return_value
        instance.dimension = dim
        instance.encode_corpus.return_value = mock_embeddings
        instance.encode_query.return_value = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        retriever = SemanticRetriever(corpus_chunks=synthetic_corpus)
        result = retriever.search("auth jwt tokens", k=2)

        assert len(result.chunks) == 2
        assert result.chunks[0].chunk_id == "chunk_auth"
        assert result.chunks[0].rank == 1
        assert pytest.approx(result.chunks[0].score, rel=1e-4) == 1.0
        assert result.chunks[0].section_title == "Security"
        assert result.metrics.retrieval_time_ns > 0
        assert result.metrics.chunks_scored == 3


def test_semantic_retriever_edge_k_bounds(synthetic_corpus):
    dim = 2
    with patch("src.retrieval.semantic_retriever.EmbeddingGenerator") as MockGen:
        instance = MockGen.return_value
        instance.dimension = dim
        instance.encode_corpus.return_value = np.zeros((3, dim), dtype=np.float32)
        instance.encode_query.return_value = np.zeros(dim, dtype=np.float32)

        retriever = SemanticRetriever(corpus_chunks=synthetic_corpus)

        # k = 0
        assert retriever.search("test", k=0).chunks == []

        # k > N (N = 3, k = 10)
        res_large_k = retriever.search("test", k=10)
        assert len(res_large_k.chunks) == 3