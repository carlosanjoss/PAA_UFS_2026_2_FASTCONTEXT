"""
tests/test_tfidf.py
Testes unitarios para o vetorizador TF-IDF manual e similaridade de cosseno.
"""

import math
from src.representations.tfidf import TFIDFVectorizer


def test_tfidf_empty_corpus():
    """Valida ajuste e transformacao em corpus vazio."""
    tfidf = TFIDFVectorizer()
    tfidf.fit([])
    assert tfidf.doc_count == 0
    assert tfidf.transform("fastapi") == {}


def test_tfidf_vocabulary_and_idf():
    """Valida geracao de vocabulario e consistencia matematica do IDF."""
    corpus = [
        {"content": "fastapi python async"},
        {"content": "python web framework"},
    ]
    tfidf = TFIDFVectorizer()
    tfidf.fit(corpus)

    assert "python" in tfidf.vocabulary
    assert "fastapi" in tfidf.vocabulary
    # 'python' aparece em 2 docs, 'fastapi' aparece em 1 doc -> IDF(fastapi) > IDF(python)
    assert tfidf.idf_values["fastapi"] > tfidf.idf_values["python"]


def test_cosine_similarity_identical_vectors():
    """Valida que textos equivalentes possuem similaridade aproximada a 1.0."""
    corpus = [{"content": "authentication authorization jwt"}]
    tfidf = TFIDFVectorizer()
    tfidf.fit(corpus)

    vec_a = tfidf.transform("authentication authorization")
    vec_b = tfidf.transform("authentication authorization")

    sim = tfidf.cosine_similarity(vec_a, vec_b)
    assert math.isclose(sim, 1.0, rel_tol=1e-5)


def test_cosine_similarity_orthogonal_vectors():
    """Valida similaridade 0.0 entre vetores sem nenhuma intersecao."""
    corpus = [
        {"content": "database postgres sql"},
        {"content": "frontend react javascript"},
    ]
    tfidf = TFIDFVectorizer()
    tfidf.fit(corpus)

    vec_a = tfidf.transform("database")
    vec_b = tfidf.transform("frontend")

    sim = tfidf.cosine_similarity(vec_a, vec_b)
    assert sim == 0.0