"""Tests for the manual TF-IDF representation and cosine similarity."""

import math

from src.representations.tfidf import TFIDFVectorizer


def test_tfidf_empty_corpus() -> None:
    """Validate fitting and transformation with an empty corpus."""

    tfidf = TFIDFVectorizer()
    tfidf.fit([])

    assert tfidf.doc_count == 0
    assert tfidf.transform("fastapi") == {}


def test_tfidf_vocabulary_and_idf() -> None:
    """Validate vocabulary creation and IDF consistency."""

    corpus = [
        {
            "content": "fastapi python async",
        },
        {
            "content": "python web framework",
        },
    ]

    tfidf = TFIDFVectorizer()
    tfidf.fit(corpus)

    assert "python" in tfidf.vocabulary
    assert "fastapi" in tfidf.vocabulary

    assert (
        tfidf.idf_values["fastapi"]
        > tfidf.idf_values["python"]
    )


def test_cosine_similarity_identical_vectors() -> None:
    """Validate similarity near one for identical vectors."""

    corpus = [
        {
            "content": (
                "authentication authorization jwt"
            ),
        }
    ]

    tfidf = TFIDFVectorizer()
    tfidf.fit(corpus)

    vector_a = tfidf.transform(
        "authentication authorization"
    )

    vector_b = tfidf.transform(
        "authentication authorization"
    )

    similarity = tfidf.cosine_similarity(
        vector_a,
        vector_b,
    )

    assert math.isclose(
        similarity,
        1.0,
        rel_tol=1e-5,
    )


def test_cosine_similarity_orthogonal_vectors() -> None:
    """Validate zero similarity for vectors with no shared terms."""

    corpus = [
        {
            "content": "database postgres sql",
        },
        {
            "content": "frontend react javascript",
        },
    ]

    tfidf = TFIDFVectorizer()
    tfidf.fit(corpus)

    vector_a = tfidf.transform(
        "database"
    )

    vector_b = tfidf.transform(
        "frontend"
    )

    similarity = tfidf.cosine_similarity(
        vector_a,
        vector_b,
    )

    assert similarity == 0.0