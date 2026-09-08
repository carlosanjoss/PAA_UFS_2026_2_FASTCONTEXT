from __future__ import annotations

import math
import re
from typing import Any, Self

SparseVector = dict[str, float]
CorpusChunk = dict[str, Any]


class TFIDFVectorizer:
    """Manual sparse TF-IDF representation with cosine similarity."""

    def __init__(self) -> None:
        self.doc_count: int = 0
        self.document_frequencies: dict[str, int] = {}
        self.idf_values: dict[str, float] = {}
        self.vocabulary: dict[str, int] = {}

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Extract normalized alphanumeric tokens."""

        if not text:
            return []

        return re.findall(
            r"\b\w+\b",
            text.lower(),
        )

    def fit(
        self,
        corpus: list[CorpusChunk],
    ) -> Self:
        """Build the vocabulary and inverse document frequencies."""

        self.doc_count = len(corpus)

        self.document_frequencies.clear()
        self.idf_values.clear()
        self.vocabulary.clear()

        if self.doc_count == 0:
            return self

        for document in corpus:
            tokens = set(
                self.tokenize(
                    str(
                        document.get(
                            "content",
                            "",
                        )
                    )
                )
            )

            for token in tokens:
                self.document_frequencies[token] = (
                    self.document_frequencies.get(
                        token,
                        0,
                    )
                    + 1
                )

        sorted_terms = sorted(
            self.document_frequencies
        )

        for index, term in enumerate(
            sorted_terms
        ):
            self.vocabulary[term] = index

        for term, document_frequency in (
            self.document_frequencies.items()
        ):
            self.idf_values[term] = (
                math.log(
                    (1.0 + self.doc_count)
                    / (1.0 + document_frequency)
                )
                + 1.0
            )

        return self

    def transform(
        self,
        text: str,
    ) -> SparseVector:
        """Transform text into a sparse TF-IDF vector."""

        tokens = self.tokenize(text)

        if not tokens:
            return {}

        total_tokens = len(tokens)

        term_counts: dict[str, int] = {}

        for token in tokens:
            term_counts[token] = (
                term_counts.get(
                    token,
                    0,
                )
                + 1
            )

        sparse_vector: SparseVector = {}

        for term, count in term_counts.items():
            idf = self.idf_values.get(term)

            if idf is None:
                continue

            term_frequency = (
                count
                / total_tokens
            )

            sparse_vector[term] = (
                term_frequency
                * idf
            )

        return sparse_vector

    @staticmethod
    def cosine_similarity(
        vector_a: SparseVector,
        vector_b: SparseVector,
    ) -> float:
        """Calculate cosine similarity between sparse vectors."""

        if not vector_a or not vector_b:
            return 0.0

        dot_product = sum(
            weight_a * vector_b.get(term, 0.0)
            for term, weight_a in vector_a.items()
        )

        if dot_product == 0.0:
            return 0.0

        norm_a = math.sqrt(
            sum(
                weight * weight
                for weight in vector_a.values()
            )
        )

        norm_b = math.sqrt(
            sum(
                weight * weight
                for weight in vector_b.values()
            )
        )

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        return dot_product / (
            norm_a * norm_b
        )