"""
src/representations/tfidf.py
Implementacao manual do modelo TF-IDF e Similaridade de Cosseno para PAA.
"""

import math
import re
from typing import List, Dict, Any, Tuple


class TFIDFVectorizer:
    """
    Vetorizador TF-IDF implementado manualmente para analise de relevancia lexical.
    """

    def __init__(self):
        self.doc_count: int = 0
        self.document_frequencies: Dict[str, int] = {}
        self.idf_values: Dict[str, float] = {}
        self.vocabulary: Dict[str, int] = {}  # termo -> indice

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Extrai tokens alfanumericos em minusculo."""
        if not text:
            return []
        return re.findall(r"\b\w+\b", text.lower())

    def fit(self, corpus: List[Dict[str, Any]]) -> "TFIDFVectorizer":
        """
        Aprende o vocabulario e calcula as frequencias de documento (DF e IDF).
        Cada item de corpus deve conter a chave 'content'.
        """
        self.doc_count = len(corpus)
        self.document_frequencies.clear()
        self.idf_values.clear()
        self.vocabulary.clear()

        if self.doc_count == 0:
            return self

        # 1. Contagem de Document Frequency (DF)
        for doc in corpus:
            tokens = set(self.tokenize(doc.get("content", "")))
            for token in tokens:
                self.document_frequencies[token] = (
                    self.document_frequencies.get(token, 0) + 1
                )

        # 2. Criacao do vocabulario ordenado
        sorted_terms = sorted(self.document_frequencies.keys())
        for idx, term in enumerate(sorted_terms):
            self.vocabulary[term] = idx

        # 3. Calculo do IDF com suavizacao padrao: ln((1 + N) / (1 + DF)) + 1
        for term, df in self.document_frequencies.items():
            self.idf_values[term] = (
                math.log((1.0 + self.doc_count) / (1.0 + df)) + 1.0
            )

        return self

    def transform(self, text: str) -> Dict[str, float]:
        """
        Converte uma string em um vetor TF-IDF esparso (termo -> peso_tfidf).
        """
        tokens = self.tokenize(text)
        if not tokens:
            return {}

        total_tokens = len(tokens)
        term_counts: Dict[str, int] = {}
        for token in tokens:
            term_counts[token] = term_counts.get(token, 0) + 1

        sparse_vector: Dict[str, float] = {}
        for term, count in term_counts.items():
            if term in self.idf_values:
                tf = count / total_tokens
                idf = self.idf_values[term]
                sparse_vector[term] = tf * idf

        return sparse_vector

    def cosine_similarity(
        self, vec_a: Dict[str, float], vec_b: Dict[str, float]
    ) -> float:
        """
        Calcula a similaridade de cosseno entre dois vetores esparsos.
        """
        if not vec_a or not vec_b:
            return 0.0

        # Produto escalar (apenas nos termos da intersecao)
        dot_product = 0.0
        for term, weight_a in vec_a.items():
            if term in vec_b:
                dot_product += weight_a * vec_b[term]

        if dot_product == 0.0:
            return 0.0

        # Normas euclidianas (L2)
        norm_a = math.sqrt(sum(w * w for w in vec_a.values()))
        norm_b = math.sqrt(sum(w * w for w in vec_b.values()))

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        return dot_product / (norm_a * norm_b)