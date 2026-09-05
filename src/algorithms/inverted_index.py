"""
src/algorithms/inverted_index.py
Implementacao de Indice Invertido para recuperacao lexica estruturada e PAA.
"""

import time
import re
from typing import List, Dict, Set, Any, Tuple


class InvertedIndex:
    """
    Indice Invertido com listas de postagens e frequencias de termos.
    """

    def __init__(self):
        # Mapeamento: termo -> {chunk_id: frequencia_no_chunk}
        self.index: Dict[str, Dict[str, int]] = {}
        self.vocabulary: Set[str] = set()
        self.build_time_ns: int = 0
        self.total_docs: int = 0

    def build(self, corpus: List[Dict[str, Any]]) -> None:
        """
        Constroi o indice invertido a partir de uma lista de chunks.
        Cada chunk deve conter 'chunk_id' e 'content'.
        """
        start_time = time.perf_counter_ns()
        self.index.clear()
        self.vocabulary.clear()
        self.total_docs = len(corpus)

        for chunk in corpus:
            chunk_id = chunk.get("chunk_id", "")
            content = chunk.get("content", "")
            tokens = self.tokenize(content)

            for token in tokens:
                if token not in self.index:
                    self.index[token] = {}
                    self.vocabulary.add(token)
                self.index[token][chunk_id] = self.index[token].get(chunk_id, 0) + 1

        self.build_time_ns = time.perf_counter_ns() - start_time

    def get_postings(self, term: str) -> Dict[str, int]:
        """
        Retorna o dicionario de postagens para um termo especifico: {chunk_id: frequencia}.
        Retorna dicionario vazio se o termo nao existir no vocabulario.
        """
        term_clean = term.strip().lower()
        return self.index.get(term_clean, {})

    def get_candidate_chunk_ids(self, query: str) -> Set[str]:
        """
        Retorna a uniao dos chunk_ids que contem pelo menos um dos termos da consulta.
        """
        query_tokens = self.tokenize(query)
        candidates: Set[str] = set()

        for token in query_tokens:
            postings = self.get_postings(token)
            candidates.update(postings.keys())

        return candidates

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        Segmenta e normaliza o texto extraindo palavras alfanumericas em minusculo.
        """
        if not text:
            return []
        return re.findall(r"\b\w+\b", text.lower())

    @property
    def vocabulary_size(self) -> int:
        """Retorna o numero total de termos unicos indexados."""
        return len(self.vocabulary)