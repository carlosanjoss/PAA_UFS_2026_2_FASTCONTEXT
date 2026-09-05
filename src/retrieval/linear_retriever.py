"""
src/retrieval/linear_retriever.py
Implementação do LinearRetriever com busca sequencial, métricas e identificador canônico.
"""

import time
from typing import List, Dict, Any
from src.retrieval.base import Retriever, RetrievedChunk, RetrievalResult, RetrievalMetrics


class LinearRetriever(Retriever):
    """
    Recuperador Baseline: avalia todo o corpus sequencialmente.
    """

    name: str = "linear"

    def __init__(self, corpus_chunks: List[Dict[str, Any]]):
        self.corpus = corpus_chunks

    def search(self, query: str, k: int = 5) -> RetrievalResult:
        start_time = time.perf_counter_ns()
        metrics = RetrievalMetrics()

        # Tratamento de casos de borda
        if not self.corpus or k <= 0 or not query.strip():
            metrics.retrieval_time_ns = time.perf_counter_ns() - start_time
            return RetrievalResult(
                query=query,
                k=k,
                retriever_name=self.name,
                chunks=[],
                metrics=metrics,
            )

        candidates = []
        metrics.chunks_scored = len(self.corpus)

        # Percorrimento sequencial da baseline
        for chunk in self.corpus:
            score = self._compute_lexical_score(query, chunk.get("content", ""))
            if score > 0.0:
                candidates.append((score, chunk))

        metrics.candidates_found = len(candidates)

        # Ordenação com critério de desempate determinístico:
        # Score decrescente (-score) e chunk_id crescente
        sort_start = time.perf_counter_ns()
        candidates.sort(key=lambda item: (-item[0], item[1]["chunk_id"]))
        metrics.sorting_time_ns = time.perf_counter_ns() - sort_start

        # Seleção de até k elementos
        top_k = candidates[:k]

        retrieved_chunks = [
            RetrievedChunk(
                chunk_id=item[1]["chunk_id"],
                score=float(item[0]),
                rank=idx + 1,
                source_path=item[1].get("source_path", ""),
                section_title=item[1].get("section_title", ""),
                content=item[1].get("content", ""),
                token_count=item[1].get("token_count", 0),
            )
            for idx, item in enumerate(top_k)
        ]

        metrics.retrieval_time_ns = time.perf_counter_ns() - start_time

        return RetrievalResult(
            query=query,
            k=k,
            retriever_name=self.name,
            chunks=retrieved_chunks,
            metrics=metrics,
        )

    def _compute_lexical_score(self, query: str, text: str) -> float:
        """Cálculo lexical inicial baseado na sobreposição simples de termos."""
        query_tokens = set(query.lower().split())
        text_tokens = set(text.lower().split())
        if not query_tokens or not text_tokens:
            return 0.0
        return float(len(query_tokens.intersection(text_tokens)))