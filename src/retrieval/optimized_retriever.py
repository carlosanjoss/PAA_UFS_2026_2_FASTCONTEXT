"""
src/retrieval/optimized_retriever.py
Implementacao do OptimizedRetriever com filtragem por indice e poda de candidatos.
"""

import time
from typing import List, Dict, Any
from src.retrieval.base import Retriever, RetrievedChunk, RetrievalResult, RetrievalMetrics
from src.algorithms.inverted_index import InvertedIndex


class OptimizedRetriever(Retriever):
    """
    Recuperador Otimizado: utiliza InvertedIndex combinado com poda antecipada
    (early filtering/pruning) para evitar a pontuacao exaustiva de chunks de baixa sobreposicao.
    """

    name: str = "optimized"

    def __init__(self, corpus_chunks: List[Dict[str, Any]]):
        self.corpus_map: Dict[str, Dict[str, Any]] = {
            chunk["chunk_id"]: chunk for chunk in corpus_chunks
        }
        self.index = InvertedIndex()
        self.index.build(corpus_chunks)
        self.index_build_time_ns = self.index.build_time_ns

    def search(self, query: str, k: int = 5) -> RetrievalResult:
        start_time = time.perf_counter_ns()
        metrics = RetrievalMetrics(index_build_time_ns=self.index_build_time_ns)

        # Borda: entrada vazia ou invalida
        if not self.corpus_map or k <= 0 or not query.strip():
            metrics.retrieval_time_ns = time.perf_counter_ns() - start_time
            return RetrievalResult(
                query=query,
                k=k,
                retriever_name=self.name,
                chunks=[],
                metrics=metrics,
            )

        query_tokens = list(set(InvertedIndex.tokenize(query)))
        if not query_tokens:
            metrics.retrieval_time_ns = time.perf_counter_ns() - start_time
            return RetrievalResult(
                query=query,
                k=k,
                retriever_name=self.name,
                chunks=[],
                metrics=metrics,
            )

        # Coleta de candidatos via indice invertido acumulando frequencia previa
        candidate_scores: Dict[str, float] = {}
        for token in query_tokens:
            postings = self.index.get_postings(token)
            for chunk_id, freq in postings.items():
                candidate_scores[chunk_id] = candidate_scores.get(chunk_id, 0.0) + 1.0

        metrics.chunks_scored = len(candidate_scores)

        # Poda: manter apenas quem tem pontuacao positiva
        candidates = [
            (score, self.corpus_map[cid])
            for cid, score in candidate_scores.items()
            if score > 0.0
        ]
        metrics.candidates_found = len(candidates)

        # Ordenacao deterministica: maior pontuacao (-score), desempate por chunk_id crescente
        sort_start = time.perf_counter_ns()
        candidates.sort(key=lambda item: (-item[0], item[1]["chunk_id"]))
        metrics.sorting_time_ns = time.perf_counter_ns() - sort_start

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