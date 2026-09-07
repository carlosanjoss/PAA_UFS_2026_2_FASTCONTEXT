"""
src/retrieval/optimized_retriever.py
Implementacao do OptimizedRetriever com filtragem por indice invertido,
busca binaria instrumentada e selecao Top-k via Min-Heap manual para PAA.
"""

import time
from typing import List, Dict, Any, Set
from src.retrieval.base import Retriever, RetrievedChunk, RetrievalResult, RetrievalMetrics
from src.algorithms.inverted_index import InvertedIndex
from src.algorithms.topk_heap import top_k


class OptimizedRetriever(Retriever):
    """
    Recuperador Otimizado (Configuracao C):
    Utiliza InvertedIndex para selecionar candidatos e emprega Min-Heap
    manual de tamanho k para evitar a ordenacao completa de todos os candidatos,
    atingindo complexidade O(C log k).
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

        # Casos de borda: corpus vazio, k <= 0 ou busca em branco
        if not self.corpus_map or k <= 0 or not query.strip():
            metrics.retrieval_time_ns = time.perf_counter_ns() - start_time
            return RetrievalResult(
                query=query,
                k=k,
                retriever_name=self.name,
                chunks=[],
                metrics=metrics,
            )

        query_tokens = self.index.tokenize(query)
        if not query_tokens:
            metrics.retrieval_time_ns = time.perf_counter_ns() - start_time
            return RetrievalResult(
                query=query,
                k=k,
                retriever_name=self.name,
                chunks=[],
                metrics=metrics,
            )

        # 1. Coleta de candidatos no indice e acumulacao de frequencias
        # Contabiliza comparacoes de chave da busca binaria no vocabulario
        candidate_scores: Dict[str, float] = {}
        total_binary_comparisons = 0

        for token in set(query_tokens):
            exists, comps = self.index.contains_term(token)
            total_binary_comparisons += comps
            if exists:
                postings = self.index.get_postings(token)
                for chunk_id, freq in postings.items():
                    candidate_scores[chunk_id] = candidate_scores.get(chunk_id, 0.0) + float(freq)

        metrics.chunks_scored = len(candidate_scores)

        # Filtra candidatos com score positivo
        candidates = [
            (score, self.corpus_map[cid])
            for cid, score in candidate_scores.items()
            if score > 0.0
        ]
        metrics.candidates_found = len(candidates)

        if not candidates:
            metrics.comparisons = total_binary_comparisons
            metrics.retrieval_time_ns = time.perf_counter_ns() - start_time
            return RetrievalResult(
                query=query,
                k=k,
                retriever_name=self.name,
                chunks=[],
                metrics=metrics,
            )

        # 2. Selecao dos Top-k melhores via Min-Heap manual da equipe (O(C log k))
        sort_start = time.perf_counter_ns()
        top_k_candidates, heap_stats = top_k(candidates, k)
        metrics.sorting_time_ns = time.perf_counter_ns() - sort_start

        # Soma as comparacoes da busca binaria com as operacoes da min-heap
        heap_comparisons = heap_stats.comparisons if hasattr(heap_stats, "comparisons") else int(heap_stats or 0)
        metrics.comparisons = total_binary_comparisons + heap_comparisons

        # 3. Montagem da lista final de chunks retornados
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
            for idx, item in enumerate(top_k_candidates)
        ]

        metrics.retrieval_time_ns = time.perf_counter_ns() - start_time

        return RetrievalResult(
            query=query,
            k=k,
            retriever_name=self.name,
            chunks=retrieved_chunks,
            metrics=metrics,
        )