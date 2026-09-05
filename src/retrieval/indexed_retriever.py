"""
src/retrieval/indexed_retriever.py
Implementacao do IndexedRetriever utilizando InvertedIndex para filtragem e pontuacao.
"""

import time
from typing import List, Dict, Any
from src.retrieval.base import Retriever, RetrievedChunk, RetrievalResult, RetrievalMetrics
from src.algorithms.inverted_index import InvertedIndex


class IndexedRetriever(Retriever):
    """
    Recuperador Indexado: utiliza InvertedIndex para avaliar apenas
    chunks que contem os termos da consulta.
    """

    name: str = "indexed"

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

        # Tratamento de borda: corpus vazio, k invalido ou query em branco
        if not self.corpus_map or k <= 0 or not query.strip():
            metrics.retrieval_time_ns = time.perf_counter_ns() - start_time
            return RetrievalResult(
                query=query,
                k=k,
                retriever_name=self.name,
                chunks=[],
                metrics=metrics,
            )

        # Filtragem com InvertedIndex: busca apenas os chunks que possuem tokens da query
        candidate_ids = self.index.get_candidate_chunk_ids(query)
        metrics.chunks_scored = len(candidate_ids)

        candidates = []
        for chunk_id in candidate_ids:
            chunk = self.corpus_map[chunk_id]
            score = self._compute_lexical_score(query, chunk.get("content", ""))
            if score > 0.0:
                candidates.append((score, chunk))

        metrics.candidates_found = len(candidates)

        # Ordenacao deterministica: maior score (-score), desempate por chunk_id crescente
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

    def _compute_lexical_score(self, query: str, text: str) -> float:
        """Calcula sobreposicao lexical simples de termos."""
        query_tokens = set(InvertedIndex.tokenize(query))
        text_tokens = set(InvertedIndex.tokenize(text))
        if not query_tokens or not text_tokens:
            return 0.0
        return float(len(query_tokens.intersection(text_tokens)))