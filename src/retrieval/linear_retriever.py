"""
src/retrieval/linear_retriever.py
Implementacao do LinearRetriever com busca sequencial, TF-IDF manual,
similaridade de cosseno e Merge Sort manual integrado para PAA.
"""

import time
from typing import List, Dict, Any
from src.retrieval.base import Retriever, RetrievedChunk, RetrievalResult, RetrievalMetrics
from src.representations.tfidf import TFIDFVectorizer
from src.algorithms.merge_sort import merge_sort


class LinearRetriever(Retriever):
    """
    Recuperador Baseline (Configuracao A):
    Avalia todo o corpus sequencialmente calculando similaridade de cosseno
    via TF-IDF manual e ordenando os candidatos com Merge Sort manual.
    """

    name: str = "linear"

    def __init__(self, corpus_chunks: List[Dict[str, Any]]):
        self.corpus = corpus_chunks
        self.vectorizer = TFIDFVectorizer()
        self.vectorizer.fit(corpus_chunks)
        
        # Pre-computa os vetores TF-IDF dos chunks do corpus
        self.doc_vectors: List[Dict[str, float]] = [
            self.vectorizer.transform(chunk.get("content", ""))
            for chunk in self.corpus
        ]

    def search(self, query: str, k: int = 5) -> RetrievalResult:
        start_time = time.perf_counter_ns()
        metrics = RetrievalMetrics()

        if not self.corpus or k <= 0 or not query.strip():
            metrics.retrieval_time_ns = time.perf_counter_ns() - start_time
            return RetrievalResult(
                query=query,
                k=k,
                retriever_name=self.name,
                chunks=[],
                metrics=metrics,
            )

        query_vec = self.vectorizer.transform(query)
        if not query_vec:
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

        # Percorrimento linear exaustivo
        for idx, chunk in enumerate(self.corpus):
            chunk_vec = self.doc_vectors[idx]
            score = self.vectorizer.cosine_similarity(query_vec, chunk_vec)
            if score > 0.0:
                candidates.append((score, chunk))

        metrics.candidates_found = len(candidates)

        # Ordenacao via Merge Sort manual da equipe (Theta(N log N))
        sort_start = time.perf_counter_ns()
        sorted_candidates, sort_stats = merge_sort(candidates)
        metrics.sorting_time_ns = time.perf_counter_ns() - sort_start

        # Contabiliza comparacoes elementares
        if hasattr(sort_stats, "comparisons"):
            metrics.comparisons = sort_stats.comparisons
        elif isinstance(sort_stats, int):
            metrics.comparisons = sort_stats

        top_k = sorted_candidates[:k]

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