"""
src/retrieval/indexed_retriever.py
Implementacao do IndexedRetriever com indice invertido, busca binaria instrumentada,
TF-IDF com similaridade de cosseno e Merge Sort manual integrado para PAA.
"""

import time
from typing import List, Dict, Any, Set
from src.retrieval.base import Retriever, RetrievedChunk, RetrievalResult, RetrievalMetrics
from src.algorithms.inverted_index import InvertedIndex
from src.representations.tfidf import TFIDFVectorizer
from src.algorithms.merge_sort import merge_sort


class IndexedRetriever(Retriever):
    """
    Recuperador Indexado (Configuracao B):
    Utiliza InvertedIndex com busca binaria para selecionar candidatos,
    calcula TF-IDF e cosseno apenas nos candidatos filtrados e
    ordena usando Merge Sort manual (Theta(C log C)).
    """

    name: str = "indexed"

    def __init__(self, corpus_chunks: List[Dict[str, Any]]):
        self.corpus_map: Dict[str, Dict[str, Any]] = {
            chunk["chunk_id"]: chunk for chunk in corpus_chunks
        }
        
        self.index = InvertedIndex()
        self.index.build(corpus_chunks)
        self.index_build_time_ns = self.index.build_time_ns

        self.vectorizer = TFIDFVectorizer()
        self.vectorizer.fit(corpus_chunks)
        self.doc_vectors: Dict[str, Dict[str, float]] = {
            chunk["chunk_id"]: self.vectorizer.transform(chunk.get("content", ""))
            for chunk in corpus_chunks
        }

    def search(self, query: str, k: int = 5) -> RetrievalResult:
        start_time = time.perf_counter_ns()
        metrics = RetrievalMetrics(index_build_time_ns=self.index_build_time_ns)

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

        # 1. Filtra candidatos pelo indice e conta comparacoes da busca binaria
        candidate_ids: Set[str] = set()
        total_binary_comparisons = 0

        for token in query_tokens:
            exists, comps = self.index.contains_term(token)
            total_binary_comparisons += comps
            if exists:
                postings = self.index.get_postings(token)
                candidate_ids.update(postings.keys())

        metrics.chunks_scored = len(candidate_ids)

        if not candidate_ids:
            metrics.comparisons = total_binary_comparisons
            metrics.retrieval_time_ns = time.perf_counter_ns() - start_time
            return RetrievalResult(
                query=query,
                k=k,
                retriever_name=self.name,
                chunks=[],
                metrics=metrics,
            )

        # 2. Calcula TF-IDF e cosseno apenas nos candidatos filtrados
        query_vec = self.vectorizer.transform(query)
        candidates = []

        for chunk_id in candidate_ids:
            chunk = self.corpus_map[chunk_id]
            chunk_vec = self.doc_vectors.get(chunk_id, {})
            score = self.vectorizer.cosine_similarity(query_vec, chunk_vec)
            if score > 0.0:
                candidates.append((score, chunk))

        metrics.candidates_found = len(candidates)

        # 3. Ordenacao com Merge Sort manual (Theta(C log C))
        sort_start = time.perf_counter_ns()
        sorted_candidates, sort_stats = merge_sort(candidates)
        metrics.sorting_time_ns = time.perf_counter_ns() - sort_start

        sort_comps = sort_stats.comparisons if hasattr(sort_stats, "comparisons") else int(sort_stats or 0)
        metrics.comparisons = total_binary_comparisons + sort_comps

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