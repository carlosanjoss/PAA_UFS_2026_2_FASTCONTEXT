"""
src/retrieval/semantic_retriever.py
Implementacao do SemanticRetriever integrando Embeddings densos (BGE)
e indice vetorial FAISS (IndexFlatIP) ao contrato oficial de RAG para PAA.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Sequence

from src.representations.embeddings import EmbeddingGenerator
from src.retrieval.base import (
    Retriever,
    RetrievedChunk,
    RetrievalMetrics,
    RetrievalResult,
)
from src.semantic.faiss_index import FaissIndex


class SemanticRetriever(Retriever):
    """
    Recuperador Semantico (Configuracao D):
    Utiliza embeddings densos (BAAI/bge-small-en-v1.5) e busca vetorial
    baseada em produto interno (IndexFlatIP) normalizado via FAISS.
    """

    name: str = "semantic"

    def __init__(
        self,
        corpus_chunks: List[Dict[str, Any]],
        model_name: str = "BAAI/bge-small-en-v1.5",
    ) -> None:
        self.corpus_chunks = corpus_chunks
        self.corpus_map: Dict[str, Dict[str, Any]] = {
            chunk["chunk_id"]: chunk for chunk in corpus_chunks if "chunk_id" in chunk
        }

        # Inicializa gerador de representacao vetorial
        self.embedding_generator = EmbeddingGenerator(model_name=model_name, normalize=True)
        self.dimension = self.embedding_generator.dimension

        # Inicializa e popula o indice vetorial denso FAISS
        start_build = time.perf_counter_ns()
        self.index = FaissIndex(dimension=self.dimension)

        if self.corpus_chunks:
            texts = [chunk.get("content", "") for chunk in self.corpus_chunks]
            chunk_ids = [chunk.get("chunk_id", "") for chunk in self.corpus_chunks]
            embeddings = self.embedding_generator.encode_corpus(texts)
            self.index.build(embeddings, chunk_ids)

        self.index_build_time_ns = time.perf_counter_ns() - start_build

    def search(self, query: str, k: int = 5) -> RetrievalResult:
        """
        Executa a recuperacao semantica densa e retorna os k trechos mais proximos.
        """
        start_time = time.perf_counter_ns()
        metrics = RetrievalMetrics(index_build_time_ns=self.index_build_time_ns)

        # Tratamento defensivo de casos de borda
        clean_query = query.strip()
        if not self.corpus_chunks or k <= 0 or not clean_query:
            metrics.retrieval_time_ns = time.perf_counter_ns() - start_time
            return RetrievalResult(
                query=query,
                k=k,
                retriever_name=self.name,
                chunks=[],
                metrics=metrics,
            )

        # 1. Converte a query em vetor denso normalizado
        query_vector = self.embedding_generator.encode_query(clean_query)

        # 2. Busca os vizinhos mais proximos no FAISS
        faiss_results = self.index.search(query_embedding=query_vector, top_k=k)
        metrics.chunks_scored = self.index.size
        metrics.candidates_found = len(faiss_results)

        # 3. Mapeia resultados para o contrato oficial RetrievedChunk
        retrieved_chunks: List[RetrievedChunk] = []
        for rank_idx, item in enumerate(faiss_results):
            chunk_data = self.corpus_map.get(item.chunk_id, {})
            retrieved_chunks.append(
                RetrievedChunk(
                    chunk_id=item.chunk_id,
                    score=float(item.score),
                    rank=rank_idx + 1,
                    source_path=chunk_data.get("source_path", ""),
                    section_title=chunk_data.get("section_title", ""),
                    content=chunk_data.get("content", ""),
                    token_count=chunk_data.get("token_count", 0),
                    metadata=chunk_data.get("metadata", {}),
                )
            )

        metrics.retrieval_time_ns = time.perf_counter_ns() - start_time

        return RetrievalResult(
            query=query,
            k=k,
            retriever_name=self.name,
            chunks=retrieved_chunks,
            metrics=metrics,
        )