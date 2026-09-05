"""
src/retrieval/base.py
Contrato oficial entre Retrieval, Algoritmos Classicos e RAG.
Define RetrievedChunk, RetrievalMetrics, RetrievalResult e a classe base Retriever.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass(frozen=True)
class RetrievedChunk:
    """
    Representa a informacao de um trecho individual posicionado no ranking.
    """
    chunk_id: str
    score: float
    rank: int
    source_path: str
    section_title: str
    content: str
    token_count: Optional[int] = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalMetrics:
    """
    Metricas de desempenho e instrumentacao algoritmica para PAA.
    """
    retrieval_time_ns: int = 0
    sorting_time_ns: int = 0
    index_build_time_ns: int = 0
    peak_memory_mb: float = 0.0
    comparisons: int = 0
    chunks_scored: int = 0
    candidates_found: int = 0


@dataclass
class RetrievalResult:
    """
    Objeto de retorno padrao entregue pelo Retriever ao pipeline de RAG e interface.
    """
    query: str
    k: int
    retriever_name: str
    chunks: List[RetrievedChunk]
    metrics: RetrievalMetrics = field(default_factory=RetrievalMetrics)

    def is_empty(self) -> bool:
        """Indica se a busca nao retornou nenhum trecho."""
        return len(self.chunks) == 0


class Retriever(ABC):
    """
    Classe base abstrata para todos os recuperadores.
    Exige o atributo 'name' ('linear', 'indexed', 'optimized' ou 'semantic')
    e o metodo 'search(query, k)'.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Identificador textual exato do recuperador."""
        pass

    @abstractmethod
    def search(self, query: str, k: int = 5) -> RetrievalResult:
        """Executa a busca e retorna os k itens mais relevantes."""
        pass