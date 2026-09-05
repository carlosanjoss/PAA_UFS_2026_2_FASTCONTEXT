from src.retrieval.base import Retriever
from src.retrieval.models import (
    RetrievalMetrics,
    RetrievalResult,
    RetrievedChunk,
)


class FakeRetriever(Retriever):
    """Minimal retriever used to test the interface."""

    @property
    def name(self) -> str:
        return "fake"

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> RetrievalResult:
        chunk = RetrievedChunk(
            chunk_id="chunk_001",
            content="Example content.",
            source_path="example.md",
            section_title="Example",
            score=1.0,
            rank=1,
        )

        return RetrievalResult(
            query=query,
            algorithm=self.name,
            top_k=top_k,
            chunks=(chunk,),
            metrics=RetrievalMetrics(
                retrieval_time_ns=100,
                comparisons=1,
                chunks_scored=1,
                candidates_found=1,
            ),
        )


def test_retriever_contract() -> None:
    retriever = FakeRetriever()

    result = retriever.retrieve(
        query="Example query",
        top_k=5,
    )

    assert retriever.name == "fake"

    assert (
        result.algorithm
        == retriever.name
    )

    assert (
        result.query
        == "Example query"
    )

    assert (
        result.chunks[0].rank
        == 1
    )