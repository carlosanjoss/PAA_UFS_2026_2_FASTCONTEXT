import pytest

from src.retrieval.indexed_retriever import IndexedRetriever
from src.retrieval.linear_retriever import LinearRetriever
from src.retrieval.optimized_retriever import OptimizedRetriever


def test_linear_retriever_name() -> None:
    """Validate the canonical linear identifier."""

    retriever = LinearRetriever([])

    assert retriever.name == "linear"


def test_linear_retriever_rejects_empty_query() -> None:
    """Reject an empty query at the retrieval contract boundary."""

    corpus = [
        {
            "chunk_id": "c1",
            "content": "FastAPI security",
        }
    ]

    retriever = LinearRetriever(
        corpus
    )

    with pytest.raises(
        ValueError,
        match="query cannot be empty",
    ):
        retriever.retrieve(
            "",
            top_k=3,
        )


def test_linear_retriever_ranking_and_tiebreak() -> None:
    """Rank tied candidates deterministically by chunk identifier."""

    corpus = [
        {
            "chunk_id": "chunk_b",
            "content": (
                "fastapi dependencies auth"
            ),
        },
        {
            "chunk_id": "chunk_a",
            "content": (
                "fastapi dependencies tutorial"
            ),
        },
    ]

    retriever = LinearRetriever(
        corpus
    )

    result = retriever.retrieve(
        "dependencies",
        top_k=2,
    )

    assert len(result.chunks) == 2

    assert (
        result.chunks[0].chunk_id
        == "chunk_a"
    )

    assert result.chunks[0].rank == 1

    assert (
        result.chunks[1].chunk_id
        == "chunk_b"
    )

    assert result.chunks[1].rank == 2

    assert (
        result.metadata is not None
    )

    assert (
        result.metadata[
            "ranking_strategy"
        ]
        == "merge_sort"
    )


def test_indexed_retriever_name() -> None:
    """Validate the canonical indexed identifier."""

    retriever = IndexedRetriever([])

    assert retriever.name == "indexed"


def test_indexed_retriever_empty_corpus() -> None:
    """Return no chunks from an empty corpus."""

    retriever = IndexedRetriever([])

    result = retriever.retrieve(
        "security",
        top_k=5,
    )

    assert result.is_empty()


def test_indexed_retriever_rejects_empty_query() -> None:
    """Reject an empty indexed query."""

    corpus = [
        {
            "chunk_id": "c1",
            "content": "FastAPI security",
        }
    ]

    retriever = IndexedRetriever(
        corpus
    )

    with pytest.raises(
        ValueError,
        match="query cannot be empty",
    ):
        retriever.retrieve(
            "",
            top_k=2,
        )


def test_indexed_retriever_search_and_ranking() -> None:
    """Filter candidates before deterministic ranking."""

    corpus = [
        {
            "chunk_id": "chunk_b",
            "content": (
                "fastapi dependencies auth"
            ),
        },
        {
            "chunk_id": "chunk_a",
            "content": (
                "fastapi dependencies tutorial"
            ),
        },
        {
            "chunk_id": "chunk_c",
            "content": (
                "unrelated database schema"
            ),
        },
    ]

    retriever = IndexedRetriever(
        corpus
    )

    result = retriever.retrieve(
        "dependencies",
        top_k=2,
    )

    assert len(result.chunks) == 2

    assert (
        result.chunks[0].chunk_id
        == "chunk_a"
    )

    assert (
        result.chunks[1].chunk_id
        == "chunk_b"
    )

    assert (
        result.metrics.candidates_found
        == 2
    )

    assert (
        result.metrics.chunks_scored
        == 2
    )


def test_optimized_retriever_name() -> None:
    """Validate the canonical optimized identifier."""

    retriever = OptimizedRetriever([])

    assert retriever.name == "optimized"


def test_optimized_retriever_empty_corpus() -> None:
    """Return no chunks from an empty corpus."""

    retriever = OptimizedRetriever([])

    result = retriever.retrieve(
        "security",
        top_k=5,
    )

    assert result.is_empty()


def test_optimized_retriever_rejects_empty_query() -> None:
    """Reject an empty optimized query."""

    corpus = [
        {
            "chunk_id": "c1",
            "content": "FastAPI security",
        }
    ]

    retriever = OptimizedRetriever(
        corpus
    )

    with pytest.raises(
        ValueError,
        match="query cannot be empty",
    ):
        retriever.retrieve(
            "",
            top_k=2,
        )


def test_optimized_retriever_search_and_ranking() -> None:
    """Use indexed filtering and bounded Top-k ranking."""

    corpus = [
        {
            "chunk_id": "chunk_b",
            "content": (
                "fastapi dependencies auth"
            ),
        },
        {
            "chunk_id": "chunk_a",
            "content": (
                "fastapi dependencies tutorial"
            ),
        },
        {
            "chunk_id": "chunk_c",
            "content": (
                "unrelated content"
            ),
        },
    ]

    retriever = OptimizedRetriever(
        corpus
    )

    result = retriever.retrieve(
        "dependencies auth",
        top_k=2,
    )

    assert len(result.chunks) == 2

    assert (
        result.chunks[0].chunk_id
        == "chunk_b"
    )

    assert (
        result.chunks[0].score
        > result.chunks[1].score
    )

    assert (
        result.chunks[1].chunk_id
        == "chunk_a"
    )

    assert (
        result.metadata is not None
    )

    assert (
        result.metadata[
            "ranking_strategy"
        ]
        == "top_k_heap"
    )


@pytest.mark.parametrize(
    "retriever_class",
    [
        LinearRetriever,
        IndexedRetriever,
        OptimizedRetriever,
    ],
)
def test_retrievers_support_zero_top_k(
    retriever_class: type[
        LinearRetriever
        | IndexedRetriever
        | OptimizedRetriever
    ],
) -> None:
    """Return an empty result for top_k equal to zero."""

    corpus = [
        {
            "chunk_id": "c1",
            "content": "FastAPI security",
        }
    ]

    retriever = retriever_class(
        corpus
    )

    result = retriever.retrieve(
        "security",
        top_k=0,
    )

    assert result.top_k == 0
    assert result.chunks == ()


@pytest.mark.parametrize(
    "retriever_class",
    [
        LinearRetriever,
        IndexedRetriever,
        OptimizedRetriever,
    ],
)
def test_retrievers_support_top_k_greater_than_corpus(
    retriever_class: type[
        LinearRetriever
        | IndexedRetriever
        | OptimizedRetriever
    ],
) -> None:
    """Return every relevant chunk when top_k exceeds corpus size."""

    corpus = [
        {
            "chunk_id": "c1",
            "content": "FastAPI middleware",
        },
        {
            "chunk_id": "c2",
            "content": "FastAPI routing",
        },
    ]

    retriever = retriever_class(
        corpus
    )

    result = retriever.retrieve(
        "FastAPI",
        top_k=10,
    )

    assert len(result.chunks) == 2


def test_legacy_search_alias() -> None:
    """Keep Wilson's search(query, k) API available."""

    corpus = [
        {
            "chunk_id": "c1",
            "content": "FastAPI dependencies",
        }
    ]

    retriever = LinearRetriever(
        corpus
    )

    result = retriever.search(
        "dependencies",
        k=1,
    )

    assert result.algorithm == "linear"
    assert result.retriever_name == "linear"
    assert result.k == 1