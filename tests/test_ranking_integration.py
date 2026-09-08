"""Integration tests for the canonical FastContext ranking contract.

The classical ranking algorithms must produce the same deterministic order
expected by the retrieval layer:

    score descending, then chunk_id ascending.

The tests use candidates in the internal retrieval-compatible format:

    (score, chunk_dict)
"""

import random
from typing import Any

from src.algorithms.merge_sort import merge_sort
from src.algorithms.ordering import default_key
from src.algorithms.quick_sort import quick_sort
from src.algorithms.topk_heap import top_k

Candidate = tuple[float, dict[str, Any]]


def make_candidate(
    score: float,
    chunk_id: str,
    content: str = "content",
) -> Candidate:
    """Create a retrieval-style ranking candidate."""

    return (
        score,
        {
            "chunk_id": chunk_id,
            "content": content,
        },
    )


def make_random_candidate(
    rng: random.Random,
    index: int,
    *,
    minimum_score: float,
    maximum_score: float,
    maximum_chunk_id: int,
    chunk_id_width: int,
) -> Candidate:
    """Create a deterministic pseudo-random ranking candidate."""

    score = round(
        rng.uniform(
            minimum_score,
            maximum_score,
        ),
        2,
    )

    numeric_id = rng.randint(
        0,
        maximum_chunk_id,
    )

    chunk_id = (
        f"chunk_{numeric_id:0{chunk_id_width}d}_{index}"
    )

    return make_candidate(
        score,
        chunk_id,
    )


def retriever_style_order(
    candidates: list[Candidate],
) -> list[str]:
    """Return the reference order used by the retrieval layer."""

    ordered = sorted(
        candidates,
        key=lambda item: (
            -item[0],
            item[1]["chunk_id"],
        ),
    )

    return [
        str(candidate[1]["chunk_id"])
        for candidate in ordered
    ]


def test_merge_sort_matches_retriever_ordering() -> None:
    """Verify Merge Sort against the retrieval ranking contract."""

    rng = random.Random(2026)

    for _ in range(30):
        size = rng.randint(
            0,
            40,
        )

        candidates = [
            make_random_candidate(
                rng,
                index,
                minimum_score=0.0,
                maximum_score=3.0,
                maximum_chunk_id=99,
                chunk_id_width=2,
            )
            for index in range(size)
        ]

        ordered, _ = merge_sort(
            candidates
        )

        actual = [
            str(candidate[1]["chunk_id"])
            for candidate in ordered
        ]

        assert actual == retriever_style_order(
            candidates
        )


def test_quick_sort_matches_retriever_ordering() -> None:
    """Verify Quick Sort against the retrieval ranking contract."""

    rng = random.Random(4096)

    for _ in range(30):
        size = rng.randint(
            0,
            40,
        )

        candidates = [
            make_random_candidate(
                rng,
                index,
                minimum_score=0.0,
                maximum_score=3.0,
                maximum_chunk_id=99,
                chunk_id_width=2,
            )
            for index in range(size)
        ]

        ordered, _ = quick_sort(
            candidates
        )

        actual = [
            str(candidate[1]["chunk_id"])
            for candidate in ordered
        ]

        assert actual == retriever_style_order(
            candidates
        )


def test_topk_matches_retriever_slice() -> None:
    """Verify Top-k against full ranking followed by slicing."""

    rng = random.Random(777)

    for _ in range(30):
        size = rng.randint(
            0,
            40,
        )

        k = rng.randint(
            0,
            size + 3,
        )

        candidates = [
            make_random_candidate(
                rng,
                index,
                minimum_score=0.0,
                maximum_score=3.0,
                maximum_chunk_id=99,
                chunk_id_width=2,
            )
            for index in range(size)
        ]

        result, _ = top_k(
            candidates,
            k,
        )

        actual = [
            str(candidate[1]["chunk_id"])
            for candidate in result
        ]

        expected = retriever_style_order(
            candidates
        )[: max(k, 0)]

        assert actual == expected


def test_ranking_example_from_context_doc() -> None:
    """Verify the canonical ranking example used by the project."""

    candidates = [
        make_candidate(
            0.72,
            "chunk_10",
        ),
        make_candidate(
            0.91,
            "chunk_02",
        ),
        make_candidate(
            0.72,
            "chunk_04",
        ),
    ]

    expected = [
        "chunk_02",
        "chunk_04",
        "chunk_10",
    ]

    merged, _ = merge_sort(
        candidates
    )

    quicked, _ = quick_sort(
        candidates
    )

    topped, _ = top_k(
        candidates,
        3,
    )

    assert [
        candidate[1]["chunk_id"]
        for candidate in merged
    ] == expected

    assert [
        candidate[1]["chunk_id"]
        for candidate in quicked
    ] == expected

    assert [
        candidate[1]["chunk_id"]
        for candidate in topped
    ] == expected


def test_three_algorithms_agree_on_full_ranking() -> None:
    """Verify that all ranking algorithms agree on complete rankings."""

    rng = random.Random(31337)

    for _ in range(20):
        size = rng.randint(
            0,
            50,
        )

        candidates = [
            make_random_candidate(
                rng,
                index,
                minimum_score=-2.0,
                maximum_score=2.0,
                maximum_chunk_id=999,
                chunk_id_width=3,
            )
            for index in range(size)
        ]

        merged, _ = merge_sort(
            candidates
        )

        quicked, _ = quick_sort(
            candidates
        )

        topped, _ = top_k(
            candidates,
            size if size > 0 else 1,
        )

        merged_keys = [
            default_key(candidate)
            for candidate in merged
        ]

        quicked_keys = [
            default_key(candidate)
            for candidate in quicked
        ]

        topped_keys = [
            default_key(candidate)
            for candidate in topped
        ]

        assert merged_keys == quicked_keys

        if size > 0:
            assert topped_keys == merged_keys