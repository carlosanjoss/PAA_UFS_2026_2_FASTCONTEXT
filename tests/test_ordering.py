"""
tests/test_ordering.py
Testes do critério de ordenação canônico compartilhado (score DESC, chunk_id ASC).
"""

from dataclasses import dataclass

from src.algorithms.ordering import default_key, precedes


# ---------------------------------------------------------------------------
# precedes
# ---------------------------------------------------------------------------
def test_precedes_higher_score_first():
    assert precedes((0.9, "b"), (0.5, "a")) is True
    assert precedes((0.5, "a"), (0.9, "b")) is False


def test_precedes_tie_breaks_by_chunk_id_asc():
    assert precedes((0.5, "a"), (0.5, "b")) is True
    assert precedes((0.5, "b"), (0.5, "a")) is False


def test_precedes_equal_keys_not_strict():
    # Chaves idênticas: nenhum precede o outro.
    assert precedes((0.5, "a"), (0.5, "a")) is False


# ---------------------------------------------------------------------------
# default_key — caso tupla (score, payload)
# ---------------------------------------------------------------------------
def test_default_key_tuple_with_dict_payload():
    assert default_key((0.7, {"chunk_id": "c1"})) == (0.7, "c1")


def test_default_key_tuple_with_object_payload():
    @dataclass
    class Payload:
        chunk_id: str

    assert default_key((1.5, Payload("c2"))) == (1.5, "c2")


# ---------------------------------------------------------------------------
# default_key — caso objeto/dict direto (cobre o ramo do fallback)
# ---------------------------------------------------------------------------
def test_default_key_direct_dict():
    assert default_key({"score": 0.3, "chunk_id": "c3"}) == (0.3, "c3")


def test_default_key_direct_object():
    @dataclass
    class Chunk:
        score: float
        chunk_id: str

    assert default_key(Chunk(2.0, "c4")) == (2.0, "c4")
