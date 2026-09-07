"""
src/algorithms/ordering.py
Critério de ordenação canônico compartilhado pelos algoritmos clássicos.

Regra determinística única (KIRO_MATEUS_FASTCONTEXT.md, seção 8):

    1. score decrescente;
    2. em caso de empate, chunk_id crescente.

Este módulo NÃO conhece os contratos de retrieval (Wilson) nem de RAG (Carlos).
Ele opera apenas sobre chaves de ordenação puras ``(score, chunk_id)``, mantendo
os algoritmos clássicos totalmente desacoplados do restante do projeto.
"""

from __future__ import annotations

from typing import Any, Callable, Tuple

# Uma chave de ordenação é um par (score, chunk_id).
SortKey = Tuple[float, str]

# Um extrator transforma um item arbitrário em sua chave de ordenação.
KeyFunc = Callable[[Any], SortKey]


def default_key(item: Any) -> SortKey:
    """
    Extrator padrão de chave para os itens candidatos.

    Aceita as duas formas mais comuns no projeto sem se acoplar a nenhum
    contrato específico:

    - uma tupla/lista ``(score, payload)`` onde ``payload`` possui ``chunk_id``
      (formato dos retrievers do Wilson: ``(score, chunk_dict)``);
    - um objeto/dict que exponha diretamente ``score`` e ``chunk_id``.

    Returns:
        SortKey: par ``(score, chunk_id)``.
    """
    # Caso 1: (score, payload)
    if isinstance(item, (tuple, list)) and len(item) == 2:
        score, payload = item
        chunk_id = _extract_field(payload, "chunk_id")
        return float(score), str(chunk_id)

    # Caso 2: objeto/dict com score e chunk_id
    score = _extract_field(item, "score")
    chunk_id = _extract_field(item, "chunk_id")
    return float(score), str(chunk_id)


def _extract_field(obj: Any, name: str) -> Any:
    """Lê ``name`` de um dict (por chave) ou de um objeto (por atributo)."""
    if isinstance(obj, dict):
        return obj[name]
    return getattr(obj, name)


def precedes(key_a: SortKey, key_b: SortKey) -> bool:
    """
    Retorna ``True`` se ``key_a`` deve vir ANTES de ``key_b`` no ranking.

    Ordem: score decrescente; em empate, chunk_id crescente.

    Formalmente, ``a`` precede ``b`` quando::

        a.score > b.score
        ou (a.score == b.score e a.chunk_id < b.chunk_id)

    Esta é a operação elementar de comparação analisada na complexidade dos
    algoritmos clássicos.
    """
    score_a, id_a = key_a
    score_b, id_b = key_b

    if score_a != score_b:
        return score_a > score_b
    return id_a < id_b
