"""Count tokens with the tokenizer associated with the embedding model."""

from functools import lru_cache

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.representations.embeddings import DEFAULT_MODEL_NAME


@lru_cache(maxsize=1)
def get_tokenizer() -> PreTrainedTokenizerBase:
    """Load the embedding tokenizer once, on first use."""
    return AutoTokenizer.from_pretrained(
        DEFAULT_MODEL_NAME,
        use_fast=True,
    )


def count_tokens(text: str) -> int:
    """Count model tokens without adding special tokens."""
    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    return len(
        get_tokenizer()(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]
    )