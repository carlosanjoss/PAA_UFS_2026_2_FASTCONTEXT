from src.utils.config import (
    load_corpus_config,
    load_experiments_config,
    load_retrieval_config,
)


def test_corpus_config():
    config = load_corpus_config()

    assert config["name"] == "FastAPI Documentation"
    assert config["source"]["language"] == "en"
    assert config["chunking"]["max_tokens"] == 400
    assert config["chunking"]["overlap_tokens"] == 60


def test_retrieval_config():
    config = load_retrieval_config()

    assert config["sorting"]["primary"]["algorithm"] == "merge_sort"
    assert config["inverted_index"]["binary_search"] is True


def test_experiments_config():
    config = load_experiments_config()

    assert config["repetitions"] == 5
    assert 1.00 in config["corpus_sizes"]
    assert 5 in config["k_values"]