from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


def load_yaml(path: Path) -> dict[str, Any]:
    """Load and validate a YAML configuration file."""

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = yaml.safe_load(file)

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise TypeError(f"Expected a mapping in configuration file: {path}")

    return data


def load_corpus_config() -> dict[str, Any]:
    """Load the corpus configuration."""

    data = load_yaml(CONFIG_DIR / "corpus.yaml")
    return data["corpus"]


def load_retrieval_config() -> dict[str, Any]:
    """Load the retrieval configuration."""

    data = load_yaml(CONFIG_DIR / "retrieval.yaml")
    return data["retrieval"]


def load_experiments_config() -> dict[str, Any]:
    """Load the experiments configuration."""

    data = load_yaml(CONFIG_DIR / "experiments.yaml")
    return data["experiments"]
