from __future__ import annotations

from pathlib import Path
from typing import Any, cast


import yaml

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

CONFIG_DIR = (
    PROJECT_ROOT / "config"
)


def load_yaml( path: Path,) -> dict[str, Any]:
    """Load and validate a YAML configuration file."""

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}"
        )

    with path.open("r", encoding="utf-8", ) as file:
        data = yaml.safe_load(
            file
        )

    if not isinstance(
        data,
        dict,
    ):
        raise TypeError(
            "Expected a mapping in "
            f"configuration file: {path}"
        )

    return cast(
        dict[str, Any],
        data,
    )


def get_mapping(
    data: dict[str, Any],
    key: str,
) -> dict[str, Any]:
    """Extract a typed nested mapping from configuration data."""

    value = data.get(
        key
    )

    if not isinstance(
        value,
        dict,
    ):
        raise TypeError(
            f"Expected '{key}' "
            "to be a mapping."
        )

    return cast(
        dict[str, Any],
        value,
    )


def load_corpus_config() -> dict[str, Any]:
    """Load the corpus configuration section."""

    data = load_yaml(
        CONFIG_DIR / "corpus.yaml"
    )

    return get_mapping(
        data,
        "corpus",
    )


def load_retrieval_config() -> dict[str, Any]:
    """Load the retrieval configuration section."""

    data = load_yaml(
        CONFIG_DIR / "retrieval.yaml"
    )

    return get_mapping(
        data,
        "retrieval",
    )


def load_experiments_config() -> dict[str, Any]:
    """Load the experiments configuration section."""

    data = load_yaml(
        CONFIG_DIR / "experiments.yaml"
    )

    return get_mapping(
        data,
        "experiments",
    )