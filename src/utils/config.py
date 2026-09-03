from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


def load_yaml(filename: str) -> dict[str, Any]:
    """
    Carrega um arquivo YAML localizado no diretório config/.

    Parameters
    ----------
    filename:
        Nome do arquivo YAML.

    Returns
    -------
    dict
        Conteúdo do arquivo convertido para dicionário.

    Raises
    ------
    FileNotFoundError
        Caso o arquivo solicitado não exista.

    ValueError
        Caso o YAML não tenha um objeto raiz válido.
    """

    path = CONFIG_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo de configuração não encontrado: {path}"
        )

    with path.open(
        mode="r",
        encoding="utf-8"
    ) as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(
            f"Configuração inválida em {path}"
        )

    return data


def load_corpus_config() -> dict[str, Any]:
    return load_yaml("corpus.yaml")["corpus"]


def load_retrieval_config() -> dict[str, Any]:
    return load_yaml("retrieval.yaml")["retrieval"]


def load_experiments_config() -> dict[str, Any]:
    return load_yaml("experiments.yaml")["experiments"]