from unittest.mock import Mock

import requests

from src.rag.providers.base import LLMConnectionError
from src.rag.providers.ollama import OllamaProvider


def test_ollama_provider_name() -> None:
    provider = OllamaProvider(model="qwen3:4b")

    assert provider.name == "ollama"
    assert provider.model == "qwen3:4b"


def test_ollama_provider_is_available() -> None:
    session = Mock(spec=requests.Session)

    response = Mock()
    response.ok = True

    session.get.return_value = response

    provider = OllamaProvider(
        model="qwen3:4b",
        session=session,
    )

    assert provider.is_available() is True


def test_ollama_provider_generates_response() -> None:
    session = Mock(spec=requests.Session)

    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "response": "FastAPI is a Python web framework.",
        "total_duration": 100,
        "eval_count": 10,
    }

    session.post.return_value = response

    provider = OllamaProvider(
        model="qwen3:4b",
        session=session,
    )

    result = provider.generate("What is FastAPI?")

    assert result.text == "FastAPI is a Python web framework."
    assert result.provider == "ollama"
    assert result.model == "qwen3:4b"


def test_ollama_provider_raises_connection_error() -> None:
    session = Mock(spec=requests.Session)

    session.post.side_effect = requests.ConnectionError()

    provider = OllamaProvider(
        model="qwen3:4b",
        session=session,
    )

    try:
        provider.generate("What is FastAPI?")
    except LLMConnectionError:
        pass
    else:
        raise AssertionError("Expected LLMConnectionError.")