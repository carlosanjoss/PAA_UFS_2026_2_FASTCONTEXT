from unittest.mock import Mock

import pytest
import requests

from src.rag.providers.base import (
    GenerationConfig,
    LLMConnectionError,
    LLMResponseError,
)
from src.rag.providers.ollama import OllamaProvider


def test_ollama_provider_name() -> None:
    provider = OllamaProvider(model="qwen3:4b")

    assert provider.name == "ollama"
    assert provider.model == "qwen3:4b"


def test_ollama_provider_is_available() -> None:
    session = Mock(spec=requests.Session)

    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "models": [
            {
                "name": "qwen3:4b",
                "model": "qwen3:4b",
            }
        ]
    }

    session.get.return_value = response

    provider = OllamaProvider(
        model="qwen3:4b",
        session=session,
    )

    assert provider.is_available() is True


def test_ollama_provider_is_unavailable_when_model_is_missing() -> None:
    session = Mock(spec=requests.Session)

    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "models": [
            {
                "name": "another-model:latest",
                "model": "another-model:latest",
            }
        ]
    }

    session.get.return_value = response

    provider = OllamaProvider(
        model="qwen3:4b",
        session=session,
    )

    assert provider.is_available() is False


def test_ollama_provider_is_unavailable_when_server_fails() -> None:
    session = Mock(spec=requests.Session)

    session.get.side_effect = requests.ConnectionError()

    provider = OllamaProvider(
        model="qwen3:4b",
        session=session,
    )

    assert provider.is_available() is False


def test_ollama_provider_generates_response() -> None:
    session = Mock(spec=requests.Session)

    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "response": "FastAPI is a Python web framework.",
        "total_duration": 100,
        "eval_count": 10,
        "done_reason": "stop",
    }

    session.post.return_value = response

    provider = OllamaProvider(
        model="qwen3:4b",
        session=session,
    )

    result = provider.generate(
        "What is FastAPI?",
        config=GenerationConfig(
            temperature=0.0,
            max_tokens=64,
            think=False,
        ),
    )

    assert result.text == "FastAPI is a Python web framework."
    assert result.provider == "ollama"
    assert result.model == "qwen3:4b"
    assert result.metadata is not None
    assert result.metadata["thinking_enabled"] is False

    call = session.post.call_args

    assert call.kwargs["json"]["think"] is False
    assert call.kwargs["json"]["options"]["num_predict"] == 64


def test_ollama_provider_raises_connection_error() -> None:
    session = Mock(spec=requests.Session)

    session.post.side_effect = requests.ConnectionError()

    provider = OllamaProvider(
        model="qwen3:4b",
        session=session,
    )

    with pytest.raises(LLMConnectionError):
        provider.generate("What is FastAPI?")


def test_ollama_provider_rejects_empty_response() -> None:
    session = Mock(spec=requests.Session)

    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "response": "",
        "eval_count": 32,
    }

    session.post.return_value = response

    provider = OllamaProvider(
        model="qwen3:4b",
        session=session,
    )

    with pytest.raises(
        LLMResponseError,
        match="empty generated response",
    ):
        provider.generate("What is FastAPI?")