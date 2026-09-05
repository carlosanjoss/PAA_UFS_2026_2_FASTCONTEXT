from unittest.mock import Mock

import pytest
import requests

from src.rag.providers.base import (
    GenerationConfig,
    LLMConnectionError,
    LLMMessage,
)
from src.rag.providers.ollama import (
    OllamaProvider,
)


def build_user_message() -> list[LLMMessage]:
    return [
        LLMMessage(
            role="user",
            content="What is FastAPI?",
        )
    ]


def test_ollama_provider_name() -> None:
    provider = OllamaProvider(
        model="qwen3:4b"
    )

    assert provider.name == "ollama"
    assert provider.model == "qwen3:4b"


def test_ollama_provider_is_available() -> None:
    session = Mock(
        spec=requests.Session
    )

    response = Mock()
    response.raise_for_status.return_value = None

    response.json.return_value = {
        "models": [
            {
                "name": "qwen3:4b",
            }
        ]
    }

    session.get.return_value = response

    provider = OllamaProvider(
        model="qwen3:4b",
        session=session,
    )

    assert provider.is_available() is True


def test_model_missing_is_unavailable() -> None:
    session = Mock(
        spec=requests.Session
    )

    response = Mock()
    response.raise_for_status.return_value = None

    response.json.return_value = {
        "models": [
            {
                "name": (
                    "another-model:latest"
                ),
            }
        ]
    }

    session.get.return_value = response

    provider = OllamaProvider(
        model="qwen3:4b",
        session=session,
    )

    assert provider.is_available() is False


def test_ollama_provider_generates_response() -> None:
    session = Mock(
        spec=requests.Session
    )

    response = Mock()
    response.raise_for_status.return_value = None

    response.json.return_value = {
        "message": {
            "role": "assistant",
            "content": (
                "FastAPI is a Python "
                "web framework."
            ),
        },
        "done_reason": "stop",
        "total_duration": 100,
        "load_duration": 10,
        "prompt_eval_count": 20,
        "prompt_eval_duration": 30,
        "eval_count": 10,
        "eval_duration": 40,
    }

    session.post.return_value = response

    provider = OllamaProvider(
        model="qwen3:4b",
        session=session,
    )

    result = provider.generate(
        messages=build_user_message()
    )

    assert (
        result.text
        == "FastAPI is a Python web framework."
    )

    assert result.provider == "ollama"
    assert result.model == "qwen3:4b"

    assert result.metadata is not None

    assert (
        result.metadata["done_reason"]
        == "stop"
    )

    payload = (
        session.post.call_args.kwargs[
            "json"
        ]
    )

    assert payload["think"] is False

    assert (
        payload["messages"][0]["role"]
        == "user"
    )


def test_provider_preserves_length_reason() -> None:
    session = Mock(
        spec=requests.Session
    )

    response = Mock()
    response.raise_for_status.return_value = None

    response.json.return_value = {
        "message": {
            "role": "assistant",
            "content": "Incomplete answer",
        },
        "done_reason": "length",
        "eval_count": 64,
    }

    session.post.return_value = response

    provider = OllamaProvider(
        model="qwen3:4b",
        session=session,
    )

    result = provider.generate(
        messages=build_user_message(),
        config=GenerationConfig(
            max_tokens=64,
        ),
    )

    assert result.text == "Incomplete answer"

    assert result.metadata is not None

    assert (
        result.metadata["done_reason"]
        == "length"
    )

    assert (
        result.metadata[
            "requested_max_tokens"
        ]
        == 64
    )


def test_ollama_provider_can_enable_thinking() -> None:
    session = Mock(
        spec=requests.Session
    )

    response = Mock()
    response.raise_for_status.return_value = None

    response.json.return_value = {
        "message": {
            "role": "assistant",
            "content": "Final answer.",
        },
        "done_reason": "stop",
    }

    session.post.return_value = response

    provider = OllamaProvider(
        model="qwen3:4b",
        session=session,
    )

    provider.generate(
        messages=[
            LLMMessage(
                role="user",
                content="Test prompt.",
            )
        ],
        config=GenerationConfig(
            think=True,
        ),
    )

    payload = (
        session.post.call_args.kwargs[
            "json"
        ]
    )

    assert payload["think"] is True


def test_connection_error_is_wrapped() -> None:
    session = Mock(
        spec=requests.Session
    )

    session.post.side_effect = (
        requests.ConnectionError()
    )

    provider = OllamaProvider(
        model="qwen3:4b",
        session=session,
    )

    with pytest.raises(
        LLMConnectionError
    ):
        provider.generate(
            messages=build_user_message()
        )