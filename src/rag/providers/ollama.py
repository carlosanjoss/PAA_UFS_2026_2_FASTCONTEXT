from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import requests

from src.rag.providers.base import (
    GenerationConfig,
    LLMConnectionError,
    LLMMessage,
    LLMProvider,
    LLMResponse,
    LLMResponseError,
)


class OllamaProvider(LLMProvider):
    """LLM provider backed by a local Ollama server."""

    DEFAULT_TIMEOUT = (5, 120)

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        session: requests.Session | None = None,
    ) -> None:
        normalized_model = model.strip()

        if not normalized_model:
            raise ValueError(
                "Model name cannot be empty."
            )

        normalized_base_url = base_url.strip()

        if not normalized_base_url:
            raise ValueError(
                "Base URL cannot be empty."
            )

        self._model = normalized_model
        self._base_url = (
            normalized_base_url.rstrip("/")
        )
        self._session = (
            session or requests.Session()
        )

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def model(self) -> str:
        return self._model

    def is_available(self) -> bool:
        """Check whether Ollama and the model are available."""

        try:
            response = self._session.get(
                f"{self._base_url}/api/tags",
                timeout=5,
            )

            response.raise_for_status()
            data = response.json()

        except (
            requests.RequestException,
            ValueError,
        ):
            return False

        models = data.get("models")

        if not isinstance(models, list):
            return False

        installed_models: set[str] = set()

        for model in models:
            if not isinstance(model, dict):
                continue

            name = model.get("name")
            model_name = model.get("model")

            if isinstance(name, str):
                installed_models.add(name)

            if isinstance(model_name, str):
                installed_models.add(
                    model_name
                )

        return self._model in installed_models

    def generate(
        self,
        messages: Sequence[LLMMessage],
        config: GenerationConfig | None = None,
    ) -> LLMResponse:
        """Generate a chat response using Ollama."""

        if not messages:
            raise ValueError(
                "Messages cannot be empty."
            )

        generation_config = (
            config or GenerationConfig()
        )

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {
                    "role": message.role,
                    "content": message.content,
                }
                for message in messages
            ],
            "stream": False,
            "think": generation_config.think,
            "options": {
                "temperature": (
                    generation_config.temperature
                ),
                "num_predict": (
                    generation_config.max_tokens
                ),
            },
        }

        try:
            response = self._session.post(
                f"{self._base_url}/api/chat",
                json=payload,
                timeout=self.DEFAULT_TIMEOUT,
            )

        except (
            requests.ConnectionError,
            requests.Timeout,
        ) as exc:
            raise LLMConnectionError(
                "Failed to connect to Ollama at "
                f"{self._base_url}."
            ) from exc

        except requests.RequestException as exc:
            raise LLMConnectionError(
                "Unexpected error while "
                "communicating with Ollama."
            ) from exc

        try:
            response.raise_for_status()

        except requests.HTTPError as exc:
            raise LLMResponseError(
                "Ollama rejected the generation "
                "request with HTTP status "
                f"{response.status_code}."
            ) from exc

        try:
            data = response.json()

        except ValueError as exc:
            raise LLMResponseError(
                "Ollama returned an invalid "
                "JSON response."
            ) from exc

        message = data.get("message")

        if not isinstance(message, dict):
            raise LLMResponseError(
                "Ollama response does not contain "
                "a valid message."
            )

        generated_text = message.get("content")

        if not isinstance(
            generated_text,
            str,
        ):
            raise LLMResponseError(
                "Ollama response does not contain "
                "valid message content."
            )

        return LLMResponse(
            text=generated_text.strip(),
            model=self._model,
            provider=self.name,
            metadata={
                "done_reason": data.get(
                    "done_reason"
                ),
                "total_duration": data.get(
                    "total_duration"
                ),
                "load_duration": data.get(
                    "load_duration"
                ),
                "prompt_eval_count": data.get(
                    "prompt_eval_count"
                ),
                "prompt_eval_duration": data.get(
                    "prompt_eval_duration"
                ),
                "eval_count": data.get(
                    "eval_count"
                ),
                "eval_duration": data.get(
                    "eval_duration"
                ),
                "thinking_enabled": (
                    generation_config.think
                ),
                "requested_max_tokens": (
                    generation_config.max_tokens
                ),
            },
        )