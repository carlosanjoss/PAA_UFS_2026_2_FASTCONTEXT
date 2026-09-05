from __future__ import annotations

from typing import Any

import requests

from src.rag.providers.base import (
    GenerationConfig,
    LLMConnectionError,
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
        if not model.strip():
            raise ValueError("Model name cannot be empty.")

        if not base_url.strip():
            raise ValueError("Base URL cannot be empty.")

        self._model = model.strip()
        self._base_url = base_url.rstrip("/")
        self._session = session or requests.Session()

    @property
    def name(self) -> str:
        """Return the provider name."""

        return "ollama"

    @property
    def model(self) -> str:
        """Return the configured model name."""

        return self._model

    def is_available(self) -> bool:
        """Return whether Ollama and the configured model are available."""

        try:
            response = self._session.get(
                f"{self._base_url}/api/tags",
                timeout=5,
            )
            response.raise_for_status()
            data = response.json()

        except (requests.RequestException, ValueError):
            return False

        models = data.get("models")

        if not isinstance(models, list):
            return False

        installed_models: set[str] = set()

        for model in models:
            if not isinstance(model, dict):
                continue

            name = model.get("name")
            model_identifier = model.get("model")

            if isinstance(name, str):
                installed_models.add(name)

            if isinstance(model_identifier, str):
                installed_models.add(model_identifier)

        return self._model in installed_models

    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> LLMResponse:
        """Generate a response using the configured Ollama model."""

        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        generation_config = config or GenerationConfig()

        payload: dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "think": generation_config.think,
            "options": {
                "temperature": generation_config.temperature,
                "num_predict": generation_config.max_tokens,
            },
        }

        try:
            response = self._session.post(
                f"{self._base_url}/api/generate",
                json=payload,
                timeout=self.DEFAULT_TIMEOUT,
            )

        except (
            requests.ConnectionError,
            requests.Timeout,
        ) as exc:
            raise LLMConnectionError(
                f"Failed to connect to Ollama at {self._base_url}."
            ) from exc

        except requests.RequestException as exc:
            raise LLMConnectionError(
                "Unexpected error while communicating with Ollama."
            ) from exc

        try:
            response.raise_for_status()

        except requests.HTTPError as exc:
            raise LLMResponseError(
                "Ollama rejected the generation request "
                f"with HTTP status {response.status_code}."
            ) from exc

        try:
            data = response.json()

        except ValueError as exc:
            raise LLMResponseError(
                "Ollama returned an invalid JSON response."
            ) from exc

        generated_text = data.get("response")

        if not isinstance(generated_text, str):
            raise LLMResponseError(
                "Ollama response does not contain "
                "a valid 'response' field."
            )

        generated_text = generated_text.strip()

        if not generated_text:
            raise LLMResponseError(
                "Ollama returned an empty generated response."
            )

        metadata: dict[str, Any] = {
            "total_duration": data.get("total_duration"),
            "load_duration": data.get("load_duration"),
            "prompt_eval_count": data.get("prompt_eval_count"),
            "eval_count": data.get("eval_count"),
            "done_reason": data.get("done_reason"),
            "thinking_enabled": generation_config.think,
        }

        thinking = data.get("thinking")

        if isinstance(thinking, str) and thinking.strip():
            metadata["thinking"] = thinking.strip()

        return LLMResponse(
            text=generated_text,
            model=self._model,
            provider=self.name,
            metadata=metadata,
        )