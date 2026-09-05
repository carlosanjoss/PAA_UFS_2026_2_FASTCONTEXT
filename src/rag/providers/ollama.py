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

        self._model = model.strip()
        self._base_url = base_url.rstrip("/")
        self._session = session or requests.Session()

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def model(self) -> str:
        return self._model

    def is_available(self) -> bool:
        try:
            response = self._session.get(
                f"{self._base_url}/api/tags",
                timeout=5,
            )
            return response.ok
        except requests.RequestException:
            return False

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
                f"Failed to connect to Ollama at "
                f"{self._base_url}."
            ) from exc

        except requests.RequestException as exc:
            raise LLMConnectionError(
                "Unexpected error while communicating "
                "with Ollama."
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

        return LLMResponse(
            text=generated_text.strip(),
            model=self._model,
            provider=self.name,
        metadata={
            "total_duration": data.get(
                "total_duration"
            ),
            "load_duration": data.get(
                "load_duration"
            ),
            "prompt_eval_count": data.get(
                "prompt_eval_count"
            ),
            "eval_count": data.get(
                "eval_count"
            ),
        },
    )