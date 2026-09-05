from src.rag.settings import load_rag_settings


def test_load_rag_settings_from_yaml(
    monkeypatch,
) -> None:
    monkeypatch.delenv(
        "RAG_DEFAULT_PROVIDER",
        raising=False,
    )

    monkeypatch.delenv(
        "OLLAMA_BASE_URL",
        raising=False,
    )

    monkeypatch.delenv(
        "OLLAMA_MODEL",
        raising=False,
    )

    monkeypatch.delenv(
        "OLLAMA_FALLBACK_MODEL",
        raising=False,
    )

    settings = load_rag_settings(
        load_environment=False
    )

    assert (
        settings.default_provider
        == "ollama"
    )

    assert (
        settings.ollama.model
        == "qwen2.5:3b"
    )

    assert (
        settings.ollama.fallback_model
        == "qwen3:1.7b"
    )


def test_environment_overrides_yaml(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "OLLAMA_MODEL",
        "test-model:latest",
    )

    monkeypatch.setenv(
        "OLLAMA_BASE_URL",
        "http://test-host:1234",
    )

    settings = load_rag_settings(
        load_environment=False
    )

    assert (
        settings.ollama.model
        == "test-model:latest"
    )

    assert (
        settings.ollama.base_url
        == "http://test-host:1234"
    )