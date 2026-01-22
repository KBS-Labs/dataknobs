"""Integration test configuration and fixtures for LLM package."""

import os
import time
from typing import Any

import pytest
import requests


def is_ollama_available(
    host: str = "localhost", port: int = 11434, timeout: float = 2.0
) -> bool:
    """Check if Ollama is available.

    Args:
        host: Ollama host
        port: Ollama port
        timeout: Connection timeout in seconds

    Returns:
        True if Ollama is accessible, False otherwise
    """
    try:
        response = requests.get(f"http://{host}:{port}/api/tags", timeout=timeout)
        return response.status_code == 200
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        return False


def wait_for_ollama(
    host: str = "localhost", port: int = 11434, max_retries: int = 30
) -> bool:
    """Wait for Ollama to be ready.

    Args:
        host: Ollama host
        port: Ollama port
        max_retries: Maximum number of connection attempts

    Returns:
        True if Ollama became available

    Raises:
        ConnectionError: If Ollama is not accessible after max retries
    """
    for i in range(max_retries):
        if is_ollama_available(host, port):
            return True
        if i == max_retries - 1:
            raise ConnectionError(
                f"Could not connect to Ollama at {host}:{port} after {max_retries} attempts. "
                f"Please ensure Ollama is running and accessible."
            )
        time.sleep(1)
    return False


def get_available_models(
    host: str = "localhost", port: int = 11434
) -> list[str]:
    """Get list of available Ollama models.

    Args:
        host: Ollama host
        port: Ollama port

    Returns:
        List of model names
    """
    try:
        response = requests.get(f"http://{host}:{port}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return [m.get("name", "") for m in data.get("models", [])]
    except Exception:
        pass
    return []


def verify_ollama_model(
    model: str, host: str = "localhost", port: int = 11434
) -> bool:
    """Verify that a specific Ollama model is available.

    Args:
        model: Model name (e.g., "qwen3-coder")
        host: Ollama host
        port: Ollama port

    Returns:
        True if model is available, False otherwise
    """
    models = get_available_models(host, port)
    # Check if our model is in the list (handle both with and without tag)
    base_model = model.split(":")[0]
    return any(m.startswith(base_model) for m in models)


@pytest.fixture(scope="session")
def ollama_connection_params() -> dict[str, Any]:
    """Ollama connection parameters for integration tests."""
    return {
        "host": os.environ.get("OLLAMA_HOST", "localhost"),
        "port": int(os.environ.get("OLLAMA_PORT", "11434")),
    }


@pytest.fixture(scope="session")
def ensure_ollama_ready(ollama_connection_params: dict[str, Any]) -> None:
    """Ensure Ollama is ready before running tests."""
    wait_for_ollama(
        host=ollama_connection_params["host"],
        port=ollama_connection_params["port"],
    )


@pytest.fixture(scope="session")
def ollama_model(ollama_connection_params: dict[str, Any]) -> str:
    """Get the Ollama model to use for tests.

    Uses OLLAMA_MODEL environment variable, defaulting to qwen3-coder.
    Falls back to any available model if specified model is not found.
    """
    preferred_model = os.environ.get("OLLAMA_MODEL", "qwen3-coder")
    host = ollama_connection_params["host"]
    port = ollama_connection_params["port"]

    if verify_ollama_model(preferred_model, host, port):
        return preferred_model

    # Fall back to any available model
    available = get_available_models(host, port)
    # Prefer models good for extraction
    extraction_models = ["qwen3-coder", "qwen3", "llama3", "mistral", "gemma3"]
    for model in extraction_models:
        for available_model in available:
            if available_model.startswith(model):
                print(f"\nUsing fallback model: {available_model}")
                return available_model

    if available:
        print(f"\nWARNING: Using first available model: {available[0]}")
        return available[0]

    pytest.skip(f"No Ollama models available. Run: ollama pull {preferred_model}")
    return preferred_model  # Won't reach here, but satisfies type checker


@pytest.fixture
def ollama_extractor_config(
    ensure_ollama_ready: None,
    ollama_connection_params: dict[str, Any],
    ollama_model: str,
) -> dict[str, Any]:
    """Configuration for SchemaExtractor with Ollama."""
    return {
        "provider": "ollama",
        "model": ollama_model,
        "temperature": 0.0,  # Deterministic for testing
    }
