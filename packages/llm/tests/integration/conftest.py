"""Integration test configuration and fixtures for LLM package.

Postgres infrastructure fixtures (``postgres_connection_params``,
``ensure_postgres_ready``, ``wait_for_postgres``) come from the
``dataknobs_common.testing`` pytest11 plugin — no duplication here. The
:func:`postgres_test_db` wrapper below uses the ``dataknobs-llm`` package's
``test_conversations_`` table prefix.
"""

import os
import time
from typing import Any, Generator

import pytest
import requests

from dataknobs_common.testing import is_ollama_model_usable


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
        model: Model name (e.g., "llama3.1:8b")
        host: Ollama host
        port: Ollama port

    Returns:
        True if model is available, False otherwise
    """
    models = get_available_models(host, port)
    # Check if our model is in the list (handle both with and without tag).
    # Match exact name or name with any tag suffix (e.g., 'llama2' matches
    # 'llama2:latest' but NOT 'llama2-uncensored:latest').
    base_model = model.split(":", maxsplit=1)[0]
    return any(m == base_model or m.startswith(base_model + ":") for m in models)


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
    """Resolve an Ollama model that actually produces usable output.

    Uses OLLAMA_MODEL (default ``llama3.1:8b``) as the preferred model, but goes
    beyond checking a model is merely *installed*: each candidate is canaried via
    :func:`~dataknobs_common.testing.is_ollama_model_usable` (a trivial
    generation) and the first that returns non-empty output is used. This makes
    the suite resilient to environmental changes — a model that is loaded but
    mis-serving (a reasoning model exhausting its token budget, or a
    runtime/template mismatch after an Ollama upgrade) is stepped over instead of
    silently yielding empty extractions that fail every assertion.

    - If a usable model is found, it is used (a loud note is printed when it is
      not the preferred one, so a degraded environment stays visible).
    - If models are installed but NONE produce usable output, the fixture
      **hard-fails** with a diagnosis — a broken runtime is a real failure, not a
      silent skip (per the project's CI policy).
    - If no models are installed at all, it skips (nothing to test).
    """
    preferred = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
    host = ollama_connection_params["host"]
    port = ollama_connection_params["port"]

    available = get_available_models(host, port)
    if not available:
        pytest.skip(f"No Ollama models installed. Run: ollama pull {preferred}")

    # Ordered candidates: the preferred model first (respecting OLLAMA_MODEL),
    # then reliable instruct families that are fast + well-behaved for
    # deterministic extraction. Reasoning/"coder" families (qwen*) come last:
    # they emit hidden thinking tokens, are slow to canary, and can return empty
    # under some runtimes — poor extraction defaults, useful only as a fallback.
    candidates: list[str] = []

    def _add(name: str) -> None:
        if name and name not in candidates:
            candidates.append(name)

    if verify_ollama_model(preferred, host, port):
        base = preferred.split(":", maxsplit=1)[0]
        for model in available:
            if model in (preferred, base) or model.startswith(base + ":"):
                _add(model)
        _add(preferred)
    for family in ("llama3", "mistral", "gemma3", "qwen3-coder", "qwen3"):
        for model in available:
            if model.startswith(family):
                _add(model)
    # Cap the candidate set so a wholesale-broken runtime hard-fails promptly
    # rather than canarying every installed model. The preferred + reliable
    # instruct families above cover the realistic working cases.
    candidates = candidates[:6]

    tried: list[str] = []
    for model in candidates:
        # Bounded canary: a slow reasoning model that returns empty (e.g.
        # qwen3-coder) is cut off at the timeout instead of stalling the run.
        if is_ollama_model_usable(
            model, host=host, port=port, num_predict=16, timeout=12.0
        ):
            if tried:
                print(
                    f"\nOllama: preferred model unusable; recovered with "
                    f"'{model}' (tried: {', '.join(tried)})."
                )
            return model
        tried.append(model)

    pytest.fail(
        f"Ollama is reachable and {len(available)} model(s) are installed, but "
        f"NONE produced usable (non-empty) output. Tried: {', '.join(tried)}. "
        "This indicates a broken Ollama runtime (e.g. a version/template "
        "mismatch after an upgrade, or reasoning models exhausting their token "
        "budget) — not a dataknobs code defect. Fix the Ollama environment "
        "(restart/roll back Ollama, re-pull a model, or set OLLAMA_MODEL to a "
        "working instruct model) and re-run."
    )
    return preferred  # Won't reach here, but satisfies the type checker


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


@pytest.fixture
def postgres_test_db(make_postgres_test_db) -> Generator[dict[str, Any], None, None]:
    """Provide a clean PostgreSQL table per test, using the ``test_conversations_`` prefix."""
    yield from make_postgres_test_db("test_conversations_")
