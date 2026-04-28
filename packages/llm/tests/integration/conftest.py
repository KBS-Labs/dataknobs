"""Integration test configuration and fixtures for LLM package."""

import os
import time
from typing import Any, Generator

import pytest
import requests
from dataknobs_common.testing import safe_sql_ident


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


# ---------------------------------------------------------------------------
# Postgres fixtures (duplicated from packages/data/tests/integration/conftest.py)
#
# The data-package fixtures aren't part of its public API, so we duplicate the
# minimal subset needed to exercise SQL-backend routing in the LLM package's
# storage tests. If a future item consolidates this, the source of truth lives
# in packages/data/tests/integration/conftest.py.
# ---------------------------------------------------------------------------


def wait_for_postgres(
    host: str,
    port: int,
    user: str,
    password: str,
    max_retries: int = 30,
) -> bool:
    """Wait for PostgreSQL to become reachable on the maintenance database."""
    import psycopg2

    for i in range(max_retries):
        try:
            conn = psycopg2.connect(
                host=host,
                port=port,
                user=user,
                password=password,
                database="postgres",
            )
            conn.close()
            return True
        except psycopg2.OperationalError:
            if i == max_retries - 1:
                raise
            time.sleep(1)
    return False


@pytest.fixture(scope="session")
def postgres_connection_params() -> dict[str, Any]:
    """PostgreSQL connection parameters for integration tests."""
    if os.path.exists("/.dockerenv") or os.environ.get("DOCKER_CONTAINER"):
        default_host = "postgres"
    else:
        default_host = "localhost"

    return {
        "host": os.environ.get("POSTGRES_HOST", default_host),
        "port": int(os.environ.get("POSTGRES_PORT", "5432")),
        "user": os.environ.get("POSTGRES_USER", "postgres"),
        "password": os.environ.get("POSTGRES_PASSWORD", "postgres"),
        "database": os.environ.get("POSTGRES_DB", "dataknobs_test"),
    }


@pytest.fixture(scope="session")
def ensure_postgres_ready(postgres_connection_params: dict[str, Any]) -> None:
    """Ensure PostgreSQL is reachable and the test database exists."""
    import psycopg2
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

    wait_for_postgres(
        host=postgres_connection_params["host"],
        port=postgres_connection_params["port"],
        user=postgres_connection_params["user"],
        password=postgres_connection_params["password"],
    )

    conn = psycopg2.connect(
        host=postgres_connection_params["host"],
        port=postgres_connection_params["port"],
        user=postgres_connection_params["user"],
        password=postgres_connection_params["password"],
        database="postgres",
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s",
            (postgres_connection_params["database"],),
        )
        if not cursor.fetchone():
            cursor.execute(
                f"CREATE DATABASE {safe_sql_ident(postgres_connection_params['database'])}"
            )
    except psycopg2.errors.DuplicateDatabase:
        pass
    finally:
        cursor.close()
        conn.close()


@pytest.fixture
def postgres_test_db(
    ensure_postgres_ready: None,
    postgres_connection_params: dict[str, Any],
) -> Generator[dict[str, Any], None, None]:
    """Provide a clean PostgreSQL table per test, dropped on teardown."""
    import uuid

    import psycopg2

    test_id = uuid.uuid4().hex[:8]
    config = postgres_connection_params.copy()
    config["table"] = f"test_conversations_{test_id}"
    config["schema"] = "public"

    yield config

    conn = psycopg2.connect(
        host=config["host"],
        port=config["port"],
        user=config["user"],
        password=config["password"],
        database=config["database"],
    )
    try:
        cursor = conn.cursor()
        cursor.execute(
            f"DROP TABLE IF EXISTS {safe_sql_ident(config['schema'])}."
            f"{safe_sql_ident(config['table'])} CASCADE"
        )
        conn.commit()
    finally:
        cursor.close()
        conn.close()
