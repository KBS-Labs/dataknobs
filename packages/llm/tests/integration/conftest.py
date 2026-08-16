"""Integration test configuration and fixtures for LLM package.

Postgres and Ollama infrastructure — ``postgres_connection_params``,
``ensure_postgres_ready``, ``wait_for_postgres``, ``ollama_env_params``,
``wait_for_ollama``, ``list_ollama_models``, ``is_ollama_model_available``,
``is_ollama_model_usable`` — comes from ``dataknobs_common.testing``, so no
probe is defined here. The :func:`postgres_test_db` wrapper below uses the
``dataknobs-llm`` package's ``test_conversations_`` table prefix.
"""

import os
import warnings
from typing import Any, Generator

import pytest

from dataknobs_common.testing import (
    is_ollama_model_available,
    is_ollama_model_usable,
    list_ollama_models,
    ollama_env_params,
    wait_for_ollama,
)


@pytest.fixture(scope="session")
def ollama_connection_params() -> dict[str, Any]:
    """Ollama connection parameters for integration tests."""
    return ollama_env_params()


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

    available = list_ollama_models(host, port)
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

    if is_ollama_model_available(preferred, host, port):
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
        if is_ollama_model_usable(model, host=host, port=port, num_predict=16, timeout=12.0):
            if tried:
                # A warning rather than a print, and not only because the print
                # check reaches this directory now: stdout written during
                # fixture setup is captured and replayed only for a test that
                # FAILS, so the message announcing a recovery — the case where
                # everything then passes — was the one nobody saw. The warnings
                # summary is printed at the end of every run.
                warnings.warn(
                    f"Ollama: preferred model unusable; recovered with "
                    f"'{model}' (tried: {', '.join(tried)}).",
                    stacklevel=2,
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
