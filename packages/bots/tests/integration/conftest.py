"""Integration test configuration and fixtures.

The Ollama probes — ``ollama_env_params``, ``wait_for_ollama``,
``is_ollama_model_available`` — come from ``dataknobs_common.testing``. They
were reimplemented here, and the local copy of the model match accepted any
installed name that merely *began* with the requested one, so a request for
``gemma3`` was satisfied by ``gemma3-uncensored:latest`` and the suite ran
green against a model nobody asked for.
"""

import warnings

import pytest

from dataknobs_common.testing import (
    is_ollama_model_available,
    ollama_env_params,
    wait_for_ollama,
)


# =============================================================================
# Echo LLM Configuration (for tests that don't need real LLM responses)
# =============================================================================


@pytest.fixture
def echo_config() -> dict:
    """Provide Echo LLM configuration for tests that don't need real LLM."""
    return {
        "provider": "echo",
        "model": "echo-model",
        "temperature": 0.7,
        "max_tokens": 500,
    }


@pytest.fixture
def bot_config_echo(echo_config) -> dict:
    """Provide bot configuration using Echo LLM."""
    return {
        "llm": echo_config,
        "conversation_storage": {"backend": "memory"},
        "prompts": {
            "test_assistant": "You are a helpful test assistant. Keep responses very brief."
        },
        "system_prompt": {"name": "test_assistant"},
    }


@pytest.fixture
def bot_config_echo_with_memory(echo_config) -> dict:
    """Provide bot configuration with memory using Echo LLM."""
    return {
        "llm": echo_config,
        "conversation_storage": {"backend": "memory"},
        "memory": {
            "type": "buffer",
            "max_messages": 10,
        },
        "prompts": {
            "test_assistant": "You are a helpful test assistant with memory. Keep responses very brief."
        },
        "system_prompt": {"name": "test_assistant"},
    }


@pytest.fixture
def bot_config_echo_react(echo_config) -> dict:
    """Provide bot configuration with ReAct reasoning using Echo LLM."""
    return {
        "llm": echo_config,
        "conversation_storage": {"backend": "memory"},
        "reasoning": {
            "strategy": "react",
            "max_iterations": 3,
            "verbose": False,
            "store_trace": True,
        },
        "prompts": {
            "test_agent": "You are a test agent with tool access. Keep responses very brief."
        },
        "system_prompt": {"name": "test_agent"},
    }


# =============================================================================
# Ollama Configuration (for tests that need real LLM responses)
# =============================================================================


#: The model :func:`ollama_config` asks for. The readiness check below verifies
#: this exact name rather than also trying a bare ``gemma3``: the untagged retry
#: existed to work around a match that could not tell ``gemma3:1b`` from any
#: other name starting with ``gemma3``, and the shared matcher makes an untagged
#: request accept any *tag* of the model without accepting a different model.
OLLAMA_TEST_MODEL = "gemma3:1b"


@pytest.fixture(scope="session")
def ollama_connection_params():
    """Ollama connection parameters for integration tests."""
    return ollama_env_params()


@pytest.fixture(scope="session")
def ensure_ollama_ready(ollama_connection_params):
    """Ensure Ollama is ready before running tests."""
    wait_for_ollama(
        host=ollama_connection_params["host"],
        port=ollama_connection_params["port"],
    )

    if not is_ollama_model_available(
        OLLAMA_TEST_MODEL,
        host=ollama_connection_params["host"],
        port=ollama_connection_params["port"],
    ):
        warnings.warn(
            f"{OLLAMA_TEST_MODEL} model not found in Ollama. "
            f"Run: ollama pull {OLLAMA_TEST_MODEL} — tests will attempt to run "
            "but may fail if the model is unavailable.",
            stacklevel=2,
        )


@pytest.fixture
def ollama_config(ensure_ollama_ready, ollama_connection_params) -> dict:
    """Provide Ollama configuration for tests that need real LLM."""
    return {
        "provider": "ollama",
        "model": OLLAMA_TEST_MODEL,
        "temperature": 0.7,
        "max_tokens": 500,
        **ollama_connection_params,
    }


@pytest.fixture
def bot_config_simple(ollama_config) -> dict:
    """Provide simple bot configuration."""
    return {
        "llm": ollama_config,
        "conversation_storage": {"backend": "memory"},
        "prompts": {
            "test_assistant": "You are a helpful test assistant. Keep responses very brief."
        },
        "system_prompt": {"name": "test_assistant"},
    }


@pytest.fixture
def bot_config_with_memory(ollama_config) -> dict:
    """Provide bot configuration with memory."""
    return {
        "llm": ollama_config,
        "conversation_storage": {"backend": "memory"},
        "memory": {
            "type": "buffer",
            "max_messages": 10,
        },
        "prompts": {
            "test_assistant": "You are a helpful test assistant with memory. Keep responses very brief."
        },
        "system_prompt": {"name": "test_assistant"},
    }


@pytest.fixture
def bot_config_react(ollama_config) -> dict:
    """Provide bot configuration with ReAct reasoning."""
    return {
        "llm": ollama_config,
        "conversation_storage": {"backend": "memory"},
        "reasoning": {
            "strategy": "react",
            "max_iterations": 3,
            "verbose": False,
            "store_trace": True,
        },
        "prompts": {
            "test_agent": "You are a test agent with tool access. Keep responses very brief."
        },
        "system_prompt": {"name": "test_agent"},
    }


@pytest.fixture
def sample_tool():
    """Provide a sample tool for testing."""
    from dataknobs_llm.tools import Tool
    from typing import Dict, Any

    class TestTool(Tool):
        """Simple test tool that echoes input."""

        def __init__(self):
            super().__init__(
                name="test_echo",
                description="Echoes the input text",
            )

        @property
        def schema(self) -> Dict[str, Any]:
            return {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to echo",
                    }
                },
                "required": ["text"],
            }

        async def execute(self, text: str) -> str:
            """Echo the input."""
            return f"Echo: {text}"

    return TestTool()
