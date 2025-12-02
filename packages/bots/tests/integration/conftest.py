"""Integration test configuration and fixtures."""

import os
import time
from typing import Generator

import pytest
import requests


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

def wait_for_ollama(host: str = "localhost", port: int = 11434, max_retries: int = 30):
    """Wait for Ollama to be ready.

    Args:
        host: Ollama host
        port: Ollama port
        max_retries: Maximum number of connection attempts

    Raises:
        ConnectionError: If Ollama is not accessible after max retries
    """
    for i in range(max_retries):
        try:
            response = requests.get(f"http://{host}:{port}/api/tags", timeout=2)
            if response.status_code == 200:
                return True
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            if i == max_retries - 1:
                raise ConnectionError(
                    f"Could not connect to Ollama at {host}:{port} after {max_retries} attempts. "
                    f"Please ensure Ollama is running and accessible."
                )
            time.sleep(1)

    return False


def verify_ollama_model(model: str, host: str = "localhost", port: int = 11434) -> bool:
    """Verify that a specific Ollama model is available.

    Args:
        model: Model name (e.g., "gemma3:1b")
        host: Ollama host
        port: Ollama port

    Returns:
        True if model is available, False otherwise
    """
    try:
        response = requests.get(f"http://{host}:{port}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])
            # Extract model names from the response
            model_names = [m.get("name", "") for m in models]
            # Check if our model is in the list (handle both with and without tag)
            return any(
                model_name.startswith(model.split(":")[0]) for model_name in model_names
            )
    except Exception as e:
        print(f"Error verifying Ollama model: {e}")
        return False


@pytest.fixture(scope="session")
def ollama_connection_params():
    """Ollama connection parameters for integration tests."""
    return {
        "host": os.environ.get("OLLAMA_HOST", "localhost"),
        "port": int(os.environ.get("OLLAMA_PORT", "11434")),
    }


@pytest.fixture(scope="session")
def ensure_ollama_ready(ollama_connection_params):
    """Ensure Ollama is ready before running tests."""
    wait_for_ollama(
        host=ollama_connection_params["host"],
        port=ollama_connection_params["port"],
    )

    # Verify required models are available
    required_models = ["gemma3:1b", "gemma3"]  # Try both with and without tag
    model_available = False

    for model in required_models:
        if verify_ollama_model(
            model,
            host=ollama_connection_params["host"],
            port=ollama_connection_params["port"],
        ):
            model_available = True
            break

    if not model_available:
        print("\nWARNING: gemma3:1b model not found in Ollama")
        print("Run: ollama pull gemma3:1b")
        print("Tests will attempt to run but may fail if model is not available")


@pytest.fixture
def ollama_config(ensure_ollama_ready, ollama_connection_params) -> dict:
    """Provide Ollama configuration for tests that need real LLM."""
    return {
        "provider": "ollama",
        "model": "gemma3:1b",
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
