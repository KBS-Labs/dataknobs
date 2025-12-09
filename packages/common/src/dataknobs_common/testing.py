"""Test utilities for dataknobs packages.

This module provides pytest utilities for service availability checking,
test configuration factories, and fixture helpers.

Example:
    ```python
    import pytest
    from dataknobs_common.testing import (
        is_ollama_available,
        requires_ollama,
        get_test_bot_config,
    )

    # Skip test if Ollama not available
    @pytest.mark.skipif(not is_ollama_available(), reason="Ollama not available")
    def test_with_ollama():
        ...

    # Or use the marker
    @requires_ollama
    def test_with_ollama_marker():
        ...

    # Get test configuration
    config = get_test_bot_config(use_echo_llm=True)
    ```
"""

import importlib.util
import logging
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Service Availability Checks


def is_ollama_available() -> bool:
    """Check if Ollama service is available.

    Returns:
        True if Ollama is running, False otherwise
    """
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def is_ollama_model_available(model_name: str = "nomic-embed-text") -> bool:
    """Check if a specific Ollama model is available.

    Args:
        model_name: Name of the model to check (default: nomic-embed-text)

    Returns:
        True if model is available, False otherwise
    """
    if not is_ollama_available():
        return False

    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return model_name in result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def is_faiss_available() -> bool:
    """Check if FAISS is available.

    Returns:
        True if FAISS can be imported, False otherwise
    """
    return importlib.util.find_spec("faiss") is not None


def is_chromadb_available() -> bool:
    """Check if ChromaDB is available.

    Returns:
        True if ChromaDB can be imported, False otherwise
    """
    return importlib.util.find_spec("chromadb") is not None


def is_redis_available(host: str = "localhost", port: int = 6379) -> bool:
    """Check if Redis service is available.

    Args:
        host: Redis host
        port: Redis port

    Returns:
        True if Redis is available, False otherwise
    """
    try:
        import socket

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except OSError:
        return False


def is_package_available(package_name: str) -> bool:
    """Check if a Python package is available.

    Args:
        package_name: Name of the package to check

    Returns:
        True if package can be imported, False otherwise
    """
    return importlib.util.find_spec(package_name) is not None


# Pytest Markers


try:
    import pytest

    requires_ollama = pytest.mark.skipif(
        not is_ollama_available(),
        reason="Ollama service not available",
    )

    requires_faiss = pytest.mark.skipif(
        not is_faiss_available(),
        reason="FAISS not installed",
    )

    requires_chromadb = pytest.mark.skipif(
        not is_chromadb_available(),
        reason="ChromaDB not installed",
    )

    requires_redis = pytest.mark.skipif(
        not is_redis_available(),
        reason="Redis not available",
    )

    def requires_package(package_name: str) -> Any:
        """Create a skip marker for a required package.

        Args:
            package_name: Name of the required package

        Returns:
            pytest.mark.skipif marker
        """
        return pytest.mark.skipif(
            not is_package_available(package_name),
            reason=f"{package_name} not installed",
        )

    def requires_ollama_model(model_name: str = "nomic-embed-text") -> Any:
        """Create a skip marker for a required Ollama model.

        Args:
            model_name: Name of the required model

        Returns:
            pytest.mark.skipif marker
        """
        return pytest.mark.skipif(
            not is_ollama_model_available(model_name),
            reason=f"Ollama model {model_name} not available",
        )

except ImportError:
    # pytest not installed - provide placeholder markers
    requires_ollama = None  # type: ignore
    requires_faiss = None  # type: ignore
    requires_chromadb = None  # type: ignore
    requires_redis = None  # type: ignore

    def requires_package(package_name: str) -> Any:  # type: ignore
        return None

    def requires_ollama_model(model_name: str = "nomic-embed-text") -> Any:  # type: ignore
        return None


# Test Configuration Factories


def get_test_bot_config(
    use_echo_llm: bool = True,
    use_in_memory_storage: bool = True,
    include_memory: bool = False,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """Get a test bot configuration.

    Args:
        use_echo_llm: Use echo LLM instead of real LLM (default: True)
        use_in_memory_storage: Use in-memory conversation storage (default: True)
        include_memory: Include buffer memory configuration (default: False)
        system_prompt: Optional system prompt content

    Returns:
        Bot configuration dictionary suitable for DynaBot.from_config()

    Example:
        ```python
        config = get_test_bot_config(
            use_echo_llm=True,
            system_prompt="You are a test assistant."
        )
        bot = await DynaBot.from_config(config)
        ```
    """
    config: dict[str, Any] = {
        "llm": {
            "provider": "echo" if use_echo_llm else "openai",
            "model": "test" if use_echo_llm else "gpt-4o-mini",
            "temperature": 0.7,
        },
        "conversation_storage": {
            "backend": "memory" if use_in_memory_storage else "file",
        },
    }

    if include_memory:
        config["memory"] = {
            "type": "buffer",
            "max_messages": 10,
        }

    if system_prompt:
        config["system_prompt"] = system_prompt

    return config


def get_test_rag_config(
    use_in_memory_store: bool = True,
    embedding_provider: str = "ollama",
    embedding_model: str = "nomic-embed-text",
) -> dict[str, Any]:
    """Get a test RAG/knowledge base configuration.

    Args:
        use_in_memory_store: Use in-memory vector store (default: True)
        embedding_provider: Embedding provider (default: "ollama")
        embedding_model: Embedding model name (default: "nomic-embed-text")

    Returns:
        Knowledge base configuration dictionary

    Example:
        ```python
        config = get_test_rag_config(use_in_memory_store=True)
        bot_config = get_test_bot_config()
        bot_config["knowledge_base"] = config
        ```
    """
    return {
        "type": "rag",
        "vector_store": {
            "backend": "memory" if use_in_memory_store else "faiss",
            "dimensions": 768,
            "metric": "cosine",
        },
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
        "chunking": {
            "max_chunk_size": 800,
            "chunk_overlap": 100,
        },
        "retrieval": {
            "top_k": 5,
            "score_threshold": 0.7,
        },
    }


# Test File Helpers


def create_test_markdown_files(tmp_path: Path) -> list[str]:
    """Create test markdown files for ingestion.

    Args:
        tmp_path: Temporary directory path (from pytest fixture)

    Returns:
        List of created file paths as strings

    Example:
        ```python
        def test_ingestion(tmp_path):
            files = create_test_markdown_files(tmp_path)
            # files contains paths to test markdown documents
        ```
    """
    files = []

    # Create test markdown file 1
    md1 = tmp_path / "test_doc1.md"
    md1.write_text(
        """# Test Document 1

## Introduction

This is a test document for validating ingestion and retrieval.

### Key Points

1. First important point
2. Second important point
3. Third important point

## Details

More detailed information about the topic goes here.
"""
    )
    files.append(str(md1))

    # Create test markdown file 2
    md2 = tmp_path / "test_doc2.md"
    md2.write_text(
        """# Test Document 2

## Overview

Another test document with different content.

## Content

- Item A: Description of item A
- Item B: Description of item B
- Item C: Description of item C

## Summary

This concludes the second test document.
"""
    )
    files.append(str(md2))

    return files


def create_test_json_files(tmp_path: Path) -> list[str]:
    """Create test JSON files.

    Args:
        tmp_path: Temporary directory path (from pytest fixture)

    Returns:
        List of created file paths as strings
    """
    import json

    files = []

    # Create test JSON file 1
    json1 = tmp_path / "test_data1.json"
    json1.write_text(
        json.dumps(
            {
                "title": "Test Data 1",
                "items": [
                    {"id": 1, "name": "Item 1", "value": 100},
                    {"id": 2, "name": "Item 2", "value": 200},
                ],
                "metadata": {"version": "1.0", "created": "2024-01-01"},
            },
            indent=2,
        )
    )
    files.append(str(json1))

    # Create test JSON file 2
    json2 = tmp_path / "test_data2.json"
    json2.write_text(
        json.dumps(
            {
                "title": "Test Data 2",
                "items": [
                    {"id": 3, "name": "Item 3", "value": 300},
                    {"id": 4, "name": "Item 4", "value": 400},
                ],
                "metadata": {"version": "1.0", "created": "2024-01-02"},
            },
            indent=2,
        )
    )
    files.append(str(json2))

    return files
