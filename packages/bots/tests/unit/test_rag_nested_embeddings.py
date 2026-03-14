"""Tests for RAG embedding configuration handling (nested vs legacy).

Verifies that RAGKnowledgeBase.from_config correctly delegates to
create_embedding_provider() for both nested and legacy config formats.
Uses EchoProvider (registered as "echo") instead of mocks.
"""

from typing import Any

import pytest
from dataknobs_common.testing import requires_ollama

from dataknobs_bots.knowledge.rag import RAGKnowledgeBase


class TestRAGEmbeddingsConfig:
    """Tests for from_config embedding config resolution."""

    @pytest.mark.asyncio
    async def test_nested_embedding_config(self) -> None:
        """Nested embedding config is used when present."""
        config: dict[str, Any] = {
            "vector_store": {"backend": "memory", "dimensions": 384},
            "embedding": {"provider": "echo", "model": "test-embed-nested"},
        }

        kb = await RAGKnowledgeBase.from_config(config)
        try:
            assert kb.embedding_provider.config.provider == "echo"
            assert kb.embedding_provider.config.model == "test-embed-nested"
        finally:
            await kb.close()

    @pytest.mark.asyncio
    async def test_legacy_embedding_config(self) -> None:
        """Legacy embedding_provider/embedding_model fallback is used."""
        config: dict[str, Any] = {
            "vector_store": {"backend": "memory", "dimensions": 384},
            "embedding_provider": "echo",
            "embedding_model": "test-embed-legacy",
        }

        kb = await RAGKnowledgeBase.from_config(config)
        try:
            assert kb.embedding_provider.config.provider == "echo"
            assert kb.embedding_provider.config.model == "test-embed-legacy"
        finally:
            await kb.close()

    @requires_ollama
    @pytest.mark.asyncio
    async def test_defaults_when_no_embedding_config(self) -> None:
        """Default Ollama provider used when no embedding config given."""
        config: dict[str, Any] = {
            "vector_store": {"backend": "memory", "dimensions": 768},
        }
        kb = await RAGKnowledgeBase.from_config(config)
        try:
            assert kb.embedding_provider.config.provider == "ollama"
            # Ollama normalizes model names to include ":latest" tag
            assert kb.embedding_provider.config.model in (
                "nomic-embed-text",
                "nomic-embed-text:latest",
            )
        finally:
            await kb.close()
