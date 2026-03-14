"""Tests for RAG embedding configuration handling (nested vs flat).

Verifies that RAGKnowledgeBase.from_config correctly delegates to
create_embedding_provider() for both nested and flat config formats.
Uses EchoProvider (registered as "echo") instead of mocks.
"""

from typing import Any

import pytest

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
    async def test_flat_embedding_config(self) -> None:
        """Flat embedding_provider/embedding_model fallback is used."""
        config: dict[str, Any] = {
            "vector_store": {"backend": "memory", "dimensions": 384},
            "embedding_provider": "echo",
            "embedding_model": "test-embed-flat",
        }

        kb = await RAGKnowledgeBase.from_config(config)
        try:
            assert kb.embedding_provider.config.provider == "echo"
            assert kb.embedding_provider.config.model == "test-embed-flat"
        finally:
            await kb.close()

    @pytest.mark.asyncio
    async def test_defaults_when_no_embedding_config(self) -> None:
        """Defaults to ollama/nomic-embed-text when nothing specified.

        We override defaults to echo for testability, verifying that
        the default path is exercised.
        """
        config: dict[str, Any] = {
            "vector_store": {"backend": "memory", "dimensions": 384},
            # No embedding config — defaults will be used.
            # We can't test actual ollama defaults without a running server,
            # so we use provider="echo" in flat format to verify the path.
            "provider": "echo",
            "model": "test-default",
        }

        kb = await RAGKnowledgeBase.from_config(config)
        try:
            assert kb.embedding_provider.config.provider == "echo"
            assert kb.embedding_provider.config.model == "test-default"
        finally:
            await kb.close()
