"""Tests for RAG embedding configuration handling (nested vs flat)."""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from dataknobs_bots.knowledge.rag import RAGKnowledgeBase


class TestRAGEmbeddingsConfig:
    """Tests for from_config embedding config resolution."""

    @pytest.mark.asyncio
    async def test_nested_embedding_config(self) -> None:
        """Nested embedding config is used when present."""
        config: dict[str, Any] = {
            "vector_store": {"backend": "memory", "dimensions": 1536},
            "embedding": {"provider": "openai", "model": "text-embedding-3-small"},
        }

        mock_vs = AsyncMock()
        mock_ep = AsyncMock()

        with (
            patch(
                "dataknobs_data.vector.stores.VectorStoreFactory"
            ) as mock_vs_factory_cls,
            patch(
                "dataknobs_llm.llm.LLMProviderFactory"
            ) as mock_llm_factory_cls,
        ):
            mock_vs_factory_cls.return_value.create.return_value = mock_vs
            mock_llm_factory = mock_llm_factory_cls.return_value
            mock_llm_factory.create.return_value = mock_ep

            await RAGKnowledgeBase.from_config(config)

            mock_llm_factory.create.assert_called_once_with(
                {"provider": "openai", "model": "text-embedding-3-small"}
            )

    @pytest.mark.asyncio
    async def test_flat_embedding_config(self) -> None:
        """Flat embedding_provider/embedding_model fallback is used."""
        config: dict[str, Any] = {
            "vector_store": {"backend": "memory", "dimensions": 768},
            "embedding_provider": "ollama",
            "embedding_model": "nomic-embed-text",
        }

        mock_vs = AsyncMock()
        mock_ep = AsyncMock()

        with (
            patch(
                "dataknobs_data.vector.stores.VectorStoreFactory"
            ) as mock_vs_factory_cls,
            patch(
                "dataknobs_llm.llm.LLMProviderFactory"
            ) as mock_llm_factory_cls,
        ):
            mock_vs_factory_cls.return_value.create.return_value = mock_vs
            mock_llm_factory = mock_llm_factory_cls.return_value
            mock_llm_factory.create.return_value = mock_ep

            await RAGKnowledgeBase.from_config(config)

            mock_llm_factory.create.assert_called_once_with(
                {"provider": "ollama", "model": "nomic-embed-text"}
            )

    @pytest.mark.asyncio
    async def test_defaults_when_no_embedding_config(self) -> None:
        """Defaults to openai/text-embedding-ada-002 when nothing specified."""
        config: dict[str, Any] = {
            "vector_store": {"backend": "memory", "dimensions": 1536},
        }

        mock_vs = AsyncMock()
        mock_ep = AsyncMock()

        with (
            patch(
                "dataknobs_data.vector.stores.VectorStoreFactory"
            ) as mock_vs_factory_cls,
            patch(
                "dataknobs_llm.llm.LLMProviderFactory"
            ) as mock_llm_factory_cls,
        ):
            mock_vs_factory_cls.return_value.create.return_value = mock_vs
            mock_llm_factory = mock_llm_factory_cls.return_value
            mock_llm_factory.create.return_value = mock_ep

            await RAGKnowledgeBase.from_config(config)

            mock_llm_factory.create.assert_called_once_with(
                {"provider": "openai", "model": "text-embedding-ada-002"}
            )
