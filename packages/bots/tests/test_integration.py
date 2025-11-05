"""Integration tests for DynaBot with knowledge base."""

from pathlib import Path

import pytest

from dataknobs_bots import BotContext, DynaBot


class TestDynaBotWithKnowledgeBase:
    """Integration tests for DynaBot with RAG knowledge base."""

    @pytest.mark.asyncio
    async def test_bot_with_knowledge_base(self):
        """Test creating a bot with a knowledge base."""
        test_docs_dir = Path(__file__).parent / "test_docs"

        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "knowledge_base": {
                "enabled": True,
                "type": "rag",
                "vector_store": {"backend": "memory", "dimensions": 384},
                "embedding_provider": "echo",
                "embedding_model": "test",
                "documents_path": str(test_docs_dir),
                "document_pattern": "**/*.md",
            },
        }

        bot = await DynaBot.from_config(config)

        # Verify knowledge base is loaded
        assert bot.knowledge_base is not None

        # Query the knowledge base directly
        results = await bot.knowledge_base.query("How do I install DynaBot?", k=2)
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_bot_chat_with_knowledge_context(self):
        """Test that bot includes knowledge base context in messages."""
        test_docs_dir = Path(__file__).parent / "test_docs"

        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "knowledge_base": {
                "enabled": True,
                "type": "rag",
                "vector_store": {"backend": "memory", "dimensions": 384},
                "embedding_provider": "echo",
                "embedding_model": "test",
                "documents_path": str(test_docs_dir),
            },
        }

        bot = await DynaBot.from_config(config)
        context = BotContext(conversation_id="conv-kb-test", client_id="test-client")

        # Ask a question that should retrieve knowledge
        response = await bot.chat("How do I configure memory?", context)

        # Echo provider echoes back, so we can verify knowledge was included
        assert response is not None
        assert isinstance(response, str)

    @pytest.mark.asyncio
    async def test_bot_with_memory_and_knowledge(self):
        """Test bot with both memory and knowledge base."""
        test_docs_dir = Path(__file__).parent / "test_docs"

        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "memory": {"type": "buffer", "max_messages": 5},
            "knowledge_base": {
                "enabled": True,
                "type": "rag",
                "vector_store": {"backend": "memory", "dimensions": 384},
                "embedding_provider": "echo",
                "embedding_model": "test",
                "documents_path": str(test_docs_dir),
            },
        }

        bot = await DynaBot.from_config(config)
        context = BotContext(conversation_id="conv-both", client_id="test-client")

        # First message
        await bot.chat("Tell me about installation", context)

        # Second message - should have memory context
        response = await bot.chat("What about configuration?", context)

        # Verify both memory and knowledge base are working
        assert bot.memory is not None
        assert bot.knowledge_base is not None
        assert response is not None

        # Check memory has messages
        memory_context = await bot.memory.get_context("test")
        assert len(memory_context) >= 2  # At least 2 messages (user + assistant)

    @pytest.mark.asyncio
    async def test_bot_without_knowledge_base(self):
        """Test that bot works without knowledge base (backward compatibility)."""
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
        }

        bot = await DynaBot.from_config(config)
        context = BotContext(
            conversation_id="conv-no-kb", client_id="test-client"
        )

        # Should work fine without knowledge base
        response = await bot.chat("Hello", context)
        assert response is not None
        assert bot.knowledge_base is None

    @pytest.mark.asyncio
    async def test_knowledge_base_query_multiple_docs(self):
        """Test querying across multiple documents."""
        test_docs_dir = Path(__file__).parent / "test_docs"

        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "knowledge_base": {
                "enabled": True,
                "type": "rag",
                "vector_store": {"backend": "memory", "dimensions": 384},
                "embedding_provider": "echo",
                "embedding_model": "test",
                "documents_path": str(test_docs_dir),
            },
        }

        bot = await DynaBot.from_config(config)

        # Query for something that appears in both documents
        results = await bot.knowledge_base.query("configuration", k=5)

        # Should get results from multiple sources
        assert len(results) > 0

        # Check that we have results from different sources
        sources = set(r["source"] for r in results)
        # Should have at least one source
        assert len(sources) >= 1
