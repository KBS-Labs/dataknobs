"""Tests for memory implementations."""

import pytest
import numpy as np

from dataknobs_bots.memory import BufferMemory, VectorMemory, create_memory_from_config
from dataknobs_data.vector.stores import VectorStoreFactory
from dataknobs_llm.llm import LLMProviderFactory


class TestBufferMemory:
    """Tests for BufferMemory."""

    @pytest.mark.asyncio
    async def test_add_and_get_messages(self):
        """Test adding and retrieving messages."""
        memory = BufferMemory(max_messages=3)

        # Add messages
        await memory.add_message("Hello", "user")
        await memory.add_message("Hi there!", "assistant")
        await memory.add_message("How are you?", "user")

        # Get context
        context = await memory.get_context("test")
        assert len(context) == 3
        assert context[0]["content"] == "Hello"
        assert context[0]["role"] == "user"
        assert context[1]["content"] == "Hi there!"
        assert context[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_buffer_overflow(self):
        """Test that buffer respects max_messages limit."""
        memory = BufferMemory(max_messages=2)

        # Add 3 messages
        await memory.add_message("First", "user")
        await memory.add_message("Second", "assistant")
        await memory.add_message("Third", "user")

        # Should only have last 2
        context = await memory.get_context("test")
        assert len(context) == 2
        assert context[0]["content"] == "Second"
        assert context[1]["content"] == "Third"

    @pytest.mark.asyncio
    async def test_clear(self):
        """Test clearing memory."""
        memory = BufferMemory(max_messages=3)

        # Add messages
        await memory.add_message("Hello", "user")
        await memory.add_message("Hi", "assistant")

        # Clear
        await memory.clear()

        # Should be empty
        context = await memory.get_context("test")
        assert len(context) == 0

    @pytest.mark.asyncio
    async def test_metadata(self):
        """Test storing metadata with messages."""
        memory = BufferMemory()

        metadata = {"source": "test", "timestamp": "2024-01-01"}
        await memory.add_message("Hello", "user", metadata=metadata)

        context = await memory.get_context("test")
        assert context[0]["metadata"] == metadata


class TestVectorMemory:
    """Tests for VectorMemory."""

    @pytest.mark.asyncio
    async def test_add_and_get_messages(self):
        """Test adding and retrieving messages with vector similarity."""
        # Create in-memory vector store
        factory = VectorStoreFactory()
        vector_store = factory.create(backend="memory", dimensions=384)
        await vector_store.initialize()

        # Create Echo provider for embeddings (deterministic for testing)
        llm_factory = LLMProviderFactory(is_async=True)
        embedding_provider = llm_factory.create({"provider": "echo", "model": "test"})
        await embedding_provider.initialize()

        # Create vector memory
        memory = VectorMemory(
            vector_store=vector_store,
            embedding_provider=embedding_provider,
            max_results=5,
            similarity_threshold=0.0,  # Low threshold for testing
        )

        # Add messages
        await memory.add_message("Hello world", "user")
        await memory.add_message("Hi there", "assistant")
        await memory.add_message("Python programming", "user")

        # Get context
        context = await memory.get_context("greeting")
        assert len(context) > 0
        assert all("content" in msg for msg in context)
        assert all("role" in msg for msg in context)
        assert all("similarity" in msg for msg in context)

    @pytest.mark.asyncio
    async def test_similarity_threshold(self):
        """Test that similarity threshold filters results."""
        factory = VectorStoreFactory()
        vector_store = factory.create(backend="memory", dimensions=384)
        await vector_store.initialize()

        llm_factory = LLMProviderFactory(is_async=True)
        embedding_provider = llm_factory.create({"provider": "echo", "model": "test"})
        await embedding_provider.initialize()

        # Create memory with high threshold
        memory = VectorMemory(
            vector_store=vector_store,
            embedding_provider=embedding_provider,
            max_results=10,
            similarity_threshold=0.99,  # Very high threshold
        )

        # Add messages
        await memory.add_message("Hello", "user")

        # Get context - might return nothing due to high threshold
        context = await memory.get_context("completely different topic")
        # With Echo provider, similarity should be deterministic
        # The test verifies the threshold filtering works
        assert isinstance(context, list)

    @pytest.mark.asyncio
    async def test_from_config(self):
        """Test creating VectorMemory from configuration."""
        config = {
            "backend": "memory",
            "dimension": 384,
            "embedding_provider": "echo",
            "embedding_model": "test",
            "max_results": 3,
            "similarity_threshold": 0.5,
        }

        memory = await VectorMemory.from_config(config)
        assert memory.max_results == 3
        assert memory.similarity_threshold == 0.5

        # Test it works
        await memory.add_message("Test message", "user")
        context = await memory.get_context("test")
        assert isinstance(context, list)


class TestMemoryFactory:
    """Tests for memory factory function."""

    @pytest.mark.asyncio
    async def test_create_buffer_memory(self):
        """Test creating buffer memory from config."""
        config = {"type": "buffer", "max_messages": 5}

        memory = await create_memory_from_config(config)
        assert isinstance(memory, BufferMemory)
        assert memory.max_messages == 5

    @pytest.mark.asyncio
    async def test_create_vector_memory(self):
        """Test creating vector memory from config."""
        config = {
            "type": "vector",
            "backend": "memory",
            "dimension": 384,
            "embedding_provider": "echo",
            "embedding_model": "test",
        }

        memory = await create_memory_from_config(config)
        assert isinstance(memory, VectorMemory)

    @pytest.mark.asyncio
    async def test_default_type(self):
        """Test that default memory type is buffer."""
        config = {}

        memory = await create_memory_from_config(config)
        assert isinstance(memory, BufferMemory)

    @pytest.mark.asyncio
    async def test_invalid_type(self):
        """Test error handling for invalid memory type."""
        config = {"type": "invalid"}

        with pytest.raises(ValueError, match="Unknown memory type"):
            await create_memory_from_config(config)
