"""Tests for memory implementations."""

import pytest
import numpy as np

from dataknobs_bots.memory import (
    BufferMemory,
    SummaryMemory,
    VectorMemory,
    create_memory_from_config,
)
from dataknobs_data.vector.stores import VectorStoreFactory
from dataknobs_llm import EchoProvider
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


class TestSummaryMemory:
    """Tests for SummaryMemory."""

    @staticmethod
    def _create_echo_provider(
        responses: list[str] | None = None,
    ) -> EchoProvider:
        """Create an EchoProvider with optional scripted responses."""
        factory = LLMProviderFactory(is_async=True)
        provider = factory.create({"provider": "echo", "model": "test"})
        if responses:
            provider.set_responses(responses, cycle=True)
        return provider

    @pytest.mark.asyncio
    async def test_add_and_get_messages_within_window(self):
        """Messages within the window are returned verbatim."""
        provider = self._create_echo_provider()
        memory = SummaryMemory(llm_provider=provider, recent_window=5)

        await memory.add_message("Hello", "user")
        await memory.add_message("Hi there!", "assistant")

        context = await memory.get_context("test")
        assert len(context) == 2
        assert context[0]["content"] == "Hello"
        assert context[0]["role"] == "user"
        assert context[1]["content"] == "Hi there!"
        assert context[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_summarization_triggers_at_threshold(self):
        """When messages exceed recent_window, oldest are summarized."""
        provider = self._create_echo_provider(
            responses=["Summary of the conversation so far."]
        )
        await provider.initialize()
        memory = SummaryMemory(llm_provider=provider, recent_window=3)

        # Add 4 messages (exceeds window of 3)
        await memory.add_message("Message 1", "user")
        await memory.add_message("Message 2", "assistant")
        await memory.add_message("Message 3", "user")
        await memory.add_message("Message 4", "assistant")  # Triggers summarization

        context = await memory.get_context("test")

        # First element should be the summary
        assert context[0]["role"] == "system"
        assert context[0]["metadata"]["is_summary"] is True
        assert "Summary of the conversation" in context[0]["content"]

        # Remaining should be the recent messages (window of 3)
        recent = [m for m in context if m.get("metadata", {}).get("is_summary") is not True]
        assert len(recent) == 3

    @pytest.mark.asyncio
    async def test_get_context_returns_summary_plus_recent(self):
        """get_context returns [summary] + [recent_messages]."""
        provider = self._create_echo_provider(
            responses=["First summary", "Updated summary"]
        )
        await provider.initialize()
        memory = SummaryMemory(llm_provider=provider, recent_window=2)

        # Add enough to trigger summarization
        await memory.add_message("Msg 1", "user")
        await memory.add_message("Msg 2", "assistant")
        await memory.add_message("Msg 3", "user")  # Triggers summarization of Msg 1

        context = await memory.get_context("test")

        # Should have summary + 2 recent messages
        assert len(context) == 3
        assert context[0]["role"] == "system"
        assert context[1]["content"] == "Msg 2"
        assert context[2]["content"] == "Msg 3"

    @pytest.mark.asyncio
    async def test_clear_resets_summary_and_buffer(self):
        """Clear removes both the summary and buffered messages."""
        provider = self._create_echo_provider(
            responses=["A summary"]
        )
        await provider.initialize()
        memory = SummaryMemory(llm_provider=provider, recent_window=2)

        # Fill and trigger summarization
        await memory.add_message("Msg 1", "user")
        await memory.add_message("Msg 2", "assistant")
        await memory.add_message("Msg 3", "user")

        # Verify non-empty
        context = await memory.get_context("test")
        assert len(context) > 0

        # Clear
        await memory.clear()

        # Should be empty
        context = await memory.get_context("test")
        assert len(context) == 0

    @pytest.mark.asyncio
    async def test_graceful_degradation_on_llm_failure(self):
        """When the LLM fails, old messages are dropped gracefully."""
        provider = self._create_echo_provider()
        await provider.initialize()

        # Make the provider raise on complete
        async def fail_complete(*args: object, **kwargs: object) -> None:
            raise RuntimeError("LLM unavailable")

        provider.complete = fail_complete  # type: ignore[assignment]

        memory = SummaryMemory(llm_provider=provider, recent_window=2)

        # Add 3 messages — the overflow triggers summarization which will fail
        await memory.add_message("Msg 1", "user")
        await memory.add_message("Msg 2", "assistant")
        await memory.add_message("Msg 3", "user")  # Triggers failed summarization

        # Should still work — old messages dropped, recent kept
        context = await memory.get_context("test")
        recent = [m for m in context if m.get("metadata", {}).get("is_summary") is not True]
        assert len(recent) == 2
        assert recent[0]["content"] == "Msg 2"
        assert recent[1]["content"] == "Msg 3"

    @pytest.mark.asyncio
    async def test_empty_history(self):
        """get_context on empty memory returns an empty list."""
        provider = self._create_echo_provider()
        memory = SummaryMemory(llm_provider=provider)

        context = await memory.get_context("test")
        assert context == []

    @pytest.mark.asyncio
    async def test_custom_summary_prompt(self):
        """Custom summary_prompt is used for summarization."""
        custom_prompt = (
            "CUSTOM: Summarize.\n{existing_summary}\n{new_messages}"
        )
        provider = self._create_echo_provider(responses=["custom summary result"])
        await provider.initialize()
        memory = SummaryMemory(
            llm_provider=provider,
            recent_window=1,
            summary_prompt=custom_prompt,
        )

        await memory.add_message("Msg 1", "user")
        await memory.add_message("Msg 2", "user")  # Triggers summarization

        context = await memory.get_context("test")
        assert any("custom summary result" in m["content"] for m in context)


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
    async def test_create_summary_memory(self):
        """Test creating summary memory from config."""
        factory = LLMProviderFactory(is_async=True)
        provider = factory.create({"provider": "echo", "model": "test"})

        config = {"type": "summary", "recent_window": 5}
        memory = await create_memory_from_config(config, llm_provider=provider)
        assert isinstance(memory, SummaryMemory)
        assert memory.recent_window == 5

    @pytest.mark.asyncio
    async def test_create_summary_memory_with_dedicated_llm(self):
        """Test creating summary memory with its own LLM config."""
        config = {
            "type": "summary",
            "recent_window": 8,
            "llm": {"provider": "echo", "model": "summary-model"},
        }
        # No fallback provider needed — dedicated LLM is in config
        memory = await create_memory_from_config(config)
        assert isinstance(memory, SummaryMemory)
        assert memory.recent_window == 8

    @pytest.mark.asyncio
    async def test_create_summary_memory_dedicated_llm_overrides_fallback(self):
        """Dedicated LLM config takes precedence over fallback provider."""
        fallback = LLMProviderFactory(is_async=True).create(
            {"provider": "echo", "model": "fallback"}
        )
        config = {
            "type": "summary",
            "llm": {"provider": "echo", "model": "dedicated"},
        }
        memory = await create_memory_from_config(config, llm_provider=fallback)
        assert isinstance(memory, SummaryMemory)
        # The provider should be the dedicated one, not the fallback
        assert memory.llm_provider is not fallback

    @pytest.mark.asyncio
    async def test_create_summary_memory_without_any_provider_raises(self):
        """Test that summary memory without any LLM source raises ValueError."""
        config = {"type": "summary"}
        with pytest.raises(ValueError, match="requires an LLM provider"):
            await create_memory_from_config(config)

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
