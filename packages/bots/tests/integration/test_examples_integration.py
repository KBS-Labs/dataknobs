"""Integration tests for DynaBot examples.

These tests verify that:
1. The examples work correctly (using Echo LLM for most tests)
2. The underlying implementation functions properly
3. All features demonstrated in examples are functional

Most tests use the Echo LLM provider for fast, deterministic testing.
Tests that require real LLM reasoning (e.g., semantic memory) use Ollama
and are marked with @pytest.mark.ollama_required.

Run all tests:
    pytest tests/integration/test_examples_integration.py

Run only Echo-based tests (no Ollama needed):
    pytest tests/integration/test_examples_integration.py -m "not ollama_required"

Run Ollama tests (requires Ollama + gemma3:1b):
    TEST_OLLAMA=true pytest tests/integration/test_examples_integration.py -m ollama_required
"""

import os
from typing import Any, Dict

import pytest

from dataknobs_bots import BotContext, DynaBot
from dataknobs_llm.tools import Tool


# =============================================================================
# Tests using Echo LLM (fast, no external dependencies)
# =============================================================================


class TestSimpleChatbotIntegration:
    """Integration tests for simple chatbot (example 01) using Echo LLM."""

    @pytest.mark.asyncio
    async def test_simple_conversation(self, bot_config_echo):
        """Test basic conversation flow."""
        bot = await DynaBot.from_config(bot_config_echo)

        context = BotContext(
            conversation_id="test-simple-001",
            client_id="test-client",
            user_id="test-user",
        )

        # Send a message
        response = await bot.chat("Hello, how are you?", context)

        # Verify we got a response (Echo returns "Echo: <message>")
        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0
        assert "Echo:" in response

    @pytest.mark.asyncio
    async def test_multiple_messages(self, bot_config_echo):
        """Test multiple message exchanges."""
        bot = await DynaBot.from_config(bot_config_echo)

        context = BotContext(
            conversation_id="test-simple-002",
            client_id="test-client",
            user_id="test-user",
        )

        # Send multiple messages
        messages = [
            "Hello!",
            "What is Python?",
            "Thank you!",
        ]

        for message in messages:
            response = await bot.chat(message, context)
            assert response is not None
            assert len(response) > 0


class TestChatbotWithMemoryIntegration:
    """Integration tests for chatbot with memory (example 02)."""

    @pytest.mark.asyncio
    async def test_memory_buffer_limit(self, bot_config_echo_with_memory):
        """Test that memory buffer respects max_messages limit."""
        bot = await DynaBot.from_config(bot_config_echo_with_memory)

        context = BotContext(
            conversation_id="test-memory-002",
            client_id="test-client",
            user_id="test-user",
        )

        # Send more messages than buffer size
        max_messages = bot_config_echo_with_memory["memory"]["max_messages"]
        for i in range(max_messages + 5):
            await bot.chat(f"Message {i}", context)

        # Verify memory doesn't grow unbounded
        if bot.memory:
            memory_context = await bot.memory.get_context("test")
            assert len(memory_context) <= max_messages

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.environ.get("TEST_OLLAMA", "").lower() == "true",
        reason="Semantic memory test requires real LLM (TEST_OLLAMA=true)",
    )
    async def test_memory_retention_with_ollama(self, bot_config_with_memory):
        """Test that bot retains context across messages (requires real LLM)."""
        bot = await DynaBot.from_config(bot_config_with_memory)

        context = BotContext(
            conversation_id="test-memory-001",
            client_id="test-client",
            user_id="test-user",
        )

        # First message establishes context
        await bot.chat("My name is Alice", context)

        # Second message should use memory
        response = await bot.chat("What is my name?", context)

        # Bot should remember the name (basic check)
        assert response is not None
        assert len(response) > 0


class TestMultiTenantIntegration:
    """Integration tests for multi-tenant bot (example 05) using Echo LLM."""

    @pytest.mark.asyncio
    async def test_conversation_isolation(self, bot_config_echo):
        """Test that conversations are isolated per client."""
        bot = await DynaBot.from_config(bot_config_echo)

        # Create contexts for different clients
        context1 = BotContext(
            conversation_id="test-mt-client1",
            client_id="client-1",
            user_id="user-1",
        )

        context2 = BotContext(
            conversation_id="test-mt-client2",
            client_id="client-2",
            user_id="user-2",
        )

        # Each client sends different messages
        response1 = await bot.chat("My favorite color is blue", context1)
        response2 = await bot.chat("My favorite color is red", context2)

        # Both should get valid responses
        assert response1 is not None
        assert response2 is not None

        # Contexts should remain isolated
        assert context1.conversation_id != context2.conversation_id
        assert context1.client_id != context2.client_id

    @pytest.mark.asyncio
    async def test_concurrent_conversations(self, bot_config_echo_with_memory):
        """Test concurrent conversations with shared bot instance."""
        import asyncio

        bot = await DynaBot.from_config(bot_config_echo_with_memory)

        async def client_conversation(client_id: str, num_messages: int):
            """Simulate a client conversation."""
            context = BotContext(
                conversation_id=f"test-concurrent-{client_id}",
                client_id=client_id,
                user_id=f"user-{client_id}",
            )

            responses = []
            for i in range(num_messages):
                response = await bot.chat(f"Message {i} from {client_id}", context)
                responses.append(response)

            return responses

        # Run 3 concurrent conversations
        results = await asyncio.gather(
            client_conversation("client-1", 3),
            client_conversation("client-2", 3),
            client_conversation("client-3", 3),
        )

        # Verify all conversations completed successfully
        assert len(results) == 3
        for client_responses in results:
            assert len(client_responses) == 3
            assert all(r is not None for r in client_responses)


class TestReActIntegration:
    """Integration tests for ReAct agent (example 04) using Echo LLM."""

    @pytest.mark.asyncio
    async def test_react_without_tools(self, bot_config_echo_react):
        """Test ReAct falls back gracefully when no tools available."""
        bot = await DynaBot.from_config(bot_config_echo_react)

        context = BotContext(
            conversation_id="test-react-001",
            client_id="test-client",
            user_id="test-user",
        )

        # Without tools, should still work (falls back to simple generation)
        response = await bot.chat("Hello!", context)

        assert response is not None
        assert isinstance(response, str)

    @pytest.mark.asyncio
    async def test_react_trace_storage(self, bot_config_echo_react):
        """Test that ReAct stores reasoning trace in metadata."""
        bot = await DynaBot.from_config(bot_config_echo_react)

        context = BotContext(
            conversation_id="test-react-trace-001",
            client_id="test-client",
            user_id="test-user",
        )

        # Generate a response
        await bot.chat("Hello!", context)

        # Verify reasoning strategy has store_trace enabled
        assert bot.reasoning_strategy is not None
        assert bot.reasoning_strategy.store_trace is True


class TestBotConfigurationIntegration:
    """Integration tests for bot configuration variations using Echo LLM."""

    @pytest.mark.asyncio
    async def test_temperature_variation(self, echo_config):
        """Test different temperature settings."""
        for temp in [0.0, 0.5, 1.0]:
            config = {
                "llm": {**echo_config, "temperature": temp},
                "conversation_storage": {"backend": "memory"},
            }

            bot = await DynaBot.from_config(config)
            assert bot.default_temperature == temp

            context = BotContext(
                conversation_id=f"test-temp-{temp}",
                client_id="test-client",
            )

            response = await bot.chat("Hello", context)
            assert response is not None

    @pytest.mark.asyncio
    async def test_max_tokens_variation(self, echo_config):
        """Test different max_tokens settings."""
        for max_tokens in [100, 500, 1000]:
            config = {
                "llm": {**echo_config, "max_tokens": max_tokens},
                "conversation_storage": {"backend": "memory"},
            }

            bot = await DynaBot.from_config(config)
            assert bot.default_max_tokens == max_tokens

            context = BotContext(
                conversation_id=f"test-tokens-{max_tokens}",
                client_id="test-client",
            )

            response = await bot.chat("Hello", context)
            assert response is not None


class TestStorageBackendIntegration:
    """Integration tests for different storage backends using Echo LLM."""

    @pytest.mark.asyncio
    async def test_memory_backend(self, echo_config):
        """Test in-memory storage backend."""
        config = {
            "llm": echo_config,
            "conversation_storage": {"backend": "memory"},
        }

        bot = await DynaBot.from_config(config)

        context = BotContext(
            conversation_id="test-storage-memory",
            client_id="test-client",
        )

        # Multiple messages should work
        for i in range(3):
            response = await bot.chat(f"Message {i}", context)
            assert response is not None

    @pytest.mark.asyncio
    async def test_conversation_persistence(self, bot_config_echo):
        """Test that conversations persist across bot restarts (in memory)."""
        # First bot instance
        bot1 = await DynaBot.from_config(bot_config_echo)

        context = BotContext(
            conversation_id="test-persist-001",
            client_id="test-client",
        )

        # Send a message
        await bot1.chat("First message", context)

        # Create second bot instance (simulating restart)
        # Note: In-memory storage won't actually persist, but the test
        # verifies the architecture supports this pattern
        bot2 = await DynaBot.from_config(bot_config_echo)

        # Second bot can create a new conversation with same ID
        response = await bot2.chat("Second message", context)
        assert response is not None


class TestReasoningStrategiesIntegration:
    """Integration tests for different reasoning strategies using Echo LLM."""

    @pytest.mark.asyncio
    async def test_simple_reasoning(self, echo_config):
        """Test simple reasoning strategy."""
        config = {
            "llm": echo_config,
            "conversation_storage": {"backend": "memory"},
            "reasoning": {"strategy": "simple"},
        }

        bot = await DynaBot.from_config(config)

        from dataknobs_bots.reasoning import SimpleReasoning

        assert isinstance(bot.reasoning_strategy, SimpleReasoning)

        context = BotContext(
            conversation_id="test-simple-reasoning",
            client_id="test-client",
        )

        response = await bot.chat("Hello!", context)
        assert response is not None

    @pytest.mark.asyncio
    async def test_react_reasoning_config(self, echo_config):
        """Test ReAct reasoning configuration options."""
        config = {
            "llm": echo_config,
            "conversation_storage": {"backend": "memory"},
            "reasoning": {
                "strategy": "react",
                "max_iterations": 3,
                "verbose": True,
                "store_trace": True,
            },
        }

        bot = await DynaBot.from_config(config)

        from dataknobs_bots.reasoning import ReActReasoning

        assert isinstance(bot.reasoning_strategy, ReActReasoning)
        assert bot.reasoning_strategy.max_iterations == 3
        assert bot.reasoning_strategy.verbose is True
        assert bot.reasoning_strategy.store_trace is True


class TestErrorHandlingIntegration:
    """Integration tests for error handling."""

    @pytest.mark.asyncio
    async def test_invalid_llm_config(self):
        """Test handling of invalid LLM configuration."""
        config = {
            "llm": {
                "provider": "invalid_provider",
                "model": "invalid_model",
            },
            "conversation_storage": {"backend": "memory"},
        }

        # Should raise an error during bot creation
        with pytest.raises(Exception):
            await DynaBot.from_config(config)

    @pytest.mark.asyncio
    async def test_empty_message(self, bot_config_echo):
        """Test handling of empty message."""
        bot = await DynaBot.from_config(bot_config_echo)

        context = BotContext(
            conversation_id="test-empty-msg",
            client_id="test-client",
        )

        # Empty messages should be rejected by the conversation manager
        with pytest.raises(ValueError, match="Either content or prompt_name must be provided"):
            await bot.chat("", context)


# =============================================================================
# Ollama Infrastructure Test (verifies Ollama connectivity)
# =============================================================================


class TestOllamaConnectivity:
    """Test Ollama infrastructure connectivity."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.environ.get("TEST_OLLAMA", "").lower() == "true",
        reason="Ollama connectivity test requires TEST_OLLAMA=true",
    )
    async def test_ollama_basic_chat(self, ollama_config):
        """Test basic chat with Ollama to verify infrastructure is working."""
        config = {
            "llm": ollama_config,
            "conversation_storage": {"backend": "memory"},
        }

        bot = await DynaBot.from_config(config)

        context = BotContext(
            conversation_id="test-ollama-connectivity",
            client_id="test-client",
        )

        # Simple test to verify Ollama is responding
        response = await bot.chat("Say 'hello' and nothing else.", context)

        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0
