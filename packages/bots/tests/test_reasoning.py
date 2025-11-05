"""Tests for reasoning strategies."""

import pytest

from dataknobs_bots import BotContext, DynaBot
from dataknobs_bots.reasoning import (
    ReActReasoning,
    ReasoningStrategy,
    SimpleReasoning,
    create_reasoning_from_config,
)


class TestSimpleReasoning:
    """Tests for SimpleReasoning strategy."""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test simple reasoning initialization."""
        strategy = SimpleReasoning()
        assert isinstance(strategy, ReasoningStrategy)

    @pytest.mark.asyncio
    async def test_generate(self):
        """Test simple reasoning generation."""
        # Create a bot with simple reasoning
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "reasoning": {"strategy": "simple"},
        }

        bot = await DynaBot.from_config(config)
        context = BotContext(
            conversation_id="conv-simple", client_id="test-client"
        )

        # Generate response
        response = await bot.chat("Hello", context)
        assert response is not None
        assert isinstance(response, str)


class TestReActReasoning:
    """Tests for ReActReasoning strategy."""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test ReAct reasoning initialization."""
        strategy = ReActReasoning(max_iterations=3, verbose=False)
        assert isinstance(strategy, ReasoningStrategy)
        assert strategy.max_iterations == 3
        assert strategy.verbose is False

    @pytest.mark.asyncio
    async def test_default_initialization(self):
        """Test ReAct with default parameters."""
        strategy = ReActReasoning()
        assert strategy.max_iterations == 5
        assert strategy.verbose is False

    @pytest.mark.asyncio
    async def test_generate_without_tools(self):
        """Test ReAct falls back when no tools available."""
        # Create bot with ReAct but no tools
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "reasoning": {"strategy": "react", "max_iterations": 3},
        }

        bot = await DynaBot.from_config(config)
        context = BotContext(
            conversation_id="conv-react-no-tools", client_id="test-client"
        )

        # Should work fine without tools (falls back to simple)
        response = await bot.chat("Hello", context)
        assert response is not None

    @pytest.mark.asyncio
    async def test_verbose_mode(self):
        """Test verbose mode doesn't break execution."""
        strategy = ReActReasoning(max_iterations=2, verbose=True)
        assert strategy.verbose is True

        # Create bot with verbose ReAct
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "reasoning": {"strategy": "react", "verbose": True, "max_iterations": 2},
        }

        bot = await DynaBot.from_config(config)
        context = BotContext(
            conversation_id="conv-react-verbose", client_id="test-client"
        )

        # Should work with verbose output
        response = await bot.chat("Test", context)
        assert response is not None


class TestReasoningFactory:
    """Tests for reasoning factory function."""

    def test_create_simple_reasoning(self):
        """Test creating simple reasoning from config."""
        config = {"strategy": "simple"}
        strategy = create_reasoning_from_config(config)
        assert isinstance(strategy, SimpleReasoning)

    def test_create_react_reasoning(self):
        """Test creating ReAct reasoning from config."""
        config = {"strategy": "react", "max_iterations": 3, "verbose": True}
        strategy = create_reasoning_from_config(config)
        assert isinstance(strategy, ReActReasoning)
        assert strategy.max_iterations == 3
        assert strategy.verbose is True

    def test_default_strategy(self):
        """Test that default strategy is simple."""
        config = {}
        strategy = create_reasoning_from_config(config)
        assert isinstance(strategy, SimpleReasoning)

    def test_invalid_strategy(self):
        """Test error handling for invalid strategy."""
        config = {"strategy": "invalid"}
        with pytest.raises(ValueError, match="Unknown reasoning strategy"):
            create_reasoning_from_config(config)


class TestReasoningIntegration:
    """Integration tests for reasoning with bots."""

    @pytest.mark.asyncio
    async def test_bot_with_simple_reasoning(self):
        """Test bot with explicit simple reasoning."""
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "reasoning": {"strategy": "simple"},
        }

        bot = await DynaBot.from_config(config)
        assert bot.reasoning_strategy is not None
        assert isinstance(bot.reasoning_strategy, SimpleReasoning)

    @pytest.mark.asyncio
    async def test_bot_with_react_reasoning(self):
        """Test bot with ReAct reasoning."""
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "reasoning": {"strategy": "react", "max_iterations": 3},
        }

        bot = await DynaBot.from_config(config)
        assert bot.reasoning_strategy is not None
        assert isinstance(bot.reasoning_strategy, ReActReasoning)

    @pytest.mark.asyncio
    async def test_bot_without_reasoning(self):
        """Test bot works without explicit reasoning strategy."""
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
        }

        bot = await DynaBot.from_config(config)
        # Without reasoning config, strategy should be None
        # and bot should use default ConversationManager.complete()
        assert bot.reasoning_strategy is None

        context = BotContext(
            conversation_id="conv-no-reasoning", client_id="test-client"
        )
        response = await bot.chat("Hello", context)
        assert response is not None

    @pytest.mark.asyncio
    async def test_reasoning_with_memory(self):
        """Test reasoning strategies work with memory."""
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "memory": {"type": "buffer", "max_messages": 5},
            "reasoning": {"strategy": "simple"},
        }

        bot = await DynaBot.from_config(config)
        context = BotContext(
            conversation_id="conv-reasoning-memory", client_id="test-client"
        )

        # Multiple interactions
        await bot.chat("First message", context)
        response = await bot.chat("Second message", context)

        assert response is not None
        # Check memory was updated
        memory_context = await bot.memory.get_context("test")
        assert len(memory_context) >= 2

    @pytest.mark.asyncio
    async def test_react_with_store_trace(self):
        """Test ReAct with store_trace enabled."""
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "reasoning": {
                "strategy": "react",
                "max_iterations": 2,
                "store_trace": True,
            },
        }

        bot = await DynaBot.from_config(config)
        assert bot.reasoning_strategy is not None
        assert isinstance(bot.reasoning_strategy, ReActReasoning)
        assert bot.reasoning_strategy.store_trace is True

        context = BotContext(
            conversation_id="conv-react-trace", client_id="test-client"
        )

        # Generate response (will have no tools, so will complete immediately)
        response = await bot.chat("Test message", context)
        assert response is not None

        # Note: The trace would be stored in conversation metadata
        # In a real scenario with tools, we would verify the trace structure

    @pytest.mark.asyncio
    async def test_react_verbose_mode(self):
        """Test ReAct verbose mode uses logging instead of print."""
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "reasoning": {
                "strategy": "react",
                "max_iterations": 2,
                "verbose": True,
            },
        }

        bot = await DynaBot.from_config(config)
        assert bot.reasoning_strategy.verbose is True

        context = BotContext(
            conversation_id="conv-react-verbose", client_id="test-client"
        )

        # This should generate log messages (not print to stdout)
        # In production, these would go to your logging infrastructure
        response = await bot.chat("Test message", context)
        assert response is not None
