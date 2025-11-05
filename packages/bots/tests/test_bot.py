"""Tests for DynaBot core functionality."""

import pytest

from dataknobs_bots import BotContext, DynaBot
from dataknobs_bots.memory import BufferMemory


class TestBotContext:
    """Tests for BotContext dataclass."""

    def test_basic_context(self):
        """Test basic context creation."""
        context = BotContext(conversation_id="conv-1", client_id="client-1")

        assert context.conversation_id == "conv-1"
        assert context.client_id == "client-1"
        assert context.user_id is None
        assert context.session_metadata == {}
        assert context.request_metadata == {}

    def test_context_with_metadata(self):
        """Test context with metadata."""
        session_metadata = {"session_key": "value"}
        request_metadata = {"request_id": "123"}

        context = BotContext(
            conversation_id="conv-1",
            client_id="client-1",
            user_id="user-1",
            session_metadata=session_metadata,
            request_metadata=request_metadata,
        )

        assert context.user_id == "user-1"
        assert context.session_metadata == session_metadata
        assert context.request_metadata == request_metadata


class TestDynaBot:
    """Tests for DynaBot."""

    @pytest.mark.asyncio
    async def test_from_config_minimal(self):
        """Test creating DynaBot from minimal configuration."""
        config = {
            "llm": {"provider": "echo", "model": "test", "temperature": 0.7},
            "conversation_storage": {"backend": "memory"},
        }

        bot = await DynaBot.from_config(config)
        assert bot is not None
        assert bot.default_temperature == 0.7
        assert bot.default_max_tokens == 1000

    @pytest.mark.asyncio
    async def test_from_config_with_memory(self):
        """Test creating DynaBot with memory configuration."""
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "memory": {"type": "buffer", "max_messages": 5},
        }

        bot = await DynaBot.from_config(config)
        assert bot.memory is not None
        assert isinstance(bot.memory, BufferMemory)
        assert bot.memory.max_messages == 5

    @pytest.mark.asyncio
    async def test_from_config_with_system_prompt(self):
        """Test creating DynaBot with system prompt."""
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "system_prompt": {"name": "helpful_assistant"},
        }

        bot = await DynaBot.from_config(config)
        assert bot.system_prompt_name == "helpful_assistant"

    @pytest.mark.asyncio
    async def test_chat_basic(self):
        """Test basic chat functionality."""
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
        }

        bot = await DynaBot.from_config(config)
        context = BotContext(conversation_id="conv-test-1", client_id="test-client")

        # Echo provider should echo the message back
        response = await bot.chat("Hello, bot!", context)
        assert response is not None
        assert isinstance(response, str)

    @pytest.mark.asyncio
    async def test_chat_with_memory(self):
        """Test chat with memory context."""
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "memory": {"type": "buffer", "max_messages": 10},
        }

        bot = await DynaBot.from_config(config)
        context = BotContext(conversation_id="conv-test-2", client_id="test-client")

        # First message
        await bot.chat("First message", context)

        # Memory should have the message
        memory_context = await bot.memory.get_context("test")
        assert len(memory_context) >= 1

    @pytest.mark.asyncio
    async def test_chat_temperature_override(self):
        """Test overriding temperature in chat."""
        config = {
            "llm": {"provider": "echo", "model": "test", "temperature": 0.5},
            "conversation_storage": {"backend": "memory"},
        }

        bot = await DynaBot.from_config(config)
        context = BotContext(conversation_id="conv-test-3", client_id="test-client")

        # Chat with temperature override
        response = await bot.chat("Hello", context, temperature=0.9)
        assert response is not None

    @pytest.mark.asyncio
    async def test_multiple_conversations(self):
        """Test managing multiple conversations."""
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
        }

        bot = await DynaBot.from_config(config)

        # Create two different contexts
        context1 = BotContext(conversation_id="conv-1", client_id="client-1")
        context2 = BotContext(conversation_id="conv-2", client_id="client-1")

        # Chat in both conversations
        await bot.chat("Message in conv 1", context1)
        await bot.chat("Message in conv 2", context2)

        # Verify both conversations are cached
        assert "conv-1" in bot._conversation_managers
        assert "conv-2" in bot._conversation_managers

    @pytest.mark.asyncio
    async def test_build_message_with_context_no_memory(self):
        """Test building message without memory or knowledge."""
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
        }

        bot = await DynaBot.from_config(config)

        # Message should be unchanged
        message = await bot._build_message_with_context("Test message")
        assert message == "Test message"

    @pytest.mark.asyncio
    async def test_conversation_persistence(self):
        """Test that conversations can be resumed."""
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
        }

        bot = await DynaBot.from_config(config)
        context = BotContext(conversation_id="conv-persist", client_id="test-client")

        # First interaction
        await bot.chat("First message", context)

        # Clear the cache to simulate a fresh bot instance
        bot._conversation_managers.clear()

        # Second interaction should resume the conversation
        response = await bot.chat("Second message", context)
        assert response is not None
