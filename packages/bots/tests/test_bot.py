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
    async def test_from_config_system_prompt_dict_with_content(self):
        """Test system prompt with dict containing content key."""
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "system_prompt": {"content": "You are a helpful assistant."},
        }

        bot = await DynaBot.from_config(config)
        assert bot.system_prompt_name is None
        assert bot.system_prompt_content == "You are a helpful assistant."

    @pytest.mark.asyncio
    async def test_from_config_system_prompt_short_string_as_template_name(self):
        """Test short string is used as template name when it exists in library."""
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "prompts": {
                "helpful_assistant": "You are a helpful assistant."
            },
            "system_prompt": "helpful_assistant",
        }

        bot = await DynaBot.from_config(config)
        # Since "helpful_assistant" exists in prompts, it should be used as template name
        assert bot.system_prompt_name == "helpful_assistant"
        assert bot.system_prompt_content is None

    @pytest.mark.asyncio
    async def test_from_config_system_prompt_string_not_in_library_as_inline(self):
        """Test string NOT in library is treated as inline content (smart detection)."""
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "system_prompt": "helpful_assistant",  # Not in prompts library
        }

        bot = await DynaBot.from_config(config)
        # Since "helpful_assistant" does NOT exist in prompts, it's inline content
        assert bot.system_prompt_name is None
        assert bot.system_prompt_content == "helpful_assistant"

    @pytest.mark.asyncio
    async def test_from_config_system_prompt_multiline_as_inline_content(self):
        """Test multi-line string is treated as inline content."""
        multiline_prompt = """You are a helpful assistant.
You should be concise and accurate.
Always be polite."""
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "system_prompt": multiline_prompt,
        }

        bot = await DynaBot.from_config(config)
        assert bot.system_prompt_name is None
        assert bot.system_prompt_content == multiline_prompt

    @pytest.mark.asyncio
    async def test_from_config_system_prompt_long_string_as_inline_content(self):
        """Test long string (>100 chars) is treated as inline content."""
        long_prompt = "You are a helpful assistant. " * 5  # 150 characters
        assert len(long_prompt) > 100  # Verify it's long enough

        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "system_prompt": long_prompt,
        }

        bot = await DynaBot.from_config(config)
        assert bot.system_prompt_name is None
        assert bot.system_prompt_content == long_prompt

    @pytest.mark.asyncio
    async def test_from_config_system_prompt_any_length_as_inline_if_not_in_library(self):
        """Test any length string is treated as inline if not in library (smart detection)."""
        # With smart detection, length doesn't matter - only library membership does
        exact_100_prompt = "x" * 100
        assert len(exact_100_prompt) == 100

        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "system_prompt": exact_100_prompt,  # Not in prompts library
        }

        bot = await DynaBot.from_config(config)
        # Since it's not in the library, it's treated as inline content
        assert bot.system_prompt_name is None
        assert bot.system_prompt_content == exact_100_prompt

    @pytest.mark.asyncio
    async def test_system_prompt_content_used_in_conversation(self):
        """Test that inline system prompt content is added to conversation."""
        system_content = "You are a helpful coding assistant."
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "system_prompt": {"content": system_content},
        }

        bot = await DynaBot.from_config(config)
        context = BotContext(conversation_id="conv-system-test", client_id="test-client")

        # Chat to trigger conversation creation
        await bot.chat("Hello", context)

        # Verify conversation was created with system prompt
        conversation_state = await bot.get_conversation("conv-system-test")
        assert conversation_state is not None

        # Get messages from the tree using find_nodes to find all system messages
        tree = conversation_state.message_tree
        system_nodes = tree.find_nodes(
            lambda node: node.data.message and node.data.message.role == "system"
        )

        # Verify at least one system message exists
        assert len(system_nodes) >= 1

        # The system message should contain our inline content
        system_messages = [node.data.message for node in system_nodes]
        assert any(m.content == system_content for m in system_messages)

    @pytest.mark.asyncio
    async def test_system_prompt_multiline_yaml_style(self):
        """Test multi-line system prompt as would appear in YAML config."""
        # This simulates how YAML multi-line content would be loaded
        yaml_style_prompt = """You are a helpful AI assistant specialized in customer support.

Key responsibilities:
- Answer questions accurately
- Be polite and professional
- Escalate complex issues

Remember to always verify customer identity before sharing sensitive information."""

        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "system_prompt": yaml_style_prompt,
        }

        bot = await DynaBot.from_config(config)
        assert bot.system_prompt_name is None
        assert bot.system_prompt_content == yaml_style_prompt

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

    @pytest.mark.asyncio
    async def test_get_conversation(self):
        """Test retrieving conversation history."""
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
        }

        bot = await DynaBot.from_config(config)
        context = BotContext(conversation_id="conv-get-test", client_id="test-client")

        # Create a conversation with some messages
        await bot.chat("Hello", context)
        await bot.chat("How are you?", context)

        # Retrieve the conversation
        conversation_state = await bot.get_conversation("conv-get-test")

        # Verify we got a conversation state
        assert conversation_state is not None
        assert conversation_state.conversation_id == "conv-get-test"
        assert conversation_state.message_tree is not None

    @pytest.mark.asyncio
    async def test_get_conversation_not_found(self):
        """Test retrieving a non-existent conversation returns None."""
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
        }

        bot = await DynaBot.from_config(config)

        # Try to get a conversation that doesn't exist
        conversation_state = await bot.get_conversation("non-existent-conv")

        # Should return None
        assert conversation_state is None

    @pytest.mark.asyncio
    async def test_clear_conversation(self):
        """Test clearing a conversation."""
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
        }

        bot = await DynaBot.from_config(config)
        context = BotContext(conversation_id="conv-clear-test", client_id="test-client")

        # Create a conversation
        await bot.chat("Hello", context)

        # Verify conversation exists in cache
        assert "conv-clear-test" in bot._conversation_managers

        # Clear the conversation
        deleted = await bot.clear_conversation("conv-clear-test")

        # Verify deletion
        assert deleted is True
        assert "conv-clear-test" not in bot._conversation_managers

        # Try to get the deleted conversation (should return None)
        conversation_state = await bot.get_conversation("conv-clear-test")
        assert conversation_state is None

    @pytest.mark.asyncio
    async def test_clear_nonexistent_conversation(self):
        """Test clearing a conversation that doesn't exist."""
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
        }

        bot = await DynaBot.from_config(config)

        # Clear a conversation that doesn't exist
        deleted = await bot.clear_conversation("non-existent-conv")

        # Should return False
        assert deleted is False

    @pytest.mark.asyncio
    async def test_clear_conversation_then_new_chat(self):
        """Test that after clearing, a new chat starts fresh."""
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
        }

        bot = await DynaBot.from_config(config)
        context = BotContext(conversation_id="conv-fresh-test", client_id="test-client")

        # Create a conversation
        await bot.chat("First message", context)

        # Clear the conversation
        await bot.clear_conversation("conv-fresh-test")

        # Start a new chat with same conversation_id
        response = await bot.chat("Fresh start", context)

        # Should succeed and create a new conversation
        assert response is not None

        # Verify new conversation was created
        assert "conv-fresh-test" in bot._conversation_managers

    @pytest.mark.asyncio
    async def test_stream_chat_basic(self):
        """Test basic streaming chat functionality."""
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
        }

        bot = await DynaBot.from_config(config)
        context = BotContext(conversation_id="conv-stream-1", client_id="test-client")

        # Stream response
        chunks = []
        async for chunk in bot.stream_chat("Hello, bot!", context):
            chunks.append(chunk)

        # Should receive multiple chunks (echo provider streams char by char)
        assert len(chunks) > 0

        # Concatenated chunks should form a non-empty response
        full_response = "".join(chunks)
        assert len(full_response) > 0

        await bot.close()

    @pytest.mark.asyncio
    async def test_stream_chat_updates_conversation_history(self):
        """Test that streaming updates conversation history."""
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
        }

        bot = await DynaBot.from_config(config)
        context = BotContext(conversation_id="conv-stream-history", client_id="test-client")

        # Stream a response
        full_response = ""
        async for chunk in bot.stream_chat("Test message", context):
            full_response += chunk

        # Verify conversation was updated with both user and assistant messages
        conversation_state = await bot.get_conversation("conv-stream-history")
        assert conversation_state is not None

        # Get all messages from the tree
        tree = conversation_state.message_tree
        all_nodes = tree.find_nodes(lambda node: node.data.message is not None)

        # Should have at least user and assistant messages
        messages = [node.data.message for node in all_nodes]
        roles = [m.role for m in messages if m]

        assert "user" in roles
        assert "assistant" in roles

        await bot.close()

    @pytest.mark.asyncio
    async def test_stream_chat_with_memory(self):
        """Test streaming chat with memory context."""
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "memory": {"type": "buffer", "max_messages": 10},
        }

        bot = await DynaBot.from_config(config)
        context = BotContext(conversation_id="conv-stream-memory", client_id="test-client")

        # Stream first message
        async for _ in bot.stream_chat("First message", context):
            pass

        # Memory should have been updated
        memory_context = await bot.memory.get_context("test")
        assert len(memory_context) >= 1

        await bot.close()

    @pytest.mark.asyncio
    async def test_stream_chat_yields_incremental_content(self):
        """Test that stream_chat yields incremental content."""
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
        }

        bot = await DynaBot.from_config(config)
        context = BotContext(conversation_id="conv-stream-chunks", client_id="test-client")

        # Collect chunks to verify streaming
        chunks = []
        async for chunk in bot.stream_chat("Hello world", context):
            chunks.append(chunk)
            # Each chunk should be a string
            assert isinstance(chunk, str)

        # Echo provider yields character by character, so we should have multiple chunks
        assert len(chunks) > 1

        await bot.close()

    @pytest.mark.asyncio
    async def test_stream_chat_multiple_messages(self):
        """Test streaming multiple messages in same conversation."""
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
        }

        bot = await DynaBot.from_config(config)
        context = BotContext(conversation_id="conv-stream-multi", client_id="test-client")

        # First streamed message
        response1_chunks = []
        async for chunk in bot.stream_chat("First", context):
            response1_chunks.append(chunk)

        # Second streamed message
        response2_chunks = []
        async for chunk in bot.stream_chat("Second", context):
            response2_chunks.append(chunk)

        # Both should have produced responses
        assert len(response1_chunks) > 0
        assert len(response2_chunks) > 0

        # Conversation should have multiple exchanges
        conversation_state = await bot.get_conversation("conv-stream-multi")
        tree = conversation_state.message_tree
        user_nodes = tree.find_nodes(
            lambda node: node.data.message and node.data.message.role == "user"
        )
        assert len(user_nodes) >= 2

        await bot.close()

    @pytest.mark.asyncio
    async def test_stream_chat_temperature_override(self):
        """Test streaming with temperature override."""
        config = {
            "llm": {"provider": "echo", "model": "test", "temperature": 0.5},
            "conversation_storage": {"backend": "memory"},
        }

        bot = await DynaBot.from_config(config)
        context = BotContext(conversation_id="conv-stream-temp", client_id="test-client")

        # Stream with temperature override
        chunks = []
        async for chunk in bot.stream_chat("Hello", context, temperature=0.9):
            chunks.append(chunk)

        assert len(chunks) > 0

        await bot.close()

    @pytest.mark.asyncio
    async def test_stream_chat_with_system_prompt(self):
        """Test streaming with system prompt configured."""
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "system_prompt": {"content": "You are a helpful assistant."},
        }

        bot = await DynaBot.from_config(config)
        context = BotContext(conversation_id="conv-stream-sysprompt", client_id="test-client")

        # Stream response
        chunks = []
        async for chunk in bot.stream_chat("Hello", context):
            chunks.append(chunk)

        assert len(chunks) > 0

        # Verify conversation has system prompt
        conversation_state = await bot.get_conversation("conv-stream-sysprompt")
        tree = conversation_state.message_tree
        system_nodes = tree.find_nodes(
            lambda node: node.data.message and node.data.message.role == "system"
        )
        assert len(system_nodes) >= 1

        await bot.close()

    @pytest.mark.asyncio
    async def test_stream_chat_post_stream_middleware_hook(self):
        """Test that post_stream middleware hook is called after streaming."""
        from dataknobs_bots.middleware.base import Middleware
        from typing import Any

        # Create a test middleware that tracks post_stream calls
        class TestMiddleware(Middleware):
            def __init__(self):
                self.post_stream_calls = []
                self.before_message_calls = []

            async def before_message(self, message: str, context: BotContext) -> None:
                self.before_message_calls.append(message)

            async def after_message(
                self, response: str, context: BotContext, **kwargs: Any
            ) -> None:
                pass  # Not called for streaming

            async def post_stream(
                self, message: str, response: str, context: BotContext
            ) -> None:
                self.post_stream_calls.append({
                    "message": message,
                    "response": response,
                    "conversation_id": context.conversation_id,
                })

            async def on_error(
                self, error: Exception, message: str, context: BotContext
            ) -> None:
                pass

        # Create bot with test middleware
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
        }
        bot = await DynaBot.from_config(config)

        # Add test middleware
        test_mw = TestMiddleware()
        bot.middleware.append(test_mw)

        context = BotContext(conversation_id="conv-post-stream", client_id="test-client")

        # Stream a message
        chunks = []
        async for chunk in bot.stream_chat("Hello middleware!", context):
            chunks.append(chunk)

        full_response = "".join(chunks)

        # Verify before_message was called
        assert len(test_mw.before_message_calls) == 1
        assert test_mw.before_message_calls[0] == "Hello middleware!"

        # Verify post_stream was called with correct arguments
        assert len(test_mw.post_stream_calls) == 1
        call = test_mw.post_stream_calls[0]
        assert call["message"] == "Hello middleware!"
        assert call["response"] == full_response
        assert call["conversation_id"] == "conv-post-stream"

        await bot.close()
