"""Tests for conversation-level metadata enrichment in DynaBot."""

import pytest

from dataknobs_bots import BotContext, DynaBot


class TestConversationMetadata:
    """Verify that DynaBot populates conversation metadata with model, provider, tools."""

    @pytest.mark.asyncio
    async def test_conversation_metadata_includes_model_provider_tools(self):
        """New conversations include model, provider, and tools in metadata."""
        config = {
            "llm": {"provider": "echo", "model": "test-model"},
            "conversation_storage": {"backend": "memory"},
        }
        bot = await DynaBot.from_config(config)
        context = BotContext(
            conversation_id="conv-meta-test",
            client_id="test-client",
        )

        await bot.chat("Hello", context)

        conv = await bot.get_conversation("conv-meta-test")
        assert conv is not None
        metadata = conv.metadata
        assert metadata["model"] == "test-model"
        assert metadata["provider"] == "echo"
        assert metadata["tools"] == []

    @pytest.mark.asyncio
    async def test_conversation_metadata_includes_registered_tools(self):
        """Conversations include tool names when tools are registered on the bot."""
        config = {
            "llm": {"provider": "echo", "model": "test-model"},
            "conversation_storage": {"backend": "memory"},
        }
        bot = await DynaBot.from_config(config)

        # Register a simple tool
        from dataknobs_llm.tools.base import Tool

        class DummyTool(Tool):
            def __init__(self) -> None:
                super().__init__(
                    name="dummy_tool",
                    description="A dummy tool for testing",
                )

            @property
            def schema(self) -> dict:
                return {"type": "object", "properties": {}}

            async def execute(self, **kwargs) -> str:
                return "ok"

        bot.tool_registry.register_tool(DummyTool())

        context = BotContext(
            conversation_id="conv-tools-test",
            client_id="test-client",
        )
        await bot.chat("Hello", context)

        conv = await bot.get_conversation("conv-tools-test")
        assert conv is not None
        assert "dummy_tool" in conv.metadata["tools"]

    @pytest.mark.asyncio
    async def test_session_metadata_can_override_auto_fields(self):
        """Session metadata (bot_id etc.) spread last, overriding auto fields."""
        config = {
            "llm": {"provider": "echo", "model": "test-model"},
            "conversation_storage": {"backend": "memory"},
        }
        bot = await DynaBot.from_config(config)
        context = BotContext(
            conversation_id="conv-override-test",
            client_id="test-client",
            session_metadata={"bot_id": "my-bot", "model": "custom-override"},
        )

        await bot.chat("Hello", context)

        conv = await bot.get_conversation("conv-override-test")
        assert conv is not None
        # session_metadata spreads last, so it overrides auto-populated model
        assert conv.metadata["model"] == "custom-override"
        assert conv.metadata["bot_id"] == "my-bot"
