"""Tests for ReasoningStrategy.stream_generate() contract.

Validates:
- Base default wraps generate() as a single yield
- SimpleReasoning streams via manager.stream_complete()
- WizardReasoning uses default single-chunk behavior
- DynaBot.stream_chat() delegates to stream_generate()
"""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_bots.reasoning.simple import SimpleReasoning
from dataknobs_bots.reasoning.wizard import WizardReasoning
from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader
from dataknobs_llm import LLMStreamResponse
from dataknobs_llm.conversations import ConversationManager
from dataknobs_llm.llm.providers.echo import EchoProvider


class TestSimpleReasoningStreamGenerate:
    """SimpleReasoning.stream_generate() yields streaming chunks."""

    @pytest.mark.asyncio
    async def test_stream_generate_yields_chunks(
        self,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """stream_generate() yields LLMStreamResponse chunks from manager."""
        manager, provider = conversation_manager_pair
        # EchoProvider's stream_complete yields chunks
        provider.set_responses(["Hello streaming world"])

        # Add a user message so the conversation has content
        await manager.add_message("user", "Hi")

        strategy = SimpleReasoning()
        chunks: list[Any] = []
        async for chunk in strategy.stream_generate(manager, llm=None):
            chunks.append(chunk)

        assert len(chunks) > 0
        # All chunks should be LLMStreamResponse
        assert all(isinstance(c, LLMStreamResponse) for c in chunks)
        # Reconstruct full text
        full_text = "".join(c.delta for c in chunks)
        assert "Hello streaming world" in full_text

    @pytest.mark.asyncio
    async def test_stream_generate_vs_generate_same_content(
        self,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """stream_generate() produces the same content as generate()."""
        manager, provider = conversation_manager_pair
        provider.set_responses([
            "Response for streaming",
            "Response for streaming",  # Second call for generate()
        ])

        await manager.add_message("user", "Hi")

        strategy = SimpleReasoning()

        # Collect streamed content
        chunks: list[str] = []
        async for chunk in strategy.stream_generate(manager, llm=None):
            chunks.append(chunk.delta)
        streamed = "".join(chunks)

        # Get non-streamed content (need fresh conversation for fair comparison)
        response = await strategy.generate(manager, llm=None)
        generated = response.content

        assert streamed == generated


class TestWizardReasoningStreamGenerate:
    """WizardReasoning uses default single-chunk stream_generate()."""

    @pytest.mark.asyncio
    async def test_wizard_stream_generate_yields_single_response(
        self,
        conversation_manager: ConversationManager,
    ) -> None:
        """Wizard stream_generate() yields a single complete response."""
        config: dict[str, Any] = {
            "name": "stream-wizard",
            "version": "1.0",
            "stages": [
                {
                    "name": "start",
                    "is_start": True,
                    "prompt": "Ask the user a question",
                    "response_template": "Welcome to the wizard!",
                    "transitions": [{"target": "done"}],
                },
                {
                    "name": "done",
                    "is_end": True,
                    "prompt": "Done",
                },
            ],
        }
        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(config)
        reasoning = WizardReasoning(wizard_fsm=fsm, strict_validation=False)

        # Greet first to initialize state
        await reasoning.greet(conversation_manager, llm=None)

        # Add a user message for generate
        await conversation_manager.add_message("user", "hello")

        results: list[Any] = []
        async for item in reasoning.stream_generate(
            conversation_manager, llm=None, tools=[]
        ):
            results.append(item)

        # Wizard yields a single complete response (not stream chunks)
        assert len(results) == 1
        assert hasattr(results[0], "content")


class TestDynaBotStreamChatIntegration:
    """DynaBot.stream_chat() delegates to stream_generate()."""

    @pytest.mark.asyncio
    async def test_stream_chat_with_simple_strategy(self) -> None:
        """stream_chat() with SimpleReasoning yields LLMStreamResponse chunks."""
        from dataknobs_bots.bot.base import DynaBot
        from dataknobs_bots.bot.context import BotContext

        config: dict[str, Any] = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "reasoning": {"strategy": "simple"},
        }
        bot = await DynaBot.from_config(config)
        try:
            context = BotContext(
                conversation_id="test-stream-1",
                client_id="test",
            )

            chunks: list[LLMStreamResponse] = []
            async for chunk in bot.stream_chat("Hello", context):
                chunks.append(chunk)

            assert len(chunks) > 0
            assert all(isinstance(c, LLMStreamResponse) for c in chunks)
            full_text = "".join(c.delta for c in chunks)
            assert len(full_text) > 0
        finally:
            await bot.close()

    @pytest.mark.asyncio
    async def test_stream_chat_with_wizard_strategy(self) -> None:
        """stream_chat() with WizardReasoning wraps response as stream chunk."""
        from dataknobs_bots.bot.base import DynaBot
        from dataknobs_bots.bot.context import BotContext

        config: dict[str, Any] = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "reasoning": {
                "strategy": "wizard",
                "wizard_config": {
                    "name": "stream-test",
                    "stages": [
                        {
                            "name": "start",
                            "is_start": True,
                            "prompt": "Ask user something",
                            "response_template": "Welcome!",
                            "transitions": [{"target": "done"}],
                        },
                        {
                            "name": "done",
                            "is_end": True,
                            "prompt": "Done",
                        },
                    ],
                },
            },
        }
        bot = await DynaBot.from_config(config)
        try:
            context = BotContext(
                conversation_id="test-stream-2",
                client_id="test",
            )

            chunks: list[LLMStreamResponse] = []
            async for chunk in bot.stream_chat("Hello", context):
                chunks.append(chunk)

            # Wizard produces a single chunk (wrapped complete response)
            assert len(chunks) >= 1
            assert all(isinstance(c, LLMStreamResponse) for c in chunks)
            full_text = "".join(c.delta for c in chunks)
            assert len(full_text) > 0
        finally:
            await bot.close()

    @pytest.mark.asyncio
    async def test_stream_chat_without_strategy(self) -> None:
        """stream_chat() without reasoning strategy streams directly."""
        from dataknobs_bots.bot.base import DynaBot
        from dataknobs_bots.bot.context import BotContext

        config: dict[str, Any] = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
        }
        bot = await DynaBot.from_config(config)
        try:
            context = BotContext(
                conversation_id="test-stream-3",
                client_id="test",
            )

            chunks: list[LLMStreamResponse] = []
            async for chunk in bot.stream_chat("Hello", context):
                chunks.append(chunk)

            assert len(chunks) > 0
            assert all(isinstance(c, LLMStreamResponse) for c in chunks)
        finally:
            await bot.close()
