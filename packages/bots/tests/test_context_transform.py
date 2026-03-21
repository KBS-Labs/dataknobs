"""Tests for Gap 10: context_transform parameter on DynaBot."""

import re

import pytest

from dataknobs_bots.testing import BotTestHarness


def strip_system_tags(content: str) -> str:
    """Sample context transform that strips <system> tags from content."""
    return re.sub(r"</?system>", "", content)


def fence_content(content: str) -> str:
    """Sample transform that wraps content in data fences."""
    return f"[DATA_START]{content}[DATA_END]"


def _get_last_user_content(bot: object) -> str:
    """Extract the last user message content from EchoProvider call history."""
    last_call = bot.llm.get_last_call()
    messages = last_call["messages"]
    for msg in reversed(messages):
        if hasattr(msg, "role") and msg.role == "user":
            return msg.content
        if isinstance(msg, dict) and msg.get("role") == "user":
            return msg["content"]
    raise AssertionError("No user message found in last call")


@pytest.mark.asyncio
async def test_context_transform_applied_to_memory_content() -> None:
    """context_transform sanitizes memory content before prompt injection."""
    async with await BotTestHarness.create(
        bot_config={
            "llm": {"provider": "echo", "model": "echo-test"},
            "conversation_storage": {"backend": "memory"},
            "memory": {"type": "buffer", "max_messages": 20},
            "context_transform": fence_content,
        },
        main_responses=["First response.", "Second response."],
    ) as harness:
        # First turn seeds memory
        await harness.chat("Hello there")
        # Second turn — memory context should be fenced
        await harness.chat("What did I say?")

        content = _get_last_user_content(harness.bot)
        assert "[DATA_START]" in content
        assert "[DATA_END]" in content


@pytest.mark.asyncio
async def test_no_context_transform_default_behaviour() -> None:
    """Without context_transform, content passes through unchanged."""
    async with await BotTestHarness.create(
        bot_config={
            "llm": {"provider": "echo", "model": "echo-test"},
            "conversation_storage": {"backend": "memory"},
            "memory": {"type": "buffer", "max_messages": 20},
        },
        main_responses=["First.", "Second."],
    ) as harness:
        await harness.chat("Hello")
        await harness.chat("Recall")

        content = _get_last_user_content(harness.bot)
        # Memory content present but not fenced
        assert "[DATA_START]" not in content
        assert "<conversation_history>" in content


@pytest.mark.asyncio
async def test_context_transform_via_callable_in_config() -> None:
    """context_transform callable passed directly in config dict."""
    async with await BotTestHarness.create(
        bot_config={
            "llm": {"provider": "echo", "model": "echo-test"},
            "conversation_storage": {"backend": "memory"},
            "context_transform": strip_system_tags,
        },
    ) as harness:
        assert harness.bot._context_transform is strip_system_tags


@pytest.mark.asyncio
async def test_context_transform_constructor_param() -> None:
    """context_transform can be passed directly to the constructor."""
    from dataknobs_llm.conversations import DataknobsConversationStorage
    from dataknobs_llm.prompts import AsyncPromptBuilder
    from dataknobs_llm.prompts.implementations import CompositePromptLibrary

    from dataknobs_bots.bot.base import DynaBot
    from dataknobs_llm import EchoProvider

    provider = EchoProvider({"provider": "echo", "model": "echo-test"})
    storage = await DataknobsConversationStorage.create({"backend": "memory"})
    prompt_builder = AsyncPromptBuilder(CompositePromptLibrary())

    bot = DynaBot(
        llm=provider,
        prompt_builder=prompt_builder,
        conversation_storage=storage,
        context_transform=strip_system_tags,
    )
    assert bot._context_transform is strip_system_tags
