"""Lifecycle tests for DynaBot's per-conversation in-memory caches.

DynaBot keeps two per-conversation structures in memory: the cached
``ConversationManager`` (``_conversation_managers``) and the per-turn undo
checkpoints (``_turn_checkpoints``). They share a single lifetime — a
conversation that is cleared should reclaim *both*. Historically only the
manager was dropped by ``clear_conversation`` while the checkpoints were
left behind (an unbounded process-lifetime leak).

These tests assert on the cache internals directly (as the public leak
contract is *about* those structures), which is the sanctioned exception
for a test that verifies internal resource-lifecycle behavior.
"""

from __future__ import annotations

import pytest

from dataknobs_bots.testing import BotTestHarness

_BOT_CONFIG = {
    "llm": {"provider": "echo", "model": "test"},
    "conversation_storage": {"backend": "memory"},
    "reasoning": {"strategy": "simple"},
}


class TestClearConversationPrunesBothCaches:
    """``clear_conversation`` must reclaim the manager AND its checkpoints."""

    @pytest.mark.asyncio
    async def test_clear_conversation_drops_manager_and_checkpoints(self):
        async with await BotTestHarness.create(
            bot_config=_BOT_CONFIG,
            main_responses=["Hi there", "Second reply"],
        ) as harness:
            conv_id = harness.context.conversation_id

            # Drive a turn so both caches hold an entry for this conversation.
            await harness.chat("Hello")
            assert conv_id in harness.bot._conversation_managers
            assert conv_id in harness.bot._turn_checkpoints

            # Clear must reclaim BOTH structures, not just the manager.
            deleted = await harness.bot.clear_conversation(conv_id)
            assert deleted is True

            assert conv_id not in harness.bot._conversation_managers
            assert conv_id not in harness.bot._turn_checkpoints
