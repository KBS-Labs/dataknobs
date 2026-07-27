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


class TestDropConversationCacheIdempotency:
    """The teardown choke point must absorb the asymmetric/repeat cases.

    ``_drop_conversation_cache`` pops both dicts unconditionally
    (``pop(..., None)``). That is load-bearing because the two caches
    populate at *different* sites: the manager is cached in
    ``_get_or_create_conversation`` on every turn, but the checkpoint is
    appended in ``_prepare_turn`` only *after* a greet early-return — so a
    greet-only conversation has a cached manager and **no** checkpoint entry.
    Clearing an unknown or already-cleared id must likewise be a clean no-op.
    """

    @pytest.mark.asyncio
    async def test_clear_greet_only_conversation_reclaims_cleanly(self):
        async with await BotTestHarness.create(
            bot_config=_BOT_CONFIG,
            main_responses=["Hello there"],
        ) as harness:
            conv_id = harness.context.conversation_id

            # A greet caches the manager but returns before the checkpoint
            # append — the asymmetric case (manager present, no checkpoint).
            await harness.greet()
            assert conv_id in harness.bot._conversation_managers
            assert conv_id not in harness.bot._turn_checkpoints

            # Clear must reclaim the manager without tripping on the absent
            # checkpoint entry (the unconditional ``pop(None)`` absorbs it).
            # (The storage-delete return value is orthogonal — a greet-only
            # conversation is never persisted — so we assert only the cache
            # reclaim, which is the teardown contract under test.)
            await harness.bot.clear_conversation(conv_id)
            assert conv_id not in harness.bot._conversation_managers
            assert conv_id not in harness.bot._turn_checkpoints

    @pytest.mark.asyncio
    async def test_clear_unknown_and_repeated_ids_are_noops(self):
        async with await BotTestHarness.create(
            bot_config=_BOT_CONFIG,
            main_responses=["Hi there"],
        ) as harness:
            # Clearing a conversation that was never seen must not raise and
            # must leave both caches empty of it.
            unseen = "never-seen-conversation"
            await harness.bot.clear_conversation(unseen)
            assert unseen not in harness.bot._conversation_managers
            assert unseen not in harness.bot._turn_checkpoints

            # A second clear on an already-cleared id must also be a no-op.
            conv_id = harness.context.conversation_id
            await harness.chat("Hello")
            await harness.bot.clear_conversation(conv_id)
            await harness.bot.clear_conversation(conv_id)  # must not raise
            assert conv_id not in harness.bot._conversation_managers
            assert conv_id not in harness.bot._turn_checkpoints
