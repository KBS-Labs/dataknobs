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

import asyncio

import pytest

from dataknobs_bots.bot.context import BotContext
from dataknobs_bots.middleware import Middleware
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


class TestBoundedManagerCache:
    """``max_cached_conversations`` bounds the manager cache (access-LRU).

    Eviction is access-ordered and co-drops the evicted conversation's undo
    checkpoints through the single teardown choke point, and the in-flight
    conversation of an active turn is never evicted (it is pinned for the
    duration of its turn).
    """

    @pytest.mark.asyncio
    async def test_lru_eviction_co_drops_manager_and_checkpoints(self):
        # Bound the cache at 2; drive turns on three distinct conversations.
        # The least-recently-used conversation (the first) must be evicted
        # when the third is inserted, taking BOTH its cached manager and its
        # undo checkpoints with it; the two most-recent survive.
        bot_config = {**_BOT_CONFIG, "max_cached_conversations": 2}
        async with await BotTestHarness.create(
            bot_config=bot_config,
            main_responses=["reply-a", "reply-b", "reply-c"],
        ) as harness:
            bot = harness.bot
            for conv_id in ("conv-a", "conv-b", "conv-c"):
                ctx = BotContext(conversation_id=conv_id, client_id="test")
                await bot.chat("hello", ctx)

            # conv-a is the LRU entry -> evicted when conv-c was inserted.
            assert "conv-a" not in bot._conversation_managers
            assert "conv-a" not in bot._turn_checkpoints  # co-dropped

            # The two most-recent survive, in both structures.
            assert "conv-b" in bot._conversation_managers
            assert "conv-c" in bot._conversation_managers
            assert "conv-b" in bot._turn_checkpoints
            assert "conv-c" in bot._turn_checkpoints
            assert len(bot._conversation_managers) == 2

    @pytest.mark.asyncio
    async def test_in_flight_conversation_is_not_evicted(self):
        # Reproduce-first: without the in-flight pin, inserting conv-b into a
        # size-1 cache while conv-a's turn is still running would evict conv-a
        # (its LRU victim) mid-turn. The pin (taken in
        # _get_or_create_conversation, released in
        # _call_finally_turn_middleware) protects it, so the bound is
        # transiently exceeded rather than evicting the live conversation.
        reached_gate = asyncio.Event()
        release_gate = asyncio.Event()

        class _AfterTurnGate(Middleware):
            """Suspends conv-a inside its (pinned) turn until released.

            ``after_turn`` fires from ``_finalize_turn`` — after the pin is
            taken in ``_prepare_turn`` and before it is released in the
            turn driver's ``finally`` — so blocking here keeps conv-a
            in-flight (and pinned) while conv-b's turn runs to completion.
            """

            async def after_turn(self, turn):
                if turn.context.conversation_id == "conv-a":
                    reached_gate.set()
                    await release_gate.wait()

        bot_config = {**_BOT_CONFIG, "max_cached_conversations": 1}
        async with await BotTestHarness.create(
            bot_config=bot_config,
            main_responses=["reply-a", "reply-b"],
            middleware=[_AfterTurnGate()],
        ) as harness:
            bot = harness.bot
            ctx_a = BotContext(conversation_id="conv-a", client_id="test")
            ctx_b = BotContext(conversation_id="conv-b", client_id="test")

            task_a = asyncio.create_task(bot.chat("hello a", ctx_a))
            # Wait until conv-a is pinned and suspended inside its turn.
            await asyncio.wait_for(reached_gate.wait(), timeout=5.0)

            # conv-b's turn inserts into the size-1 cache. conv-a is pinned
            # (in-flight) so it must survive; the cache exceeds its bound
            # transiently rather than evicting the live conversation.
            await bot.chat("hello b", ctx_b)
            assert "conv-a" in bot._conversation_managers  # not evicted
            assert "conv-b" in bot._conversation_managers

            # Release conv-a; it completes its own turn intact.
            release_gate.set()
            await asyncio.wait_for(task_a, timeout=5.0)
            assert "conv-a" in bot._conversation_managers


class TestDefaultUnboundedNoRegression:
    """With neither bound set, both caches grow unbounded exactly as before."""

    @pytest.mark.asyncio
    async def test_no_bound_never_evicts(self):
        async with await BotTestHarness.create(
            bot_config=_BOT_CONFIG,
        ) as harness:
            bot = harness.bot
            # Default: the cache is unbounded (opt-in bounding only).
            assert bot._conversation_managers.max_size is None

            for i in range(10):
                ctx = BotContext(
                    conversation_id=f"conv-{i}", client_id="test"
                )
                await bot.chat("hello", ctx)

            # Nothing evicted — every conversation is retained in both caches.
            assert len(bot._conversation_managers) == 10
            assert len(bot._turn_checkpoints) == 10
