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
from typing import Any

import pytest

from dataknobs_bots.bot.context import BotContext
from dataknobs_bots.middleware import Middleware
from dataknobs_bots.testing import BotTestHarness
from dataknobs_llm.conversations.storage import get_node_by_id
from dataknobs_llm.testing import text_response, tool_call_response
from dataknobs_llm.tools.base import Tool

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

    @pytest.mark.asyncio
    async def test_over_unpin_from_same_id_turn_spares_the_pinned_turn(self):
        # Reproduce-first for the pin/unpin refcount asymmetry: a SECOND turn
        # on the same conversation id whose ``_prepare_turn`` raises BEFORE it
        # pins (here an ``on_turn_start`` that raises) still reaches the turn
        # driver's ``finally``. Pins are a global per-key refcount, so an
        # unconditional release there would decrement the pin the FIRST (still
        # in-flight) turn holds — dropping conv-x to zero and letting its live
        # conversation be evicted mid-turn. The per-turn ``pinned_conversation``
        # flag releases only the releasing turn's OWN pin, so the failing turn
        # (which never pinned) leaves the in-flight turn's pin intact.
        #
        # Without the per-turn guard this test fails twice over: the
        # ``is_pinned`` assertion trips immediately, and conv-x is then evicted
        # by conv-y's insert under the size-1 bound.
        reached_gate = asyncio.Event()
        release_gate = asyncio.Event()

        class _GateAndRaise(Middleware):
            """Gate the keep-alive turn; fail the raise-before-pin turn early.

            ``on_turn_start`` runs in ``_prepare_turn`` *before* the pin, so
            raising there models a turn that reaches the driver ``finally``
            without ever pinning. ``after_turn`` runs after the pin is taken,
            so gating there keeps the first turn in-flight and pinned.
            """

            async def on_turn_start(self, turn):
                if turn.message == "raise-before-pin":
                    raise RuntimeError("boom before pin")
                return None

            async def after_turn(self, turn):
                if turn.message == "keep-alive":
                    reached_gate.set()
                    await release_gate.wait()

        bot_config = {**_BOT_CONFIG, "max_cached_conversations": 1}
        async with await BotTestHarness.create(
            bot_config=bot_config,
            main_responses=["reply-a", "reply-y"],
            middleware=[_GateAndRaise()],
        ) as harness:
            bot = harness.bot
            ctx_x = BotContext(conversation_id="conv-x", client_id="test")

            # Turn A pins conv-x and suspends mid-turn (after_turn gate).
            task_a = asyncio.create_task(bot.chat("keep-alive", ctx_x))
            await asyncio.wait_for(reached_gate.wait(), timeout=5.0)
            assert bot._conversation_managers.is_pinned("conv-x")

            # Turn B on the SAME id fails before it can pin; its ``finally``
            # must not release the pin turn A still holds.
            with pytest.raises(RuntimeError, match="boom before pin"):
                await bot.chat("raise-before-pin", ctx_x)
            assert bot._conversation_managers.is_pinned("conv-x")  # A's pin held

            # Consequence: a new conversation inserted under the size-1 bound
            # must not evict the still-pinned, in-flight conv-x (nor co-drop
            # its checkpoints) — the bound is exceeded transiently instead.
            ctx_y = BotContext(conversation_id="conv-y", client_id="test")
            await bot.chat("normal-y", ctx_y)
            assert "conv-x" in bot._conversation_managers  # pinned -> survived
            assert "conv-x" in bot._turn_checkpoints  # checkpoints intact

            # Release conv-x; it completes its own turn intact.
            release_gate.set()
            await asyncio.wait_for(task_a, timeout=5.0)
            assert "conv-x" in bot._conversation_managers


class TestCheckpointCap:
    """``max_undo_checkpoints`` tail-caps the retained undo checkpoints.

    Only the most-recent N checkpoints are kept; the oldest are trimmed from
    the front (tracked in ``dropped``) so ``rewind_to_turn`` can still map an
    absolute turn index correctly and reject a target older than the window.
    """

    @pytest.mark.asyncio
    async def test_checkpoint_cap_tail_retains(self):
        # Cap at 3; drive 5 turns on one conversation. Only the 3 most-recent
        # checkpoints survive; the 2 oldest are trimmed from the front and
        # counted in ``dropped`` so absolute turn numbering is preserved.
        bot_config = {**_BOT_CONFIG, "max_undo_checkpoints": 3}
        async with await BotTestHarness.create(
            bot_config=bot_config,
            main_responses=[f"reply-{i}" for i in range(5)],
        ) as harness:
            bot = harness.bot
            conv_id = harness.context.conversation_id
            for _ in range(5):
                await harness.chat("hello")

            log = bot._turn_checkpoints[conv_id]
            assert len(log.entries) == 3  # tail-retained
            assert log.dropped == 2  # 2 trimmed from the front
            assert log.total == 5  # absolute turn count preserved

    @pytest.mark.asyncio
    async def test_rewind_offset_and_undo_after_cap(self):
        # With the front trimmed, rewind must map absolute turn indices through
        # the dropped offset: a retained turn lands correctly, a dropped turn
        # raises the clear "beyond the retained undo window" error (never a
        # silent wrong-node rewind), and relative undo_last_turn is unaffected.
        bot_config = {**_BOT_CONFIG, "max_undo_checkpoints": 3}
        async with await BotTestHarness.create(
            bot_config=bot_config,
            main_responses=[f"reply-{i}" for i in range(6)],
        ) as harness:
            bot = harness.bot
            ctx = harness.context
            conv_id = ctx.conversation_id
            for _ in range(5):
                await harness.chat("hello")

            log = bot._turn_checkpoints[conv_id]
            assert log.dropped == 2 and len(log.entries) == 3

            # Turn 0 and turn -1 (the start) had their checkpoints trimmed —
            # the window guard fires. It raises BEFORE running any undo, so
            # these assertions are non-mutating. (Without the guard, the stale
            # turns_to_undo would instead exhaust the retained entries and
            # raise the wrong "Nothing to undo" — hence matching the specific
            # message.)
            with pytest.raises(ValueError, match="beyond the retained undo window"):
                await bot.rewind_to_turn(ctx, 0)
            with pytest.raises(ValueError, match="beyond the retained undo window"):
                await bot.rewind_to_turn(ctx, -1)

            # An out-of-range-high turn still reports the true (absolute)
            # conversation length, unaffected by the front trim.
            with pytest.raises(ValueError, match="conversation has 5 turns"):
                await bot.rewind_to_turn(ctx, 9)

            # Relative undo pops the tail and leaves the dropped offset intact.
            await bot.undo_last_turn(ctx)
            assert len(log.entries) == 2
            assert log.dropped == 2

            # A retained turn still rewinds through the offset without error.
            result = await bot.rewind_to_turn(ctx, 2)
            assert result is not None


class _EchoTool(Tool):
    """Minimal tool: records its calls, echoes its input back.

    Drives a multi-iteration ReAct loop so in-loop history compaction has a
    body to compact.
    """

    def __init__(self) -> None:
        super().__init__(name="echo", description="Echoes the input back")
        self.calls: list[dict[str, Any]] = []

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }

    async def execute(self, **kwargs: Any) -> Any:
        kwargs.pop("_context", None)
        self.calls.append(kwargs)
        return {"echoed": kwargs.get("text", "")}


class TestCompactionUndoInteroperation:
    """In-loop history compaction must never dangle an undo checkpoint.

    A turn's undo checkpoint is the ``current_node_id`` recorded *before* that
    turn's user message is added — i.e. the prior turn's terminal node. ReAct
    in-loop compaction (``compact_history``) anchors at the *last user message*
    and only compacts the body after it (this turn's tool-loop iterations),
    retaining the entire head verbatim, and is non-destructive (dropped nodes
    stay in the tree as an abandoned branch). So a checkpoint anchored in the
    retained head can never be pruned by a later turn's compaction.

    This is a regression *guard*: it passes today and after. If a future
    compaction variant became destructive or moved its anchor past a
    checkpoint node, ``switch_to_node`` would raise "Node not found" and
    ``undo_last_turn`` here would fail — that is the guard firing.
    """

    @pytest.mark.asyncio
    async def test_undo_resolves_checkpoint_after_compacting_tool_turn(self):
        bot_config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "reasoning": {
                "strategy": "react",
                "max_iterations": 8,
                # A low absolute budget forces the loop to compact. The
                # ``window`` strategy needs no extra LLM calls, so the main
                # scripted queue is consumed only by the turn itself.
                "history_compaction": {
                    "enabled": True,
                    "history_token_budget": 30,
                    "keep_recent_iterations": 1,
                    "strategy": "window",
                },
            },
        }
        # Turn 1: a plain text answer (no tools) -> ends at a non-root node
        # that becomes turn 2's checkpoint anchor. Turn 2: a 5-iteration tool
        # loop whose body compaction trims.
        main_responses = [
            text_response("turn one done"),
            *(tool_call_response("echo", {"text": f"step {i}"}) for i in range(5)),
            text_response("turn two done"),
            text_response("branched follow-up"),  # the post-undo turn
        ]
        tool = _EchoTool()
        async with await BotTestHarness.create(
            bot_config=bot_config,
            main_responses=main_responses,
            tools=[tool],
        ) as harness:
            bot = harness.bot
            ctx = harness.context
            conv_id = ctx.conversation_id

            await harness.chat("first message")  # turn 0
            await harness.chat("do the multi-step task")  # turn 1 (compacts)

            manager = bot.get_conversation_manager(conv_id)
            history = await manager.get_history()

            # Confirm compaction actually fired on the tool turn: fewer tool
            # observations survive than the iterations driven. Pin that the
            # loop genuinely ran all 5 iterations first, so a shortfall in
            # ``tool_msgs`` is unambiguously attributable to compaction rather
            # than to an early-terminating or reshaped loop (which would let
            # ``len(tool_msgs) < 5`` pass with no compaction at all).
            assert len(tool.calls) == 5, (
                "the tool loop did not run 5 iterations -> the < 5 check "
                "below would no longer prove compaction fired"
            )
            tool_msgs = [m for m in history if m.role == "tool"]
            assert len(tool_msgs) < 5, "compaction did not fire -> the guard would be vacuous"

            # The checkpoint recorded for turn 1 is turn 0's terminal node,
            # which lives in the retained head of turn 1's compaction. It must
            # still resolve in the tree (non-destructive compaction).
            log = bot._turn_checkpoints[conv_id]
            checkpoint_node_id = log.entries[-1][0]
            # A real intermediate node (turn 0's terminal), not the trivially
            # always-resolvable root — the guard exercises a node compaction
            # could plausibly have touched.
            assert checkpoint_node_id != ""
            assert get_node_by_id(manager.state.message_tree, checkpoint_node_id) is not None, (
                "compaction dangled the undo checkpoint node"
            )

            # End to end: undo navigates to that checkpoint node without
            # raising "Node not found" -> the anchor is resolvable post-compaction.
            result = await bot.undo_last_turn(ctx)
            assert result is not None
            # The tree is intact enough to branch a fresh turn from the checkpoint.
            follow_up = await harness.chat("alternative follow-up")
            assert follow_up is not None


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
                ctx = BotContext(conversation_id=f"conv-{i}", client_id="test")
                await bot.chat("hello", ctx)

            # Nothing evicted — every conversation is retained in both caches.
            assert len(bot._conversation_managers) == 10
            assert len(bot._turn_checkpoints) == 10
