"""Tests for DynaBot.undo_last_turn() and rewind_to_turn().

Phase 2b of the conversation undo plan: per-node FSM state, checkpoint
recording, and coordinated undo across tree, memory, wizard state, and banks.
"""

from __future__ import annotations

import pytest

from dataknobs_bots import BotContext, DynaBot
from dataknobs_bots.bot.base import UndoResult, _node_depth


# =====================================================================
# Helpers
# =====================================================================


async def _make_bot(*, with_memory: bool = True) -> DynaBot:
    """Create a DynaBot with EchoProvider + BufferMemory for testing."""
    config: dict = {
        "llm": {"provider": "echo", "model": "test"},
        "conversation_storage": {"backend": "memory"},
    }
    if with_memory:
        config["memory"] = {"type": "buffer", "max_messages": 50}
    return await DynaBot.from_config(config)


def _ctx(conv_id: str = "conv-undo-1") -> BotContext:
    return BotContext(conversation_id=conv_id, client_id="test")


# =====================================================================
# _node_depth helper
# =====================================================================


class TestNodeDepth:
    def test_root(self):
        assert _node_depth("") == 0

    def test_depth_one(self):
        assert _node_depth("0") == 1

    def test_depth_three(self):
        assert _node_depth("0.0.0") == 3


# =====================================================================
# Simple chat undo (no wizard)
# =====================================================================


class TestSimpleChatUndo:
    """Undo in a non-wizard conversation."""

    @pytest.mark.asyncio
    async def test_undo_single_turn(self):
        bot = await _make_bot()
        ctx = _ctx()

        await bot.chat("Hello", ctx)
        result = await bot.undo_last_turn(ctx)

        assert isinstance(result, UndoResult)
        assert result.undone_user_message == "Hello"
        assert result.branching is True

    @pytest.mark.asyncio
    async def test_undo_restores_memory(self):
        bot = await _make_bot()
        ctx = _ctx()

        await bot.chat("First", ctx)
        await bot.chat("Second", ctx)

        # Memory should have 4 messages (2 user + 2 assistant)
        mem_before = await bot.memory.get_context("test")
        assert len(mem_before) == 4

        await bot.undo_last_turn(ctx)

        # Memory should now have 2 messages (1 user + 1 assistant)
        mem_after = await bot.memory.get_context("test")
        assert len(mem_after) == 2

    @pytest.mark.asyncio
    async def test_undo_creates_sibling_branch(self):
        """After undo, the next chat() creates a new branch."""
        bot = await _make_bot()
        ctx = _ctx()

        await bot.chat("Hello", ctx)
        await bot.undo_last_turn(ctx)

        # Next message should work and create a sibling branch
        response = await bot.chat("Hello again", ctx)
        assert response is not None

    @pytest.mark.asyncio
    async def test_undo_nothing_raises(self):
        bot = await _make_bot()
        ctx = _ctx()

        # Start a conversation, undo the only turn, then try undo again
        await bot.chat("Hello", ctx)
        await bot.undo_last_turn(ctx)

        with pytest.raises(ValueError, match="Nothing to undo"):
            await bot.undo_last_turn(ctx)

    @pytest.mark.asyncio
    async def test_undo_no_conversation_raises(self):
        bot = await _make_bot()
        ctx = _ctx("nonexistent")

        with pytest.raises(ValueError, match="No active conversation"):
            await bot.undo_last_turn(ctx)

    @pytest.mark.asyncio
    async def test_remaining_turns_count(self):
        bot = await _make_bot()
        ctx = _ctx()

        await bot.chat("First", ctx)
        await bot.chat("Second", ctx)
        await bot.chat("Third", ctx)

        result = await bot.undo_last_turn(ctx)
        assert result.remaining_turns == 2  # First and Second remain

    @pytest.mark.asyncio
    async def test_undo_without_memory(self):
        """Undo works even when no memory is configured."""
        bot = await _make_bot(with_memory=False)
        ctx = _ctx()

        await bot.chat("Hello", ctx)
        result = await bot.undo_last_turn(ctx)
        assert result.undone_user_message == "Hello"


# =====================================================================
# Multi-turn rewind
# =====================================================================


class TestRewindToTurn:
    """rewind_to_turn() for multi-turn undo."""

    @pytest.mark.asyncio
    async def test_rewind_to_first_turn(self):
        bot = await _make_bot()
        ctx = _ctx()

        await bot.chat("First", ctx)
        await bot.chat("Second", ctx)
        await bot.chat("Third", ctx)

        result = await bot.rewind_to_turn(ctx, 0)
        assert result.remaining_turns == 1  # Only "First" remains

        mem = await bot.memory.get_context("test")
        assert len(mem) == 2  # 1 user + 1 assistant

    @pytest.mark.asyncio
    async def test_rewind_to_start(self):
        bot = await _make_bot()
        ctx = _ctx()

        await bot.chat("First", ctx)
        await bot.chat("Second", ctx)

        result = await bot.rewind_to_turn(ctx, -1)
        # All user turns undone — memory should be empty
        mem = await bot.memory.get_context("test")
        assert len(mem) == 0
        # The undone message should be "First" (the last undo in the sequence)
        assert result.undone_user_message == "First"

    @pytest.mark.asyncio
    async def test_rewind_invalid_turn_raises(self):
        bot = await _make_bot()
        ctx = _ctx()

        await bot.chat("First", ctx)

        with pytest.raises(ValueError, match="Invalid turn"):
            await bot.rewind_to_turn(ctx, 5)


# =====================================================================
# Checkpoint recording
# =====================================================================


class TestCheckpointRecording:
    """Verify checkpoints are recorded correctly per conversation."""

    @pytest.mark.asyncio
    async def test_checkpoints_accumulate(self):
        bot = await _make_bot()
        ctx = _ctx()

        await bot.chat("First", ctx)
        await bot.chat("Second", ctx)
        await bot.chat("Third", ctx)

        checkpoints = bot._turn_checkpoints.get(ctx.conversation_id, [])
        assert len(checkpoints) == 3

    @pytest.mark.asyncio
    async def test_checkpoints_per_conversation(self):
        """Each conversation tracks its own checkpoints."""
        bot = await _make_bot()
        ctx1 = _ctx("conv-1")
        ctx2 = _ctx("conv-2")

        await bot.chat("Hello", ctx1)
        await bot.chat("Hello", ctx2)
        await bot.chat("Again", ctx1)

        assert len(bot._turn_checkpoints["conv-1"]) == 2
        assert len(bot._turn_checkpoints["conv-2"]) == 1

    @pytest.mark.asyncio
    async def test_undo_pops_checkpoint(self):
        bot = await _make_bot()
        ctx = _ctx()

        await bot.chat("First", ctx)
        await bot.chat("Second", ctx)
        assert len(bot._turn_checkpoints[ctx.conversation_id]) == 2

        await bot.undo_last_turn(ctx)
        assert len(bot._turn_checkpoints[ctx.conversation_id]) == 1


# =====================================================================
# Non-interference tests
# =====================================================================


class TestNonInterference:
    """Checkpoint recording must not affect normal conversation flow."""

    @pytest.mark.asyncio
    async def test_normal_chat_unaffected(self):
        """A normal multi-turn chat works identically with checkpointing."""
        bot = await _make_bot()
        ctx = _ctx()

        r1 = await bot.chat("Hello", ctx)
        r2 = await bot.chat("How are you?", ctx)
        r3 = await bot.chat("Tell me a joke", ctx)

        assert all(isinstance(r, str) for r in [r1, r2, r3])
        assert all(len(r) > 0 for r in [r1, r2, r3])

        mem = await bot.memory.get_context("test")
        assert len(mem) == 6  # 3 user + 3 assistant

    @pytest.mark.asyncio
    async def test_undo_then_continue(self):
        """After undo, conversation continues normally."""
        bot = await _make_bot()
        ctx = _ctx()

        await bot.chat("First", ctx)
        await bot.chat("Second", ctx)
        await bot.undo_last_turn(ctx)

        # Continue from after "First"
        r = await bot.chat("Alternative second", ctx)
        assert isinstance(r, str)

        mem = await bot.memory.get_context("test")
        # Should have: First(user), First(assistant), Alternative(user), Alternative(assistant)
        assert len(mem) == 4

    @pytest.mark.asyncio
    async def test_multiple_undo_redo_cycles(self):
        """Multiple undo/redo cycles work correctly."""
        bot = await _make_bot()
        ctx = _ctx()

        await bot.chat("First", ctx)
        await bot.chat("Second", ctx)

        # Undo second
        await bot.undo_last_turn(ctx)
        # Redo with different message
        await bot.chat("Second v2", ctx)
        # Undo again
        await bot.undo_last_turn(ctx)
        # Redo again
        r = await bot.chat("Second v3", ctx)
        assert isinstance(r, str)
