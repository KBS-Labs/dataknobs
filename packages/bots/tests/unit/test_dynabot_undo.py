"""Tests for DynaBot.undo_last_turn() and rewind_to_turn().

Conversation undo: per-node FSM state, checkpoint
recording, and coordinated undo across tree, memory, wizard state, and banks.
"""

from __future__ import annotations

from itertools import pairwise
from typing import Any

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


def _role_of(message: Any) -> str:
    """Role of a message dict or object (memory/tree items use either shape)."""
    if isinstance(message, dict):
        return message.get("role", "")
    return getattr(message, "role", "")


def _user_message_count(manager: Any) -> int:
    """Number of user messages on the manager's active tree path (LLM-visible)."""
    return sum(1 for m in manager.messages if _role_of(m) == "user")


def _no_consecutive_user_messages(manager: Any) -> bool:
    """True if no two adjacent tree-path messages are both user-role.

    Two consecutive user messages is the shape a strict provider (Anthropic)
    rejects with a 400 — the downstream symptom of the phantom leading message.
    """
    roles = [_role_of(m) for m in manager.messages]
    return not any(
        a == "user" and b == "user" for a, b in pairwise(roles)
    )


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
        # Undoing the only turn (turn 0) resets the conversation to empty and
        # discards the turn-0 branch (there is nothing before it to branch
        # from), so this undo is non-branching. Later-turn undo still branches
        # (see TestLaterTurnUndoUnchanged).
        assert result.branching is False

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

        # The tree-path channel (what the LLM sees) must empty in lock-step
        # with memory — no phantom leading user message survives the
        # rewind-to-start. (Before the fix this retained "First" while memory
        # rolled back, so these three assertions FAILED at the turn-0 boundary.)
        manager = bot._conversation_managers.get(ctx.conversation_id)
        assert _user_message_count(manager) == 0
        # No off-by-one: an emptied conversation reports zero remaining turns
        # (the retained phantom used to make this report 1).
        assert result.remaining_turns == 0
        # The turn-0 branch is discarded, so this final undo is non-branching.
        assert result.branching is False

    @pytest.mark.asyncio
    async def test_rewind_invalid_turn_raises(self):
        bot = await _make_bot()
        ctx = _ctx()

        await bot.chat("First", ctx)

        with pytest.raises(ValueError, match="Invalid turn"):
            await bot.rewind_to_turn(ctx, 5)


class TestUndoToStartClearsTreePath:
    """Undo/rewind back through the first turn empties the tree-path channel.

    Regression guard for the phantom-leading-message defect. With **no system
    prompt** the first user message *becomes* the conversation-tree root, so a
    turn-0 checkpoint anchored on the (then-empty) tree used to be reoccupied
    by that message: undo-to-start switched back onto it and left a stale
    leading user message in ``manager.messages`` (the LLM-visible path) while
    memory rolled back correctly. The next turn then sent two consecutive user
    messages (Anthropic 400). The fix anchors turn-0 on a ``None`` sentinel and
    resets the manager to its empty pre-message state.

    All tests here use the no-system-prompt ``_make_bot`` — the bug's exact
    precondition. ``TestSystemPromptUndoToStart`` guards that the seeded-system
    case (which never took the sentinel path) is unchanged.
    """

    @pytest.mark.asyncio
    async def test_undo_only_turn_clears_both_channels(self):
        # Undoing the sole turn empties BOTH the tree path and memory, resets
        # the manager, and reports zero remaining turns (no off-by-one).
        bot = await _make_bot()
        ctx = _ctx()

        await bot.chat("First", ctx)
        result = await bot.undo_last_turn(ctx)

        manager = bot._conversation_managers.get(ctx.conversation_id)
        assert manager.messages == []          # tree path emptied
        assert manager.state is None           # manager reset to pre-message
        assert len(await bot.memory.get_context("test")) == 0  # memory emptied
        assert result.remaining_turns == 0     # no off-by-one
        assert result.branching is False       # turn-0 branch discarded

    @pytest.mark.asyncio
    async def test_no_phantom_on_next_chat_after_rewind_to_start(self):
        # The defining symptom: a fresh chat after rewind-to-start must not
        # carry the undone first user message into the new branch.
        bot = await _make_bot()
        ctx = _ctx()

        await bot.chat("First", ctx)
        await bot.chat("Second", ctx)
        await bot.rewind_to_turn(ctx, -1)

        await bot.chat("Fresh", ctx)

        manager = bot._conversation_managers.get(ctx.conversation_id)
        user_msgs = [
            m["content"] for m in manager.messages if _role_of(m) == "user"
        ]
        # Only the fresh turn's user message — no phantom "First", and no
        # consecutive-user shape a 400-strict provider would reject.
        assert user_msgs == ["Fresh"]
        assert _no_consecutive_user_messages(manager)

    @pytest.mark.asyncio
    async def test_memory_tree_user_count_invariant_after_each_undo(self):
        # Memory and the tree path agree on user-message count after every
        # undo — asserted generally, through the turn-0 boundary. The divergence
        # at turn 0 *is* the bug.
        bot = await _make_bot()
        ctx = _ctx()

        await bot.chat("First", ctx)
        await bot.chat("Second", ctx)
        await bot.chat("Third", ctx)

        manager = bot._conversation_managers.get(ctx.conversation_id)
        for _ in range(3):
            await bot.undo_last_turn(ctx)
            mem = await bot.memory.get_context("test")
            mem_users = sum(1 for m in mem if _role_of(m) == "user")
            assert mem_users == _user_message_count(manager)
        # Fully unwound: both channels empty.
        assert _user_message_count(manager) == 0

    @pytest.mark.asyncio
    async def test_undo_only_turn_without_memory(self):
        # The reset path holds with no memory configured (mirrors the existing
        # test_undo_without_memory for the turn-0 boundary).
        bot = await _make_bot(with_memory=False)
        ctx = _ctx()

        await bot.chat("Hello", ctx)
        result = await bot.undo_last_turn(ctx)

        manager = bot._conversation_managers.get(ctx.conversation_id)
        assert manager.messages == []
        assert result.remaining_turns == 0
        assert result.branching is False


class TestLaterTurnUndoUnchanged:
    """Undo of a non-first turn keeps the pre-fix behavior byte-for-byte.

    Only the turn-0 anchor is a ``None`` sentinel; every later checkpoint is a
    real node id and takes the unchanged ``switch_to_node`` path (sibling branch
    preserved, ``branching=True``).
    """

    @pytest.mark.asyncio
    async def test_later_undo_preserves_branch_and_flags_branching(self):
        bot = await _make_bot()
        ctx = _ctx()

        await bot.chat("First", ctx)
        await bot.chat("Second", ctx)

        result = await bot.undo_last_turn(ctx)  # undo turn 1, not turn 0

        manager = bot._conversation_managers.get(ctx.conversation_id)
        # Back to after turn 0: "First" survives on the tree path.
        user_msgs = [
            m["content"] for m in manager.messages if _role_of(m) == "user"
        ]
        assert user_msgs == ["First"]
        assert manager.state is not None        # NOT reset — real node switch
        assert result.remaining_turns == 1
        assert result.branching is True         # sibling branch preserved

    @pytest.mark.asyncio
    async def test_bounded_undo_still_functions(self):
        # FU-cache bounding: with max_undo_checkpoints, later-turn undo still
        # works and the turn-0 fix does not disturb the cap/dropped bookkeeping.
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "memory": {"type": "buffer", "max_messages": 50},
            "max_undo_checkpoints": 3,
        }
        bot = await DynaBot.from_config(config)
        ctx = _ctx("conv-cap-later")

        for _ in range(5):
            await bot.chat("hello", ctx)

        log = bot._turn_checkpoints[ctx.conversation_id]
        assert log.dropped == 2 and len(log.entries) == 3  # cap active

        result = await bot.undo_last_turn(ctx)
        assert result.branching is True
        assert result.remaining_turns == 4


class TestSystemPromptUndoToStart:
    """A seeded system prompt keeps undo-to-start on the system root (unchanged).

    With a system prompt the system message occupies root ``""`` and the first
    *user* message becomes child ``"0"``; the turn-0 checkpoint records the real
    system-root ``""`` (state is not ``None``), so it never takes the sentinel
    path. Undo-to-start lands on ``[system]``, not empty.
    """

    async def _make_system_bot(self) -> DynaBot:
        config: dict[str, Any] = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "memory": {"type": "buffer", "max_messages": 50},
            "system_prompt": "You are a helpful assistant.",
        }
        return await DynaBot.from_config(config)

    @pytest.mark.asyncio
    async def test_undo_to_start_retains_system_message(self):
        bot = await self._make_system_bot()
        ctx = _ctx("conv-sysprompt-undo")

        await bot.chat("First", ctx)
        result = await bot.undo_last_turn(ctx)

        manager = bot._conversation_managers.get(ctx.conversation_id)
        # System root survives; not reset to empty.
        assert manager.state is not None
        roles = [_role_of(m) for m in manager.messages]
        assert roles == ["system"]
        assert _user_message_count(manager) == 0  # zero user turns remain
        # Real node switch (system root ""), so branching is preserved.
        assert result.branching is True


class TestRewindToCurrentTurnIsNoop:
    """Rewinding to the current/newest turn is a legal no-op, not an error.

    ``rewind_to_turn`` to the turn the conversation already sits at computes
    zero undo work (``turns_to_undo == 0``).  Previously the trailing
    ``if result is None`` raised a misleading ``"Nothing to undo"`` for this
    case — and, worse, gave that same wrong message for a never-started
    conversation.  The proper behavior: a zero-work rewind of an *active*
    conversation returns a well-formed no-op ``UndoResult`` (nothing undone,
    no new branch, all turns retained), while a conversation with no active
    manager still reports the clear ``"No active conversation"``.
    """

    @pytest.mark.asyncio
    async def test_rewind_to_current_turn_returns_noop_result(self):
        bot = await _make_bot()
        ctx = _ctx()

        await bot.chat("First", ctx)
        await bot.chat("Second", ctx)
        await bot.chat("Third", ctx)

        conv_id = ctx.conversation_id
        entries_before = list(bot._turn_checkpoints[conv_id].entries)
        mem_before = await bot.memory.get_context("test")

        # Rewind to turn 2 — the newest turn, i.e. where we already are.
        result = await bot.rewind_to_turn(ctx, 2)

        # A well-formed no-op: nothing undone, no new branch, all turns remain.
        assert isinstance(result, UndoResult)
        assert result.undone_user_message == ""
        assert result.undone_bot_response == ""
        assert result.branching is False
        assert result.remaining_turns == 3

        # And nothing actually changed — no checkpoint popped, memory intact.
        assert bot._turn_checkpoints[conv_id].entries == entries_before
        assert len(await bot.memory.get_context("test")) == len(mem_before)

    @pytest.mark.asyncio
    async def test_rewind_to_current_turn_noop_under_cap(self):
        # Bounded undo history: rewinding to the newest turn is still a no-op,
        # and it must not disturb the retained entries or the dropped offset.
        # remaining_turns reflects the true conversation length (5), not the
        # retained-checkpoint count (3).
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "memory": {"type": "buffer", "max_messages": 50},
            "max_undo_checkpoints": 3,
        }
        bot = await DynaBot.from_config(config)
        ctx = _ctx("conv-cap-noop")

        for _ in range(5):
            await bot.chat("hello", ctx)

        log = bot._turn_checkpoints[ctx.conversation_id]
        assert log.dropped == 2 and len(log.entries) == 3  # cap active

        # Newest turn is 4 (total == 5). Rewind to it: zero work.
        result = await bot.rewind_to_turn(ctx, 4)
        assert result.undone_user_message == ""
        assert result.branching is False
        assert result.remaining_turns == 5

        # The tail-cap state is untouched by the no-op.
        assert log.dropped == 2
        assert len(log.entries) == 3

    @pytest.mark.asyncio
    async def test_rewind_zero_work_on_active_but_emptied_conversation(self):
        # After undoing the only turn, the conversation is still active
        # (manager + state present) but holds zero checkpoints. A rewind to
        # the start is then a zero-work no-op, not "Nothing to undo".
        bot = await _make_bot()
        ctx = _ctx("conv-emptied")

        await bot.chat("Only", ctx)
        await bot.undo_last_turn(ctx)
        assert bot._turn_checkpoints[ctx.conversation_id].entries == []

        # total == 0 but the manager is still active -> a no-op result, not a
        # raise. (remaining_turns is left to undo's own tree-path accounting;
        # the contract under test here is "no-op, no spurious rollback".)
        result = await bot.rewind_to_turn(ctx, -1)
        assert isinstance(result, UndoResult)
        assert result.undone_user_message == ""
        assert result.undone_bot_response == ""
        assert result.branching is False
        assert bot._turn_checkpoints[ctx.conversation_id].entries == []

    @pytest.mark.asyncio
    async def test_rewind_never_started_conversation_reports_no_active(self):
        # A zero-work target on a conversation with no active manager is NOT a
        # silent no-op — it reports the clear "No active conversation"
        # (previously the misleading "Nothing to undo").
        bot = await _make_bot()
        ctx = _ctx("conv-never-started")

        with pytest.raises(ValueError, match="No active conversation"):
            await bot.rewind_to_turn(ctx, -1)


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

        log = bot._turn_checkpoints.get(ctx.conversation_id)
        assert log is not None
        assert len(log.entries) == 3

    @pytest.mark.asyncio
    async def test_checkpoints_per_conversation(self):
        """Each conversation tracks its own checkpoints."""
        bot = await _make_bot()
        ctx1 = _ctx("conv-1")
        ctx2 = _ctx("conv-2")

        await bot.chat("Hello", ctx1)
        await bot.chat("Hello", ctx2)
        await bot.chat("Again", ctx1)

        assert len(bot._turn_checkpoints["conv-1"].entries) == 2
        assert len(bot._turn_checkpoints["conv-2"].entries) == 1

    @pytest.mark.asyncio
    async def test_undo_pops_checkpoint(self):
        bot = await _make_bot()
        ctx = _ctx()

        await bot.chat("First", ctx)
        await bot.chat("Second", ctx)
        assert len(bot._turn_checkpoints[ctx.conversation_id].entries) == 2

        await bot.undo_last_turn(ctx)
        assert len(bot._turn_checkpoints[ctx.conversation_id].entries) == 1


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


# =====================================================================
# Wizard undo (greet → chat → undo)
# =====================================================================


def _wizard_config() -> dict[str, Any]:
    """Minimal wizard config: greeting stage → collect_info → done."""
    return {
        "name": "undo-test-wizard",
        "version": "1.0",
        "stages": [
            {
                "name": "greeting",
                "is_start": True,
                "prompt": "Greet the user and ask for their name",
                "response_template": "Hello! What is your name?",
                "schema": {
                    "type": "object",
                    "properties": {"user_name": {"type": "string"}},
                    "required": ["user_name"],
                },
                "transitions": [
                    {"target": "done", "condition": "data.get('user_name')"},
                ],
            },
            {
                "name": "done",
                "is_end": True,
                "prompt": "All done!",
            },
        ],
    }


async def _make_wizard_bot() -> DynaBot:
    """Create a DynaBot with EchoProvider + wizard reasoning for testing."""
    config: dict[str, Any] = {
        "llm": {"provider": "echo", "model": "test"},
        "conversation_storage": {"backend": "memory"},
        "memory": {"type": "buffer", "max_messages": 50},
        "reasoning": {
            "strategy": "wizard",
            "wizard_config": _wizard_config(),
            "strict_validation": False,
        },
    }
    return await DynaBot.from_config(config)


class TestWizardUndo:
    """Undo in a wizard conversation after greet → chat.

    Bug: _restore_wizard_from_node reads wizard_fsm_state from the
    checkpoint node's metadata. If the checkpoint node is a greeting
    node and wizard FSM state wasn't saved on it, restore silently
    does nothing — leaving wizard state at the post-chat stage instead
    of reverting to the greeting stage.
    """

    @pytest.mark.asyncio
    async def test_undo_after_greet_restores_wizard_stage(self):
        """After greet→chat→undo, wizard stage should revert to greeting."""
        bot = await _make_wizard_bot()
        ctx = _ctx("conv-wizard-undo")

        # Greet — wizard starts at "greeting" stage
        greeting = await bot.greet(ctx)
        assert greeting is not None

        state_after_greet = await bot.get_wizard_state(ctx.conversation_id)
        assert state_after_greet is not None
        assert state_after_greet["current_stage"] == "greeting"

        # Chat — EchoProvider echoes the message; wizard may extract data
        # and transition to next stage
        await bot.chat("My name is Alice", ctx)

        # Undo — should revert wizard state to greeting stage
        result = await bot.undo_last_turn(ctx)
        assert isinstance(result, UndoResult)

        state_after_undo = await bot.get_wizard_state(ctx.conversation_id)
        assert state_after_undo is not None
        assert state_after_undo["current_stage"] == "greeting", (
            f"Expected wizard stage 'greeting' after undo, "
            f"got '{state_after_undo['current_stage']}'. "
            f"Wizard FSM state was not restored from the greeting node."
        )
