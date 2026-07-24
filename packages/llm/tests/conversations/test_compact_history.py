"""Tests for ``ConversationManager.compact_history`` (in-loop history compaction).

The primitive re-roots the active conversation path, dropping (windowing) or
summarizing the oldest tool-iteration pairs while preserving system prompt, the
current-turn user message, and the most recent K iterations. The load-bearing
invariant it must never violate: a ``tool_use`` is never separated from its
``tool_result`` — compaction happens only at whole-iteration boundaries (the
exact condition an Anthropic 400 enforces on the re-sent history).

Reproduce-first: ``compact_history`` is a new primitive; against HEAD before it
existed these exercises raised ``AttributeError`` (no capability to bound a long
tool loop). Real constructs only — a real ``ConversationManager`` over an
``AsyncMemoryDatabase`` + ``EchoProvider``; the summarize path uses a real
``LLMSummarizer`` (no mocks).
"""

from __future__ import annotations

import pytest

from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_llm import LLMSummarizer, Summarizer
from dataknobs_llm.conversations import (
    ConversationManager,
    DataknobsConversationStorage,
)
from dataknobs_llm.llm import EchoProvider, LLMConfig
from dataknobs_llm.llm.base import ToolCall
from dataknobs_llm.prompts import AsyncPromptBuilder

pytestmark = pytest.mark.asyncio


async def _make_manager() -> ConversationManager:
    """A minimal real manager: EchoProvider + in-memory storage."""
    llm = EchoProvider(LLMConfig(provider="echo", model="echo-model"))
    builder = AsyncPromptBuilder(library=None)
    storage = DataknobsConversationStorage(AsyncMemoryDatabase())
    return await ConversationManager.create(
        llm=llm, prompt_builder=builder, storage=storage
    )


async def _append_iteration(manager: ConversationManager, i: int) -> None:
    """Append one tool iteration: an assistant ``tool_use`` + its ``tool_result``.

    ``add_message`` reconstructs a bare ``LLMMessage`` (dropping ``tool_calls``),
    so the assistant's tool call is stamped onto the stored message directly —
    exactly the fidelity ``compact_history`` must preserve by *moving* nodes
    rather than rebuilding them.
    """
    call_id = f"call_{i}"
    node = await manager.add_message(role="assistant", content=f"step {i}")
    node.message.tool_calls = [
        ToolCall(name="lookup", parameters={"n": i}, id=call_id)
    ]
    await manager.add_message(
        role="tool", content=f"observation {i}", tool_call_id=call_id
    )


async def _build_loop(manager: ConversationManager, n_iterations: int) -> None:
    """System prompt, user, then n copies of (assistant tool_use + tool_result)."""
    await manager.add_message(role="system", content="You are a tool user.")
    await manager.add_message(role="user", content="Do the task.")
    for i in range(n_iterations):
        await _append_iteration(manager, i)


def _assert_no_dangling_tool_use(messages) -> None:
    """Every assistant ``tool_use`` id is answered by a later ``tool_result``.

    The exact Anthropic-400 well-formedness condition on the re-sent history.
    """
    open_ids: set[str] = set()
    for msg in messages:
        if msg.role == "assistant" and msg.tool_calls:
            open_ids.update(tc.id for tc in msg.tool_calls if tc.id)
        elif msg.role == "tool" and msg.tool_call_id:
            open_ids.discard(msg.tool_call_id)
    assert not open_ids, f"dangling tool_use ids: {open_ids}"


class TestCompactHistoryWindowing:
    async def test_windows_old_iterations(self) -> None:
        """keep_recent=2 over 5 iterations drops the oldest 3, keeps last 2."""
        manager = await _make_manager()
        await _build_loop(manager, 5)
        before = await manager.get_history()
        assert len(before) == 2 + 5 * 2  # system + user + 5 pairs

        dropped = await manager.compact_history(2)

        assert dropped == 3
        after = await manager.get_history()
        # system + user + 2 kept pairs (2 messages each)
        assert len(after) == 2 + 2 * 2
        contents = [m.content for m in after]
        # Oldest three iterations gone; the most recent two retained.
        assert "observation 0" not in contents
        assert "observation 2" not in contents
        assert "observation 3" in contents
        assert "observation 4" in contents

    async def test_preserves_pairing(self) -> None:
        """No retained ``tool_use`` is separated from its ``tool_result``."""
        manager = await _make_manager()
        await _build_loop(manager, 6)
        await manager.compact_history(2)
        _assert_no_dangling_tool_use(await manager.get_history())

    async def test_keep_zero_windows_all(self) -> None:
        """keep_recent=0 drops the entire current loop body."""
        manager = await _make_manager()
        await _build_loop(manager, 4)
        dropped = await manager.compact_history(0)
        assert dropped == 4
        after = await manager.get_history()
        assert [m.role for m in after] == ["system", "user"]

    async def test_noop_when_within_keep(self) -> None:
        """Fewer units than must be kept → no-op (returns 0, unchanged)."""
        manager = await _make_manager()
        await _build_loop(manager, 2)
        before = await manager.get_history()
        dropped = await manager.compact_history(3)
        assert dropped == 0
        assert await manager.get_history() == before

    async def test_noop_without_state(self) -> None:
        """A fresh manager (no messages yet) compacts to nothing."""
        manager = await _make_manager()
        assert await manager.compact_history(2) == 0

    async def test_negative_keep_raises(self) -> None:
        manager = await _make_manager()
        with pytest.raises(ValueError):
            await manager.compact_history(-1)

    async def test_current_position_stays_valid(self) -> None:
        """After compaction the loop can continue: appends land on the tail."""
        manager = await _make_manager()
        await _build_loop(manager, 5)
        await manager.compact_history(2)
        # The next iteration appends onto the compacted tail without error.
        await _append_iteration(manager, 99)
        history = await manager.get_history()
        assert history[-1].content == "observation 99"
        _assert_no_dangling_tool_use(history)


class TestCompactHistorySummarizing:
    async def test_summarizes_dropped_span(self) -> None:
        """With a summarizer, dropped iterations fold into one summary node."""
        manager = await _make_manager()
        await _build_loop(manager, 5)
        summarizer = LLMSummarizer(
            EchoProvider(LLMConfig(provider="echo", model="echo-model"))
        )
        assert isinstance(summarizer, Summarizer)

        dropped = await manager.compact_history(2, summarizer=summarizer)

        assert dropped == 3
        after = await manager.get_history()
        # system, user, <one summary>, then the last 2 pairs (4 messages).
        assert len(after) == 2 + 1 + 2 * 2
        # Exactly one summary message inserted, at the head of the loop body.
        summaries = [
            m
            for m in after
            if m.role == "system" and "observation 0" in (m.content or "")
        ]
        assert len(summaries) == 1  # the echoed summary carries the folded text
        contents = [m.content for m in after]
        assert "observation 3" in contents
        assert "observation 4" in contents
        _assert_no_dangling_tool_use(after)
