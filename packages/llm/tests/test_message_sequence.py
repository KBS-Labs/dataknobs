"""Tests for the shared message-sequence invariants.

``pair_orphan_tool_calls`` was hoisted into ``dataknobs_llm.llm`` from the
bots ReAct strategy so the same ``tool_use`` ↔ ``tool_result`` pairing
invariant can be reused by the provider adapters' message-consolidation path
without an import cycle. Its exhaustive behavioral coverage lives with the
ReAct finalize tests (the original consumer); this module pins the core at its
new home in the ``dataknobs-llm`` package.
"""

from __future__ import annotations

from dataknobs_llm.llm.base import LLMMessage, ToolCall
from dataknobs_llm.llm.message_sequence import pair_orphan_tool_calls


def _assistant_call(name: str, params: dict, call_id: str | None) -> LLMMessage:
    return LLMMessage(
        role="assistant",
        content="",
        tool_calls=[ToolCall(name=name, parameters=params, id=call_id)],
    )


def test_well_formed_history_is_noop() -> None:
    messages = [
        LLMMessage(role="user", content="Hi"),
        _assistant_call("search", {"q": "x"}, "t1"),
        LLMMessage(role="tool", content="ok", name="search", tool_call_id="t1"),
    ]
    assert pair_orphan_tool_calls(messages) == []


def test_orphan_tool_use_is_paired() -> None:
    messages = [
        LLMMessage(role="user", content="Hi"),
        _assistant_call("search", {"q": "x"}, "t1"),
    ]
    results = pair_orphan_tool_calls(messages)
    assert len(results) == 1
    assert results[0].role == "tool"
    assert results[0].tool_call_id == "t1"
    assert results[0].name == "search"
    assert "loop ended" in results[0].content.lower()


def test_input_is_not_mutated() -> None:
    messages = [_assistant_call("search", {"q": "x"}, "t1")]
    original_len = len(messages)
    pair_orphan_tool_calls(messages)
    assert len(messages) == original_len


def test_duplicate_signature_gets_duplicate_guidance() -> None:
    """An orphan repeating an already-answered call gets the duplicate hint."""
    messages = [
        _assistant_call("search", {"q": "x"}, "t1"),
        LLMMessage(role="tool", content="ok", name="search", tool_call_id="t1"),
        # A second, unanswered call with identical name+params (id t2).
        _assistant_call("search", {"q": "x"}, "t2"),
    ]
    results = pair_orphan_tool_calls(messages)
    assert len(results) == 1
    assert results[0].tool_call_id == "t2"
    assert "already called" in results[0].content.lower()


def test_idempotent_over_partially_paired_history() -> None:
    messages = [_assistant_call("search", {"q": "x"}, "t1")]
    first = pair_orphan_tool_calls(messages)
    assert len(first) == 1
    # Re-running over the now-complete history yields nothing new.
    assert pair_orphan_tool_calls(messages + first) == []
