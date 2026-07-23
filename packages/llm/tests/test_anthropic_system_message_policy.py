"""Reproduce-first tests for the mid-conversation system-message policy.

``AnthropicAdapter.adapt_messages`` historically hoisted **every**
``role="system"`` message — leading OR mid-conversation — into the top-level
``system`` param. That is correct for a leading system prompt (Anthropic's
Messages API has no inline ``system`` role) but **lossy** for a
mid-conversation notice: a positional, in-context instruction becomes a
standing global instruction, silently.

This module pins the configurable policy that replaces the silent hoist:

- ``hoist`` — merge into the top-level ``system`` param (legacy behavior).
- ``inline`` — convert a mid-conversation system message to a ``user`` message
  at its position, consolidating content blocks so role-alternation and
  ``tool_use`` ↔ ``tool_result`` adjacency stay valid (**the default**).
- ``warn`` — log a warning naming the offending message, then hoist (fallback).
- ``reject`` — raise ``ValidationError``.

Leading system messages ALWAYS hoist, under every policy. Whether a family
accepts an inline system message at all is read from the S1
``ModelConstraints.accepts_inline_system`` datum (``False`` for Anthropic).

The behavioral reproduce-first proof is
``TestDefaultPolicy.test_default_inlines_mid_conversation_system``: against the
pre-policy adapter a mid-conversation system message is hoisted (present in
``system_content``); after the fix the default inlines it.
"""

from __future__ import annotations

import logging

import pytest

from dataknobs_common.exceptions import ValidationError
from dataknobs_llm.llm.base import LLMConfig, LLMMessage, ToolCall
from dataknobs_llm.llm.providers.anthropic import (
    AnthropicAdapter,
    AnthropicProvider,
)

_ANTHROPIC_LOGGER = "dataknobs_llm.llm.providers.anthropic"


def _flatten_text(msgs: list[dict]) -> str:
    """Concatenate every text/tool_result payload across adapted messages."""
    parts: list[str] = []
    for m in msgs:
        content = m["content"]
        if isinstance(content, str):
            parts.append(content)
            continue
        for block in content:
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif block.get("type") == "tool_result":
                parts.append(str(block.get("content", "")))
    return "\n".join(parts)


def _no_consecutive_same_role(msgs: list[dict]) -> bool:
    roles = [m["role"] for m in msgs]
    return all(roles[i] != roles[i + 1] for i in range(len(roles) - 1))


def _provider(**options: object) -> AnthropicProvider:
    """Construct an AnthropicProvider without initializing the SDK client."""
    config = LLMConfig(
        provider="anthropic",
        model="claude-3-sonnet-20240229",
        api_key="test-key",
        options=dict(options),
    )
    return AnthropicProvider(config)


# ---------------------------------------------------------------------------
# Leading system messages ALWAYS hoist, regardless of policy
# ---------------------------------------------------------------------------


class TestLeadingSystemAlwaysHoists:
    @pytest.mark.parametrize("policy", ["hoist", "inline", "warn", "reject"])
    def test_leading_system_hoisted(self, policy: str) -> None:
        adapter = AnthropicAdapter(system_message_policy=policy)
        messages = [
            LLMMessage(role="system", content="You are helpful."),
            LLMMessage(role="user", content="Hi"),
        ]
        system, msgs = adapter.adapt_messages(messages)
        assert system == "You are helpful."
        assert [m["role"] for m in msgs] == ["user"]

    def test_consecutive_leading_system_all_hoist(self) -> None:
        adapter = AnthropicAdapter(system_message_policy="reject")
        messages = [
            LLMMessage(role="system", content="First."),
            LLMMessage(role="system", content="Second."),
            LLMMessage(role="user", content="Hi"),
        ]
        system, msgs = adapter.adapt_messages(messages)
        assert "First." in system
        assert "Second." in system
        assert [m["role"] for m in msgs] == ["user"]


# ---------------------------------------------------------------------------
# hoist policy — legacy behavior, back-compat
# ---------------------------------------------------------------------------


class TestHoistPolicy:
    def test_mid_conversation_system_hoisted(self) -> None:
        adapter = AnthropicAdapter(system_message_policy="hoist")
        messages = [
            LLMMessage(role="user", content="Hi"),
            LLMMessage(role="system", content="Mid notice."),
            LLMMessage(role="assistant", content="OK"),
        ]
        system, msgs = adapter.adapt_messages(messages)
        assert "Mid notice." in system
        assert all(m["role"] != "system" for m in msgs)
        assert [m["role"] for m in msgs] == ["user", "assistant"]


# ---------------------------------------------------------------------------
# inline policy — convert to a user message at position, consolidated
# ---------------------------------------------------------------------------


class TestInlinePolicy:
    def test_mid_conversation_system_becomes_user(self) -> None:
        adapter = AnthropicAdapter(system_message_policy="inline")
        messages = [
            LLMMessage(role="user", content="Hi"),
            LLMMessage(role="assistant", content="Hello"),
            LLMMessage(role="system", content="Mid notice."),
            LLMMessage(role="user", content="Continue"),
        ]
        system, msgs = adapter.adapt_messages(messages)
        assert "Mid notice." not in system
        assert _no_consecutive_same_role(msgs)
        assert "Mid notice." in _flatten_text(msgs)

    def test_leading_system_still_hoisted_under_inline(self) -> None:
        adapter = AnthropicAdapter(system_message_policy="inline")
        messages = [
            LLMMessage(role="system", content="Leading."),
            LLMMessage(role="user", content="Hi"),
            LLMMessage(role="system", content="Mid."),
            LLMMessage(role="assistant", content="OK"),
        ]
        system, msgs = adapter.adapt_messages(messages)
        assert system == "Leading."
        assert "Mid." not in system
        assert "Mid." in _flatten_text(msgs)

    def test_inline_merges_into_preceding_user(self) -> None:
        """A notice after a plain user message merges into that user turn."""
        adapter = AnthropicAdapter(system_message_policy="inline")
        messages = [
            LLMMessage(role="user", content="Hi"),
            LLMMessage(role="system", content="Mid notice."),
            LLMMessage(role="assistant", content="OK"),
        ]
        system, msgs = adapter.adapt_messages(messages)
        assert "Mid notice." not in system
        assert [m["role"] for m in msgs] == ["user", "assistant"]
        assert "Hi" in _flatten_text(msgs)
        assert "Mid notice." in _flatten_text(msgs)

    def test_inline_tool_pairing_safety(self) -> None:
        """The safety test: a notice between tool_use and tool_result.

        ``[user, assistant(tool_use), system(notice), tool(result)]`` under
        ``inline`` MUST NOT produce consecutive same-role messages, and the
        ``tool_result`` MUST stay adjacent/paired to the ``tool_use`` — the
        exact structural condition Anthropic's API rejects with a 400.

        Reproduce-first: Anthropic requires every ``tool_result`` block to come
        **first** in a user turn's content array (text after). Before the
        ordering fix the inlined-notice ``text`` block landed *before* the
        consolidated ``tool_result`` — a documented 400 — so the
        ``tool_result``-before-``text`` assertion below fails against the
        pre-fix adapter and passes after.
        """
        adapter = AnthropicAdapter(system_message_policy="inline")
        messages = [
            LLMMessage(role="user", content="Do the thing"),
            LLMMessage(
                role="assistant",
                content="",
                tool_calls=[
                    ToolCall(name="search", parameters={"q": "x"}, id="t1"),
                ],
            ),
            LLMMessage(role="system", content="Loop timed out; use results."),
            LLMMessage(
                role="tool",
                content='{"r": 1}',
                name="search",
                tool_call_id="t1",
            ),
        ]
        system, msgs = adapter.adapt_messages(messages)

        assert "Loop timed out" not in system
        assert _no_consecutive_same_role(msgs), [m["role"] for m in msgs]
        assert [m["role"] for m in msgs] == ["user", "assistant", "user"]

        # The tool_use lives in the assistant turn.
        assert any(b["type"] == "tool_use" for b in msgs[1]["content"])

        # The following user turn carries BOTH the inlined notice text and the
        # paired tool_result — merged into one message so the result stays
        # adjacent to the tool_use.
        final_blocks = msgs[2]["content"]
        types = [b["type"] for b in final_blocks]
        assert "text" in types
        assert "tool_result" in types
        # Anthropic requires tool_result blocks first, text after — else 400.
        assert types.index("tool_result") < types.index("text"), types
        result = next(b for b in final_blocks if b["type"] == "tool_result")
        assert result["tool_use_id"] == "t1"
        assert "Loop timed out" in _flatten_text(msgs)

    def test_inline_notice_after_tool_result_keeps_result_first(self) -> None:
        """Adversarial ordering: notice arriving *after* the tool_result.

        ``[user, assistant(tool_use), tool(result), system(notice)]`` — the
        notice inlines onto the same user turn that already holds the
        ``tool_result``. The result must stay first, the notice text after.
        """
        adapter = AnthropicAdapter(system_message_policy="inline")
        messages = [
            LLMMessage(role="user", content="Do the thing"),
            LLMMessage(
                role="assistant",
                content="",
                tool_calls=[
                    ToolCall(name="search", parameters={"q": "x"}, id="t1"),
                ],
            ),
            LLMMessage(
                role="tool",
                content='{"r": 1}',
                name="search",
                tool_call_id="t1",
            ),
            LLMMessage(role="system", content="Now summarize."),
        ]
        _, msgs = adapter.adapt_messages(messages)

        assert _no_consecutive_same_role(msgs), [m["role"] for m in msgs]
        assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
        types = [b["type"] for b in msgs[2]["content"]]
        assert types.index("tool_result") < types.index("text"), types
        assert "Now summarize." in _flatten_text(msgs)

    def test_inline_notice_between_multi_tool_use_and_results(self) -> None:
        """Adversarial ordering: notice before *two* tool results.

        ``[user, assistant(tool_use t1, tool_use t2), system(notice),
        tool(t1), tool(t2)]`` — both results must land first, in first-seen
        order, with the notice text last.
        """
        adapter = AnthropicAdapter(system_message_policy="inline")
        messages = [
            LLMMessage(role="user", content="Do two things"),
            LLMMessage(
                role="assistant",
                content="",
                tool_calls=[
                    ToolCall(name="a", parameters={}, id="t1"),
                    ToolCall(name="b", parameters={}, id="t2"),
                ],
            ),
            LLMMessage(role="system", content="Loop ended; use results."),
            LLMMessage(role="tool", content="r1", name="a", tool_call_id="t1"),
            LLMMessage(role="tool", content="r2", name="b", tool_call_id="t2"),
        ]
        _, msgs = adapter.adapt_messages(messages)

        assert _no_consecutive_same_role(msgs), [m["role"] for m in msgs]
        final_blocks = msgs[-1]["content"]
        types = [b["type"] for b in final_blocks]
        # Both tool_results first, in order; the notice text after them.
        assert types == ["tool_result", "tool_result", "text"], types
        assert [
            b["tool_use_id"] for b in final_blocks if b["type"] == "tool_result"
        ] == ["t1", "t2"]
        assert "Loop ended" in _flatten_text(msgs)

    def test_consecutive_tool_results_still_consolidated(self) -> None:
        """The pre-existing tool_result consolidation is preserved."""
        adapter = AnthropicAdapter(system_message_policy="inline")
        messages = [
            LLMMessage(role="tool", content="r1", name="s", tool_call_id="t1"),
            LLMMessage(role="tool", content="r2", name="c", tool_call_id="t2"),
        ]
        _, msgs = adapter.adapt_messages(messages)
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert [b["tool_use_id"] for b in msgs[0]["content"]] == ["t1", "t2"]


# ---------------------------------------------------------------------------
# warn policy — log + fall back to hoist
# ---------------------------------------------------------------------------


class TestWarnPolicy:
    def test_mid_conversation_system_warns_and_hoists(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        adapter = AnthropicAdapter(system_message_policy="warn")
        messages = [
            LLMMessage(role="user", content="Hi"),
            LLMMessage(role="system", content="Mid notice."),
        ]
        with caplog.at_level(logging.WARNING, logger=_ANTHROPIC_LOGGER):
            system, msgs = adapter.adapt_messages(messages)
        assert "Mid notice." in system  # fallback disposition = hoist
        assert all(m["role"] != "system" for m in msgs)
        assert any(
            "system" in r.message.lower() for r in caplog.records
        ), [r.message for r in caplog.records]

    def test_leading_system_does_not_warn(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        adapter = AnthropicAdapter(system_message_policy="warn")
        messages = [
            LLMMessage(role="system", content="Leading."),
            LLMMessage(role="user", content="Hi"),
        ]
        with caplog.at_level(logging.WARNING, logger=_ANTHROPIC_LOGGER):
            adapter.adapt_messages(messages)
        assert not caplog.records


# ---------------------------------------------------------------------------
# reject policy — raise ValidationError
# ---------------------------------------------------------------------------


class TestRejectPolicy:
    def test_mid_conversation_system_raises(self) -> None:
        adapter = AnthropicAdapter(system_message_policy="reject")
        messages = [
            LLMMessage(role="user", content="Hi"),
            LLMMessage(role="system", content="Mid notice."),
        ]
        with pytest.raises(ValidationError):
            adapter.adapt_messages(messages)

    def test_leading_system_not_rejected(self) -> None:
        adapter = AnthropicAdapter(system_message_policy="reject")
        messages = [
            LLMMessage(role="system", content="Leading."),
            LLMMessage(role="user", content="Hi"),
        ]
        system, _ = adapter.adapt_messages(messages)
        assert system == "Leading."


# ---------------------------------------------------------------------------
# Default policy + accepts_inline_system constraint
# ---------------------------------------------------------------------------


class TestDefaultPolicy:
    def test_default_is_inline(self) -> None:
        assert AnthropicAdapter().system_message_policy == "inline"

    def test_default_inlines_mid_conversation_system(self) -> None:
        """Reproduce-first: pre-policy this text lands in ``system_content``."""
        adapter = AnthropicAdapter()
        messages = [
            LLMMessage(role="user", content="Hi"),
            LLMMessage(role="system", content="Mid notice."),
            LLMMessage(role="user", content="More"),
        ]
        system, msgs = adapter.adapt_messages(messages)
        assert "Mid notice." not in system
        assert "Mid notice." in _flatten_text(msgs)
        assert _no_consecutive_same_role(msgs)


class TestAcceptsInlineSystemConstraint:
    def test_accepts_inline_leaves_system_message_inline(self) -> None:
        """If the family accepts inline system, the message stays a system turn.

        The policy governs only families that forbid inline system messages
        (``accepts_inline_system=False`` — Anthropic's default). When a
        consumer declares the family accepts them, no conversion happens.
        """
        adapter = AnthropicAdapter(
            system_message_policy="hoist", accepts_inline_system=True
        )
        messages = [
            LLMMessage(role="user", content="Hi"),
            LLMMessage(role="system", content="Mid notice."),
        ]
        system, msgs = adapter.adapt_messages(messages)
        assert "Mid notice." not in system
        assert any(
            m["role"] == "system" and m["content"] == "Mid notice."
            for m in msgs
        )


# ---------------------------------------------------------------------------
# Provider wiring — policy read from LLMConfig.options
# ---------------------------------------------------------------------------


class TestProviderWiring:
    def test_default_options_policy_is_inline(self) -> None:
        assert _provider().adapter.system_message_policy == "inline"

    def test_options_policy_propagates(self) -> None:
        provider = _provider(system_message_policy="hoist")
        assert provider.adapter.system_message_policy == "hoist"

    def test_anthropic_forbids_inline_system(self) -> None:
        assert _provider().adapter.accepts_inline_system is False

    def test_constraints_override_accepts_inline_system(self) -> None:
        config = LLMConfig(
            provider="anthropic",
            model="claude-3-sonnet-20240229",
            api_key="test-key",
            constraints={"accepts_inline_system": True},
        )
        provider = AnthropicProvider(config)
        assert provider.adapter.accepts_inline_system is True

    def test_invalid_policy_raises(self) -> None:
        with pytest.raises(ValidationError):
            _provider(system_message_policy="bogus")
