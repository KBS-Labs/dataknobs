"""Reproduce-first tests: orphan ``tool_use`` pairing at the turn-finalize chokepoint.

Background
----------
When a non-ReAct tool turn ends *abnormally* — the DynaBot monolithic tool loop
hits its iteration cap or a wall-clock ``tool_loop_timeout`` with a pending,
unexecuted ``tool_call`` — the last assistant ``tool_use`` is left in
conversation history with no following ``tool_result``.  On the *next* turn the
manager re-sends that history to the provider; Anthropic's Messages API rejects
a dangling ``tool_use`` with a hard 400 (other providers tolerate it).

This is the same defect class as the ReAct finalize gap (fixed earlier at the
``dataknobs_llm.llm.message_sequence.pair_orphan_tool_calls`` layer), but reached
through ``reasoning: simple`` + a tool registry rather than through ReAct.
ReAct pairs *before* its mid-turn synthesis re-completion; the ``simple``
monolithic loop performs no such re-completion, so the orphan is only paired
when the fix runs at the *universal persistence chokepoint*,
``DynaBot._finalize_turn`` — the single method every turn type and both delivery
modes (buffered ``chat`` + streaming ``stream_chat``) funnel through before the
turn is persisted.

(``grounded`` accepts ``tools`` for ABC compliance but never forwards them to
its synthesis call, so it does not drive the monolithic tool loop and cannot
produce an orphan — it is covered here as a pass-through no-op guard, T3, not as
a defect path.  ``hybrid`` is the tool-using grounded variant and self-pairs via
its ReAct phase.)

The pairing is a thin ``ConversationManager`` adapter
(``pair_orphan_tool_calls_on_manager``) over the pure ``pair_orphan_tool_calls``
core, called once at the top of ``_finalize_turn``.  It is idempotent: a no-op on
an already-paired history (every happy-path, wizard, and ReAct-already-paired
turn).

Assertion strategy — the *next-turn replay*
--------------------------------------------
The defect is a persisted conversation-state invariant whose real consequence is
the next turn's provider request.  Unlike the ReAct tests (which read the
in-turn synthesis call, because ReAct re-completes within the turn), the
``simple``/``grounded`` loop makes *no* provider call after the break — so the
faithful, fully-public assertion is a **two-turn replay**:

1. Turn 1 ends abnormally, leaving (on unfixed code) a dangling ``tool_use`` in
   the persisted history.
2. Turn 2 is an ordinary text turn; its single ``complete()`` re-sends the full
   persisted history.  ``EchoProvider.get_last_call()`` captures exactly the
   ``list[LLMMessage]`` the provider receives on that replay.
3. Run those messages through ``AnthropicAdapter.adapt_messages(...)`` — the
   exact conversion the API 400 validates — and assert no ``tool_use`` block is
   left unpaired.

These tests FAIL against unfixed HEAD (turn 2 replays a dangling ``tool_use``)
and PASS once ``_finalize_turn`` pairs the orphan.
"""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_bots.testing import BotTestHarness
from dataknobs_llm import LLMMessage
from dataknobs_llm.llm.message_sequence import (
    _UNEXECUTED_TOOL_RESULT,
    pair_orphan_tool_calls,
)
from dataknobs_llm.llm.providers.anthropic import AnthropicAdapter
from dataknobs_llm.llm.providers.echo import EchoProvider
from dataknobs_llm.testing import text_response, tool_call_response
from dataknobs_llm.tools.base import Tool

# ---------------------------------------------------------------------------
# Test tool
# ---------------------------------------------------------------------------


class EchoTool(Tool):
    """Simple tool that returns its input for testing."""

    def __init__(self) -> None:
        super().__init__(name="echo_tool", description="Echoes input back")
        self.call_count = 0

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"message": {"type": "string"}},
        }

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        self.call_count += 1
        return {"echoed": kwargs.get("message", ""), "call": self.call_count}


# ---------------------------------------------------------------------------
# Structural assertion helpers (shared shape with the ReAct pairing tests)
# ---------------------------------------------------------------------------

#: Both synthetic guidance strings share this marker prefix, so the tests can
#: recognise a pairing tool_result structurally without hardcoding the
#: route-aware full text.
_SYNTHETIC_PREFIX = "[Tool result unavailable:"


def _assert_no_dangling_tool_use(messages: list[LLMMessage]) -> None:
    """Assert every ``tool_use`` block pairs with a ``tool_result``.

    Runs ``messages`` through the Anthropic adapter (the exact conversion the
    API 400 validates) and asserts no ``tool_use`` id is left without a
    matching ``tool_result`` — the precise dangling-``tool_use`` condition.
    """
    _system, anthropic_messages = AnthropicAdapter().adapt_messages(messages)
    tool_use_ids: list[str] = []
    tool_result_ids: set[str] = set()
    for m in anthropic_messages:
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if block.get("type") == "tool_use":
                tool_use_ids.append(block["id"])
            elif block.get("type") == "tool_result":
                tool_result_ids.add(block["tool_use_id"])

    unpaired = [tid for tid in tool_use_ids if tid not in tool_result_ids]
    assert not unpaired, (
        "Dangling tool_use blocks after adaptation (the exact Anthropic 400 "
        f"condition): {unpaired}. Adapted messages: {anthropic_messages}"
    )


def _last_call_messages(provider: EchoProvider) -> list[LLMMessage]:
    """Return the message list the most recent completion received."""
    last = provider.get_last_call()
    assert last is not None, "expected at least one provider call"
    return list(last["messages"])


def _synthetic_pairing_contents(messages: list[LLMMessage]) -> list[str]:
    """Contents of the synthetic pairing ``tool_result`` messages, if any."""
    return [
        m.content
        for m in messages
        if m.role == "tool"
        and isinstance(m.content, str)
        and m.content.startswith(_SYNTHETIC_PREFIX)
    ]


def _has_synthetic_pairing(messages: list[LLMMessage]) -> bool:
    """Whether any synthetic pairing ``tool_result`` was appended."""
    return bool(_synthetic_pairing_contents(messages))


def _bot_config(
    strategy: str,
    *,
    reasoning_extra: dict[str, Any] | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Minimal echo-backed bot config for a monolithic-loop strategy.

    ``reasoning_extra`` is merged into the ``reasoning`` sub-block (e.g.
    grounded's ``intent``); other kwargs are top-level ``DynaBotConfig``
    fields (e.g. ``max_tool_iterations``, ``tool_loop_timeout``).
    """
    reasoning: dict[str, Any] = {"strategy": strategy}
    if reasoning_extra:
        reasoning.update(reasoning_extra)
    return {
        "llm": {"provider": "echo", "model": "test"},
        "conversation_storage": {"backend": "memory"},
        "reasoning": reasoning,
        **overrides,
    }


# =========================================================================
# T1 — simple + tools, cap-hit (buffered)
# =========================================================================


class TestSimpleCapHitBuffered:
    """``reasoning: simple`` + a tool, never satisfied, hits the iteration
    cap with a pending unexecuted call — the last ``tool_use`` dangles."""

    @pytest.mark.asyncio
    async def test_cap_hit_pairs_orphan_on_replay(self) -> None:
        tool = EchoTool()

        async with await BotTestHarness.create(
            # cap=2 → exactly 3 tool-call completions consumed on turn 1
            # (initial + 2 re-calls), the 3rd left dangling.
            bot_config=_bot_config("simple", max_tool_iterations=2),
            main_responses=[
                tool_call_response("echo_tool", {"message": "a"}),
                tool_call_response("echo_tool", {"message": "b"}),
                tool_call_response("echo_tool", {"message": "c"}),
                text_response("Follow-up answer"),
            ],
            tools=[tool],
        ) as harness:
            # Turn 1: abnormal termination — no raise.
            first = await harness.chat("Use the echo tool")
            assert first.response is not None
            # Turn 2: ordinary text turn re-sends the full persisted history.
            second = await harness.chat("And now answer plainly")

        assert second.response == "Follow-up answer"

        replayed = _last_call_messages(harness.provider)
        # The replayed history adapts to a paired Anthropic sequence — no 400.
        _assert_no_dangling_tool_use(replayed)
        # A synthetic pairing tool_result was persisted for the orphan; the
        # never-reached call ("c") carries the generic "loop ended" guidance.
        contents = _synthetic_pairing_contents(replayed)
        assert _UNEXECUTED_TOOL_RESULT in contents


# =========================================================================
# T2 — simple + tools, wall-clock timeout break (buffered)
# =========================================================================


class TestSimpleTimeoutBuffered:
    """A ``tool_loop_timeout`` break leaves the first ``tool_use`` dangling —
    a *different* break path than the cap (Part 4 both-routes requirement)."""

    @pytest.mark.asyncio
    async def test_timeout_pairs_orphan_on_replay(self) -> None:
        tool = EchoTool()

        async with await BotTestHarness.create(
            bot_config=_bot_config("simple", tool_loop_timeout=0.0),
            main_responses=[
                tool_call_response("echo_tool", {"message": "hi"}),
                text_response("Answer after timeout"),
            ],
            tools=[tool],
        ) as harness:
            await harness.chat("Use the echo tool")
            # Timeout fired before the tool executed.
            assert tool.call_count == 0
            second = await harness.chat("And now answer plainly")

        assert second.response == "Answer after timeout"

        replayed = _last_call_messages(harness.provider)
        _assert_no_dangling_tool_use(replayed)
        assert _UNEXECUTED_TOOL_RESULT in _synthetic_pairing_contents(replayed)


# =========================================================================
# T3 — grounded pass-through (a second monolithic strategy through the
#      same chokepoint; Layer A runs and no-ops)
# =========================================================================


class TestGroundedPassThrough:
    """``grounded`` is a second non-phased strategy that routes through the
    same ``_generate_response`` → monolithic-loop → ``_finalize_turn`` path.

    Unlike ``simple``, grounded does *not* drive tools — it accepts ``tools``
    for ABC compliance but never forwards them to its synthesis ``complete()``
    (``grounded.py`` docstring: "tools ... unused in this strategy"; HybridReasoning
    is the tool-using grounded variant, and its ReAct phase self-pairs).  So a
    grounded turn cannot produce a monolithic-loop orphan.  What this pins is
    that the Layer-A pairing call still *runs* on a grounded turn (a tool is
    registered, so the guard passes) and is a correct **no-op** — it must not
    inject a spurious ``tool_result`` into a strategy that carries no
    LLM-history orphan.
    """

    @pytest.mark.asyncio
    async def test_grounded_turn_appends_no_synthetic_pairing(self) -> None:
        async with await BotTestHarness.create(
            # ``intent: static`` → no LLM query-generation call; the single
            # provider call is the conversational synthesis.
            bot_config=_bot_config(
                "grounded", reasoning_extra={"intent": {"mode": "static"}}
            ),
            main_responses=[text_response("Grounded answer")],
            tools=[EchoTool()],
        ) as harness:
            result = await harness.chat("Tell me something")

            manager = harness.bot._conversation_managers[
                harness.context.conversation_id
            ]
            history = await manager.get_history()

        assert result.response == "Grounded answer"
        assert not _has_synthetic_pairing(history)
        _assert_no_dangling_tool_use(history)


# =========================================================================
# T4 — simple + tools, cap-hit (streaming)
# =========================================================================


class TestSimpleCapHitStreaming:
    """The streaming loop finalizes through the same ``_finalize_turn`` — one
    chokepoint covers both delivery modes."""

    @pytest.mark.asyncio
    async def test_streaming_cap_hit_pairs_orphan_on_replay(self) -> None:
        tool = EchoTool()

        async with await BotTestHarness.create(
            bot_config=_bot_config("simple", max_tool_iterations=2),
            main_responses=[
                tool_call_response("echo_tool", {"message": "a"}),
                tool_call_response("echo_tool", {"message": "b"}),
                tool_call_response("echo_tool", {"message": "c"}),
                text_response("Streamed follow-up"),
            ],
            tools=[tool],
        ) as harness:
            # Turn 1 streams and ends abnormally.
            await harness.stream_chat("Use the echo tool")
            # Turn 2 (buffered) re-sends the streamed turn's persisted history.
            second = await harness.chat("And now answer plainly")

        assert second.response == "Streamed follow-up"

        replayed = _last_call_messages(harness.provider)
        _assert_no_dangling_tool_use(replayed)
        assert _has_synthetic_pairing(replayed)


# =========================================================================
# T5 — simple + tools, timeout break (streaming)
# =========================================================================


class TestSimpleTimeoutStreaming:
    """Streaming wall-clock timeout break — the streaming counterpart of T2."""

    @pytest.mark.asyncio
    async def test_streaming_timeout_pairs_orphan_on_replay(self) -> None:
        tool = EchoTool()

        async with await BotTestHarness.create(
            bot_config=_bot_config("simple", tool_loop_timeout=0.0),
            main_responses=[
                tool_call_response("echo_tool", {"message": "hi"}),
                text_response("Streamed answer after timeout"),
            ],
            tools=[tool],
        ) as harness:
            await harness.stream_chat("Use the echo tool")
            second = await harness.chat("And now answer plainly")

        assert second.response == "Streamed answer after timeout"

        replayed = _last_call_messages(harness.provider)
        _assert_no_dangling_tool_use(replayed)
        assert _has_synthetic_pairing(replayed)


# =========================================================================
# T6 — ReAct unchanged (Layer B already paired; Layer A no-ops)
# =========================================================================


class TestReActUnchanged:
    """A ReAct cap-hit turn stays green: ReAct pairs before its in-turn
    synthesis (Layer B), and the new ``_finalize_turn`` call (Layer A) is an
    idempotent no-op on the already-paired history."""

    @pytest.mark.asyncio
    async def test_react_turn_still_paired_and_unchanged(self) -> None:
        tool = EchoTool()

        async with await BotTestHarness.create(
            bot_config=_bot_config("react"),
            main_responses=[
                tool_call_response("echo_tool", {"message": "same"}),
                tool_call_response("echo_tool", {"message": "same"}),
                text_response("Synthesized answer"),
            ],
            tools=[tool],
        ) as harness:
            result = await harness.chat("Use the echo tool")

        # ReAct re-completes in-turn; its synthesis call carries a paired
        # history (the existing #184 behavior, unchanged).
        assert result.response == "Synthesized answer"
        synthesis = _last_call_messages(harness.provider)
        _assert_no_dangling_tool_use(synthesis)


# =========================================================================
# T7 — no false positive (tool satisfied, and no-tools bot)
# =========================================================================


class TestNoFalsePositive:
    """A normal turn appends no synthetic pairing; a no-tools bot never
    reaches the pairing call (guard short-circuits)."""

    @pytest.mark.asyncio
    async def test_tool_satisfied_turn_no_synthetic_pairing(self) -> None:
        tool = EchoTool()

        async with await BotTestHarness.create(
            bot_config=_bot_config("simple"),
            main_responses=[
                tool_call_response("echo_tool", {"message": "go"}),
                text_response("The tool answered"),
                text_response("Plain follow-up"),
            ],
            tools=[tool],
        ) as harness:
            first = await harness.chat("Use the echo tool")
            assert first.response == "The tool answered"
            assert tool.call_count == 1
            second = await harness.chat("And now answer plainly")

        assert second.response == "Plain follow-up"
        replayed = _last_call_messages(harness.provider)
        _assert_no_dangling_tool_use(replayed)
        # The tool_use was genuinely answered — no synthetic pairing added.
        assert not _has_synthetic_pairing(replayed)

    @pytest.mark.asyncio
    async def test_no_tools_bot_no_synthetic_pairing(self) -> None:
        async with await BotTestHarness.create(
            bot_config=_bot_config("simple"),
            main_responses=[
                text_response("Hello there"),
                text_response("Goodbye"),
            ],
        ) as harness:
            await harness.chat("Hi")
            second = await harness.chat("Bye")

        assert second.response == "Goodbye"
        replayed = _last_call_messages(harness.provider)
        _assert_no_dangling_tool_use(replayed)
        assert not _has_synthetic_pairing(replayed)


# =========================================================================
# T8 — wizard unaffected (Layer A runs at the phased finalize and no-ops)
# =========================================================================


class TestWizardUnaffected:
    """A wizard turn reaches ``_finalize_turn`` (phased buffered path) with a
    tool registry present, so the Layer-A pairing call *runs* — and no-ops,
    because the wizard routes tool results through state
    (``add_observations=False``) rather than leaving an LLM-history orphan.

    Asserts on the *persisted* conversation history (what the next turn would
    replay): no synthetic pairing ``tool_result`` was injected.
    """

    @pytest.mark.asyncio
    async def test_wizard_turn_appends_no_synthetic_pairing(self) -> None:
        from dataknobs_bots.testing import WizardConfigBuilder

        config = (
            WizardConfigBuilder("t8")
            .stage("gather", is_start=True, prompt="Tell me your name.")
            .field("name", field_type="string", required=True)
            .transition("done", "data.get('name')")
            .stage("done", is_end=True, prompt="All done!")
            .build()
        )

        # Register a tool so ``self.tool_registry`` is truthy and the Layer-A
        # guard passes — proving the pairing call runs (and no-ops) on a
        # wizard, rather than being skipped by an empty registry.
        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Got it!"],
            extraction_results=[[{"name": "Alice"}]],
            tools=[EchoTool()],
        ) as harness:
            await harness.chat("My name is Alice")

            # Read the persisted history the next turn would replay.  The
            # cached manager's public ``get_history()`` returns the exact
            # ``list[LLMMessage]`` — a legitimate persisted-state read for a
            # state-invariant assertion (no public ``list[LLMMessage]``
            # history accessor exists on the bot).
            manager = harness.bot._conversation_managers[
                harness.context.conversation_id
            ]
            history = await manager.get_history()

        assert harness.wizard_stage == "done"
        assert not _has_synthetic_pairing(history)
        # And the persisted history is structurally well-formed.
        _assert_no_dangling_tool_use(history)


# =========================================================================
# Pure-core sanity (idempotency the chokepoint design relies on)
# =========================================================================


class TestPureCoreIdempotency:
    """The Layer-A call at ``_finalize_turn`` runs on *every* tool-bot turn,
    so its idempotency on an already-paired history is load-bearing."""

    def test_paired_history_is_noop(self) -> None:
        from dataknobs_llm.llm.base import ToolCall

        tc = ToolCall(name="echo_tool", parameters={"m": "a"}, id="X1")
        messages = [
            LLMMessage(role="user", content="hi"),
            LLMMessage(role="assistant", content="", tool_calls=[tc]),
            LLMMessage(
                role="tool", content="ok", name="echo_tool", tool_call_id="X1"
            ),
        ]
        assert pair_orphan_tool_calls(messages) == []
