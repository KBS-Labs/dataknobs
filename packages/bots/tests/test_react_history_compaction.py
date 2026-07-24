"""Tests for opt-in in-loop history compaction in the ReAct strategy.

A long ReAct tool loop grows conversation history unboundedly and can trip a
vendor input-context overflow. ``HistoryCompactionConfig`` opts a bot into
bounding that growth: proactively (estimate the path's tokens and compact when
over budget) and reactively (a caught ``ContextLengthExceededError`` compacts
once and retries). Both loop sites — the phased ``process_input`` path DynaBot
drives and the monolithic ``generate`` — share one helper (D5).

Reproduce-first: against HEAD before this feature a long loop kept every
iteration in the sent history (no bound). Real constructs only — ``BotTestHarness``
+ ``EchoProvider`` for the phased/e2e paths; a real ``ConversationManager`` +
``EchoProvider`` for the direct-loop and reactive-backstop unit paths (the
overflow is a real ``ContextLengthExceededError``, not a mock).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from dataknobs_bots.reasoning import ReActReasoning
from dataknobs_bots.testing import BotTestHarness
from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_llm.conversations import (
    ConversationManager,
    DataknobsConversationStorage,
)
from dataknobs_llm.exceptions import ContextLengthExceededError
from dataknobs_llm.llm import EchoProvider, LLMConfig
from dataknobs_llm.llm.base import ToolCall
from dataknobs_llm.prompts import AsyncPromptBuilder
from dataknobs_llm.testing import text_response, tool_call_response
from dataknobs_llm.tools.base import Tool

pytestmark = pytest.mark.asyncio


class _EchoTool(Tool):
    """Minimal tool: records its calls, echoes its input back."""

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


def _react_config(compaction: dict[str, Any] | None) -> dict[str, Any]:
    reasoning: dict[str, Any] = {"strategy": "react", "max_iterations": 8}
    if compaction is not None:
        reasoning["history_compaction"] = compaction
    return {
        "llm": {"provider": "echo", "model": "test"},
        "conversation_storage": {"backend": "memory"},
        "reasoning": reasoning,
    }


def _tool_loop_script(n_iterations: int) -> list[Any]:
    """n tool-call turns (drives the loop) then a final text answer."""
    script = [
        tool_call_response("echo", {"text": f"step {i}"})
        for i in range(n_iterations)
    ]
    script.append(text_response("All done."))
    return script


async def _history_after_run(compaction: dict[str, Any] | None) -> list[Any]:
    """Run a 5-iteration ReAct loop via the phased DynaBot path; return history."""
    async with await BotTestHarness.create(
        bot_config=_react_config(compaction),
        main_responses=_tool_loop_script(5),
        tools=[_EchoTool()],
    ) as harness:
        await harness.chat("please do the multi-step task")
        manager = harness.bot.get_conversation_manager(
            harness.context.conversation_id
        )
        return await manager.get_history()


class TestPhasedProactiveCompaction:
    async def test_default_off_keeps_full_history(self) -> None:
        """No compaction config → every iteration retained (byte-identical)."""
        history = await _history_after_run(None)
        # system + user + 5 pairs (assistant tool_use + tool obs) + final text.
        tool_msgs = [m for m in history if m.role == "tool"]
        assert len(tool_msgs) == 5

    async def test_proactive_window_bounds_history(self) -> None:
        """A low absolute budget compacts the loop; far fewer tool obs survive."""
        compacted = await _history_after_run(
            {
                "enabled": True,
                # EchoProvider resolves no input ceiling → absolute fallback.
                "history_token_budget": 30,
                "keep_recent_iterations": 1,
                "strategy": "window",
            }
        )
        baseline = await _history_after_run(None)
        compacted_tool_msgs = [m for m in compacted if m.role == "tool"]
        baseline_tool_msgs = [m for m in baseline if m.role == "tool"]
        assert len(compacted_tool_msgs) < len(baseline_tool_msgs)
        # No dangling tool_use survives compaction (the pairing invariant).
        _assert_no_dangling_tool_use(compacted)

    async def test_summarize_inserts_summary_node(self) -> None:
        """The summarize strategy folds dropped iterations into a summary node."""
        compacted = await _history_after_run(
            {
                "enabled": True,
                "history_token_budget": 30,
                "keep_recent_iterations": 1,
                "strategy": "summarize",
                # A dedicated summary provider (built + owned by the strategy),
                # so summary calls don't consume the main scripted queue.
                "summary_llm": {"provider": "echo", "model": "summary"},
            }
        )
        summary_nodes = [
            m
            for m in compacted
            if m.role == "system" and "step 0" in (m.content or "")
        ]
        assert summary_nodes, "expected a compaction summary node"
        _assert_no_dangling_tool_use(compacted)


# ---------------------------------------------------------------------------
# Monolithic ``generate`` site (the other loop path, D5) + reactive backstop
# ---------------------------------------------------------------------------


async def _make_manager(script: list[Any]) -> tuple[ConversationManager, EchoProvider]:
    llm = EchoProvider(LLMConfig(provider="echo", model="echo-model"))
    llm.set_responses(script)
    storage = DataknobsConversationStorage(AsyncMemoryDatabase())
    manager = await ConversationManager.create(
        llm=llm, prompt_builder=AsyncPromptBuilder(library=None), storage=storage
    )
    await manager.add_message(role="system", content="You are a tool user.")
    await manager.add_message(role="user", content="Do the multi-step task.")
    return manager, llm


def _assert_no_dangling_tool_use(messages) -> None:
    open_ids: set[str] = set()
    for msg in messages:
        if msg.role == "assistant" and msg.tool_calls:
            open_ids.update(tc.id for tc in msg.tool_calls if tc.id)
        elif msg.role == "tool" and msg.tool_call_id:
            open_ids.discard(msg.tool_call_id)
    assert not open_ids, f"dangling tool_use ids: {open_ids}"


class TestMonolithicGenerateCompaction:
    async def test_proactive_window_bounds_history(self) -> None:
        """The monolithic ``generate`` site compacts via the same helper."""
        manager, llm = await _make_manager(_tool_loop_script(5))
        strategy = ReActReasoning.from_config(
            {
                "max_iterations": 8,
                "history_compaction": {
                    "enabled": True,
                    "history_token_budget": 30,
                    "keep_recent_iterations": 1,
                    "strategy": "window",
                },
            }
        )
        await strategy.generate(manager, llm, tools=[_EchoTool()])
        history = await manager.get_history()
        tool_msgs = [m for m in history if m.role == "tool"]
        assert len(tool_msgs) < 5  # compaction bounded the loop
        _assert_no_dangling_tool_use(history)


async def _append_iteration(manager: ConversationManager, i: int) -> None:
    node = await manager.add_message(role="assistant", content=f"step {i}")
    node.message.tool_calls = [ToolCall(name="t", parameters={}, id=f"c{i}")]
    await manager.add_message(role="tool", content=f"obs {i}", tool_call_id=f"c{i}")


class TestReactiveBackstop:
    async def test_context_overflow_compacts_and_retries(self) -> None:
        """A context overflow compacts once and retries the completion."""
        manager, llm = await _make_manager([])
        for i in range(5):
            await _append_iteration(manager, i)
        before = len(await manager.get_history())

        strategy = ReActReasoning.from_config(
            {
                "history_compaction": {
                    "enabled": True,
                    "history_token_budget": 1_000_000,  # proactive won't fire
                    "keep_recent_iterations": 1,
                    "strategy": "window",
                }
            }
        )

        calls = {"n": 0}

        async def complete() -> str:
            calls["n"] += 1
            if calls["n"] == 1:
                raise ContextLengthExceededError("input too long")
            return "recovered"

        result = await strategy._complete_with_reactive_compaction(
            manager, llm, complete
        )

        assert result == "recovered"
        assert calls["n"] == 2  # exactly one retry
        assert len(await manager.get_history()) < before  # compaction happened

    async def test_overflow_propagates_when_disabled(self) -> None:
        """With compaction disabled the overflow propagates unchanged."""
        manager, llm = await _make_manager([])
        strategy = ReActReasoning.from_config({})  # no compaction

        async def complete() -> str:
            raise ContextLengthExceededError("input too long")

        with pytest.raises(ContextLengthExceededError):
            await strategy._complete_with_reactive_compaction(
                manager, llm, complete
            )


# ---------------------------------------------------------------------------
# Proactive budget resolution (input-ceiling * fraction, capped by the budget)
# ---------------------------------------------------------------------------


def _echo_with_input_ceiling(ceiling: int) -> EchoProvider:
    """A real EchoProvider whose resolved constraints report an input ceiling.

    The ``constraints`` override rides the real ``get_constraints`` template
    method (``_resolve_constraints`` overlays it per field), so this exercises
    the actual ceiling-resolution path — not a stubbed value.
    """
    return EchoProvider(
        LLMConfig(
            provider="echo",
            model="echo-model",
            constraints={"max_input_tokens": ceiling},
        )
    )


class TestBudgetResolution:
    @staticmethod
    def _strategy(compaction: dict[str, Any]) -> ReActReasoning:
        return ReActReasoning.from_config({"history_compaction": compaction})

    async def test_absolute_budget_caps_resolved_ceiling(self) -> None:
        """``history_token_budget`` caps ceiling*fraction, not only backstops it.

        Reproduce-first: on HEAD a resolved ceiling won the budget outright
        (``int(1_000_000 * 0.75) == 750_000``), ignoring an explicit
        ``history_token_budget``. A consumer whose *effective* window is smaller
        than the model's *advertised maximum* could therefore never make
        proactive compaction fire before the reactive backstop. The configured
        budget must now cap the resolved one.
        """
        llm = _echo_with_input_ceiling(1_000_000)
        strategy = self._strategy(
            {
                "enabled": True,
                "budget_fraction": 0.75,
                "history_token_budget": 50_000,
            }
        )
        assert strategy._resolve_history_budget(llm) == 50_000

    async def test_resolved_ceiling_used_when_no_cap(self) -> None:
        """Without an absolute budget the resolved ceiling*fraction is used."""
        llm = _echo_with_input_ceiling(1_000_000)
        strategy = self._strategy({"enabled": True, "budget_fraction": 0.75})
        assert strategy._resolve_history_budget(llm) == 750_000

    async def test_absolute_budget_sole_threshold_without_ceiling(self) -> None:
        """No resolved ceiling (plain EchoProvider) → budget used as-is."""
        llm = EchoProvider(LLMConfig(provider="echo", model="echo-model"))
        strategy = self._strategy(
            {"enabled": True, "history_token_budget": 50_000}
        )
        assert strategy._resolve_history_budget(llm) == 50_000


# ---------------------------------------------------------------------------
# Concurrent lazy strategy build (double-checked lock, no leaked provider)
# ---------------------------------------------------------------------------


class TestConcurrentStrategyBuild:
    async def test_concurrent_first_compaction_builds_one_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two concurrent first-compactions build exactly one summary provider.

        Reproduce-first: before the double-checked lock, two turns hitting their
        first ``summarize`` compaction concurrently both passed the
        ``_compaction_strategy is None`` check (the interleave opens at
        ``await initialize()``), so both built + opened a dedicated provider and
        one was silently overwritten and leaked. The lock makes the build happen
        exactly once.

        Real constructs: the summary provider is a real ``EchoProvider``
        subclass whose ``initialize`` adds a genuine suspension point (the base
        ``EchoProvider.initialize`` never yields, so the race can't otherwise be
        driven deterministically). The factory seam is the real module-level
        ``create_llm_provider`` the strategy calls.
        """
        built: list[EchoProvider] = []

        class _SlowInitEcho(EchoProvider):
            async def initialize(self) -> None:
                await asyncio.sleep(0)  # real suspension → force the interleave
                await super().initialize()

        def _counting_factory(_cfg: Any) -> EchoProvider:
            provider = _SlowInitEcho(
                LLMConfig(provider="echo", model="summary")
            )
            built.append(provider)
            return provider

        monkeypatch.setattr(
            "dataknobs_bots.reasoning.react.create_llm_provider",
            _counting_factory,
        )

        _manager, llm = await _make_manager([])
        strategy = ReActReasoning.from_config(
            {
                "history_compaction": {
                    "enabled": True,
                    "strategy": "summarize",
                    "summary_llm": {"provider": "echo", "model": "summary"},
                    "keep_recent_iterations": 1,
                }
            }
        )
        first, second = await asyncio.gather(
            strategy._get_compaction_strategy(llm),
            strategy._get_compaction_strategy(llm),
        )
        assert len(built) == 1, (
            f"expected one summary provider, built {len(built)}"
        )
        assert first is second  # both got the one cached strategy
        await strategy.close()
