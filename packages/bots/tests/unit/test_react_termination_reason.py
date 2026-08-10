"""Structured ReAct termination reason (always-on metadata + callback topic).

A ReAct tool loop can end for six distinct reasons, but historically *why* it
ended was log-only (and, with ``store_trace=True``, buried in the last trace
``status`` entry — a gated, positional dig). This surfaces the reason as
first-class, always-on ``reasoning_termination`` conversation metadata plus an
opt-in ``react:turn:end`` callback topic (EventBus-composable).

Reproduce-first: against HEAD before this feature, a terminated turn left NO
machine-readable termination reason in ``manager.metadata`` — the
``test_reproduce_*`` cases assert the reason IS present and FAIL on HEAD.

Real constructs only — ``BotTestHarness`` + ``EchoProvider`` for the phased
DynaBot path, a real ``ConversationManager`` + ``EchoProvider`` for the
monolithic ``generate`` path and the ToolsNotSupported case, and a real
``CallbackRegistry`` / ``InMemoryEventBus`` for the fan-out. No mocks.
"""

from __future__ import annotations

from typing import Any

from dataknobs_bots.reasoning import (
    REACT_TERMINATION_TOPIC,
    ReActReasoning,
    ReActTerminationReason,
)
from dataknobs_bots.testing import BotTestHarness
from dataknobs_common import Capability
from dataknobs_common.events import Event, InMemoryEventBus
from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_llm.conversations import (
    ConversationManager,
    DataknobsConversationStorage,
)
from dataknobs_llm.exceptions import ToolsNotSupportedError
from dataknobs_llm.llm import EchoProvider, LLMConfig
from dataknobs_llm.prompts import AsyncPromptBuilder
from dataknobs_llm.testing import text_response, tool_call_response
from dataknobs_llm.tools.base import Tool


class _EchoTool(Tool):
    """Minimal tool: echoes its input back so the ReAct loop runs."""

    def __init__(self) -> None:
        super().__init__(name="echo", description="Echoes the input back")

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }

    async def execute(self, **kwargs: Any) -> Any:
        kwargs.pop("_context", None)
        return {"echoed": kwargs.get("text", "")}


def _bot_config(reasoning_extra: dict[str, Any]) -> dict[str, Any]:
    return {
        "llm": {"provider": "echo", "model": "test"},
        "conversation_storage": {"backend": "memory"},
        "reasoning": {"strategy": "react", **reasoning_extra},
    }


# A trace left by a hypothetical earlier tool-using turn; the no-loop terminal
# branches must overwrite (not retain) this when store_trace is on.
_STALE_TRACE = [{"iteration": 1, "status": "prior_tool_turn"}]


async def _phased_termination(
    reasoning_extra: dict[str, Any],
    main_responses: list[Any],
) -> dict[str, Any] | None:
    """Drive one phased DynaBot turn; return the ``reasoning_termination`` dict."""
    async with await BotTestHarness.create(
        bot_config=_bot_config(reasoning_extra),
        main_responses=main_responses,
        tools=[_EchoTool()],
    ) as harness:
        await harness.chat("please do the multi-step task")
        conv = await harness.bot.get_conversation(harness.context.conversation_id)
        assert conv is not None
        return conv.metadata.get("reasoning_termination")


async def _make_manager(
    script: list[Any] | None,
) -> tuple[ConversationManager, EchoProvider]:
    """A real manager + EchoProvider for the monolithic / direct paths."""
    llm = EchoProvider(LLMConfig(provider="echo", model="echo-model"))
    if script is not None:
        llm.set_responses(script)
    storage = DataknobsConversationStorage(AsyncMemoryDatabase())
    manager = await ConversationManager.create(
        llm=llm,
        prompt_builder=AsyncPromptBuilder(library=None),
        storage=storage,
    )
    await manager.add_message(role="system", content="You are a tool user.")
    await manager.add_message(role="user", content="Do the multi-step task.")
    return manager, llm


class _CountingStorage(DataknobsConversationStorage):
    """Real storage that counts metadata persist round-trips.

    A spy over the real backend (not a mock): ``update_metadata`` is the sole
    persist point ReAct uses (via ``_persist_metadata``), so its call count is
    exactly the number of storage round-trips a terminal branch performs.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.persist_calls = 0

    async def update_metadata(self, *args: Any, **kwargs: Any) -> Any:
        self.persist_calls += 1
        return await super().update_metadata(*args, **kwargs)


# ---------------------------------------------------------------------------
# Reproduce-first: a terminated turn now carries a machine-readable reason.
# ---------------------------------------------------------------------------


class TestReproduce:
    async def test_reproduce_max_iterations_records_reason(self) -> None:
        """A max-iterations turn writes ``reasoning_termination`` (FAILS on HEAD).

        Before this feature the reason was log-only — nothing machine-readable
        in ``manager.metadata``. This asserts the always-on key is present with
        the correct reason and iteration count.
        """
        term = await _phased_termination(
            {"max_iterations": 2},
            [
                tool_call_response("echo", {"text": "a"}),
                tool_call_response("echo", {"text": "b"}),
                text_response("synth"),
            ],
        )
        assert term is not None, "reasoning_termination missing (the gap)"
        assert term["strategy"] == "react"
        assert term["reason"] == ReActTerminationReason.MAX_ITERATIONS.value
        assert term["iterations_used"] == 2


# ---------------------------------------------------------------------------
# One test per reason (all six), phased DynaBot path.
# ---------------------------------------------------------------------------


class TestPerReasonPhased:
    async def test_completed(self) -> None:
        term = await _phased_termination({"max_iterations": 5}, [text_response("Done immediately")])
        assert term is not None
        assert term["reason"] == ReActTerminationReason.COMPLETED.value
        assert term["iterations_used"] == 1

    async def test_max_iterations(self) -> None:
        term = await _phased_termination(
            {"max_iterations": 2},
            [
                tool_call_response("echo", {"text": "a"}),
                tool_call_response("echo", {"text": "b"}),
                text_response("synth"),
            ],
        )
        assert term is not None
        assert term["reason"] == ReActTerminationReason.MAX_ITERATIONS.value
        assert term["iterations_used"] == 2

    async def test_truncated_tool_call(self) -> None:
        term = await _phased_termination(
            {"max_iterations": 5},
            [
                tool_call_response("echo", {"text": "x"}, truncated=True),
                text_response("synth"),
            ],
        )
        assert term is not None
        assert term["reason"] == ReActTerminationReason.TRUNCATED_TOOL_CALL.value
        assert term["iterations_used"] == 1

    async def test_duplicate_tool_calls(self) -> None:
        term = await _phased_termination(
            {"max_iterations": 5},
            [
                tool_call_response("echo", {"text": "same"}),
                tool_call_response("echo", {"text": "same"}),
                text_response("synth"),
            ],
        )
        assert term is not None
        assert term["reason"] == ReActTerminationReason.DUPLICATE_TOOL_CALLS.value
        assert term["iterations_used"] == 2

    async def test_tools_not_supported(self) -> None:
        """ToolsNotSupportedError terminal branch records the reason.

        Uses a direct manager + a provider that raises (set_response_function)
        — the ToolsNotSupported branch previously had no trace entry at all.
        """
        manager, llm = await _make_manager(None)

        def _raise(_messages: Any) -> Any:
            raise ToolsNotSupportedError(model="test-model")

        llm.set_response_function(_raise)
        strategy = ReActReasoning()
        handle = await strategy.begin_turn(manager, llm, tools=[_EchoTool()])
        result = await strategy.process_input(handle)

        assert result.action == "tools_not_supported"
        term = manager.metadata.get("reasoning_termination")
        assert term is not None
        assert term["reason"] == ReActTerminationReason.TOOLS_NOT_SUPPORTED.value
        assert term["iterations_used"] == 1

    async def test_truncation_retry_exhausted(self) -> None:
        """Retry enabled + still-truncated → the more specific reason wins.

        Closes the FU5-B1 seam. The retry helper does not record; the caller's
        truncated terminal branch sees retry was enabled and records
        TRUNCATION_RETRY_EXHAUSTED (a single recorder, no double-write).
        """
        term = await _phased_termination(
            {"max_iterations": 5, "truncation_retry_max_tokens": 2048},
            [
                tool_call_response("echo", {"text": "x"}, truncated=True),
                tool_call_response("echo", {"text": "x"}, truncated=True),
                text_response("synth"),
            ],
        )
        assert term is not None
        assert term["reason"] == ReActTerminationReason.TRUNCATION_RETRY_EXHAUSTED.value
        assert term["iterations_used"] == 1


# ---------------------------------------------------------------------------
# Both paths (fix-scope pin): the monolithic ``generate`` fires the same helper.
# ---------------------------------------------------------------------------


class TestMonolithicGenerate:
    async def test_completed(self) -> None:
        manager, llm = await _make_manager([text_response("Done immediately")])
        strategy = ReActReasoning.from_config({"max_iterations": 5})
        await strategy.generate(manager, llm, tools=[_EchoTool()])
        term = manager.metadata.get("reasoning_termination")
        assert term is not None
        assert term["reason"] == ReActTerminationReason.COMPLETED.value
        assert term["iterations_used"] == 1

    async def test_max_iterations(self) -> None:
        manager, llm = await _make_manager(
            [
                tool_call_response("echo", {"text": "a"}),
                tool_call_response("echo", {"text": "b"}),
                text_response("synth"),
            ]
        )
        strategy = ReActReasoning.from_config({"max_iterations": 2})
        await strategy.generate(manager, llm, tools=[_EchoTool()])
        term = manager.metadata.get("reasoning_termination")
        assert term is not None
        assert term["reason"] == ReActTerminationReason.MAX_ITERATIONS.value
        assert term["iterations_used"] == 2

    async def test_truncation_retry_exhausted(self) -> None:
        """The reason-choice (retry enabled) fires in the monolithic path too."""
        manager, llm = await _make_manager(
            [
                tool_call_response("echo", {"text": "x"}, truncated=True),
                tool_call_response("echo", {"text": "x"}, truncated=True),
                text_response("synth"),
            ]
        )
        strategy = ReActReasoning.from_config(
            {"max_iterations": 5, "truncation_retry_max_tokens": 2048}
        )
        await strategy.generate(manager, llm, tools=[_EchoTool()])
        term = manager.metadata.get("reasoning_termination")
        assert term is not None
        assert term["reason"] == ReActTerminationReason.TRUNCATION_RETRY_EXHAUSTED.value

    async def test_duplicate_tool_calls(self) -> None:
        """DUPLICATE_TOOL_CALLS terminal branch fires in the monolithic path.

        Closes the both-paths pin: the duplicate-break was only asserted via
        the phased ``process_input`` path before.
        """
        manager, llm = await _make_manager(
            [
                tool_call_response("echo", {"text": "same"}),
                tool_call_response("echo", {"text": "same"}),
                text_response("synth"),
            ]
        )
        strategy = ReActReasoning.from_config({"max_iterations": 5})
        await strategy.generate(manager, llm, tools=[_EchoTool()])
        term = manager.metadata.get("reasoning_termination")
        assert term is not None
        assert term["reason"] == ReActTerminationReason.DUPLICATE_TOOL_CALLS.value
        assert term["iterations_used"] == 2

    async def test_truncated_tool_call(self) -> None:
        """Plain TRUNCATED_TOOL_CALL (retry disabled) via ``generate``.

        The retry-disabled reason-choice branch was only covered on the phased
        path; this drives it through the monolithic loop.
        """
        manager, llm = await _make_manager(
            [
                tool_call_response("echo", {"text": "x"}, truncated=True),
                text_response("synth"),
            ]
        )
        strategy = ReActReasoning.from_config({"max_iterations": 5})
        await strategy.generate(manager, llm, tools=[_EchoTool()])
        term = manager.metadata.get("reasoning_termination")
        assert term is not None
        assert term["reason"] == ReActTerminationReason.TRUNCATED_TOOL_CALL.value
        assert term["iterations_used"] == 1

    async def test_tools_not_supported(self) -> None:
        """TOOLS_NOT_SUPPORTED terminal branch fires in the monolithic path."""
        manager, llm = await _make_manager(None)

        def _raise(_messages: Any) -> Any:
            raise ToolsNotSupportedError(model="test-model")

        llm.set_response_function(_raise)
        strategy = ReActReasoning.from_config({"max_iterations": 5})
        await strategy.generate(manager, llm, tools=[_EchoTool()])
        term = manager.metadata.get("reasoning_termination")
        assert term is not None
        assert term["reason"] == ReActTerminationReason.TOOLS_NOT_SUPPORTED.value
        assert term["iterations_used"] == 1


# ---------------------------------------------------------------------------
# Always-on (D2): the reason is recorded even with store_trace=False (default).
# ---------------------------------------------------------------------------


class TestAlwaysOn:
    async def test_store_trace_false_still_records_reason(self) -> None:
        # store_trace omitted → defaults False.
        term = await _phased_termination({"max_iterations": 5}, [text_response("Done")])
        assert term is not None
        assert term["reason"] == ReActTerminationReason.COMPLETED.value


# ---------------------------------------------------------------------------
# Callback fan-out (D3): the topic fires once per terminated turn.
# ---------------------------------------------------------------------------


class TestCallbackFanout:
    async def test_registered_callback_fires_once(self) -> None:
        captured: list[dict[str, Any]] = []

        async with await BotTestHarness.create(
            bot_config=_bot_config({"max_iterations": 5}),
            main_responses=[text_response("Done")],
            tools=[_EchoTool()],
        ) as harness:
            strategy = harness.bot.reasoning_strategy
            assert isinstance(strategy, ReActReasoning)

            def _cb(payload: dict[str, Any]) -> None:
                captured.append(payload)

            strategy.termination_callbacks.register(REACT_TERMINATION_TOPIC, _cb)
            await harness.chat("do the task")

        assert len(captured) == 1
        assert captured[0]["reason"] == ReActTerminationReason.COMPLETED.value
        assert captured[0]["strategy"] == "react"

    async def test_event_bus_fanout(self) -> None:
        bus = InMemoryEventBus()
        await bus.connect()
        received: list[Event] = []

        async def _handler(event: Event) -> None:
            received.append(event)

        # The topic is already fully namespaced (``react:turn:end``), so no
        # topic_prefix is needed — the raw topic is published as-is.
        await bus.subscribe(REACT_TERMINATION_TOPIC, _handler)

        async with await BotTestHarness.create(
            bot_config=_bot_config({"max_iterations": 5}),
            main_responses=[text_response("Done")],
            tools=[_EchoTool()],
        ) as harness:
            strategy = harness.bot.reasoning_strategy
            assert isinstance(strategy, ReActReasoning)
            strategy.termination_callbacks.also_publish_to(bus)
            await harness.chat("do the task")

        assert len(received) == 1
        assert received[0].topic == REACT_TERMINATION_TOPIC
        assert received[0].payload["reason"] == ReActTerminationReason.COMPLETED.value

    async def test_no_callback_registered_is_zero_overhead(self) -> None:
        """No subscriber → no error, metadata path still works (lazy registry)."""
        term = await _phased_termination({"max_iterations": 5}, [text_response("Done")])
        assert term is not None
        assert term["reason"] == ReActTerminationReason.COMPLETED.value


# ---------------------------------------------------------------------------
# No-drift (D1/D4): metadata reason == the last trace status string.
# ---------------------------------------------------------------------------


class TestNoDrift:
    async def test_completed_trace_matches_metadata(self) -> None:
        async with await BotTestHarness.create(
            bot_config=_bot_config({"max_iterations": 5, "store_trace": True}),
            main_responses=[
                tool_call_response("echo", {"text": "a"}),
                text_response("Done"),
            ],
            tools=[_EchoTool()],
        ) as harness:
            await harness.chat("do the task")
            conv = await harness.bot.get_conversation(harness.context.conversation_id)
            trace = conv.metadata.get("reasoning_trace")
            term = conv.metadata.get("reasoning_termination")

        assert trace is not None and term is not None
        assert trace[-1]["status"] == term["reason"]
        assert term["reason"] == ReActTerminationReason.COMPLETED.value

    async def test_max_iterations_trace_matches_metadata(self) -> None:
        async with await BotTestHarness.create(
            bot_config=_bot_config({"max_iterations": 2, "store_trace": True}),
            main_responses=[
                tool_call_response("echo", {"text": "a"}),
                tool_call_response("echo", {"text": "b"}),
                text_response("synth"),
            ],
            tools=[_EchoTool()],
        ) as harness:
            await harness.chat("do the task")
            conv = await harness.bot.get_conversation(harness.context.conversation_id)
            trace = conv.metadata.get("reasoning_trace")
            term = conv.metadata.get("reasoning_termination")

        assert trace is not None and term is not None
        assert trace[-1]["status"] == term["reason"]
        assert term["reason"] == ReActTerminationReason.MAX_ITERATIONS.value


# ---------------------------------------------------------------------------
# Capability advertisement (D3-cap): CALLBACK_REGISTRY is machine-queryable.
# ---------------------------------------------------------------------------


class TestCapability:
    def test_supports_callback_registry(self) -> None:
        strategy = ReActReasoning()
        assert strategy.supports(Capability.CALLBACK_REGISTRY)
        assert Capability.CALLBACK_REGISTRY in ReActReasoning.supported_capabilities()


# ---------------------------------------------------------------------------
# No-tools fast path: a ReAct turn with no tools still records COMPLETED.
# ---------------------------------------------------------------------------


class TestNoToolsFastPath:
    """The always-on contract holds even when no tool loop runs.

    Reproduce-first: before recording was added to the two no-tools fast paths,
    a ReAct turn dispatched with no tools wrote NO ``reasoning_termination`` —
    a consumer reading the key unconditionally (the pattern the docs encourage)
    hit a missing key. Both fast paths now record COMPLETED with
    ``iterations_used=0`` (the loop body never executed).
    """

    async def test_phased_no_tools_records_completed(self) -> None:
        async with await BotTestHarness.create(
            bot_config=_bot_config({"max_iterations": 5}),
            main_responses=[text_response("Direct answer, no tools needed")],
            tools=[],
        ) as harness:
            await harness.chat("just answer directly")
            conv = await harness.bot.get_conversation(harness.context.conversation_id)
        assert conv is not None
        term = conv.metadata.get("reasoning_termination")
        assert term is not None, "no-tools turn wrote no termination reason (the gap)"
        assert term["reason"] == ReActTerminationReason.COMPLETED.value
        assert term["iterations_used"] == 0

    async def test_monolithic_no_tools_records_completed(self) -> None:
        manager, llm = await _make_manager([text_response("Direct answer")])
        strategy = ReActReasoning.from_config({"max_iterations": 5})
        result = await strategy.generate(manager, llm, tools=None)
        assert result is not None
        term = manager.metadata.get("reasoning_termination")
        assert term is not None, "no-tools generate() wrote no termination reason (the gap)"
        assert term["reason"] == ReActTerminationReason.COMPLETED.value
        assert term["iterations_used"] == 0


# ---------------------------------------------------------------------------
# Single persist: a terminal branch round-trips storage once, both keys.
# ---------------------------------------------------------------------------


class TestSinglePersist:
    """The recorder writes reason + trace in one storage round-trip.

    Reproduce-first: before folding the trace store into ``_record_termination``,
    a ``store_trace=True`` terminal branch persisted twice (reason, then trace)
    — this asserts exactly one flush, so the count is 2 on the pre-fix code.
    ``storage.update_metadata`` is the sole persist point ReAct uses.
    """

    async def _run_generate(self, *, store_trace: bool) -> tuple[_CountingStorage, Any]:
        db = AsyncMemoryDatabase()
        storage = _CountingStorage(db)
        llm = EchoProvider(LLMConfig(provider="echo", model="echo-model"))
        llm.set_responses([text_response("Done immediately")])
        manager = await ConversationManager.create(
            llm=llm,
            prompt_builder=AsyncPromptBuilder(library=None),
            storage=storage,
        )
        await manager.add_message(role="system", content="You are a tool user.")
        await manager.add_message(role="user", content="Do the task.")
        storage.persist_calls = 0  # ignore any setup writes
        strategy = ReActReasoning.from_config({"max_iterations": 5, "store_trace": store_trace})
        await strategy.generate(manager, llm, tools=[_EchoTool()])
        return storage, manager

    async def test_store_trace_true_persists_once(self) -> None:
        storage, manager = await self._run_generate(store_trace=True)
        assert storage.persist_calls == 1, (
            "store_trace=True must write both metadata keys in one round-trip"
        )
        # Both keys landed from the single flush.
        assert manager.metadata.get("reasoning_termination") is not None
        assert manager.metadata.get("reasoning_trace") is not None

    async def test_store_trace_false_persists_once(self) -> None:
        storage, manager = await self._run_generate(store_trace=False)
        assert storage.persist_calls == 1  # the always-on metadata write only
        assert manager.metadata.get("reasoning_termination") is not None
        assert manager.metadata.get("reasoning_trace") is None


# ---------------------------------------------------------------------------
# No-loop terminal branches (no-tools, tools-not-supported) write a FRESH trace.
# ---------------------------------------------------------------------------


class TestNoLoopTraceStore:
    """store_trace=True: a branch that runs no tool loop still refreshes the trace.

    Reproduce-first: before the fix, the no-tools and tools-not-supported
    terminal branches recorded ``reasoning_termination`` but never touched
    ``reasoning_trace`` — so an earlier tool-using turn's trace survived. A
    consumer reading both keys together saw ``reasoning_termination`` reporting
    "completed this turn" while ``reasoning_trace`` still showed the *previous*
    turn's tool steps. We pre-seed a stale trace to stand in for that earlier
    turn and assert each of the four branches overwrites it with a fresh,
    status-only trace (mirroring the MAX_ITERATIONS status-only append). On the
    pre-fix code the stale trace survives and every assertion FAILS.
    """

    async def test_phased_no_tools_overwrites_stale_trace(self) -> None:
        manager, llm = await _make_manager([text_response("Direct answer")])
        manager.update_metadata({"reasoning_trace": list(_STALE_TRACE)})
        strategy = ReActReasoning.from_config({"max_iterations": 5, "store_trace": True})
        handle = await strategy.begin_turn(manager, llm, tools=[])
        assert handle.early_response is not None
        trace = manager.metadata.get("reasoning_trace")
        assert trace == [{"status": ReActTerminationReason.COMPLETED.value}], (
            "no-tools turn left a stale reasoning_trace (the gap)"
        )

    async def test_monolithic_no_tools_overwrites_stale_trace(self) -> None:
        manager, llm = await _make_manager([text_response("Direct answer")])
        manager.update_metadata({"reasoning_trace": list(_STALE_TRACE)})
        strategy = ReActReasoning.from_config({"max_iterations": 5, "store_trace": True})
        await strategy.generate(manager, llm, tools=None)
        trace = manager.metadata.get("reasoning_trace")
        assert trace == [{"status": ReActTerminationReason.COMPLETED.value}]

    async def test_phased_tools_not_supported_overwrites_stale_trace(
        self,
    ) -> None:
        manager, llm = await _make_manager(None)

        def _raise(_messages: Any) -> Any:
            raise ToolsNotSupportedError(model="test-model")

        llm.set_response_function(_raise)
        manager.update_metadata({"reasoning_trace": list(_STALE_TRACE)})
        strategy = ReActReasoning.from_config({"store_trace": True})
        handle = await strategy.begin_turn(manager, llm, tools=[_EchoTool()])
        result = await strategy.process_input(handle)
        assert result.action == "tools_not_supported"
        trace = manager.metadata.get("reasoning_trace")
        assert trace == [{"status": ReActTerminationReason.TOOLS_NOT_SUPPORTED.value}]

    async def test_monolithic_tools_not_supported_overwrites_stale_trace(
        self,
    ) -> None:
        manager, llm = await _make_manager(None)

        def _raise(_messages: Any) -> Any:
            raise ToolsNotSupportedError(model="test-model")

        llm.set_response_function(_raise)
        manager.update_metadata({"reasoning_trace": list(_STALE_TRACE)})
        strategy = ReActReasoning.from_config({"store_trace": True})
        await strategy.generate(manager, llm, tools=[_EchoTool()])
        trace = manager.metadata.get("reasoning_trace")
        assert trace == [{"status": ReActTerminationReason.TOOLS_NOT_SUPPORTED.value}]
