"""Reproducing tests: what a wizard tool may read and write.

A ``ContextAwareTool`` running inside a wizard turn reaches wizard state
through ``ToolExecutionContext``. These tests observe the two things that
seam is supposed to guarantee and does not:

* a write the tool makes is still there after the turn, and
* a read the tool makes sees values extracted earlier in the *same* turn.

Every assertion here observes an effect **across a boundary** -- the
persisted wizard data after the turn, or the value the shipped extraction
pipeline put in state before the tool ran. None asserts on an object the
test constructed.
"""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_llm.tools.context import ToolExecutionContext
from dataknobs_llm.tools.context_aware import ContextAwareTool

from dataknobs_bots.testing import BotTestHarness, WizardConfigBuilder


class NoteRecordingTool(ContextAwareTool):
    """Writes into wizard data through the public accessor, and records reads.

    Deliberately writes a key that no ``tool_result_mapping`` maps, so the
    assertion is about the *write channel* rather than about the mapping of
    the tool's return value -- those are two different paths into state and
    only one of them is under test.
    """

    def __init__(self) -> None:
        super().__init__(
            name="record_note",
            description="Record a note in wizard state",
        )
        self.reads: list[dict[str, Any] | None] = []

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"note": {"type": "string"}},
        }

    async def execute_with_context(
        self,
        context: ToolExecutionContext,
        note: str = "",
        **_kwargs: Any,
    ) -> dict[str, Any]:
        data = context.wizard_data()
        self.reads.append(None if data is None else dict(data))
        if data is None:
            return {"recorded": False}
        data.setdefault("_notes", []).append(note or "unnamed")
        return {"recorded": True}


def _wizard_with_tool_stage() -> dict[str, Any]:
    """A two-stage wizard whose first stage runs the tool after extraction."""
    return (
        WizardConfigBuilder("tool-state")
        .stage(
            "gather",
            is_start=True,
            prompt="What product are you looking for?",
            tool_result_mapping=[
                {
                    "tool": "record_note",
                    "params": {"note": "product_name"},
                    "mapping": {"recorded": "_recorded"},
                },
            ],
        )
        .field("product_name", field_type="string", required=True)
        .transition("done", "has('_confirmed')")
        .stage("done", is_end=True, prompt="All set.")
        .build()
    )


@pytest.mark.asyncio
async def test_tool_write_to_wizard_data_survives_the_turn() -> None:
    """A tool's write to wizard data is still there after the turn ends.

    The tool appends to ``_notes`` through ``context.wizard_data()``. The
    assertion reads the wizard's own data after the turn, so it observes
    the write across the save boundary rather than in the dict the tool
    was handed.
    """
    tool = NoteRecordingTool()

    async with await BotTestHarness.create(
        wizard_config=_wizard_with_tool_stage(),
        main_responses=["Looking that up...", "Anything else?"],
        extraction_results=[[{"product_name": "Widget"}]],
        tools=[tool],
    ) as harness:
        await harness.chat("I want a Widget")

        assert harness.wizard_data.get("_notes") == ["Widget"]


@pytest.mark.asyncio
async def test_tool_sees_this_turns_extraction() -> None:
    """A tool reads values extracted earlier in the same turn.

    ``params`` on the tool_result_mapping entry reads live wizard state, so
    the tool is *called* with this turn's value. What it reads back out of
    the context must agree with that -- one call must not carry two
    different answers to the same question.
    """
    tool = NoteRecordingTool()

    async with await BotTestHarness.create(
        wizard_config=_wizard_with_tool_stage(),
        main_responses=["Looking that up...", "Anything else?"],
        extraction_results=[[{"product_name": "Widget"}]],
        tools=[tool],
    ) as harness:
        await harness.chat("I want a Widget")

        assert tool.reads, "the tool never ran"
        assert tool.reads[0] is not None, "the tool got no wizard state at all"
        assert tool.reads[0].get("product_name") == "Widget"


@pytest.mark.asyncio
async def test_tool_write_survives_a_later_turn() -> None:
    """The write is still discarded once wizard state exists to write into.

    The tool writes on both turns, so the assertion is about accumulation
    across a save boundary rather than a single write surviving: turn 1's
    note has to still be there after turn 2 rewrote the wizard's state
    from its own copy, which is the overwrite this item is about.
    """
    tool = NoteRecordingTool()

    async with await BotTestHarness.create(
        wizard_config=_wizard_with_tool_stage(),
        main_responses=["One moment...", "Got it.", "One moment...", "Got it."],
        extraction_results=[
            [{"product_name": "Widget"}],
            [{"product_name": "Gadget"}],
        ],
        tools=[tool],
    ) as harness:
        await harness.chat("I want a Widget")
        await harness.chat("Actually, a Gadget")

        assert len(tool.reads) == 2, "the tool did not run on both turns"
        assert tool.reads[1] is not None, "no wizard state on the second turn"
        assert harness.wizard_data.get("_notes") == ["Widget", "Gadget"]


@pytest.mark.asyncio
async def test_tool_read_is_not_a_turn_behind() -> None:
    """On turn 2 the tool must see turn 2's extraction, not turn 1's.

    This is the half a consumer is most likely to misdiagnose: the tool's
    arguments are right, because ``params`` reads live state, while the
    context it reads carries the previous turn's values. One call, two
    channels, one turn apart.
    """
    tool = NoteRecordingTool()

    async with await BotTestHarness.create(
        wizard_config=_wizard_with_tool_stage(),
        main_responses=["One moment...", "Got it.", "One moment...", "Got it."],
        extraction_results=[
            [{"product_name": "Widget"}],
            [{"product_name": "Gadget"}],
        ],
        tools=[tool],
    ) as harness:
        await harness.chat("I want a Widget")
        await harness.chat("Actually, a Gadget")

        assert len(tool.reads) == 2, "the tool did not run on both turns"
        assert tool.reads[1] is not None, "no wizard state on the second turn"
        assert tool.reads[1].get("product_name") == "Gadget"


def _wizard_pushing_a_subflow() -> dict[str, Any]:
    """A wizard that enters a subflow, whose stage then runs the tool."""
    subflow = (
        WizardConfigBuilder("details")
        .stage(
            "specifics",
            is_start=True,
            is_end=True,
            prompt="Any specifics?",
            tool_result_mapping=[
                {
                    "tool": "record_note",
                    "params": {"note": "product_name"},
                    "mapping": {"recorded": "_recorded"},
                },
            ],
        )
        .field("detail", field_type="string", required=False)
        .build()
    )

    return (
        WizardConfigBuilder("tool-state-subflow")
        .subflow("details", subflow)
        .stage("gather", is_start=True, prompt="What product?")
        .field("product_name", field_type="string", required=True)
        .transition(
            "wrapup",
            "has('product_name')",
            subflow_network="details",
            return_stage="wrapup",
        )
        .stage("wrapup", is_end=True, prompt="All set.")
        .build()
    )


@pytest.mark.asyncio
async def test_tool_write_lands_in_the_subflows_data_after_a_push() -> None:
    """After a subflow push, the tool writes into the subflow's data.

    A push replaces the collected data, and it does so through
    ``WizardState.replace_data`` so the dict the turn's channel holds is
    emptied and refilled rather than swapped out. Measured, this config
    pushes at the end of the turn that triggers it and runs the tool on
    the next one, so what this pins is the across-turn half.

    The mid-turn ordering -- a replacement between the publish and the
    tool call -- is covered by
    ``test_tool_write_survives_an_auto_restart_earlier_in_the_turn``,
    which reaches it through ``begin_turn``'s auto-restart arm.
    """
    tool = NoteRecordingTool()

    async with await BotTestHarness.create(
        wizard_config=_wizard_pushing_a_subflow(),
        main_responses=["One moment...", "Got it.", "One moment...", "Got it."],
        extraction_results=[
            [{"product_name": "Widget"}],
            [{"detail": "blue"}],
        ],
        tools=[tool],
    ) as harness:
        await harness.chat("I want a Widget")
        await harness.chat("Make it blue")

        assert tool.reads, "the tool never ran"
        assert harness.wizard_data.get("_notes"), (
            "the tool's write did not reach the data the wizard kept: "
            f"wizard_data={harness.wizard_data}"
        )


def _wizard_that_completes_then_restarts() -> dict[str, Any]:
    """A wizard that completes in one turn, so the next turn auto-restarts.

    ``allow_post_completion_edits`` defaults to off, and a completed
    wizard receiving another message therefore takes ``begin_turn``'s
    auto-restart arm -- which resets ``WizardState.data`` *after* the
    turn's live channel has been published and *before* any tool runs.
    """
    return (
        WizardConfigBuilder("tool-state-restart")
        .stage(
            "gather",
            is_start=True,
            prompt="What product are you looking for?",
            tool_result_mapping=[
                {
                    "tool": "record_note",
                    "params": {"note": "product_name"},
                    "mapping": {"recorded": "_recorded"},
                },
            ],
        )
        .field("product_name", field_type="string", required=True)
        .transition("done", "has('product_name')")
        .stage("done", is_end=True, prompt="All set.")
        .build()
    )


@pytest.mark.asyncio
async def test_tool_write_survives_an_auto_restart_earlier_in_the_turn() -> None:
    """A restart between the publish and the tool call must not strand the tool.

    This is the mid-turn rebinding case. ``begin_turn`` publishes the live
    channel, then auto-restarts because the wizard completed on the
    previous turn and amendments are off, then falls through so the user's
    message is processed by the fresh first stage -- tool call included.

    If the restart *rebinds* ``WizardState.data`` rather than clearing it,
    the channel published moments earlier still points at the abandoned
    pre-restart dict, so the tool writes somewhere nothing will ever read
    and reports success.
    """
    tool = NoteRecordingTool()

    async with await BotTestHarness.create(
        wizard_config=_wizard_that_completes_then_restarts(),
        main_responses=["One moment...", "Got it.", "One moment...", "Got it."],
        extraction_results=[
            [{"product_name": "Widget"}],
            [{"product_name": "Gizmo"}],
        ],
        tools=[tool],
    ) as harness:
        await harness.chat("I want a Widget")
        await harness.chat("Now I want a Gizmo")

        assert len(tool.reads) == 2, f"the tool did not run on both turns: {tool.reads}"
        assert harness.wizard_data.get("_notes") == ["Gizmo"], (
            "the post-restart write did not reach the data the wizard kept: "
            f"wizard_data={harness.wizard_data}"
        )


@pytest.mark.asyncio
async def test_tool_does_not_read_pre_restart_data() -> None:
    """After an auto-restart the tool must see the fresh state, not the old one.

    The read half of the same rebinding. A channel left pointing at the
    pre-restart dict hands the tool the *previous* run's answers, which is
    worse than handing it nothing: the values look plausible and belong to
    a wizard run the user has already finished.
    """
    tool = NoteRecordingTool()

    async with await BotTestHarness.create(
        wizard_config=_wizard_that_completes_then_restarts(),
        main_responses=["One moment...", "Got it.", "One moment...", "Got it."],
        extraction_results=[
            [{"product_name": "Widget"}],
            [{"product_name": "Gizmo"}],
        ],
        tools=[tool],
    ) as harness:
        await harness.chat("I want a Widget")
        await harness.chat("Now I want a Gizmo")

        assert len(tool.reads) == 2, f"the tool did not run on both turns: {tool.reads}"
        second = tool.reads[1]
        assert second is not None, "no wizard state on the second turn"
        assert second.get("_notes") is None, (
            "the tool read the abandoned pre-restart dict, which still carried "
            f"the first run's notes: {second}"
        )
        assert second.get("product_name") == "Gizmo"


# ---------------------------------------------------------------------------
# The channel's lifetime: published for a turn, and only for a turn
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_abandoning_a_stream_still_unpublishes_the_channel() -> None:
    """A turn that ends without finalizing must not leave the channel set.

    The channel aliases the strategy's own ``WizardState.data``, and the
    manager holding it is cached across turns. Leaving it published hands
    the *next* turn's tools a dict belonging to a turn that is over.

    Abandoning a stream reaches this without any error: ``stream_chat``
    finalizes only when the stream was fully consumed, so a caller that
    breaks out -- a disconnected client, a UI cancelling a response --
    skips the teardown that clears the channel.
    """
    tool = NoteRecordingTool()

    async with await BotTestHarness.create(
        wizard_config=_wizard_with_tool_stage(),
        main_responses=["One moment...", "Got it."],
        extraction_results=[[{"product_name": "Widget"}]],
        tools=[tool],
    ) as harness:
        stream = harness.bot.stream_chat("I want a Widget", harness.context)

        published = None
        async for _chunk in stream:
            manager = harness.bot.get_conversation_manager(harness.context.conversation_id)
            published = manager.state.live_wizard_state
            break
        await stream.aclose()

        assert published is not None, (
            "the channel was never published during the stream, so this test "
            "would pass whether or not teardown clears it"
        )

        manager = harness.bot.get_conversation_manager(harness.context.conversation_id)
        assert manager.state.live_wizard_state is None, (
            "the abandoned stream left the live channel published, so the next "
            "turn's tools would read and write an abandoned turn's data"
        )
