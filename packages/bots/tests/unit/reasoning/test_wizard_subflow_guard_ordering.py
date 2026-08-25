"""When a subflow guard is evaluated, and by which of the two transition paths.

A subflow guard is a transition condition. Every *other* transition
condition is evaluated after the stage's pre-transition preparation --
``derive:`` blocks and ``routing_transforms`` -- because that preparation
exists to compute what conditions read. The subflow guard was evaluated
before it, so a guard reading a key its own stage prepares could not fire
on the turn the key was written; it fired on the next one, against a
message the user wrote in answer to a prompt they never saw.

The second half is the reason the fix is a shared step rather than a moved
statement: ``advance()``, the non-conversational API, never asked the
question at all. It could leave a subflow it could not enter.

Both halves are pinned here, and so is the constraint the original
ordering was protecting: a push must still pre-empt the FSM step, because
a subflow transition compiles to a self-loop arc that would otherwise
consume the turn.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import pytest

from dataknobs_bots.reasoning.wizard_types import WizardState
from dataknobs_bots.testing import BotTestHarness, WizardConfigBuilder

# ---------------------------------------------------------------------------
# Fixtures: one wizard whose subflow guard reads a key its own stage prepares
# ---------------------------------------------------------------------------

_DETAIL_SUBFLOW: dict[str, Any] = {
    "name": "detail",
    "stages": [
        {
            "name": "sub_start",
            "is_start": True,
            "prompt": "Which detail?",
            "response_template": "Entering detail.",
            "confirm_first_render": False,
            "schema": {
                "type": "object",
                "properties": {"detail": {"type": "string"}},
            },
            "transitions": [{"target": "sub_done", "condition": "has('detail')"}],
        },
        {
            "name": "sub_done",
            "is_end": True,
            "prompt": "Detail captured.",
            "response_template": "Detail captured.",
        },
    ],
}


def flag_route(data: dict[str, Any]) -> dict[str, Any]:
    """A routing transform: the shape a real one has, minus the classifier."""
    data["_route_flag"] = True
    return data


def _guard_config(
    *,
    guard_key: str,
    routing_transforms: list[str] | None = None,
    derive: dict[str, Any] | None = None,
    confirm_on_new_data: bool = False,
    also_unconditional: bool = False,
) -> dict[str, Any]:
    """A wizard whose ``gather`` stage guards a subflow on ``guard_key``.

    ``guard_key`` is written by whichever preparation step the caller
    turns on -- a routing transform or the transition's own ``derive:``
    block. Both run inside ``_prepare_transition``; neither was visible
    to the guard.
    """
    builder = WizardConfigBuilder("subflow-guard-ordering")
    builder.stage(
        "gather",
        is_start=True,
        prompt="Tell me your name.",
        response_template="Noted.",
        confirm_first_render=False,
        confirm_on_new_data=confirm_on_new_data or None,
        routing_transforms=routing_transforms,
    )
    builder.field("name", field_type="string", required=True)
    if also_unconditional:
        # Declared FIRST so the FSM step would take it if the push were
        # evaluated after the step rather than before it.
        builder.transition("wrap")
    builder.transition(
        "wrap",
        condition=f"data.get('{guard_key}')",
        derive=derive,
        subflow_network="detail",
        return_stage="wrap",
    )
    builder.stage("wrap", is_end=True, prompt="All done.", response_template="All done.")
    builder.subflow("detail", _DETAIL_SUBFLOW)
    return builder.build()


# ---------------------------------------------------------------------------
# 1-2. The guard sees what its own stage prepared, on the turn it is prepared
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_subflow_guard_sees_a_routing_transform_key_on_the_turn_it_is_written() -> None:
    """A routing transform's key reaches the guard on turn 1, not turn 2."""
    async with await BotTestHarness.create(
        wizard_config=_guard_config(guard_key="_route_flag", routing_transforms=["flag_route"]),
        custom_functions={"flag_route": flag_route},
        main_responses=["Entering detail.", "Entering detail."],
        extraction_results=[[{"name": "Alice"}], [{"name": "Alice"}]],
    ) as harness:
        await harness.chat("I'm Alice")

        assert harness.wizard_stage == "sub_start", (
            "the guard reads a key this stage's own routing transform writes; "
            "it must not be one turn late"
        )


@pytest.mark.asyncio
async def test_a_subflow_guard_sees_a_derived_key_on_the_turn_it_is_written() -> None:
    """``derive:`` is the second writer, with the same defect.

    The brief names only ``routing_transforms``. Both run in
    ``_prepare_transition``, so a fix scoped to one would leave the other
    broken for subflow guards while fixing it for ordinary ones -- the
    same asymmetry the item exists to remove.
    """
    async with await BotTestHarness.create(
        wizard_config=_guard_config(
            guard_key="_derived_flag",
            derive={"_derived_flag": True},
        ),
        main_responses=["Entering detail.", "Entering detail."],
        extraction_results=[[{"name": "Alice"}], [{"name": "Alice"}]],
    ) as harness:
        await harness.chat("I'm Alice")

        assert harness.wizard_stage == "sub_start"


# ---------------------------------------------------------------------------
# 3. The message that used to be eaten
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_message_that_used_to_be_eaten_reaches_the_subflow() -> None:
    """Turn 2 is extracted against the subflow's schema, not the parent's.

    This is the user-visible half of the defect. With the push a turn
    late, the user answers the prompt they can see -- the parent's --
    while the wizard is still on the parent stage; that message is
    extracted against the parent's schema and its content is discarded
    when the push finally happens and replaces the data.
    """
    async with await BotTestHarness.create(
        wizard_config=_guard_config(guard_key="_route_flag", routing_transforms=["flag_route"]),
        custom_functions={"flag_route": flag_route},
        main_responses=["Entering detail.", "Detail captured."],
        extraction_results=[[{"name": "Alice"}], [{"detail": "blue"}]],
    ) as harness:
        await harness.chat("I'm Alice")
        await harness.chat("blue")

        assert harness.extractor is not None
        schemas = [call["schema"] for call in harness.extractor.extract_calls]
        assert len(schemas) >= 2, f"expected two extractions, got {len(schemas)}"
        second = schemas[1].get("properties", {})
        assert "detail" in second, (
            "the second message was extracted against the parent's schema; "
            f"the subflow never saw it (schema properties: {sorted(second)})"
        )


# ---------------------------------------------------------------------------
# 4-5. advance() -- the path that never asked
# ---------------------------------------------------------------------------


def _state_at_start(reasoning: Any) -> WizardState:
    stage = reasoning.initial_stage
    return WizardState(current_stage=stage, history=[stage], stage_entry_time=time.time())


@pytest.mark.asyncio
async def test_advance_can_push_a_subflow() -> None:
    """The non-conversational API can enter a subflow, not only leave one.

    ``advance()`` reaches ``should_pop`` through the shared
    post-transition sequence but had no ``should_push`` call anywhere, so
    a headless consumer could be carried *out* of a subflow it had no way
    to be carried *into*.
    """
    async with await BotTestHarness.create(
        wizard_config=_guard_config(guard_key="_route_flag", routing_transforms=["flag_route"]),
        custom_functions={"flag_route": flag_route},
        main_responses=["Entering detail."],
        extraction_results=[[{"name": "Alice"}]],
    ) as harness:
        reasoning = harness.bot.reasoning_strategy
        state = _state_at_start(reasoning)

        result = await reasoning.advance({"name": "Alice"}, state)

        assert state.is_in_subflow, "advance() did not push the subflow"
        assert result.stage_name == "sub_start"
        assert result.transitioned is True
        # The envelope must describe the subflow the caller is now in, not
        # the parent it came from: a headless caller has no other channel.
        assert result.from_stage == "gather"
        assert set((result.stage_schema or {}).get("properties", {})) == {"detail"}


@pytest.mark.asyncio
async def test_advance_and_chat_agree_on_the_push_turn() -> None:
    """Two APIs, one config, one data value, one answer.

    Parity is the assertion, not either path's absolute behaviour: the
    defect was that the two disagreed, and a test that pins only one of
    them lets them drift apart again.
    """
    config = _guard_config(guard_key="_route_flag", routing_transforms=["flag_route"])

    async with await BotTestHarness.create(
        wizard_config=config,
        custom_functions={"flag_route": flag_route},
        main_responses=["Entering detail."],
        extraction_results=[[{"name": "Alice"}]],
    ) as chat_harness:
        await chat_harness.chat("I'm Alice")
        via_chat = chat_harness.wizard_stage

    async with await BotTestHarness.create(
        wizard_config=config,
        custom_functions={"flag_route": flag_route},
        main_responses=["Entering detail."],
        extraction_results=[[{"name": "Alice"}]],
    ) as advance_harness:
        reasoning = advance_harness.bot.reasoning_strategy
        state = _state_at_start(reasoning)
        result = await reasoning.advance({"name": "Alice"}, state)
        via_advance = result.stage_name

    assert via_advance == via_chat, (
        f"same config and same data: chat() reached {via_chat!r}, advance() reached {via_advance!r}"
    )
    # Anti-vacuity: parity alone is satisfied by both paths being broken
    # in the same way, which is what they did before the guard moved.
    assert via_chat == "sub_start"


# ---------------------------------------------------------------------------
# 6. The constraint the original ordering was protecting
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_push_still_pre_empts_the_self_loop_arc() -> None:
    """The guard moved past the preparation, not past the FSM step.

    A subflow transition compiles to a *self-loop* arc carrying its
    condition, so the FSM step cannot perform the push. If the guard ran
    after the step, an ordinary transition declared alongside it would
    consume the turn first and the subflow would never be entered. The
    unconditional ``wrap`` transition below is declared first for exactly
    that reason: it is what the FSM step would take.
    """
    async with await BotTestHarness.create(
        wizard_config=_guard_config(
            guard_key="_route_flag",
            routing_transforms=["flag_route"],
            also_unconditional=True,
        ),
        custom_functions={"flag_route": flag_route},
        main_responses=["Entering detail."],
        extraction_results=[[{"name": "Alice"}]],
    ) as harness:
        await harness.chat("I'm Alice")

        assert harness.wizard_stage == "sub_start", (
            "the FSM step consumed the turn before the guard was asked"
        )


# ---------------------------------------------------------------------------
# 7. The regression the fix could plausibly introduce
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_confirm_on_new_data_snapshot_is_taken_before_a_push_replaces_the_data() -> None:
    """A push turn now runs all three preparation steps, including the snapshot.

    This is the regression the fix could plausibly introduce. Before it, a
    push turn returned before ``_prepare_transition`` and the third step --
    ``confirm_on_new_data``'s snapshot -- never ran on one. It runs now,
    and it runs while the parent's data is still in place: the snapshot is
    keyed by the *parent* stage, and a snapshot taken one step later would
    record the subflow's empty dict and tell the parent, on return, that
    every value it had collected was new.

    Observed across the whole push/pop cycle, which is the only place the
    parent's snapshot is reachable again.
    """
    async with await BotTestHarness.create(
        wizard_config=_guard_config(
            guard_key="_route_flag",
            routing_transforms=["flag_route"],
            confirm_on_new_data=True,
        ),
        custom_functions={"flag_route": flag_route},
        main_responses=["Entering detail.", "Detail captured."],
        extraction_results=[[{"name": "Alice"}], [{"detail": "blue"}]],
    ) as harness:
        await harness.chat("I'm Alice")
        assert harness.wizard_stage == "sub_start"

        await harness.chat("blue")

        assert harness.wizard_stage == "wrap", "the subflow did not pop"
        snapshot = harness.wizard_data.get("_stage_rendered_snapshot", {}).get("gather")
        assert snapshot == {"name": "Alice"}, (
            "the parent's confirm_on_new_data snapshot did not survive the "
            f"push/pop cycle with the parent's own values: {snapshot!r}"
        )


# ---------------------------------------------------------------------------
# 8-9. What the logs say happened
# ---------------------------------------------------------------------------
#
# The two halves of a subflow decision are invisible in opposite ways. The
# FSM asserts nothing matched when a subflow transition *did* match -- it
# compiles to a self-loop, which is indistinguishable from standing still.
# And a declined push leaves no trace at all: its only evidence is the
# absence of a push.


def _fsm_with_a_subflow_transition() -> Any:
    """A two-stage wizard whose start stage guards a subflow, always true."""
    from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader

    config = (
        WizardConfigBuilder("fsm-logging")
        .stage("gather", is_start=True, prompt="Name?")
        .field("name", field_type="string", required=True)
        .transition("wrap", condition="has('name')", subflow_network="detail", return_stage="wrap")
        .stage("wrap", is_end=True, prompt="Done.")
        .subflow("detail", _DETAIL_SUBFLOW)
        .build()
    )
    return WizardConfigLoader().load_from_dict(config)


@pytest.mark.parametrize("step_is_async", [False, True], ids=["step", "step_async"])
@pytest.mark.asyncio
async def test_none_matched_is_not_logged_when_a_subflow_transition_matched(
    caplog: pytest.LogCaptureFixture,
    step_is_async: bool,
) -> None:
    """A matched subflow transition is not "none matched" -- in both copies.

    ``step`` and ``step_async`` carry the same log block, written twice.
    Parametrising over them is what keeps a fix to one from passing while
    the other still says the false thing.
    """
    fsm = _fsm_with_a_subflow_transition()
    try:
        with caplog.at_level(logging.DEBUG, logger="dataknobs_bots.reasoning.wizard_fsm"):
            if step_is_async:
                await fsm.step_async({"name": "Alice"})
            else:
                fsm.step({"name": "Alice"})
    finally:
        fsm.close()

    messages = [record.getMessage() for record in caplog.records]
    none_matched = [m for m in messages if "none matched" in m]
    assert not none_matched, (
        "a subflow transition matched and compiled to a self-loop; the FSM "
        f"reported that nothing matched: {none_matched}"
    )
    assert any("subflow" in m.lower() for m in messages), (
        f"the FSM said nothing about the subflow transition it holds: {messages}"
    )


@pytest.mark.parametrize("step_is_async", [False, True], ids=["step", "step_async"])
@pytest.mark.asyncio
async def test_a_declined_guard_is_reported_as_nothing_matching(
    caplog: pytest.LogCaptureFixture,
    step_is_async: bool,
) -> None:
    """The converse: holding a subflow transition is not taking one.

    This is the ordinary case rather than the exotic one. A guard that
    carries pushes the subflow, and a push skips the FSM step entirely --
    so a step that runs at all, on a stage that declares a subflow
    transition, is nearly always a step where the guard declined and
    nothing matched.

    Deciding the message from what the stage *declares* describes every
    one of those turns as a self-loop absorbing the turn, which sends a
    reader looking for a push that was never attempted. The step's own
    ``transition`` is what tells the two apart.
    """
    fsm = _fsm_with_a_subflow_transition()
    try:
        with caplog.at_level(logging.DEBUG, logger="dataknobs_bots.reasoning.wizard_fsm"):
            # No 'name', so has('name') is false and the guard declines.
            if step_is_async:
                await fsm.step_async({})
            else:
                fsm.step({})

        assert fsm.current_stage == "gather", "the FSM moved; nothing should have matched"
    finally:
        fsm.close()

    messages = [record.getMessage() for record in caplog.records]
    assert any("none matched" in m for m in messages), (
        f"the guard declined and nothing matched; the FSM did not say so: {messages}"
    )
    self_loop_claims = [m for m in messages if "self-loop" in m]
    assert not self_loop_claims, (
        "nothing matched, but the FSM described the turn as a subflow "
        f"self-loop because the stage merely declares one: {self_loop_claims}"
    )


@pytest.mark.asyncio
async def test_a_declined_push_says_so(caplog: pytest.LogCaptureFixture) -> None:
    """A guard that declines leaves a trace naming itself.

    Today the only evidence of a decline is that no push happened, which
    is the same evidence as there being no subflow transition at all, as
    a misspelled condition, and as a condition that raised.
    """
    async with await BotTestHarness.create(
        wizard_config=_guard_config(guard_key="_never_set"),
        main_responses=["Noted."],
        extraction_results=[[{"name": "Alice"}]],
    ) as harness:
        with caplog.at_level(logging.DEBUG, logger="dataknobs_bots.reasoning.wizard_subflows"):
            await harness.chat("I'm Alice")

    messages = [record.getMessage() for record in caplog.records]
    assert any("_never_set" in m for m in messages), (
        f"the declined guard is not named anywhere in the log: {messages}"
    )


# ---------------------------------------------------------------------------
# The documented visibility boundary
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_subflow_guard_sees_an_extracted_key_on_the_turn_it_is_written() -> None:
    """The other row of WIZARD_SUBFLOWS.md's visibility table.

    Extraction lands before the pre-transition sequence and was never
    affected by the guard's position -- which is exactly why it belongs in
    the table beside the two writers that were. A reader told only that
    "a subflow guard is one turn late" would look for the fault here.
    """
    async with await BotTestHarness.create(
        wizard_config=_guard_config(guard_key="name"),
        main_responses=["Entering detail."],
        extraction_results=[[{"name": "Alice"}]],
    ) as harness:
        await harness.chat("I'm Alice")

        assert harness.wizard_stage == "sub_start"
