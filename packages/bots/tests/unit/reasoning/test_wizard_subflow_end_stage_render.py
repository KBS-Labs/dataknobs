"""What an ``is_end`` subflow stage says on the turn it is reached.

A subflow's end stage is entered and left inside one turn: ``should_pop``
needs only a non-empty stack and ``is_end``, so the pop runs in the same
post-transition step that landed on the stage, and the parent's return
stage renders instead. The end stage's own ``response_template`` was
therefore dead -- parsed, validated, and never on screen.

The cost is not symmetric with an ordinary missing message. A subflow
that can fail ends on a stage whose entire job is to say *nothing was
saved, and here is why*; that refusal is the one message that never
rendered, so the flow discarded the user's work and reported success.

Two things make this suite more than one assertion:

* **There are two pop sites.** ``_run_post_transition_lifecycle`` pops
  the stage a transition landed on; ``run_auto_advance_loop`` pops the
  stage its own step landed on. A fix at either alone leaves the other,
  and the second is the path an end stage reached by ``auto_advance``
  takes -- the common case.
* **The order is the fix.** ``handle_pop`` swaps the active FSM and
  replaces ``state.data`` with the parent's, so a render placed after it
  reads the parent's data and names the parent's stage. Rendering must
  precede the pop, and a template interpolating a subflow-only value is
  what pins that down rather than asserting it.

The last two tests are coverage this file owes regardless of the change:
``should_pop`` and ``handle_pop`` had no test in the repository that
called them, which is the whole explanation for how an unreachable
render survived.
"""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_bots.testing import BotTestHarness, WizardConfigBuilder

# The subflow's end stage interpolates ``detail`` -- a key that exists
# only in the subflow's data and that ``handle_pop`` replaces with the
# parent's. A render placed after the pop produces the bare prefix.
_END_TEMPLATE = "SUBFLOW-END: captured {{ detail }}"
_END_RENDERED = "SUBFLOW-END: captured blue"
_PARENT_TEMPLATE = "WRAP-PARENT"


def _detail_subflow(*, via_auto_advance: bool = False) -> dict[str, Any]:
    """A subflow that collects one field and ends.

    With *via_auto_advance* the end stage is reached from a message
    stage inside the auto-advance loop, which is the second pop site.
    Without it the end stage is reached by the turn's own transition,
    which is the first.
    """
    builder = WizardConfigBuilder("detail")
    builder.stage(
        "sub_start",
        is_start=True,
        prompt="Which detail?",
        response_template="Entering detail.",
        confirm_first_render=False,
    )
    builder.field("detail", field_type="string", required=True)
    builder.transition("sub_relay" if via_auto_advance else "sub_done", condition="has('detail')")
    if via_auto_advance:
        builder.stage(
            "sub_relay",
            prompt="Relaying.",
            response_template="SUB-RELAY",
            auto_advance=True,
            confirm_first_render=False,
        )
        builder.transition("sub_done")
    builder.stage("sub_done", is_end=True, prompt="Done.", response_template=_END_TEMPLATE)
    return builder.build()


def _refusal_subflow() -> dict[str, Any]:
    """A subflow with two exits, one of which refuses.

    The brief's shape: an end stage that builds the artifact and an end
    stage that explains why nothing was built. Both are ``is_end``, so
    before the fix neither could speak and the refusal was silent.
    """
    builder = WizardConfigBuilder("detail")
    builder.stage(
        "sub_start",
        is_start=True,
        prompt="Which detail?",
        response_template="Entering detail.",
        confirm_first_render=False,
    )
    builder.field("detail", field_type="string", required=True)
    builder.transition("sub_saved", condition="data.get('detail') != 'nothing'")
    builder.transition("sub_refused", condition="data.get('detail') == 'nothing'")
    builder.stage("sub_saved", is_end=True, prompt="Saved.", response_template="SAVED-BRANCH")
    builder.stage(
        "sub_refused",
        is_end=True,
        prompt="Refused.",
        response_template="NOTHING-SAVED: {{ detail }} could not be built",
    )
    return builder.build()


def _parent(
    subflow: dict[str, Any], *, result_mapping: dict[str, str] | None = None
) -> dict[str, Any]:
    """A parent that pushes *subflow* and returns to an end stage."""
    builder = WizardConfigBuilder("subflow-end-render")
    builder.stage(
        "gather",
        is_start=True,
        prompt="Tell me your name.",
        response_template="Noted.",
        confirm_first_render=False,
    )
    builder.field("name", field_type="string", required=True)
    builder.transition(
        "wrap",
        condition="has('name')",
        subflow_network="detail",
        return_stage="wrap",
        result_mapping=result_mapping,
    )
    builder.stage("wrap", is_end=True, prompt="All done.", response_template=_PARENT_TEMPLATE)
    builder.subflow("detail", subflow)
    return builder.build()


async def _run_to_pop(
    config: dict[str, Any],
    *,
    detail: str = "blue",
    relay: bool = False,
) -> tuple[BotTestHarness, str]:
    """Push the subflow on turn 1, pop it on turn 2; return the harness and turn 2."""
    responses = ["Entering detail.", "Popping."]
    harness = await BotTestHarness.create(
        wizard_config=config,
        main_responses=responses,
        extraction_results=[[{"name": "Alice"}], [{"detail": detail}]],
    )
    await harness.chat("I'm Alice")
    assert harness.wizard_stage == "sub_start", (
        f"turn 1 did not push the subflow (stage={harness.wizard_stage})"
    )
    result = await harness.chat(detail)
    assert relay or harness.wizard_stage == "wrap", (
        f"turn 2 did not pop back to the parent (stage={harness.wizard_stage})"
    )
    return harness, result.response


# ---------------------------------------------------------------------------
# 1-2. Both pop sites
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_is_end_subflow_stages_template_appears_in_the_turn() -> None:
    """The first pop site: the stage a transition landed on."""
    harness, response = await _run_to_pop(_parent(_detail_subflow()))
    try:
        assert _END_RENDERED in response, (
            "the subflow's end stage never spoke; the turn shows only the "
            f"parent's return render: {response!r}"
        )
    finally:
        await harness.close()


@pytest.mark.asyncio
async def test_an_is_end_subflow_stage_reached_by_auto_advance_also_renders() -> None:
    """The second pop site, in ``run_auto_advance_loop``.

    Named in no brief and in no register row. An end stage reached by
    ``auto_advance`` is the common case, and it pops through a copy of
    the same two lines -- so a fix at the other site alone would leave
    exactly the configs most likely to hit it.
    """
    harness, response = await _run_to_pop(
        _parent(_detail_subflow(via_auto_advance=True)),
        relay=True,
    )
    try:
        assert harness.wizard_stage == "wrap", (
            f"auto-advance did not carry the subflow to its end and pop "
            f"(stage={harness.wizard_stage})"
        )
        assert _END_RENDERED in response, (
            f"the end stage reached by auto-advance never spoke: {response!r}"
        )
    finally:
        await harness.close()


# ---------------------------------------------------------------------------
# 3. Order, and what it proves about the render happening before the pop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_end_stages_message_precedes_the_parents_return_render() -> None:
    """Both messages, in the order the user lived them.

    The subflow ends, then the parent resumes. A turn that said them the
    other way round would read as the parent finishing before the
    subflow it was waiting on.
    """
    harness, response = await _run_to_pop(_parent(_detail_subflow()))
    try:
        assert _END_RENDERED in response and _PARENT_TEMPLATE in response, (
            f"expected both the end stage and the parent's return render: {response!r}"
        )
        assert response.index(_END_RENDERED) < response.index(_PARENT_TEMPLATE), (
            f"the parent's return render came first: {response!r}"
        )
    finally:
        await harness.close()


@pytest.mark.asyncio
async def test_the_end_stage_renders_against_its_own_data_not_the_parents() -> None:
    """The ordering, pinned structurally rather than asserted.

    ``handle_pop`` replaces ``state.data`` with the parent's, and
    ``detail`` exists only in the subflow's. So a render placed after the
    pop yields the template's bare prefix with an empty interpolation --
    which is a passing "the message appeared" assertion and a wrong
    message. Requiring the interpolated value makes that indistinguishable
    case fail.
    """
    harness, response = await _run_to_pop(_parent(_detail_subflow()))
    try:
        assert "SUBFLOW-END: captured" in response, f"the end stage said nothing: {response!r}"
        assert _END_RENDERED in response, (
            f"the end stage rendered after the pop, against the parent's data: {response!r}"
        )
    finally:
        await harness.close()


# ---------------------------------------------------------------------------
# 4. The cost the brief measured: a refusal nobody saw
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_refusal_branch_on_an_end_stage_is_visible() -> None:
    """A subflow's two exits say different things, and both are heard.

    This is the defect's actual shape rather than its mechanism: the
    message that never rendered was the one explaining that nothing had
    been saved, so a flow that discarded the user's work reported
    success.
    """
    harness, response = await _run_to_pop(_parent(_refusal_subflow()), detail="nothing")
    try:
        assert "NOTHING-SAVED: nothing could not be built" in response, (
            f"the refusal branch was silent; the turn reported only success: {response!r}"
        )
        assert "SAVED-BRANCH" not in response, f"the wrong exit rendered: {response!r}"
    finally:
        await harness.close()


# ---------------------------------------------------------------------------
# 9-10. Pop-path coverage, owed regardless of the change
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pop_restores_the_parents_data_and_active_fsm() -> None:
    """``should_pop`` / ``handle_pop`` had no calling test in the repository.

    That absence is the whole explanation for how an unreachable render
    shipped: the exit half of the subflow lifecycle was never executed.
    This asserts the contract the render must not disturb -- the parent's
    stage, the parent's data, an empty stack, and the main FSM active
    again.
    """
    harness, _ = await _run_to_pop(_parent(_detail_subflow()))
    try:
        assert harness.wizard_stage == "wrap"
        assert harness.wizard_data.get("name") == "Alice", (
            f"the parent's data was not restored: {harness.wizard_data}"
        )
        assert "detail" not in harness.wizard_data, (
            "the subflow's data survived the pop with no result mapping asking "
            f"for it: {harness.wizard_data}"
        )
        subflows = harness.bot.reasoning_strategy._subflows
        assert subflows.active_subflow_fsm is None, "the subflow FSM is still active"
    finally:
        await harness.close()


@pytest.mark.asyncio
async def test_result_mapping_survives_the_render() -> None:
    """Rendering before the pop must not disturb what the pop carries out.

    ``handle_pop`` reads ``state.data`` to build the result mapping. The
    render runs first and touches the same state -- it increments the
    stage's render count -- so this pins that the mapped value still
    arrives in the parent.
    """
    harness, response = await _run_to_pop(
        _parent(_detail_subflow(), result_mapping={"detail": "chosen_detail"})
    )
    try:
        assert harness.wizard_data.get("chosen_detail") == "blue", (
            f"the result mapping did not reach the parent: {harness.wizard_data}"
        )
        assert _END_RENDERED in response
    finally:
        await harness.close()


@pytest.mark.asyncio
async def test_an_end_stage_with_only_a_prompt_still_says_nothing() -> None:
    """Anti-overreach: the pop renders a *template*, not the LLM's turn.

    ``prompt`` is the stage's instruction to the model, and the pop has no
    turn to give it -- the collector cannot call the LLM any more than the
    auto-advance loop can. So an end stage carrying only a ``prompt`` is
    still silent, which is worth pinning because it is the shape the
    subflow documentation's own example used and the one an author
    reaching for a completion message is most likely to write.
    """
    builder = WizardConfigBuilder("detail")
    builder.stage(
        "sub_start",
        is_start=True,
        prompt="Which detail?",
        response_template="Entering detail.",
        confirm_first_render=False,
    )
    builder.field("detail", field_type="string", required=True)
    builder.transition("sub_done", condition="has('detail')")
    builder.stage("sub_done", is_end=True, prompt="PROMPT-ONLY-END")
    harness, response = await _run_to_pop(_parent(builder.build()))
    try:
        assert "PROMPT-ONLY-END" not in response, (
            f"a prompt is not a template and must not be rendered here: {response!r}"
        )
        assert _PARENT_TEMPLATE in response
    finally:
        await harness.close()


# ---------------------------------------------------------------------------
# 8. A template that raises must not take the pop down with it
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_raising_end_stage_template_still_pops() -> None:
    """The pop is structural; the message it collects is decoration.

    Rendering before the pop is what makes the message the subflow's own,
    but it also puts a render in front of a structural step that never
    had one. The render raises on more than a typo: ``{{ data.x }}``
    raises ``UndefinedError`` even under ``strict=False``, because the
    context exposes collected values as top-level names and defines no
    ``data`` -- and that spelling is what the subflow guide taught until
    the commit that fixed it, so it is what a consumer's config written
    against that guide contains.

    Before the guard the exception escaped ahead of ``handle_pop``: the
    subflow never popped, the turn never saved, and the next turn re-ran
    the same transition and raised again. An end stage with a bad
    template made its subflow un-exitable rather than quiet, which is a
    worse failure than the silence this suite exists to fix.
    """
    builder = WizardConfigBuilder("detail")
    builder.stage(
        "sub_start",
        is_start=True,
        prompt="Which detail?",
        response_template="Entering detail.",
        confirm_first_render=False,
    )
    builder.field("detail", field_type="string", required=True)
    builder.transition("sub_done", condition="has('detail')")
    builder.stage(
        "sub_done",
        is_end=True,
        prompt="Done.",
        # The pre-fix guide's spelling. `detail` alone would render.
        response_template="SUBFLOW-END: captured {{ data.detail }}",
    )
    harness, response = await _run_to_pop(_parent(builder.build()))
    try:
        assert harness.wizard_stage == "wrap", (
            "a raising end-stage template blocked the pop; the wizard is "
            f"stuck inside the subflow at {harness.wizard_stage!r}"
        )
        assert _PARENT_TEMPLATE in response, (
            f"the parent never resumed after the failed render: {response!r}"
        )
    finally:
        await harness.close()


@pytest.mark.asyncio
async def test_a_raising_template_mid_auto_advance_still_advances() -> None:
    """The same guard, at the other caller.

    ``run_auto_advance_loop`` collects a stage's message *before* it
    steps past it, so an unguarded raise there strands the chain on a
    stage it had already decided to leave -- the auto-advance analogue of
    the stranded pop, and the reason the guard belongs in
    ``render_departing_stage`` rather than at the pop site.
    """
    builder = WizardConfigBuilder("detail")
    builder.stage(
        "sub_start",
        is_start=True,
        prompt="Which detail?",
        response_template="Entering detail.",
        confirm_first_render=False,
    )
    builder.field("detail", field_type="string", required=True)
    builder.transition("sub_relay", condition="has('detail')")
    builder.stage(
        "sub_relay",
        prompt="Relaying.",
        response_template="RELAY: {{ data.detail }}",
        auto_advance=True,
        confirm_first_render=False,
    )
    builder.transition("sub_done")
    builder.stage("sub_done", is_end=True, prompt="Done.", response_template=_END_TEMPLATE)
    harness, response = await _run_to_pop(_parent(builder.build()), relay=True)
    try:
        assert harness.wizard_stage == "wrap", (
            "a raising template on an auto-advanced stage stranded the "
            f"chain at {harness.wizard_stage!r}"
        )
        assert _END_RENDERED in response, (
            f"the surviving stages' messages were lost with it: {response!r}"
        )
    finally:
        await harness.close()
