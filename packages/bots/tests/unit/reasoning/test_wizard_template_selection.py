"""Which template a stage renders, and when — on both response paths.

The suite that existed before this file tested the *renderer*: that
Jinja substitution works, that ``_``-prefixed keys are filtered, that a
template response object carries the right fields.  Nothing tested the
*selection* — which of a stage's templates is chosen on turn N — and
nothing drove a second render at all, so every selection rule could be
wrong on one of the two response paths without a single failure.

Every test here that goes through a bot crosses at least one turn
boundary and asserts on what the bot actually said, buffered and
streamed.  The last one is a unit test, because the collector it covers
is only reachable while the wizard is mid-advance.
"""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_bots.testing import BotTestHarness, WizardConfigBuilder

GREETING = "Hi there!"
CLARIFY = "Was that a yes or no?"


def _conversation_wizard(**stage_kwargs: Any) -> dict[str, Any]:
    """One conversation-mode start stage that nothing transitions away from."""
    return (
        WizardConfigBuilder("template-selection")
        .stage(
            "opening",
            is_start=True,
            mode="conversation",
            prompt="Ask the user whether they agree.",
            **stage_kwargs,
        )
        .transition("done", "data.get('finished')")
        .stage("done", is_end=True, prompt="Finished.")
        .build()
    )


# ---------------------------------------------------------------------------
# clarification_template on the streaming path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_clarification_template_renders_on_second_render_buffered() -> None:
    """Buffered: first render is the response_template, later ones clarify."""
    config = _conversation_wizard(
        response_template=GREETING,
        clarification_template=CLARIFY,
    )
    async with await BotTestHarness.create(
        wizard_config=config,
        main_responses=["LLM-1", "LLM-2"],
    ) as harness:
        greeting = await harness.greet()
        first = await harness.chat("hello")

        assert greeting.response == GREETING
        assert first.response == CLARIFY
        assert harness.provider.call_count == 0


@pytest.mark.asyncio
async def test_clarification_template_renders_on_second_render_streamed() -> None:
    """Streamed: same config, same rule.

    Fails before the selection helper existed: the streaming path had no
    clarification branch at all, so it fell through to the LLM and
    answered ``"LLM-1"``.
    """
    config = _conversation_wizard(
        response_template=GREETING,
        clarification_template=CLARIFY,
    )
    async with await BotTestHarness.create(
        wizard_config=config,
        main_responses=["LLM-1", "LLM-2"],
    ) as harness:
        greeting = await harness.greet()
        first = await harness.stream_chat("hello")

        assert greeting.response == GREETING
        assert first.response == CLARIFY
        assert harness.provider.call_count == 0


# ---------------------------------------------------------------------------
# clarification_template with no response_template beside it
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_clarification_template_alone_renders_buffered() -> None:
    """A clarification_template is not inert without a response_template.

    Fails before the tracking predicate existed: the render counter only
    incremented for stages that set a ``response_template``, so this
    stage was permanently on its "first" render and the clarification
    branch was unreachable on every turn.
    """
    config = _conversation_wizard(clarification_template=CLARIFY)
    async with await BotTestHarness.create(
        wizard_config=config,
        main_responses=["LLM-1", "LLM-2", "LLM-3"],
    ) as harness:
        greeting = await harness.greet()
        first = await harness.chat("hello")
        second = await harness.chat("still here")

        assert greeting.response == "LLM-1"
        assert first.response == CLARIFY
        assert second.response == CLARIFY


@pytest.mark.asyncio
async def test_clarification_template_alone_renders_streamed() -> None:
    """The same stage, streamed."""
    config = _conversation_wizard(clarification_template=CLARIFY)
    async with await BotTestHarness.create(
        wizard_config=config,
        main_responses=["LLM-1", "LLM-2", "LLM-3"],
    ) as harness:
        greeting = await harness.greet()
        first = await harness.stream_chat("hello")

        assert greeting.response == "LLM-1"
        assert first.response == CLARIFY


# ---------------------------------------------------------------------------
# The rules that already held, pinned so the extraction cannot lose them
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_structured_stage_renders_its_template_every_turn() -> None:
    """A structured stage's response_template is the response, every turn.

    Deliberate — a review summary re-renders as the data changes.  Pinned
    because the selection extraction must not generalise the
    conversation-mode first-render rule to stages that never opted in.
    """
    config = (
        WizardConfigBuilder("template-selection")
        .stage(
            "opening",
            is_start=True,
            prompt="Collect a name.",
            response_template="Recorded: {{ name }}",
            confirm_first_render=False,
        )
        .field("name", field_type="string", required=False)
        .transition("done", "data.get('finished')")
        .stage("done", is_end=True, prompt="Finished.")
        .build()
    )
    async with await BotTestHarness.create(
        wizard_config=config,
        main_responses=["LLM-1", "LLM-2"],
        extraction_results=[[{"name": "Alice"}], [{"name": "Bob"}]],
    ) as harness:
        first = await harness.chat("I am Alice")
        second = await harness.chat("actually Bob")

        assert first.response == "Recorded: Alice"
        assert second.response == "Recorded: Bob"
        assert harness.provider.call_count == 0


@pytest.mark.asyncio
async def test_conversation_stage_without_clarification_falls_through_to_llm() -> None:
    """No clarification_template means the stage converses after render 1."""
    config = _conversation_wizard(response_template=GREETING)
    async with await BotTestHarness.create(
        wizard_config=config,
        main_responses=["LLM-1", "LLM-2"],
    ) as harness:
        greeting = await harness.greet()
        first = await harness.chat("hello")
        second = await harness.chat("again")

        assert greeting.response == GREETING
        assert first.response == "LLM-1"
        assert second.response == "LLM-2"


# ---------------------------------------------------------------------------
# The auto-advance collector selects the same way the turn would have
# ---------------------------------------------------------------------------


def _responder_for(**stage_kwargs: Any) -> tuple[Any, dict[str, Any]]:
    """A responder and its start stage, built without a bot around them."""
    from dataknobs_bots.reasoning.wizard import WizardReasoning
    from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader

    fsm = WizardConfigLoader().load_from_dict(_conversation_wizard(**stage_kwargs))
    return WizardReasoning(wizard_fsm=fsm)._response, fsm.current_metadata


def test_auto_advance_collects_the_template_the_turn_would_have_rendered() -> None:
    """The collector goes through the same selection rule as the turn.

    A unit test rather than a harness test: the collector is only reached
    while the wizard is mid-advance, and what is being pinned is which
    template it picks, not the advance.  It used to read
    ``response_template`` directly, so a conversation-mode stage being
    advanced past contributed its opening line again however many times
    it had already spoken.
    """
    from dataknobs_bots.reasoning.wizard import WizardState

    responder, stage = _responder_for(
        response_template=GREETING,
        clarification_template=CLARIFY,
    )

    fresh = WizardState(current_stage="opening", data={})
    assert responder._render_auto_advance_template(stage, fresh) == GREETING

    spoken = WizardState(current_stage="opening", data={})
    spoken.increment_render_count("opening")
    assert responder._render_auto_advance_template(stage, spoken) == CLARIFY


def test_auto_advance_drops_a_spoken_stage_rather_than_repeating_it() -> None:
    """A stage with nothing left to say contributes nothing, not its opening.

    The same conversation-mode stage without a ``clarification_template``.
    Had the turn stopped here the stage would have fallen through to LLM
    mode, which a collector cannot do — so the honest contribution is
    none.  Before the selection rule was shared, the collector read
    ``response_template`` directly and re-contributed the opening line on
    every pass, which is the shape an ``intent_confirm:`` stage takes when
    ``on_no_match.clarification_template`` is left unset.
    """
    from dataknobs_bots.reasoning.wizard import WizardState

    responder, stage = _responder_for(response_template=GREETING)

    fresh = WizardState(current_stage="opening", data={})
    assert responder._render_auto_advance_template(stage, fresh) == GREETING
    assert fresh.get_render_count("opening") == 1

    spoken = WizardState(current_stage="opening", data={})
    spoken.increment_render_count("opening")
    assert responder._render_auto_advance_template(stage, spoken) is None
    # Nothing rendered, so nothing counted.
    assert spoken.get_render_count("opening") == 1
