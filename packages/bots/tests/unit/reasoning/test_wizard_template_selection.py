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


# ---------------------------------------------------------------------------
# greeting_template — an opening line for a stage that is not conversational
# ---------------------------------------------------------------------------


GREETING_TEMPLATE = "Welcome! What is your name?"


def _structured_greeting_wizard(**stage_kwargs: Any) -> dict[str, Any]:
    """A structured start stage whose opening line is a greeting_template."""
    return (
        WizardConfigBuilder("greeting-template")
        .stage(
            "opening",
            is_start=True,
            prompt="Ask the user for their name.",
            greeting_template=GREETING_TEMPLATE,
            **stage_kwargs,
        )
        .field("name", field_type="string", required=False)
        .transition("done", "data.get('finished')")
        .stage("done", is_end=True, prompt="Finished.")
        .build()
    )


@pytest.mark.asyncio
async def test_structured_stage_greets_then_converses_buffered() -> None:
    """A greeting_template opens a structured stage without canning it.

    A fixed opening line on a stage that also extracts.  Before the
    field existed the only way to get one was a ``response_template``,
    which a structured stage re-renders on every turn — so the bot said
    the same sentence for as long as the wizard stayed there.
    """
    config = _structured_greeting_wizard()
    async with await BotTestHarness.create(
        wizard_config=config,
        main_responses=["LLM-1", "LLM-2"],
        extraction_results=[[{"name": "Alice"}]],
    ) as harness:
        greeting = await harness.greet()
        assert greeting.response == GREETING_TEMPLATE
        assert harness.provider.call_count == 0

        first = await harness.chat("I am Alice")
        assert first.response == "LLM-1"
        assert harness.wizard_data.get("name") == "Alice"


@pytest.mark.asyncio
async def test_structured_stage_greets_then_converses_streamed() -> None:
    """The greeting is not repeated on a streamed turn.

    Selection is asked the same question on the streaming path, and the
    greeting's "already shown" fact has to survive the switch of paths —
    ``greet()`` is buffered, so every streamed turn that follows it reads
    a count written by the other path.
    """
    config = _structured_greeting_wizard()
    async with await BotTestHarness.create(
        wizard_config=config,
        main_responses=["LLM-1", "LLM-2"],
        extraction_results=[[{"name": "Alice"}]],
    ) as harness:
        greeting = await harness.greet()
        assert greeting.response == GREETING_TEMPLATE

        first = await harness.stream_chat("I am Alice")
        assert first.response == "LLM-1"


# ---------------------------------------------------------------------------
# The whole rule, as a table
# ---------------------------------------------------------------------------


G = "GREET"
R = "RESPOND"
C = "CLARIFY-2"

#: ``(mode, greeting, response, clarification, [render 1, 2, 3])``.
#:
#: Render 1 is ``greet()``; renders 2 and 3 are user turns.  ``None`` in
#: the expected column means "the LLM answered", checked against the
#: scripted reply for that turn.
#:
#: The eight field combinations in each mode are the selection rule
#: written out — the guard the module went without, and the reason a
#: template kind could be added to one of the two response paths and
#: nobody noticed for two months.
_SELECTION_TABLE: list[tuple[str, str | None, str | None, str | None, list[str | None]]] = [
    # Structured stages: the template IS the response, so it re-renders.
    # A clarification_template never applies — there are no "later turns"
    # to distinguish, which is why the greeting had to be its own field.
    ("structured", None, None, None, [None, None, None]),
    ("structured", None, None, C, [None, None, None]),
    ("structured", None, R, None, [R, R, R]),
    ("structured", None, R, C, [R, R, R]),
    ("structured", G, None, None, [G, None, None]),
    ("structured", G, None, C, [G, None, None]),
    ("structured", G, R, None, [G, R, R]),
    ("structured", G, R, C, [G, R, R]),
    # Conversation stages: one opening line, then the stage converses —
    # or clarifies, when a clarification_template says how.
    ("conversation", None, None, None, [None, None, None]),
    ("conversation", None, None, C, [None, C, C]),
    ("conversation", None, R, None, [R, None, None]),
    ("conversation", None, R, C, [R, C, C]),
    ("conversation", G, None, None, [G, None, None]),
    ("conversation", G, None, C, [G, C, C]),
    # The greeting takes the opening, so response_template never renders.
    # The loader warns about exactly this pair.
    ("conversation", G, R, None, [G, None, None]),
    ("conversation", G, R, C, [G, C, C]),
]


def _table_id(case: tuple[Any, ...]) -> str:
    mode, greeting, response, clarification, _ = case
    fields = "".join(
        letter
        for letter, value in (("g", greeting), ("r", response), ("c", clarification))
        if value
    )
    return f"{mode}-{fields or 'none'}"


@pytest.mark.parametrize("case", _SELECTION_TABLE, ids=_table_id)
@pytest.mark.asyncio
async def test_selection_table(
    case: tuple[str, str | None, str | None, str | None, list[str | None]],
) -> None:
    """Three consecutive renders of one stage, for every field combination."""
    mode, greeting, response, clarification, expected = case
    stage_kwargs: dict[str, Any] = {}
    if mode == "conversation":
        stage_kwargs["mode"] = "conversation"
    if greeting:
        stage_kwargs["greeting_template"] = greeting
    if response:
        stage_kwargs["response_template"] = response
    if clarification:
        stage_kwargs["clarification_template"] = clarification

    config = (
        WizardConfigBuilder("selection-table")
        # confirm_first_render=False isolates selection: a structured
        # stage that greets still has its confirmation pending on the
        # user's first turn, which would answer that turn instead of the
        # template.  That interaction is the subject of its own test.
        .stage(
            "opening",
            is_start=True,
            prompt="Say something.",
            confirm_first_render=False,
            **stage_kwargs,
        )
        .transition("done", "data.get('finished')")
        .stage("done", is_end=True, prompt="Finished.")
        .build()
    )
    scripted = ["LLM-1", "LLM-2", "LLM-3"]
    async with await BotTestHarness.create(
        wizard_config=config,
        main_responses=list(scripted),
    ) as harness:
        actual = [
            (await harness.greet()).response,
            (await harness.chat("one")).response,
            (await harness.chat("two")).response,
        ]

    llm_turns = 0
    for index, want in enumerate(expected):
        if want is None:
            assert actual[index] == scripted[llm_turns], (
                f"render {index + 1}: expected the LLM's "
                f"{scripted[llm_turns]!r}, got {actual[index]!r}"
            )
            llm_turns += 1
        else:
            assert actual[index] == want, f"render {index + 1}: expected {want!r}"


# ---------------------------------------------------------------------------
# The greeting has its own count, and both reasons it must
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_greeting_does_not_spend_the_confirmation_s_first_render() -> None:
    """Greeting a structured stage leaves ``confirm_first_render`` intact.

    The render count is read twice: by selection, as "has this stage
    produced output?", and by the confirmation evaluator, as "has this
    stage rendered its ``response_template`` yet?".  Those were the same
    question until a stage could greet.  Counting a greeting in the
    shared counter would leave the evaluator looking at render 1 on the
    user's first turn, skipping the first-render branch — a default-True
    behaviour switched off because an unrelated field was set.

    This test fails against any implementation that shares the counter,
    which is the point of it.  Do not relax what it asserts: the
    confirmation disappearing is the defect, not the assertion.
    """
    config = (
        WizardConfigBuilder("greeting-confirmation")
        .stage(
            "opening",
            is_start=True,
            prompt="Ask for a name.",
            greeting_template=GREETING_TEMPLATE,
            response_template="Recorded: {{ name }}",
        )
        .field("name", field_type="string", required=True)
        .transition("done", "data.get('name')")
        .stage("done", is_end=True, prompt="Finished.")
        .build()
    )
    async with await BotTestHarness.create(
        wizard_config=config,
        main_responses=["LLM-1"],
        extraction_results=[[{"name": "Alice"}]],
    ) as harness:
        greeting = await harness.greet()
        assert greeting.response == GREETING_TEMPLATE
        # The two counts are separate, and only the greeting one moved.
        assert harness.wizard_data.get("_stage_greeting_counts") == {"opening": 1}
        assert harness.wizard_data.get("_stage_render_counts", {}).get("opening", 0) == 0

        answer = await harness.chat("I am Alice")

        assert "Is that correct?" in answer.response, (
            "the stage's confirmation did not fire on the user's first "
            f"turn; it answered {answer.response!r}"
        )
        assert harness.wizard_stage == "opening"


@pytest.mark.asyncio
async def test_a_greeting_is_not_repeated_after_a_subflow_push() -> None:
    """A pushed stage greets once, though its render is deliberately uncounted.

    The subflow-push path renders with ``track_render=False`` so the
    pushed stage's template counts as a question the user has not
    answered — the invariant
    ``test_render_count_zero_after_subflow_push`` pins.  A greeting
    sharing that counter would therefore still read as unsaid on the
    next turn and be said again, so the greeting count ignores
    ``track_render``: whether a greeting has been delivered is a fact,
    not a pending question.
    """
    from tests.unit.test_wizard_subflow import _build_subflow_confirmation_config

    config = _build_subflow_confirmation_config()
    subflow_stages = config["subflows"]["team_details"]["stages"]
    team_lead = next(s for s in subflow_stages if s["name"] == "team_lead")
    del team_lead["response_template"]
    team_lead["greeting_template"] = "Now, who is the team lead?"

    async with await BotTestHarness.create(
        wizard_config=config,
        main_responses=["LLM-1", "LLM-2"],
        extraction_results=[
            [{"project_name": "Alpha Project"}],
            [{"lead_name": "Alice Johnson"}],
        ],
    ) as harness:
        await harness.greet()
        pushed = await harness.chat("Alpha Project")
        assert harness.wizard_stage == "team_lead"
        assert pushed.response == "Now, who is the team lead?"

        # The push left the render count at 0, as it must — and the
        # greeting was still recorded, in its own counter.
        assert harness.wizard_data.get("_stage_render_counts", {}).get("team_lead", 0) == 0
        assert harness.wizard_data.get("_stage_greeting_counts", {}).get("team_lead") == 1

        answered = await harness.chat("Alice Johnson")
        assert answered.response != "Now, who is the team lead?"


def test_auto_advance_collects_a_greeting_from_a_message_stage() -> None:
    """A message stage whose only template is a greeting still contributes.

    A unit test for the same reason the collector's other tests are: it
    is reachable only while the wizard is mid-advance.  Before the
    greeting was a field, such a stage read as having no template and
    contributed nothing, silently shortening the turn.
    """
    from dataknobs_bots.reasoning.wizard import WizardReasoning, WizardState
    from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader

    config = (
        WizardConfigBuilder("auto-advance-greeting")
        .stage(
            "notice",
            is_start=True,
            prompt="Say the notice.",
            greeting_template="One moment while I look that up.",
            auto_advance=True,
        )
        .transition("done", "True")
        .stage("done", is_end=True, prompt="Finished.")
        .build()
    )
    fsm = WizardConfigLoader().load_from_dict(config)
    responder = WizardReasoning(wizard_fsm=fsm)._response
    stage = fsm.current_metadata

    state = WizardState(current_stage="notice", data={})
    assert responder._render_auto_advance_template(stage, state) == (
        "One moment while I look that up."
    )
    assert state.get_greeting_count("notice") == 1
    assert state.get_render_count("notice") == 0

    # Advanced past a second time, it has nothing left to add.
    assert responder._render_auto_advance_template(stage, state) is None
