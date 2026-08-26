"""Which FSM the navigator resolves a stage against, once a subflow is pushed.

``WizardNavigator`` holds two references: the **main** FSM, and the subflow
manager that owns the active one. Every method that needs a stage picked
one by hand, and six of them picked the main FSM -- which, inside a push,
does not have the stage. ``_stage_metadata.get(name, {})`` then answers
from an empty dict, and an empty dict is indistinguishable from a stage
that deliberately declared nothing:

* ``can_skip()`` reads ``False``, so a stage declaring ``can_skip: true``
  is told it is required;
* the stage's own ``navigation.skip.keywords`` are never found, so the
  wizard-level defaults apply and the subflow's words are dead;
* ``current_metadata`` is ``{}``, so back renders a stage with no name,
  no prompt, no schema and no template;
* an amendment jump to a subflow stage is not found; and
* restart resets the main FSM while leaving the subflow stack loaded,
  which wedges the wizard -- it can neither push again nor pop.

The same stage config is therefore correct standalone and a dead end when
pushed, with nothing in the config to say so. Every test here observes an
effect **across a push boundary**, which is the coverage the suite had
none of: the existing navigator tests construct a ``SubflowManager`` and
never push through it.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import pytest

from dataknobs_bots.reasoning.wizard import WizardReasoning
from dataknobs_bots.reasoning.wizard_types import WizardState
from dataknobs_bots.testing import BotTestHarness, WizardConfigBuilder

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

#: The stage under test, authored once. Tests 1-3 place this *same dict*
#: at the head of a subflow and at the head of a standalone wizard; the
#: item's claim is that it must mean the same thing in both positions.
#:
#: ``done``/``finished`` are deliberately absent from
#: ``DEFAULT_SKIP_KEYWORDS`` (``skip``, ``skip this``, ``use default``,
#: ``use defaults``), so recognising them proves the *stage's own* config
#: was resolved rather than the wizard-level fallback.
_SKIPPABLE_STAGE: dict[str, Any] = {
    "name": "sub_start",
    "is_start": True,
    "prompt": "Which detail?",
    "response_template": "SUB-START",
    "confirm_first_render": False,
    "can_skip": True,
    "navigation": {"skip": {"keywords": ["done", "finished"]}},
    "schema": {"type": "object", "properties": {"detail": {"type": "string"}}},
    "transitions": [
        {"target": "sub_next", "condition": "has('detail') or has('_skipped_sub_start')"},
    ],
}

#: The skip lands here rather than on an end stage, so the turn does not
#: also complete the flow. That matters for the comparison in test 3:
#: reaching an end stage inside a push pops back to the parent and swaps
#: the parent's data in, which would take ``_skipped_sub_start`` out of
#: view in one arm and not the other for a reason unrelated to this item.
_SUB_NEXT: dict[str, Any] = {
    "name": "sub_next",
    "prompt": "Anything else?",
    "response_template": "SUB-NEXT",
    "confirm_first_render": False,
    "schema": {"type": "object", "properties": {"extra": {"type": "string"}}},
    "transitions": [{"target": "sub_done", "condition": "has('extra')"}],
}

#: The same stage with no ``navigation`` of its own, which is what
#: isolates cause B: the wizard-level ``skip`` still reaches it, so a
#: refusal can only have come from ``can_skip()`` being asked of an FSM
#: without the stage.
_CAN_SKIP_ONLY_STAGE: dict[str, Any] = {
    key: value for key, value in _SKIPPABLE_STAGE.items() if key != "navigation"
}

_SUB_DONE: dict[str, Any] = {
    "name": "sub_done",
    "is_end": True,
    "prompt": "Detail captured.",
    "response_template": "SUB-DONE",
}


def _parent_pushing(subflow: dict[str, Any]) -> WizardConfigBuilder:
    """A parent whose ``gather`` stage pushes *subflow* once a name is set."""
    builder = WizardConfigBuilder("navigation-in-a-subflow")
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
        subflow_network=subflow["name"],
        return_stage="wrap",
    )
    builder.stage("wrap", is_end=True, prompt="All done.", response_template="WRAP")
    builder.subflow(subflow["name"], subflow)
    return builder


def _subflow_config_with(stage: dict[str, Any]) -> dict[str, Any]:
    """A parent pushing a subflow whose start stage is *stage*.

    Every subflow case in this file differs only in that head stage, so
    it is the one thing this takes.
    """
    return _parent_pushing(
        {"name": "detail", "stages": [stage, _SUB_NEXT, _SUB_DONE]},
    ).build()


def _skippable_subflow_config() -> dict[str, Any]:
    """A parent pushing a subflow whose start stage is the shared stage."""
    return _subflow_config_with(_SKIPPABLE_STAGE)


def _can_skip_only_subflow_config() -> dict[str, Any]:
    """A parent pushing a subflow that declares ``can_skip`` and no keywords."""
    return _subflow_config_with(_CAN_SKIP_ONLY_STAGE)


def _skippable_standalone_config() -> dict[str, Any]:
    """The same stage, at the head of a wizard that pushes nothing.

    Built by hand rather than through the builder because the point is
    that ``_SKIPPABLE_STAGE`` is carried across **unmodified** -- a
    builder call would re-author it and weaken the comparison.
    """
    return {
        "name": "navigation-standalone",
        "version": "1.0",
        "stages": [_SKIPPABLE_STAGE, _SUB_NEXT, _SUB_DONE],
    }


def _strategy(harness: BotTestHarness) -> Any:
    """The wizard reasoning strategy driving this harness's bot."""
    return harness.bot.reasoning_strategy


def _live_state(harness: BotTestHarness, stage: str | None = None) -> WizardState:
    """The wizard state the last turn persisted, optionally repositioned.

    Read back rather than constructed, because the subflow stack is the
    thing under test: a hand-built stack would assert against the fixture
    rather than against what a push actually leaves behind. ``stage``
    moves the state within the frame it is already in, for the cases that
    ask about a stage other than the current one.
    """
    manager = harness.bot.get_conversation_manager(harness.context.conversation_id)
    state = _strategy(harness)._get_wizard_state(manager)
    if stage is not None:
        state.current_stage = stage
    return state


# ---------------------------------------------------------------------------
# 1-3. Skip: cause B (the wrong FSM answers) and cause A (the wrong keywords)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_subflow_stage_declaring_can_skip_can_be_skipped() -> None:
    """Cause B: ``can_skip()`` was asked of the FSM without the stage.

    ``_execute_skip`` gated on the **main** FSM while every line after it
    -- including ``navigate_skip``'s own gate eight lines later -- used
    the active one. So the outer gate refused a skip the inner gate would
    have allowed, and the user was told a stage they had marked skippable
    was required.

    The stage here declares no keywords of its own, so the wizard-level
    ``skip`` reaches it and cause A is not in play: a refusal can only be
    ``can_skip()`` answering from the wrong FSM.
    """
    async with await BotTestHarness.create(
        wizard_config=_can_skip_only_subflow_config(),
        main_responses=["r"] * 8,
        extraction_results=[[{"name": "Alice"}], [], []],
    ) as harness:
        await harness.chat("my name is Alice")
        assert harness.wizard_stage == "sub_start", "the subflow was not pushed"

        await harness.chat("skip")

        assert harness.wizard_stage != "sub_start", (
            "the stage declares can_skip: true and the user asked to skip, "
            "but the wizard stayed put"
        )
        assert harness.wizard_data.get("_skipped_sub_start") is True


@pytest.mark.asyncio
async def test_a_subflow_stages_own_skip_keywords_are_recognised() -> None:
    """Cause A: stage-level navigation was read from the main FSM.

    ``_resolve_navigation_config`` looked the stage up in the main FSM's
    metadata, found nothing, and returned the wizard-level default --
    which does not contain ``done``. The subflow's own words never
    reached ``_execute_skip`` at all.
    """
    async with await BotTestHarness.create(
        wizard_config=_skippable_subflow_config(),
        main_responses=["r"] * 8,
        extraction_results=[[{"name": "Alice"}], [], []],
    ) as harness:
        await harness.chat("my name is Alice")
        assert harness.wizard_stage == "sub_start", "the subflow was not pushed"

        await harness.chat("done")

        assert harness.wizard_stage != "sub_start", (
            "'done' is one of the stage's own skip keywords; the wizard "
            "resolved the wizard-level defaults instead and did not "
            "recognise it"
        )


@pytest.mark.asyncio
async def test_a_subflow_stages_keywords_replace_the_defaults_as_they_do_standalone() -> None:
    """The other half of cause A, and the one consumers will notice.

    Stage-level navigation has always used **replace** semantics: the
    keywords a stage declares for a command fully replace the
    wizard-level ones for that command. Resolving that config against the
    active FSM therefore has an effect in both directions -- a subflow
    stage declaring ``[done, finished]`` gains those words and *loses*
    ``skip``, exactly as the same stage does standalone.

    Before the fix a subflow stage's override was inert in both
    directions, so ``skip`` kept working there. A consumer who declared
    subflow skip keywords and also relied on ``skip`` will see it stop;
    that is the config finally meaning what it says, and it is the same
    thing it has always meant outside a push.

    Asserted on the resolved config rather than on a turn's outcome,
    because a turn cannot tell the two states apart: before the fix
    ``skip`` was recognised and then refused by ``can_skip()``, which
    leaves the wizard on the same stage that correctly replacing the
    keywords leaves it on.
    """
    async with await BotTestHarness.create(
        wizard_config=_skippable_subflow_config(),
        main_responses=["r"] * 8,
        extraction_results=[[{"name": "Alice"}], [], []],
    ) as harness:
        await harness.chat("my name is Alice")
        assert harness.wizard_stage == "sub_start", "the subflow was not pushed"

        keywords = (
            _strategy(harness)
            ._navigator._resolve_navigation_config(_live_state(harness))
            .skip.keywords
        )

        assert "done" in keywords, (
            f"the stage's own skip keywords did not reach the resolver: {keywords}"
        )
        assert "skip" not in keywords, (
            "stage-level navigation replaces the wizard-level keywords, so "
            "a stage declaring [done, finished] must not also answer to "
            f"'skip': {keywords}"
        )


@pytest.mark.asyncio
async def test_the_default_skip_keywords_still_apply_where_a_stage_declares_none() -> None:
    """Anti-overreach: resolving against the active FSM keeps the fallback.

    The fix changes *which* FSM the stage is looked up in, not what
    happens when the stage genuinely declares no navigation of its own.
    ``sub_done`` declares none, so the wizard-level defaults must still
    reach it -- otherwise cause A would have been traded for its mirror.
    """
    async with await BotTestHarness.create(
        wizard_config=_skippable_subflow_config(),
        main_responses=["r"] * 8,
        extraction_results=[[{"name": "Alice"}], [], []],
    ) as harness:
        await harness.chat("my name is Alice")
        strategy = _strategy(harness)

        resolved = strategy._navigator._resolve_navigation_config(_live_state(harness, "sub_done"))

        assert "skip" in resolved.skip.keywords, (
            "a subflow stage declaring no navigation of its own lost the wizard-level defaults"
        )


@pytest.mark.asyncio
async def test_the_same_stage_config_skips_in_both_positions() -> None:
    """The item's actual claim, as one assertion over two runs.

    ``_SKIPPABLE_STAGE`` is the same dict object in both configs. A stage
    whose behaviour depends on whether it was reached directly or through
    a push is a config that cannot be reasoned about, and this is the
    acceptance criterion for the fix.

    What is compared is whether the *stage* was skipped, not where the
    wizard ended up. The two arms legitimately diverge afterwards: pushed,
    reaching the subflow's end stage pops back into the parent's return
    stage, so the final stage names differ for a reason that has nothing
    to do with this item.
    """
    outcomes: dict[str, tuple[bool, bool]] = {}

    for position, config in (
        ("standalone", _skippable_standalone_config()),
        ("pushed", _skippable_subflow_config()),
    ):
        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["r"] * 8,
            extraction_results=[[{"name": "Alice"}], [], []],
        ) as harness:
            if position == "pushed":
                await harness.chat("my name is Alice")
                assert harness.wizard_stage == "sub_start", "the subflow was not pushed"
            else:
                await harness.greet()

            await harness.chat("done")
            outcomes[position] = (
                harness.wizard_data.get("_skipped_sub_start") is True,
                harness.wizard_stage != "sub_start",
            )

    assert outcomes["pushed"] == outcomes["standalone"], (
        "one stage config, two meanings: standalone gave "
        f"(skipped, moved)={outcomes['standalone']}, pushed gave "
        f"{outcomes['pushed']}"
    )
    assert outcomes["standalone"] == (True, True), (
        "the standalone arm did not skip either, so this test proves nothing about the pushed one"
    )


# ---------------------------------------------------------------------------
# 4-5. The neighbouring sites the same audit found
# ---------------------------------------------------------------------------


def _two_stage_subflow() -> dict[str, Any]:
    """A subflow deep enough to go back inside, with a mappable stage name.

    ``configure_knowledge`` is one of ``map_section_to_stage``'s built-in
    section targets, which is what test 5 needs to reach the membership
    test without a custom mapping (a custom mapping returns before it).
    """
    return {
        "name": "detail",
        "stages": [
            {
                "name": "sub_a",
                "is_start": True,
                "prompt": "A?",
                "response_template": "SUB-A",
                "confirm_first_render": False,
                "schema": {"type": "object", "properties": {"a": {"type": "string"}}},
                "transitions": [{"target": "configure_knowledge", "condition": "has('a')"}],
            },
            {
                "name": "configure_knowledge",
                "prompt": "B?",
                "response_template": "SUB-B",
                "confirm_first_render": False,
                "schema": {"type": "object", "properties": {"b": {"type": "string"}}},
                "transitions": [{"target": "sub_done", "condition": "has('b')"}],
            },
            _SUB_DONE,
        ],
    }


@pytest.mark.asyncio
async def test_back_inside_a_subflow_renders_the_subflow_stage() -> None:
    """Back moved the FSM correctly and then rendered an *empty* stage.

    ``navigate_back`` already used the active FSM, so the FSM landed on
    the right stage -- but ``_execute_back`` then read
    ``self._fsm.current_metadata``, and the main FSM's ``current_stage``
    is a *subflow* stage name it does not have. The result is ``{}``: no
    name, no prompt, no schema, no template. The user gets a bare LLM
    reply where the stage's own template should be.
    """
    async with await BotTestHarness.create(
        wizard_config=_parent_pushing(_two_stage_subflow()).build(),
        main_responses=[f"ECHO-{i}" for i in range(12)],
        extraction_results=[[{"name": "Alice"}], [{"a": "AAA"}], [], []],
    ) as harness:
        await harness.chat("my name is Alice")
        await harness.chat("a is AAA")
        assert harness.wizard_stage == "configure_knowledge", "the subflow did not advance"

        await harness.chat("back")

        assert harness.wizard_stage == "sub_a", "back did not move the FSM"
        assert harness.last_response == "SUB-A", (
            "back landed on 'sub_a' but rendered someone else's stage: "
            f"{harness.last_response!r} rather than the stage's own template"
        )


@pytest.mark.asyncio
async def test_an_amendment_jump_to_a_subflow_stage_is_found() -> None:
    """``map_section_to_stage`` tested membership in the wrong FSM.

    The built-in section table maps ``kb`` to ``configure_knowledge``,
    then confirms the wizard actually has that stage before returning it.
    Asked of the main FSM while a subflow owning that stage is active, the
    membership test fails and the amendment silently finds nothing.
    """
    async with await BotTestHarness.create(
        wizard_config=_parent_pushing(_two_stage_subflow()).build(),
        main_responses=["r"] * 8,
        extraction_results=[[{"name": "Alice"}], []],
    ) as harness:
        await harness.chat("my name is Alice")
        assert harness.wizard_stage == "sub_a", "the subflow was not pushed"

        mapped = _strategy(harness)._map_section_to_stage("kb")

        assert mapped == "configure_knowledge", (
            "'kb' maps to a stage the active subflow has, but the "
            f"membership test was run against the main FSM: got {mapped!r}"
        )


@pytest.mark.asyncio
async def test_a_section_naming_no_stage_anywhere_is_still_unmapped() -> None:
    """Anti-overreach for the site above: the membership test still bites.

    Widening the lookup to the active FSM must not turn it into "return
    the built-in mapping unconditionally" -- a section whose target stage
    exists in neither FSM has to stay ``None``.
    """
    async with await BotTestHarness.create(
        wizard_config=_parent_pushing(_two_stage_subflow()).build(),
        main_responses=["r"] * 8,
        extraction_results=[[{"name": "Alice"}], []],
    ) as harness:
        await harness.chat("my name is Alice")

        assert _strategy(harness)._map_section_to_stage("behavior") is None, (
            "'behavior' maps to 'configure_behavior', which neither FSM "
            "has; the membership test stopped being applied"
        )


# ---------------------------------------------------------------------------
# 6-7. Restart: the one site where "the active FSM" is the wrong answer
# ---------------------------------------------------------------------------
#
# Restart is deliberately NOT converted. A restart returns the user to the
# *main* flow's start stage, so restarting the main FSM is correct -- what
# was missing is that the subflow state around it was left standing. The
# two tests below are the two halves of that: the stack is unwound, and
# the wizard can still be driven afterwards.


def _restartable_config() -> dict[str, Any]:
    return _parent_pushing(
        {
            "name": "detail",
            "stages": [
                {
                    "name": "sub_a",
                    "is_start": True,
                    "prompt": "A?",
                    "response_template": "SUB-A",
                    "confirm_first_render": False,
                    "schema": {"type": "object", "properties": {"a": {"type": "string"}}},
                    "transitions": [{"target": "sub_done", "condition": "has('a')"}],
                },
                _SUB_DONE,
            ],
        },
    ).build()


@pytest.mark.asyncio
async def test_restart_inside_a_subflow_leaves_the_subflow() -> None:
    """Restart reset the main FSM and left the subflow stack loaded.

    Afterwards the wizard reported ``current_stage == "gather"`` while
    ``get_active_fsm()`` still returned the subflow's FSM -- so the user
    was told they were back at the start and shown the subflow's prompt,
    schema and template.
    """
    async with await BotTestHarness.create(
        wizard_config=_restartable_config(),
        main_responses=["r"] * 8,
        extraction_results=[[{"name": "Alice"}], [], []],
    ) as harness:
        await harness.chat("my name is Alice")
        assert harness.wizard_stage == "sub_a", "the subflow was not pushed"

        await harness.chat("start over")

        assert harness.wizard_stage == "gather"
        assert (harness.wizard_state or {}).get("subflow_depth") == 0, (
            "restart returned to the main flow's start stage with the subflow stack still loaded"
        )

        subflows = _strategy(harness)._subflows
        assert subflows.active_subflow_fsm is None, (
            "the active subflow FSM outlived the restart, so the wizard "
            "renders the subflow's stage under the main stage's name"
        )


@pytest.mark.asyncio
async def test_after_a_restart_the_wizard_can_push_again() -> None:
    """The half that makes it a wedge rather than a cosmetic mismatch.

    ``should_push`` declines while ``is_in_subflow`` is true, and
    ``should_pop`` needs an end stage of the subflow -- which ``gather``
    is not. So a wizard restarted inside a subflow could neither enter
    one nor leave one, permanently. Restart is the escape hatch of last
    resort, and it was what put the wizard there.
    """
    async with await BotTestHarness.create(
        wizard_config=_restartable_config(),
        main_responses=["r"] * 12,
        extraction_results=[[{"name": "Alice"}], [], [{"name": "Bob"}], []],
    ) as harness:
        await harness.chat("my name is Alice")
        await harness.chat("start over")
        assert harness.wizard_stage == "gather", "restart did not return to the start"

        await harness.chat("my name is Bob")

        assert harness.wizard_stage == "sub_a", (
            "after a restart the wizard answered the start stage again "
            "and did not push; it can neither enter the subflow nor "
            f"leave it (stage: {harness.wizard_stage!r})"
        )


# ---------------------------------------------------------------------------
# 8-11. What resolving against the right FSM newly reaches
# ---------------------------------------------------------------------------
#
# Cause A is what kept a subflow stage's ``navigation`` block from being
# read at all: the main FSM does not have the stage, so
# ``stage_metadata_for`` answered ``{}`` and the method returned at
# ``if not stage_nav``. The block was therefore never *used*, and a
# wrong-typed one was inert rather than wrong.
#
# Resolving against the FSM that owns the stage removes that mask, and
# what it uncovers is a block read with no type check: ``stage_nav`` is
# handed straight to ``.get()``, as is each command under it, and
# ``keywords`` is iterated. Measured on this branch before the guards
# below existed, with the stage placed at the head of a subflow:
#
#     navigation: "yes"               -> AttributeError on the turn after the push
#     navigation: {skip: "yes"}       -> AttributeError, one level down
#     navigation.skip.keywords: "done" -> ('d', 'o', 'n', 'e')
#
# and, evaluated on the same runtime state, the pre-fix expression
# returns ``{}`` for all three. So these are not a new defect -- a
# *main-flow* stage has always been able to reach them -- but the fix
# above widens the set of configs that do, which is why the guards
# belong with it rather than after it.
#
# The guards fall back to the wizard-level config, which is what
# ``_stage_field`` does one file over for the same reason: a stage whose
# config cannot be read gets the documented default, and the reader is
# told once rather than every turn.


def _navigation_stage(navigation: Any) -> dict[str, Any]:
    """``_SKIPPABLE_STAGE`` with *navigation* replacing its own block."""
    return {**_SKIPPABLE_STAGE, "navigation": navigation}


@pytest.mark.asyncio
async def test_a_mistyped_navigation_block_falls_back_to_the_wizard_level() -> None:
    """A stage declaring ``navigation:`` as a scalar must not end the turn.

    Before the guard this raised ``AttributeError: 'str' object has no
    attribute 'get'`` out of ``stage_nav.get("back")`` -- on an ordinary
    turn, inside the bot, from a config that loaded without complaint.

    Falling back is asserted through *behaviour* rather than by reading
    the returned config back: the wizard-level ``skip`` still has to
    reach the stage, which is the whole content of "fall back to the
    wizard-level config".
    """
    async with await BotTestHarness.create(
        wizard_config=_subflow_config_with(_navigation_stage("yes")),
        main_responses=["r"] * 8,
        extraction_results=[[{"name": "Alice"}], [], []],
    ) as harness:
        await harness.chat("my name is Alice")
        assert harness.wizard_stage == "sub_start", "the subflow was not pushed"

        await harness.chat("skip")

        assert harness.wizard_stage != "sub_start", (
            "the stage's navigation block is unreadable, so the wizard-level "
            "'skip' should have reached it"
        )
        assert harness.wizard_data.get("_skipped_sub_start") is True


@pytest.mark.asyncio
async def test_a_mistyped_command_block_falls_back_to_the_wizard_level() -> None:
    """The same defect one level down: ``navigation.skip`` is a scalar.

    ``stage_nav`` is a mapping here and passes the outer guard, so this
    covers the second ``.get()`` -- the one inside ``_merge_command``,
    which the outer guard cannot reach. ``back`` and ``restart`` are
    absent and must keep inheriting, so a single bad command must not
    discard the two beside it.
    """
    async with await BotTestHarness.create(
        wizard_config=_subflow_config_with(_navigation_stage({"skip": "yes"})),
        main_responses=["r"] * 8,
        extraction_results=[[{"name": "Alice"}], [], []],
    ) as harness:
        await harness.chat("my name is Alice")
        assert harness.wizard_stage == "sub_start", "the subflow was not pushed"

        await harness.chat("skip")

        assert harness.wizard_stage != "sub_start"
        assert harness.wizard_data.get("_skipped_sub_start") is True


@pytest.mark.asyncio
async def test_a_keywords_string_is_not_one_keyword_per_character() -> None:
    """``keywords: "done"`` iterated to ``('d', 'o', 'n', 'e')``.

    The quiet half. Nothing raises, nothing is logged, and the stage
    acquires four one-letter skip keywords -- so a user answering ``d``
    skips a stage they meant to fill in. This is the failure mode the
    crash above is preferable to, and it is the one a config author has
    no way to notice.

    Both halves are asserted: the characters must not skip, and the
    documented fallback must.
    """
    async with await BotTestHarness.create(
        wizard_config=_subflow_config_with(
            _navigation_stage({"skip": {"keywords": "done"}}),
        ),
        main_responses=["r"] * 10,
        extraction_results=[[{"name": "Alice"}], [], [], []],
    ) as harness:
        await harness.chat("my name is Alice")
        assert harness.wizard_stage == "sub_start", "the subflow was not pushed"

        await harness.chat("d")

        assert harness.wizard_stage == "sub_start", (
            "'d' is a character of the keyword 'done', not a keyword; the "
            "stage was skipped by a letter"
        )
        assert "_skipped_sub_start" not in harness.wizard_data

        await harness.chat("skip")

        assert harness.wizard_stage != "sub_start", (
            "the keywords are unreadable, so the wizard-level 'skip' should have reached the stage"
        )


@pytest.mark.asyncio
async def test_an_unreadable_navigation_block_is_reported_once_per_stage() -> None:
    """Silence here would be the defect this item is a family of.

    The report is at WARNING and is de-duplicated per ``(stage, field)``,
    which is the discipline ``WizardFSM._stage_field`` uses for the same
    problem: this method runs on every navigation check on every turn, so
    an unthrottled line would say the same thing for the life of the
    conversation.

    A load-time check naming the stage would be better still and is not
    this PR's -- ``_validate_config`` already runs six warning checks once
    per load, and a seventh belongs with the rest of that work.
    """
    async with await BotTestHarness.create(
        wizard_config=_subflow_config_with(_navigation_stage("yes")),
        main_responses=["r"] * 10,
        extraction_results=[[{"name": "Alice"}], [], [], []],
    ) as harness:
        await harness.chat("my name is Alice")

        navigator = _strategy(harness)._navigator
        records: list[str] = []

        class _Collect(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record.getMessage())

        handler = _Collect()
        nav_logger = logging.getLogger("dataknobs_bots.reasoning.wizard_navigation")
        nav_logger.addHandler(handler)
        state = _live_state(harness)
        try:
            navigator._resolve_navigation_config(state)
            navigator._resolve_navigation_config(state)
        finally:
            nav_logger.removeHandler(handler)

        reported = [message for message in records if "navigation" in message]
        assert len(reported) == 1, f"expected one report per stage and field, got {reported}"
        assert "sub_start" in reported[0]


# ---------------------------------------------------------------------------
# 12-13. The same defect one class up: the read-only state snapshot
# ---------------------------------------------------------------------------
#
# ``WizardNavigator`` was six sites picking an FSM by hand. It is not the
# only class that picks: ``WizardReasoning.get_state_snapshot`` asks
# ``self._fsm`` for the current stage's metadata, its skippability and
# its position, and inside a push the main FSM does not have that stage.
# The snapshot is the documented way a UI reads wizard state -- the
# observability guide drives a progress bar and a skip button straight
# off these fields -- so the failure is a wrong progress bar and a
# missing skip button, not an exception anyone would notice in a log.
#
# The canonical resolver for this class already exists and is used by
# ``_build_wizard_metadata``: ``_fsm_for_state(state)``, which reads the
# subflow stack off the *state* rather than off a per-turn attribute, and
# so is also correct outside a turn -- which is exactly what a snapshot
# is.


def _snapshot_config() -> dict[str, Any]:
    """A parent whose **second** stage pushes the subflow.

    The pushing stage's index is the whole point. A subflow stage's name
    is absent from the main flow's ``stage_names``, and ``stage_position``
    reports index ``0`` for a name it cannot find -- so with the pushing
    stage at index 0 the wrong answer and the right one coincide and the
    assertion proves nothing. Here the parent stage is index 1.

    ``gather`` carries ``can_skip`` and ``suggestions`` of its own so the
    same config can also be observed from *outside* a subflow, which is
    where ``snapshot_from_metadata`` turns out to be wrong as well.
    """
    builder = WizardConfigBuilder("snapshot-in-a-subflow")
    builder.stage(
        "intro",
        is_start=True,
        prompt="Say hello.",
        response_template="INTRO",
        confirm_first_render=False,
    )
    builder.field("greeting", field_type="string", required=True)
    builder.transition("gather", condition="has('greeting')")
    builder.stage(
        "gather",
        can_skip=True,
        suggestions=["Alice", "Bob"],
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
    )
    builder.stage("wrap", is_end=True, prompt="All done.", response_template="WRAP")
    builder.subflow(
        "detail",
        {
            "name": "detail",
            "stages": [
                {**_SKIPPABLE_STAGE, "suggestions": ["Colour", "Size"]},
                _SUB_NEXT,
                _SUB_DONE,
            ],
        },
    )
    return builder.build()


def _snapshot_inside_the_subflow(harness: BotTestHarness) -> Any:
    """Drive the wizard into the subflow and take a snapshot there."""
    manager = harness.bot.get_conversation_manager(harness.context.conversation_id)
    return _strategy(harness).get_state_snapshot(manager)


@pytest.mark.asyncio
async def test_the_snapshot_describes_the_subflow_stage_it_is_standing_on() -> None:
    """The stage-derived fields came from an FSM without the stage.

    ``can_skip`` and ``suggestions`` are read off the stage's metadata,
    which the main FSM answers as ``{}`` inside a push -- so a skippable
    stage reports as required and its quick replies vanish. Both are
    documented UI inputs: the observability guide shows a skip button
    gated on ``can_skip`` and quick-reply buttons built from
    ``suggestions``.
    """
    async with await BotTestHarness.create(
        wizard_config=_snapshot_config(),
        main_responses=["r"] * 10,
        extraction_results=[[{"greeting": "hi"}], [{"name": "Alice"}], [], []],
    ) as harness:
        await harness.chat("hi")
        await harness.chat("my name is Alice")
        assert harness.wizard_stage == "sub_start", "the subflow was not pushed"

        snapshot = _snapshot_inside_the_subflow(harness)

        assert snapshot.current_stage == "sub_start"
        assert snapshot.can_skip is True, (
            "the stage declares can_skip: true; the snapshot asked an FSM "
            "that does not have the stage and got the default False"
        )
        assert snapshot.suggestions == ["Colour", "Size"], (
            "the stage's quick replies were read from the wrong FSM"
        )


@pytest.mark.asyncio
async def test_the_snapshot_reports_main_flow_progress_while_inside_a_subflow() -> None:
    """Progress stays main-flow, and it has to be asked for correctly.

    ``stage_index`` is deliberately a *main-flow* number -- a subflow is
    not a step of the outer flow, and ``_build_wizard_metadata`` reports
    the parent stage that pushed it. The snapshot passed the subflow's
    stage name to the main flow's ``stage_names`` instead, which
    ``stage_position`` cannot find and reports as index ``0``: a progress
    bar that jumps back to the start whenever a subflow opens.

    So resolving the FSM is not sufficient here. Asking the *subflow's*
    stage_names would be a second wrong answer -- right stage, wrong
    flow. The parent stage is the answer, and this pins it.
    """
    async with await BotTestHarness.create(
        wizard_config=_snapshot_config(),
        main_responses=["r"] * 10,
        extraction_results=[[{"greeting": "hi"}], [{"name": "Alice"}], [], []],
    ) as harness:
        await harness.chat("hi")
        await harness.chat("my name is Alice")
        assert harness.wizard_stage == "sub_start", "the subflow was not pushed"

        snapshot = _snapshot_inside_the_subflow(harness)

        assert snapshot.stage_index == 1, (
            "'gather' is the main-flow stage that pushed this subflow and it "
            "is index 1; index 0 is stage_position failing to find the "
            "subflow's stage name among the main flow's"
        )
        assert snapshot.total_stages == 3


# ---------------------------------------------------------------------------
# 14-16. The static sibling: snapshot_from_metadata
# ---------------------------------------------------------------------------
#
# ``WizardStateSnapshot`` has two constructors. ``get_state_snapshot()``
# asks an FSM; ``snapshot_from_metadata()`` is the documented path for
# "you have the conversation metadata but not the instance", and it
# recomputes the stage-derived fields from ``fsm_state`` plus a caller-
# supplied ``stage_definitions``.
#
# It recomputes them worse, and the right answers are one level up in the
# very dict it is handed: ``manager.metadata["wizard"]`` is
# ``_build_wizard_metadata(state)`` output with ``fsm_state`` nested
# inside it, so ``stage_index``, ``total_stages``, ``stages``,
# ``can_skip``, ``can_go_back`` and ``suggestions`` are all sitting there
# already, derived by the canonical writer and subflow-aware.
#
# Two distinct failures, and only the second is about subflows:
#
# * ``can_skip``, ``can_go_back`` and ``suggestions`` were never passed to
#   the constructor **at all**, so they took the dataclass defaults
#   (False / True / []) in every flow, subflow or not.  A UI built on this
#   path never showed a skip button and never showed a quick reply.
# * inside a push, the recomputation locates a subflow stage name among
#   the main flow's definitions, finds nothing, and reports index 0 with
#   no stage marked "current" in the roadmap.
#
# ``stage_definitions`` stays as the fallback, for metadata written before
# the canonical fields existed or hand-built by a caller.


def _wizard_metadata(harness: BotTestHarness) -> dict[str, Any]:
    """The conversation metadata a consumer would hold."""
    manager = harness.bot.get_conversation_manager(harness.context.conversation_id)
    return dict(manager.metadata)


def _main_stage_definitions() -> list[dict[str, Any]]:
    """The main flow's stage list, as the documented example passes it."""
    return _snapshot_config()["stages"]


@pytest.mark.asyncio
async def test_the_static_snapshot_reports_the_actions_the_stage_allows() -> None:
    """Not a subflow bug: these three were never populated in any flow.

    ``snapshot_from_metadata`` omitted ``can_skip``, ``can_go_back`` and
    ``suggestions`` from the constructor call entirely, so they fell to
    the dataclass defaults. The two constructors of one type therefore
    disagreed about three fields everywhere, not merely inside a push --
    and the correct values were already in the metadata being read.

    Observed in the **main** flow, before any subflow is pushed, so
    nothing here depends on the subflow machinery.
    """
    async with await BotTestHarness.create(
        wizard_config=_snapshot_config(),
        main_responses=["r"] * 10,
        extraction_results=[[{"greeting": "hi"}], [], []],
    ) as harness:
        await harness.chat("hi")
        assert harness.wizard_stage == "gather", "expected the main flow's second stage"

        snapshot = WizardReasoning.snapshot_from_metadata(
            _wizard_metadata(harness),
            stage_definitions=_main_stage_definitions(),
        )

        assert snapshot is not None
        assert snapshot.can_skip is True, (
            "'gather' declares can_skip: true and the metadata says so; the "
            "snapshot never read the field and took the dataclass default"
        )
        assert snapshot.suggestions == ["Alice", "Bob"], (
            "the stage's quick replies are in the metadata and were not read"
        )


@pytest.mark.asyncio
async def test_the_static_snapshot_locates_a_subflow_stage_in_the_main_flow() -> None:
    """Inside a push, the recomputation has nothing to match against.

    ``stage_definitions`` is the **main** flow's stage list, and the
    current stage is the subflow's. So ``stage_position`` reported index
    0 and the roadmap loop marked nothing ``"current"`` -- both fixed by
    reading what the canonical writer already derived, which reports the
    parent stage that pushed the subflow.
    """
    async with await BotTestHarness.create(
        wizard_config=_snapshot_config(),
        main_responses=["r"] * 10,
        extraction_results=[[{"greeting": "hi"}], [{"name": "Alice"}], [], []],
    ) as harness:
        await harness.chat("hi")
        await harness.chat("my name is Alice")
        assert harness.wizard_stage == "sub_start", "the subflow was not pushed"

        snapshot = WizardReasoning.snapshot_from_metadata(
            _wizard_metadata(harness),
            stage_definitions=_main_stage_definitions(),
        )

        assert snapshot is not None
        assert snapshot.stage_index == 1, (
            "'gather' pushed this subflow and is index 1; index 0 is the "
            "recomputation failing to find the subflow's stage name among "
            "the main flow's definitions"
        )
        current = [entry for entry in snapshot.stages if entry["status"] == "current"]
        assert [entry["name"] for entry in current] == ["gather"], (
            "the roadmap marked no main-flow stage current because the name "
            "it compared against belongs to the subflow"
        )
        assert snapshot.can_skip is True, "the subflow stage declares can_skip: true"


def test_metadata_without_the_derived_fields_still_uses_stage_definitions() -> None:
    """The fallback is kept, and this is what keeps it honest.

    Metadata written before the canonical fields existed -- or built by
    hand, as a caller with only ``fsm_state`` would -- has no
    ``stage_index`` to prefer. ``stage_definitions`` still drives the
    position and the roadmap there, which is the behaviour every existing
    caller has.

    A pure unit: no bot, no FSM, just the two shapes of input.
    """
    snapshot = WizardReasoning.snapshot_from_metadata(
        {"wizard": {"fsm_state": {"current_stage": "gather", "history": ["intro"]}}},
        stage_definitions=[{"name": "intro"}, {"name": "gather"}, {"name": "wrap"}],
    )

    assert snapshot is not None
    assert snapshot.stage_index == 1
    assert snapshot.total_stages == 3
    assert [entry["status"] for entry in snapshot.stages] == [
        "completed",
        "current",
        "pending",
    ]


# ---------------------------------------------------------------------------
# 17. The same defect one class over: the stage context template
# ---------------------------------------------------------------------------
#
# ``WizardResponder._render_custom_context`` builds the ``can_skip`` /
# ``can_go_back`` variables a ``settings.context_template`` renders, and it
# asks ``self._fsm`` -- the **main** FSM -- with no stage, so the answer
# comes from that FSM's live position. Inside a push the position is a
# subflow stage the main FSM does not have, and ``_stage_field`` returns
# the documented default.
#
# What makes this worth its own section rather than a footnote: fixing
# ``_execute_skip`` without fixing this makes the two *disagree*. The
# wizard now allows the skip while the system prompt it sent the model on
# the same turn says the step is required. Before, both said required.
#
# ``_build_wizard_metadata`` is the site that already does this correctly
# -- ``active_fsm.can_skip(stage)``, stage passed explicitly -- and it
# renders ``stage_prompt`` with the very same two-key ``extra_context``.


_CONTEXT_TEMPLATE = "SKIPPABLE={{ can_skip }}|STAGE={{ stage_name }}"


def _context_template_config(*, can_skip: bool) -> dict[str, Any]:
    """A parent pushing a subflow whose head stage has no template.

    ``build_stage_context`` runs on the LLM response path only, so the
    stage must not declare a ``response_template`` -- a template-mode
    stage never reaches the model and there is no system prompt to
    inspect.
    """
    stage: dict[str, Any] = {
        "name": "sub_start",
        "is_start": True,
        "prompt": "Which detail?",
        "confirm_first_render": False,
        "schema": {"type": "object", "properties": {"detail": {"type": "string"}}},
        "transitions": [{"target": "sub_done", "condition": "has('detail')"}],
    }
    if can_skip:
        stage["can_skip"] = True
    builder = _parent_pushing({"name": "detail", "stages": [stage, _SUB_DONE]})
    builder.settings(context_template=_CONTEXT_TEMPLATE)
    return builder.build()


def _last_system_prompt(harness: BotTestHarness) -> str:
    """The system prompt of the most recent model call.

    ``ConversationManager.complete(system_prompt_override=...)`` puts it
    in the first message, so this is what the model was actually told --
    the observable the context template exists to produce.
    """
    call = harness.provider.get_last_call()
    assert call is not None, "the wizard never called the model"
    for message in call.get("messages", []):
        if getattr(message, "role", None) == "system":
            return str(getattr(message, "content", ""))
    raise AssertionError("the model call carried no system message")


def _rendered_can_skip(harness: BotTestHarness) -> str:
    """The value the context template rendered for ``can_skip``.

    Read back rather than string-matched: mixed-mode rendering pads a
    substitution with spaces, and the subject here is the value, not the
    whitespace around it.
    """
    prompt = _last_system_prompt(harness)
    match = re.search(r"SKIPPABLE=\s*(\S+?)\s*\|", prompt)
    assert match is not None, f"the context template did not render: {prompt!r}"
    return match.group(1)


@pytest.mark.asyncio
async def test_the_context_template_sees_the_subflow_stages_can_skip() -> None:
    """The custom context template rendered the main FSM's answer.

    A stage declaring ``can_skip: true`` inside a push renders
    ``can_skip`` as **False** into the system prompt, so an author whose
    template says "this step is optional" tells the user the opposite --
    on the same turn the wizard would in fact accept a skip.
    """
    async with await BotTestHarness.create(
        wizard_config=_context_template_config(can_skip=True),
        main_responses=["r"] * 8,
        extraction_results=[[{"name": "Alice"}], []],
    ) as harness:
        await harness.chat("my name is Alice")
        assert harness.wizard_stage == "sub_start", "the subflow was not pushed"

        assert _rendered_can_skip(harness) == "True", (
            "the stage declares can_skip: true, but the context template "
            "was handed the main FSM's answer for a stage it does not have: "
            f"{_last_system_prompt(harness)!r}"
        )


@pytest.mark.asyncio
async def test_the_context_template_still_reports_an_unskippable_stage() -> None:
    """Anti-overreach: resolving against the right FSM is not "always True".

    The same subflow stage with no ``can_skip`` of its own must still
    render False -- otherwise the fix has replaced one constant answer
    with another.
    """
    async with await BotTestHarness.create(
        wizard_config=_context_template_config(can_skip=False),
        main_responses=["r"] * 8,
        extraction_results=[[{"name": "Alice"}], []],
    ) as harness:
        await harness.chat("my name is Alice")
        assert harness.wizard_stage == "sub_start", "the subflow was not pushed"

        assert _rendered_can_skip(harness) == "False"


# ---------------------------------------------------------------------------
# 18. Amendments across a frame boundary
# ---------------------------------------------------------------------------
#
# Widening ``map_section_to_stage``'s membership test to the active FSM
# narrowed it at the same time: while a subflow is pushed, the active FSM
# is the only one consulted, so a section naming a **main-flow** stage
# stops resolving. That is reachable -- ``CompleteWizardTool`` sets
# ``completed`` with no subflow guard, so a wizard can be completed
# inside a push and the next turn routes to ``handle_amendment``.
#
# Both frames are legitimate targets, so the lookup asks both. Landing in
# the other frame then has to unwind to it: restoring the *subflow's* FSM
# to a main-flow stage is the original defect wearing a different hat.


def _amendment_across_frames_config() -> dict[str, Any]:
    """A main flow carrying a built-in section target, plus a subflow.

    ``configure_llm`` is what the built-in table maps ``llm`` to, and it
    lives in the **main** flow -- the frame a pushed subflow hides.
    """
    builder = WizardConfigBuilder("amendment-across-frames")
    builder.stage(
        "gather",
        is_start=True,
        prompt="Tell me your name.",
        response_template="Noted.",
        confirm_first_render=False,
    )
    builder.field("name", field_type="string", required=True)
    builder.transition(
        "configure_llm",
        condition="has('name')",
        subflow_network="detail",
        return_stage="configure_llm",
    )
    builder.stage(
        "configure_llm",
        prompt="Which model?",
        response_template="MODEL",
        confirm_first_render=False,
    )
    builder.field("model", field_type="string", required=True)
    builder.transition("wrap", condition="has('model')")
    builder.stage("wrap", is_end=True, prompt="All done.", response_template="WRAP")
    builder.subflow(
        "detail",
        {"name": "detail", "stages": [_SKIPPABLE_STAGE, _SUB_NEXT, _SUB_DONE]},
    )
    builder.settings(allow_post_completion_edits=True)
    return builder.build()


@pytest.mark.asyncio
async def test_a_section_naming_a_main_flow_stage_survives_a_push() -> None:
    """The other half of the membership widening.

    ``llm`` maps to ``configure_llm``, which the main flow has and the
    subflow does not. Asking only the active FSM answers ``None`` and the
    amendment silently finds nothing -- the same silence the item fixed
    in the opposite direction.
    """
    async with await BotTestHarness.create(
        wizard_config=_amendment_across_frames_config(),
        main_responses=["r"] * 8,
        extraction_results=[[{"name": "Alice"}], []],
    ) as harness:
        await harness.chat("my name is Alice")
        assert harness.wizard_stage == "sub_start", "the subflow was not pushed"

        mapped = _strategy(harness)._map_section_to_stage("llm")

        assert mapped == "configure_llm", (
            "'llm' maps to a stage the main flow has; asking only the "
            f"active subflow's FSM lost it: got {mapped!r}"
        )


@pytest.mark.asyncio
async def test_an_amendment_to_a_main_flow_stage_leaves_the_subflow() -> None:
    """Finding the stage is not enough -- the wizard has to get to it.

    An amendment jump out of a subflow has to unwind the stack, for the
    same reason restart does: leaving it loaded means the wizard reports
    the main stage's name while the subflow's FSM answers for it, and it
    can neither push again nor pop.

    ``completed`` is set on the persisted state, which is exactly what
    ``CompleteWizardTool`` does to reach this path from inside a push.
    """
    async with await BotTestHarness.create(
        wizard_config=_amendment_across_frames_config(),
        main_responses=["r"] * 10,
        extraction_results=[
            [{"name": "Alice"}],
            [{"wants_edit": True, "target_section": "llm"}],
        ],
    ) as harness:
        await harness.chat("my name is Alice")
        assert harness.wizard_stage == "sub_start", "the subflow was not pushed"

        manager = harness.bot.get_conversation_manager(harness.context.conversation_id)
        manager.metadata["wizard"]["fsm_state"]["completed"] = True

        await harness.chat("actually, change the llm")

        assert harness.wizard_stage == "configure_llm", (
            "the amendment named a main-flow stage and did not land on it "
            f"(stage: {harness.wizard_stage!r})"
        )
        assert harness.last_response == "MODEL", (
            "the wizard landed on 'configure_llm' but rendered someone "
            f"else's stage: {harness.last_response!r}"
        )
        assert (harness.wizard_state or {}).get("subflow_depth") == 0, (
            "the amendment left the main flow's stage current with the subflow stack still loaded"
        )


@pytest.mark.asyncio
async def test_an_amendment_within_the_subflow_stays_in_the_subflow() -> None:
    """Anti-overreach: consulting the main FSM must not preempt the active one.

    A section whose target exists in **both** frames has to resolve to
    the frame the user is standing in, and must not unwind on the way.
    """
    async with await BotTestHarness.create(
        wizard_config=_amendment_across_frames_config(),
        main_responses=["r"] * 8,
        extraction_results=[[{"name": "Alice"}], []],
    ) as harness:
        await harness.chat("my name is Alice")
        assert harness.wizard_stage == "sub_start", "the subflow was not pushed"

        navigator = _strategy(harness)._navigator
        navigator._section_to_stage_mapping = {"detail": "sub_next"}

        assert _strategy(harness)._map_section_to_stage("detail") == "sub_next"


# ---------------------------------------------------------------------------
# 19. What a restart leaves behind
# ---------------------------------------------------------------------------
#
# ``restart_cleanup`` resets the stage, the data, the history, the
# completion flag, the clarification counter, the extraction flag, the
# banks, the artifact and (since this item) the subflow stack. Two pieces
# of wizard state are not in that list, and one of them is persisted.
#
# The audit trail is the second half: unwinding the stack in silence
# leaves a ``subflow_push`` record with no matching ``subflow_pop``, so a
# consumer pairing them -- or reconstructing depth from the trail --
# is wrong in exactly the case this item made reachable.


def _tasked_config() -> dict[str, Any]:
    """A wizard whose first stage carries a task completed by extraction."""
    builder = WizardConfigBuilder("restart-and-tasks")
    builder.stage(
        "gather",
        is_start=True,
        prompt="Tell me your name.",
        response_template="Noted.",
        confirm_first_render=False,
        tasks=[
            {
                "id": "collect_name",
                "description": "Collect the name",
                "completed_by": "field_extraction",
                "field_name": "name",
            }
        ],
    )
    builder.field("name", field_type="string", required=True)
    builder.transition("wrap", condition="has('name')")
    builder.stage("wrap", is_end=True, prompt="All done.", response_template="WRAP")
    return builder.build()


@pytest.mark.asyncio
async def test_restart_clears_task_completion() -> None:
    """Tasks are persisted, restored, and survived the reset.

    ``state.tasks`` round-trips through ``fsm_state``, so a restarted
    wizard reported the *previous* run's completed tasks -- a progress
    indicator that starts full on a flow the user just asked to start
    over.
    """
    async with await BotTestHarness.create(
        wizard_config=_tasked_config(),
        main_responses=["r"] * 8,
        extraction_results=[[{"name": "Alice"}], []],
    ) as harness:
        await harness.chat("my name is Alice")
        manager = harness.bot.get_conversation_manager(harness.context.conversation_id)
        assert _strategy(harness).get_state_snapshot(manager).completed_tasks == 1, (
            "the task did not complete, so the restart has nothing to reset"
        )

        await harness.chat("start over")

        snapshot = _strategy(harness).get_state_snapshot(manager)
        assert snapshot.completed_tasks == 0, (
            "the wizard restarted with the previous run's tasks still "
            f"marked complete ({snapshot.completed_tasks} of {snapshot.total_tasks})"
        )
        assert snapshot.total_tasks == 1, "the task list itself must survive the restart"


@pytest.mark.asyncio
async def test_restart_clears_transient_data() -> None:
    """``transient`` is wizard state and the reset skipped it.

    Driven through the non-conversational API because that is the path
    where a caller owns the state object across calls -- a transform
    writing an ephemeral key leaves it there, and the restart that is
    supposed to give a clean slate hands it back.
    """
    async with await BotTestHarness.create(
        wizard_config=_tasked_config(),
        main_responses=["r"] * 4,
        extraction_results=[[{"name": "Alice"}]],
    ) as harness:
        state = WizardState(current_stage="gather", data={"name": "Alice"})
        state.transient["scratch"] = "from the previous run"

        await _strategy(harness).advance(
            user_input={},
            state=state,
            navigation="restart",
        )

        assert state.transient == {}, (
            "the restart cleared data and left transient standing, so the "
            f"first stage of the new run still sees {state.transient!r}"
        )


@pytest.mark.asyncio
async def test_restart_inside_a_subflow_records_the_pop() -> None:
    """The unwind has to appear in the audit trail, like every other one.

    ``handle_pop`` records ``subflow_pop``; the restart unwind did not,
    so the trail held a push for ``detail`` that nothing ever closed.
    """
    async with await BotTestHarness.create(
        wizard_config=_restartable_config(),
        main_responses=["r"] * 8,
        extraction_results=[[{"name": "Alice"}], []],
    ) as harness:
        await harness.chat("my name is Alice")
        assert harness.wizard_stage == "sub_a", "the subflow was not pushed"

        await harness.chat("start over")

        transitions = _wizard_metadata(harness)["wizard"]["fsm_state"]["transitions"]
        pushed = [t["subflow_push"] for t in transitions if t.get("subflow_push")]
        popped = [t["subflow_pop"] for t in transitions if t.get("subflow_pop")]

        assert pushed == ["detail"], "expected exactly one push to have been recorded"
        assert popped == ["detail"], (
            "the restart unwound the subflow without recording the pop, so "
            f"the trail holds a push nothing closes: pops={popped!r}"
        )


# ---------------------------------------------------------------------------
# 20. The last field the two snapshot constructors disagreed about
# ---------------------------------------------------------------------------
#
# ``snapshot_from_metadata`` was moved onto ``normalize_wizard_state``,
# which reads what ``_build_wizard_metadata`` derived -- and that writer
# passes ``suggestions`` through ``get_stage_suggestions`` (type-checked)
# and ``render_suggestions`` (Jinja). ``get_state_snapshot`` still read
# ``stage.get("suggestions", [])`` off the raw metadata, so the same
# stage yielded a rendered list from one constructor and an unrendered
# one from the other -- and a wrong-typed value straight through.
#
# The fixtures in section 12-16 could not see this: their suggestions are
# literals, where rendered and raw coincide. That is the same hazard the
# item guards against for ``stage_index`` by putting the pushing stage at
# index 1, applied to the other field.


def _templated_suggestions_config(suggestions: Any) -> dict[str, Any]:
    """A two-stage flow whose second stage declares *suggestions*."""
    builder = WizardConfigBuilder("suggestions-render")
    builder.stage(
        "intro",
        is_start=True,
        prompt="Say hello.",
        response_template="INTRO",
        confirm_first_render=False,
    )
    builder.field("greeting", field_type="string", required=True)
    builder.transition("gather", condition="has('greeting')")
    builder.stage(
        "gather",
        prompt="Tell me your name.",
        response_template="Noted.",
        confirm_first_render=False,
        suggestions=suggestions,
    )
    builder.field("name", field_type="string", required=True)
    builder.transition("wrap", condition="has('name')")
    builder.stage("wrap", is_end=True, prompt="All done.", response_template="WRAP")
    return builder.build()


@pytest.mark.asyncio
async def test_both_snapshots_render_the_stages_suggestions() -> None:
    """One constructor rendered the quick replies and the other did not.

    A UI reading ``get_state_snapshot`` showed the user a raw Jinja
    expression as a button label, while the same stage read through
    ``snapshot_from_metadata`` showed the rendered text.
    """
    async with await BotTestHarness.create(
        wizard_config=_templated_suggestions_config(["Use {{ greeting }}", "Something else"]),
        main_responses=["r"] * 8,
        extraction_results=[[{"greeting": "hi"}], []],
    ) as harness:
        await harness.chat("hi")
        assert harness.wizard_stage == "gather", "expected the second stage"

        manager = harness.bot.get_conversation_manager(harness.context.conversation_id)
        live = _strategy(harness).get_state_snapshot(manager)
        static = WizardReasoning.snapshot_from_metadata(_wizard_metadata(harness))

        assert static is not None
        assert static.suggestions == ["Use hi", "Something else"]
        assert live.suggestions == static.suggestions, (
            "the two constructors of one type disagree about the stage's "
            f"quick replies: {live.suggestions!r} vs {static.suggestions!r}"
        )


@pytest.mark.asyncio
async def test_a_wrong_typed_suggestions_field_does_not_reach_a_snapshot() -> None:
    """``suggestions`` is declared ``list[str]`` and was handed through raw.

    A bare string is iterable, so a UI building one button per item
    rendered one button per *character*. ``get_stage_suggestions`` already
    applies the documented default; the snapshot bypassed it.
    """
    async with await BotTestHarness.create(
        wizard_config=_templated_suggestions_config("Use the default"),
        main_responses=["r"] * 8,
        extraction_results=[[{"greeting": "hi"}], []],
    ) as harness:
        await harness.chat("hi")
        assert harness.wizard_stage == "gather", "expected the second stage"

        manager = harness.bot.get_conversation_manager(harness.context.conversation_id)
        live = _strategy(harness).get_state_snapshot(manager)

        assert live.suggestions == [], (
            "a string where a list of strings belongs must fall back to the "
            f"documented default, not be handed to the caller: {live.suggestions!r}"
        )


@pytest.mark.asyncio
async def test_the_static_snapshot_does_not_alias_the_state_it_read() -> None:
    """The snapshot is documented read-only and handed out live objects.

    ``get_state_snapshot`` copies ``data`` and ``history``; this
    constructor returned the very dict and list inside
    ``manager.metadata``, so a consumer appending to ``snapshot.history``
    silently rewrote persisted wizard state.
    """
    async with await BotTestHarness.create(
        wizard_config=_templated_suggestions_config(["Alice", "Bob"]),
        main_responses=["r"] * 8,
        extraction_results=[[{"greeting": "hi"}], []],
    ) as harness:
        await harness.chat("hi")
        manager = harness.bot.get_conversation_manager(harness.context.conversation_id)

        snapshot = WizardReasoning.snapshot_from_metadata(manager.metadata)
        assert snapshot is not None
        snapshot.history.append("tampered")
        snapshot.data["injected"] = True

        persisted = manager.metadata["wizard"]["fsm_state"]
        assert "tampered" not in persisted["history"], (
            "appending to the snapshot's history rewrote the persisted state"
        )
        assert "injected" not in persisted["data"], (
            "writing to the snapshot's data rewrote the persisted state"
        )


# ---------------------------------------------------------------------------
# 17-20. The stage metadata the two constructors can and cannot supply
# ---------------------------------------------------------------------------
#
# ``ToolWizardState`` -- what a ``ContextAwareTool`` is handed -- carries
# ``stage_metadata``, and ``WizardStateSnapshot`` did not. That is the one
# tool-view field with no snapshot counterpart, so anything converting a
# snapshot into a tool view could only ever have reported ``{}``.
#
# The field is supplied by exactly one of the two constructors, and the
# asymmetry is real rather than an oversight: stage metadata is not part
# of the persisted ``fsm_state`` -- the live publisher is the only holder
# -- so ``snapshot_from_metadata`` has nothing to read. It reports ``{}``
# and says so, which is the honest answer; filling it from
# ``stage_definitions`` would make the two constructors agree by
# inventing a value the metadata route does not have, and the caller
# passing those definitions is not required to pass any.


@pytest.mark.asyncio
async def test_the_live_snapshot_carries_the_stage_metadata() -> None:
    """``get_state_snapshot`` supplies the field, from the owning FSM.

    Read for the stage the *state* says we are on, which inside a push is
    the subflow's -- the same resolution every other stage-derived field
    in this constructor uses, and the reason the assertion is taken
    inside a subflow rather than in the main flow.
    """
    async with await BotTestHarness.create(
        wizard_config=_snapshot_config(),
        main_responses=["r"] * 10,
        extraction_results=[[{"greeting": "hi"}], [{"name": "Alice"}], [], []],
    ) as harness:
        await harness.chat("hi")
        await harness.chat("my name is Alice")
        assert harness.wizard_stage == "sub_start", "the subflow was not pushed"

        snapshot = _snapshot_inside_the_subflow(harness)

        assert snapshot.stage_metadata.get("prompt") == "Which detail?", (
            "the stage metadata belongs to the subflow stage the snapshot "
            "reports; an empty dict is the main FSM answering about a stage "
            "it does not have"
        )
        assert snapshot.stage_metadata.get("can_skip") is True
        assert snapshot.stage_metadata.get("suggestions") == ["Colour", "Size"]


@pytest.mark.asyncio
async def test_the_live_snapshot_copies_the_stage_metadata() -> None:
    """The stage dict is the FSM's live one, and must not be handed out.

    ``stage_metadata_for`` documents that it returns the live dict rather
    than a copy. A snapshot is documented read-only, so writing through
    one must not reconfigure the stage for every later turn.
    """
    async with await BotTestHarness.create(
        wizard_config=_snapshot_config(),
        main_responses=["r"] * 10,
        extraction_results=[[{"greeting": "hi"}], [{"name": "Alice"}], [], []],
    ) as harness:
        await harness.chat("hi")
        await harness.chat("my name is Alice")

        snapshot = _snapshot_inside_the_subflow(harness)
        snapshot.stage_metadata["prompt"] = "tampered"

        again = _snapshot_inside_the_subflow(harness)
        assert again.stage_metadata.get("prompt") == "Which detail?", (
            "writing to the snapshot's stage metadata reconfigured the stage"
        )


@pytest.mark.asyncio
async def test_the_static_snapshot_leaves_the_stage_metadata_empty() -> None:
    """The metadata route has nothing to read, and reports that.

    ``stage_metadata`` is not written into ``fsm_state``, so this
    constructor cannot supply it. ``{}`` is the honest answer -- and it
    stays ``{}`` even when the caller passes ``stage_definitions``, which
    is a different thing (the main flow's *declarations*, not the stage
    the wizard is standing on) and is optional besides.
    """
    async with await BotTestHarness.create(
        wizard_config=_snapshot_config(),
        main_responses=["r"] * 10,
        extraction_results=[[{"greeting": "hi"}], [{"name": "Alice"}], [], []],
    ) as harness:
        await harness.chat("hi")
        await harness.chat("my name is Alice")

        snapshot = WizardReasoning.snapshot_from_metadata(
            _wizard_metadata(harness),
            stage_definitions=_main_stage_definitions(),
        )

        assert snapshot is not None
        assert snapshot.stage_metadata == {}


@pytest.mark.asyncio
async def test_the_tool_view_of_a_live_snapshot_is_complete() -> None:
    """The conversion carries every field, including the new one.

    This is the assertion that fails if ``to_tool_view()`` ships without
    the dataclass field behind it: the method would still return a
    ``ToolWizardState``, and its ``stage_metadata`` would be ``{}`` with
    nothing to say so.
    """
    async with await BotTestHarness.create(
        wizard_config=_snapshot_config(),
        main_responses=["r"] * 10,
        extraction_results=[[{"greeting": "hi"}], [{"name": "Alice"}], [], []],
    ) as harness:
        await harness.chat("hi")
        await harness.chat("my name is Alice")

        snapshot = _snapshot_inside_the_subflow(harness)
        view = snapshot.to_tool_view()

        # Every tool-view field, against its snapshot counterpart --
        # which is what "complete" means here.  Note that inside a push
        # ``data`` and ``history`` are the *subflow's*: the parent's are
        # swapped out until it pops, so "Alice" is not in view and its
        # absence is the push working, not the conversion dropping it.
        assert view.current_stage == snapshot.current_stage == "sub_start"
        assert view.collected_data == snapshot.data
        assert view.history == snapshot.history == ["sub_start"]
        assert view.completed is snapshot.completed is False
        assert view.stage_metadata == snapshot.stage_metadata
        assert view.stage_metadata.get("prompt") == "Which detail?", (
            "a tool handed this view sees the stage it is standing on; {} "
            "is the conversion running against a snapshot without the field"
        )


@pytest.mark.asyncio
async def test_the_tool_view_of_a_static_snapshot_is_honest() -> None:
    """Converting the fallback route does not invent the missing field.

    The metadata constructor leaves ``stage_metadata`` empty, and the
    conversion reports that rather than filling it from somewhere else --
    the same answer ``ToolWizardState.from_manager_metadata`` gives for
    the same reason, reached by a different path.
    """
    async with await BotTestHarness.create(
        wizard_config=_snapshot_config(),
        main_responses=["r"] * 10,
        extraction_results=[[{"greeting": "hi"}], [{"name": "Alice"}], [], []],
    ) as harness:
        await harness.chat("hi")
        await harness.chat("my name is Alice")

        snapshot = WizardReasoning.snapshot_from_metadata(
            _wizard_metadata(harness),
            stage_definitions=_main_stage_definitions(),
        )
        assert snapshot is not None

        view = snapshot.to_tool_view()

        assert view.current_stage == "sub_start"
        assert view.stage_metadata == {}


@pytest.mark.asyncio
async def test_a_second_instance_snapshots_the_stage_the_state_reports() -> None:
    """The documented case: the conversation, but not the instance that ran it.

    ``get_state_snapshot`` is taken outside a turn by definition, so it
    resolves every stage-derived field from the stage the *state*
    reports. ``stage_metadata`` has to come from the same place. An FSM
    that has not run a turn in this process answers ``current_metadata``
    with its **start** stage -- observed below, before the read -- so a
    constructor sourcing the field from that property would describe the
    wrong stage's configuration, and would do so only on the path where
    nobody has taken a turn yet.

    Both harnesses load the same wizard config; the second reads the
    first's conversation manager, which is what a restarted process or a
    second replica has.
    """
    async with await BotTestHarness.create(
        wizard_config=_snapshot_config(),
        main_responses=["r"] * 10,
        extraction_results=[[{"greeting": "hi"}], []],
    ) as first:
        await first.chat("hi")
        assert first.wizard_stage == "gather", "expected the main flow's second stage"
        manager = first.bot.get_conversation_manager(first.context.conversation_id)

        async with await BotTestHarness.create(
            wizard_config=_snapshot_config(),
            main_responses=["r"] * 10,
        ) as second:
            strategy = _strategy(second)
            assert strategy._fsm.current_metadata.get("prompt") == "Say hello.", (
                "precondition: an FSM with no turn behind it reports the "
                "start stage, which is not the stage the conversation is on"
            )

            snapshot = strategy.get_state_snapshot(manager)

            assert snapshot.current_stage == "gather"
            assert snapshot.stage_metadata.get("prompt") == "Tell me your name.", (
                "the snapshot described the start stage rather than the one "
                "its own current_stage names"
            )
            assert snapshot.stage_metadata.get("can_skip") is True
