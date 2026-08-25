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
from typing import Any

import pytest

from dataknobs_bots.reasoning.wizard import WizardReasoning
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
            _strategy(harness)._navigator._resolve_navigation_config("sub_start").skip.keywords
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

        resolved = strategy._navigator._resolve_navigation_config("sub_done")

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
        try:
            navigator._resolve_navigation_config("sub_start")
            navigator._resolve_navigation_config("sub_start")
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
