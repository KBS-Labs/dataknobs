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

from typing import Any

import pytest

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


def _skippable_subflow_config() -> dict[str, Any]:
    """A parent pushing a subflow whose start stage is the shared stage."""
    return _parent_pushing(
        {"name": "detail", "stages": [_SKIPPABLE_STAGE, _SUB_NEXT, _SUB_DONE]},
    ).build()


def _can_skip_only_subflow_config() -> dict[str, Any]:
    """A parent pushing a subflow that declares ``can_skip`` and no keywords."""
    return _parent_pushing(
        {"name": "detail", "stages": [_CAN_SKIP_ONLY_STAGE, _SUB_NEXT, _SUB_DONE]},
    ).build()


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
