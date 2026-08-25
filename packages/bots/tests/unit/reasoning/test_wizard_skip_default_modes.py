"""What ``skip_default`` does to a value the user already set.

``navigate_skip`` applied a stage's ``skip_default`` with a bare
``state.data.update(...)``, and ``dict.update`` cannot be asked to do
anything else: a key the user set five turns ago is replaced exactly as
readily as one that was never touched. There was no mode flag, no
per-key form, and no log line -- so a field the author of the config
never meant to touch was silently rewritten, and every downstream reader
(conditions, transforms, emission, templates) saw the default as though
the user had chosen it.

Both directions are legitimate and a real stage needs both **in one
block**: an option the user configured must survive the skip that saves
it, while a flag guarding an unconfigured branch must be clobbered by
the skip or the user is pushed into the branch they were trying to
leave. So the mode is per key, with a block-level default, and
``overwrite`` stays the block-level default because making ``fill``
the default would silently disarm every escape hatch that exists today.

Several tests here are **guards rather than reproductions** and pass
before the change as well as after -- the ones pinning ``overwrite`` as
the block default, the fallback for a block with no modes in it, the
block-mode fallback for an unreadable per-key mode, a foreign ``mode``
field staying a plain value, and the ``to_dict``/``from_dict`` round
trip. They are here because the failure each describes is a regression
this change could plausibly introduce.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from dataknobs_bots.reasoning.wizard import WizardReasoning
from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader
from dataknobs_bots.reasoning.wizard_skip import (
    SKIP_DEFAULT_FILL,
    SKIP_DEFAULT_OVERWRITE,
    SkipDefaultEntry,
    SkipDefaults,
)
from dataknobs_bots.reasoning.wizard_types import WizardState, field_is_present
from dataknobs_bots.testing import BotTestHarness, WizardConfigBuilder

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

#: The stage under test. Its transition fires on the skip marker alone,
#: so setting a value does **not** move the wizard on -- which is the
#: position the item is about: the user configures something, then says
#: "done" to save, and the save is what destroys it.
_CONFIGURE: dict[str, Any] = {
    "name": "configure",
    "is_start": True,
    "prompt": "Configure the options.",
    "response_template": "CONFIGURE",
    "confirm_first_render": False,
    "can_skip": True,
    "schema": {
        "type": "object",
        "properties": {
            "kb_enabled": {"type": "boolean"},
            "scenario_enabled": {"type": "boolean"},
        },
    },
    "transitions": [{"target": "review", "condition": "has('_skipped_configure')"}],
}

#: The skip lands here rather than on an end stage, so the turn does not
#: also complete the flow and take the collected data out of view.
_REVIEW: dict[str, Any] = {
    "name": "review",
    "prompt": "Anything else?",
    "response_template": "REVIEW",
    "confirm_first_render": False,
    "schema": {"type": "object", "properties": {"extra": {"type": "string"}}},
    "transitions": [{"target": "done", "condition": "has('extra')"}],
}

_DONE: dict[str, Any] = {
    "name": "done",
    "is_end": True,
    "prompt": "Complete.",
    "response_template": "DONE",
}


def _stage(**overrides: Any) -> dict[str, Any]:
    """``_CONFIGURE`` with the skip fields under test written onto it."""
    return {**_CONFIGURE, **overrides}


def _wizard(stage: dict[str, Any]) -> dict[str, Any]:
    """A two-stage wizard whose head is *stage*.

    Built by hand rather than through the builder so the stage dict the
    test authored is the one the loader sees, unmodified.
    """
    return {
        "name": "skip-defaults",
        "version": "1.0",
        "stages": [stage, _REVIEW, _DONE],
    }


async def _set_then_skip(
    config: dict[str, Any],
    extracted: dict[str, Any],
    *,
    skip_word: str = "skip",
) -> dict[str, Any]:
    """Set *extracted* on the head stage, then skip it. Returns the data.

    The two turns are the whole shape of the item: a value arrives from
    the user, and the stage is then skipped without ever leaving it.
    """
    async with await BotTestHarness.create(
        wizard_config=config,
        main_responses=["r"] * 8,
        extraction_results=[[extracted], [], []],
    ) as harness:
        await harness.greet()
        await harness.chat("set the options")
        for key, value in extracted.items():
            assert harness.wizard_data.get(key) == value, (
                f"setup: {key!r} was never set, so the test cannot say "
                "anything about what the skip did to it"
            )

        await harness.chat(skip_word)
        assert harness.wizard_data.get("_skipped_configure") is True, (
            "setup: the stage was not skipped"
        )
        return dict(harness.wizard_data)


# ---------------------------------------------------------------------------
# 1-3. The modes, through a real turn
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_skip_default_fill_preserves_a_user_set_value() -> None:
    """``fill`` writes only where the key is absent.

    This is the arm that cost a consumer a working feature: a knowledge
    base was configured, the user typed "done" to save, and the stage's
    ``kb_enabled: false`` default replaced the flag the configuration had
    just set -- so the artifact was emitted without the block the user
    had spent the session building.
    """
    data = await _set_then_skip(
        _wizard(
            _stage(
                skip_default={"kb_enabled": False},
                skip_default_mode=SKIP_DEFAULT_FILL,
            )
        ),
        {"kb_enabled": True},
    )

    assert data.get("kb_enabled") is True, (
        "skip_default_mode: fill must leave a key the user set alone; "
        "the stage's default replaced it"
    )


@pytest.mark.asyncio
async def test_skip_default_overwrite_is_still_the_default() -> None:
    """A block declaring no mode behaves exactly as it does today.

    Not a reproduction -- this passes before the fix. It is here because
    ``fill`` is the safer-*looking* default and making it the default
    would disarm every stage that relies on the clobber to leave a branch
    it cannot otherwise escape.
    """
    data = await _set_then_skip(
        _wizard(_stage(skip_default={"scenario_enabled": False})),
        {"scenario_enabled": True},
    )

    assert data.get("scenario_enabled") is False, (
        "with no mode declared the default must still overwrite"
    )


@pytest.mark.asyncio
async def test_per_key_modes_in_one_block() -> None:
    """One key filled, one clobbered, in the same ``skip_default``.

    The block-level flag alone is too coarse for the stage that motivated
    this: its two arms need opposite things from the same skip. A bare
    value keeps the block's mode, so only the key that differs has to say
    so.
    """
    data = await _set_then_skip(
        _wizard(
            _stage(
                skip_default={
                    "kb_enabled": {"value": False, "mode": SKIP_DEFAULT_FILL},
                    "scenario_enabled": False,
                },
            )
        ),
        {"kb_enabled": True, "scenario_enabled": True},
    )

    assert data.get("kb_enabled") is True, "the per-key 'fill' mode did not reach kb_enabled"
    assert data.get("scenario_enabled") is False, (
        "the bare value should keep the block mode, which is overwrite"
    )


# ---------------------------------------------------------------------------
# 4. The ordering, as a guarantee rather than an accident
# ---------------------------------------------------------------------------


class _WriteOrder(dict):  # type: ignore[type-arg]
    """A ``state.data`` that remembers the order keys were first written.

    Planted directly rather than through the harness because a turn
    rebuilds ``WizardState`` from serialized metadata -- ``data`` is a
    fresh ``copy.deepcopy`` every turn, so a probe planted between two
    turns would be discarded before the code under test ran.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.order: list[str] = []

    def __setitem__(self, key: Any, value: Any) -> None:
        if key not in self.order:
            self.order.append(key)
        super().__setitem__(key, value)

    def setdefault(self, key: Any, default: Any = None) -> Any:
        if key not in self:
            self.order.append(key)
        return super().setdefault(key, default)

    def update(self, *args: Any, **kwargs: Any) -> None:
        for key in dict(*args, **kwargs):
            if key not in self.order:
                self.order.append(key)
        super().update(*args, **kwargs)


def _navigator_for(config: dict[str, Any]) -> Any:
    """The navigator of a strategy built over *config*.

    Reaches past the harness deliberately: the two assertions below are
    about the order of two statements inside one method, which no
    sequence of turns can observe.
    """
    wizard_fsm = WizardConfigLoader().load_from_dict(config)
    reasoning = WizardReasoning(wizard_fsm=wizard_fsm, strict_validation=False)
    return reasoning._navigator


@pytest.mark.asyncio
async def test_the_skip_marker_is_observable_before_the_defaults_land() -> None:
    """The skip marker is written first, and that is a contract.

    A consumer cannot otherwise tell "this value arrived with a skip"
    from "the user said this on an ordinary turn", so every workaround
    for the gap this change closes rests on the marker being visible
    before the defaults are. The ordering is older than ``skip_default``
    -- the marker was written two weeks before the defaults were appended
    below it -- which is exactly why it needs saying: nothing chose it.
    """
    navigator = _navigator_for(
        _wizard(_stage(skip_default={"kb_enabled": False, "scenario_enabled": False}))
    )
    state = WizardState(current_stage="configure", data=_WriteOrder())

    await navigator.navigate_skip(state)

    order = state.data.order  # type: ignore[attr-defined]
    assert "_skipped_configure" in order, "the skip marker was never written"
    assert "kb_enabled" in order, "the defaults were never applied"
    assert order.index("_skipped_configure") < order.index("kb_enabled"), (
        f"the defaults landed before the skip marker: {order}"
    )


# ---------------------------------------------------------------------------
# 5. Saying so when a value is replaced
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_replaced_keys_are_logged(caplog: pytest.LogCaptureFixture) -> None:
    """An overwrite of a user-set value is findable in the log.

    The consumer who hit this found it by driving the flow and noticing
    an emitted artifact was missing a block; nothing anywhere said a
    value had been replaced.
    """
    config = _wizard(_stage(skip_default={"kb_enabled": False}))
    navigator = _navigator_for(config)
    state = WizardState(current_stage="configure", data={"kb_enabled": True})

    with caplog.at_level(logging.DEBUG, logger="dataknobs_bots.reasoning.wizard_navigation"):
        await navigator.navigate_skip(state)

    messages = [record.getMessage() for record in caplog.records]
    assert any("kb_enabled" in message and "configure" in message for message in messages), (
        f"no log line named the replaced key: {messages}"
    )


@pytest.mark.asyncio
async def test_a_default_that_agrees_with_the_user_is_not_reported(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Only a key whose value actually changed counts as replaced.

    A stage whose default happens to equal what the user chose has
    replaced nothing, and reporting it would train a reader to ignore
    the line that matters.
    """
    config = _wizard(_stage(skip_default={"kb_enabled": False}))
    navigator = _navigator_for(config)
    state = WizardState(current_stage="configure", data={"kb_enabled": False})

    with caplog.at_level(logging.DEBUG, logger="dataknobs_bots.reasoning.wizard_navigation"):
        await navigator.navigate_skip(state)

    assert not any("replaced" in record.getMessage() for record in caplog.records), (
        "a default equal to the user's own value was reported as a replacement"
    )


# ---------------------------------------------------------------------------
# 6. A block of the wrong shape is reported, not discarded
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_scalar_skip_default_is_reported(caplog: pytest.LogCaptureFixture) -> None:
    """``skip_default: "Anonymous"`` never worked, and never said so.

    The ``isinstance(..., dict)`` guard has been on this line since the
    field was introduced, so a scalar has always been dropped in silence
    -- while the builder declared the parameter ``bool | None`` and the
    package's own documentation showed a string. An author following
    either got a stage that quietly did nothing on skip.
    """
    config = _wizard(_stage(skip_default="Anonymous"))
    navigator = _navigator_for(config)
    state = WizardState(current_stage="configure", data={})

    with caplog.at_level(logging.WARNING):
        await navigator.navigate_skip(state)

    assert any(
        "skip_default" in record.getMessage()
        for record in caplog.records
        if record.levelname == "WARNING"
    ), (
        f"a wrong-typed skip_default was discarded silently: {[r.getMessage() for r in caplog.records]}"
    )


@pytest.mark.asyncio
async def test_an_unknown_mode_is_reported_and_falls_back(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A misspelled mode takes the documented default and says so.

    ``fil`` is not ``fill``, and silently treating it as one of the two
    would give the author the opposite of what they asked for on the
    key they cared enough about to annotate.
    """
    config = _wizard(
        _stage(skip_default={"kb_enabled": {"value": False, "mode": "fil"}}),
    )
    navigator = _navigator_for(config)
    state = WizardState(current_stage="configure", data={"kb_enabled": True})

    with caplog.at_level(logging.WARNING):
        await navigator.navigate_skip(state)

    assert state.data["kb_enabled"] is False, (
        "an unreadable mode must fall back to the documented default, overwrite"
    )
    assert any(
        "mode" in record.getMessage() for record in caplog.records if record.levelname == "WARNING"
    ), "the unreadable mode was accepted in silence"


# ---------------------------------------------------------------------------
# 7. Inside a subflow -- the joint case
# ---------------------------------------------------------------------------


def _subflow_wizard(stage: dict[str, Any]) -> dict[str, Any]:
    """A parent whose ``gather`` stage pushes a subflow headed by *stage*."""
    builder = WizardConfigBuilder("skip-defaults-in-a-subflow")
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
    )
    builder.stage("wrap", is_end=True, prompt="All done.", response_template="WRAP")
    builder.subflow("detail", {"name": "detail", "stages": [stage, _REVIEW, _DONE]})
    return builder.build()


@pytest.mark.asyncio
async def test_skip_default_applies_inside_a_subflow() -> None:
    """The same block, in a pushed subflow, means the same thing.

    Until the navigator resolved a stage against the FSM that owns it,
    this site was unreachable inside a push -- the outer skip gate asked
    the main FSM, which does not have the stage, and refused before
    ``skip_default`` was ever consulted. So the modes have to be shown
    working here and not only at the top level.
    """
    stage = _stage(
        skip_default={
            "kb_enabled": {"value": False, "mode": SKIP_DEFAULT_FILL},
            "scenario_enabled": False,
        },
    )
    async with await BotTestHarness.create(
        wizard_config=_subflow_wizard(stage),
        main_responses=["r"] * 8,
        extraction_results=[
            [{"name": "Alice"}],
            [{"kb_enabled": True, "scenario_enabled": True}],
            [],
        ],
    ) as harness:
        await harness.chat("my name is Alice")
        assert harness.wizard_stage == "configure", "the subflow was not pushed"

        await harness.chat("set the options")
        assert harness.wizard_data.get("kb_enabled") is True, "setup: nothing was set"

        await harness.chat("skip")

        assert harness.wizard_data.get("_skipped_configure") is True
        assert harness.wizard_data.get("kb_enabled") is True, (
            "the per-key 'fill' mode did not survive the push"
        )
        assert harness.wizard_data.get("scenario_enabled") is False, (
            "the block mode did not survive the push"
        )


# ---------------------------------------------------------------------------
# 8. The grammar, without a turn
# ---------------------------------------------------------------------------


def test_a_bare_value_takes_the_block_mode() -> None:
    """The scalar sugar: one mode for every key in the block."""
    defaults = SkipDefaults.from_stage({"a": 1, "b": 2}, SKIP_DEFAULT_FILL)

    assert defaults.entries == {
        "a": SkipDefaultEntry(value=1, mode=SKIP_DEFAULT_FILL),
        "b": SkipDefaultEntry(value=2, mode=SKIP_DEFAULT_FILL),
    }


def test_a_per_key_mode_overrides_the_block_mode() -> None:
    """The per-key form is the one the motivating stage needs."""
    defaults = SkipDefaults.from_stage(
        {"a": {"value": 1, "mode": SKIP_DEFAULT_OVERWRITE}, "b": 2},
        SKIP_DEFAULT_FILL,
    )

    assert defaults.entries["a"].mode == SKIP_DEFAULT_OVERWRITE
    assert defaults.entries["b"].mode == SKIP_DEFAULT_FILL


def test_a_mapping_without_a_value_key_is_the_value() -> None:
    """A dict is only an entry when it says ``value``.

    ``skip_default: {llm: {provider: "x"}}`` is a nested *value*, and the
    existing suite has a case for it. Reading every mapping as an entry
    would turn that config into a key with no value at all.
    """
    defaults = SkipDefaults.from_stage({"llm": {"provider": "x"}}, SKIP_DEFAULT_OVERWRITE)

    assert defaults.entries["llm"].value == {"provider": "x"}


def test_a_mapping_naming_more_than_value_and_mode_is_the_value() -> None:
    """Saying ``value`` is not enough; an entry says *only* value/mode.

    ``{value: "", label: "Email"}`` is a nested default that happens to
    have a field called ``value``. Reading it as an annotation keeps the
    value and drops ``label`` on the floor -- silently, because the shape
    is syntactically fine -- which is the loss the mode grammar exists
    to end, one level further down.
    """
    field = {"value": "", "label": "Email"}

    defaults = SkipDefaults.from_stage({"form_field": field}, SKIP_DEFAULT_OVERWRITE)

    assert defaults.entries["form_field"].value == field


def test_a_value_and_mode_nested_default_can_be_wrapped_to_say_so() -> None:
    """The one collision the grammar cannot see, and its escape hatch.

    A nested default naming *exactly* ``value`` and ``mode`` reads
    exactly like an annotation and is taken as one. Wrapping it in a
    real annotation says which was meant -- and the wrapper has to name
    both keys too, because that is what an annotation is.
    """
    literal = {"value": 3, "mode": "off"}

    defaults = SkipDefaults.from_stage(
        {"knob": {"value": literal, "mode": SKIP_DEFAULT_OVERWRITE}},
        SKIP_DEFAULT_OVERWRITE,
    )

    assert defaults.entries["knob"].value == literal


def test_apply_reports_only_the_keys_whose_value_changed() -> None:
    """``apply`` is the one place that decides what "replaced" means."""
    defaults = SkipDefaults.from_stage(
        {"changed": 2, "agreed": 1, "absent": 3},
        SKIP_DEFAULT_OVERWRITE,
    )
    data: dict[str, Any] = {"changed": 1, "agreed": 1}

    replaced = defaults.apply(data)

    assert replaced == ["changed"]
    assert data == {"changed": 2, "agreed": 1, "absent": 3}


def test_apply_in_fill_mode_replaces_nothing() -> None:
    """``fill`` cannot report a replacement, because it never makes one."""
    defaults = SkipDefaults.from_stage({"a": 2}, SKIP_DEFAULT_FILL)
    data: dict[str, Any] = {"a": 1}

    assert defaults.apply(data) == []
    assert data == {"a": 1}


# ---------------------------------------------------------------------------
# 9. What ``fill`` means by "absent"
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("existing", [None, False, 0, "", [], {}])
def test_fill_writes_exactly_where_the_package_says_a_field_is_absent(
    existing: Any,
) -> None:
    """``fill`` and ``has()`` must not disagree about what "set" means.

    This package already has one answer and states it in one place:
    :func:`field_is_present` -- "a field has been provided if its value
    is not None" -- which is what the ``has()`` condition helper, the
    confidence gate and ``WizardExtractor.apply_schema_defaults`` all use.
    ``wizard_derivations`` even carries a note naming the fork, because
    ``_apply_transition_derivations`` picked the stricter key-presence
    reading and the two have had to be told apart ever since.

    So this is a pin rather than a preference: whatever
    ``field_is_present`` says about a value is what ``fill`` does with
    the key holding it, for every value, forever.
    """
    defaults = SkipDefaults.from_stage({"a": "the default"}, SKIP_DEFAULT_FILL)
    data: dict[str, Any] = {"a": existing}

    defaults.apply(data)

    if field_is_present(existing):
        assert data["a"] == existing, (
            f"fill overwrote {existing!r}, which field_is_present() calls set"
        )
    else:
        assert data["a"] == "the default", (
            f"fill skipped {existing!r}, which field_is_present() calls absent -- "
            "so has() reports the key missing while fill reports it taken"
        )


@pytest.mark.asyncio
async def test_fill_writes_over_a_key_left_holding_none() -> None:
    """A key can be present and unset, and a real turn produces one.

    Extraction writes a property it saw mentioned but could not resolve;
    a prior stage clears one it no longer applies to. Either leaves the
    key in ``data`` with ``None`` beside it -- which every other reader
    in this package calls absent, and which ``fill`` refused to write
    because ``dict.setdefault`` asks a different question.
    """
    navigator = _navigator_for(
        _wizard(
            _stage(
                skip_default={"kb_enabled": False},
                skip_default_mode=SKIP_DEFAULT_FILL,
            )
        )
    )
    state = WizardState(current_stage="configure", data={"kb_enabled": None})

    await navigator.navigate_skip(state)

    assert state.data["kb_enabled"] is False, (
        "fill left an unset key unset: the stage's default never landed"
    )


def test_replacing_an_unset_key_is_not_reported_as_a_replacement() -> None:
    """``replaced`` names values the *user* chose, and ``None`` is not one.

    The log line exists so a consumer can find the moment their value
    was destroyed. A key holding ``None`` had no value to destroy, and
    reporting it would put noise in the one line worth reading.
    """
    defaults = SkipDefaults.from_stage({"kb_enabled": False}, SKIP_DEFAULT_OVERWRITE)
    data: dict[str, Any] = {"kb_enabled": None}

    replaced = defaults.apply(data)

    assert replaced == [], "an unset key was reported as a replaced user value"
    assert data["kb_enabled"] is False, "the default should still be written"


# ---------------------------------------------------------------------------
# 10. An annotation names both keys
# ---------------------------------------------------------------------------


def test_a_mapping_naming_only_value_is_the_value() -> None:
    """``{value: 3}`` is a nested default, and always has been.

    Reading it as an annotation changes what already-deployed YAML
    means: ``data["threshold"]`` was the mapping and would become ``3``,
    so a template reading ``threshold.value`` breaks on upgrade. The
    shape also carries no information -- an annotation naming no mode
    takes the block's, which is exactly what the bare value does -- so
    requiring both keys costs an author nothing and leaves only a
    mapping naming *both* colliding.
    """
    literal = {"value": 3}

    defaults = SkipDefaults.from_stage({"threshold": literal}, SKIP_DEFAULT_OVERWRITE)

    assert defaults.entries["threshold"].value == literal, (
        "a nested default naming only 'value' was read as an annotation"
    )


def test_a_mapping_naming_a_mode_that_is_not_ours_stays_a_value() -> None:
    """``mode`` is a common field name, and most of them are not ours.

    A guard rather than a reproduction. ``{provider: "x", mode: "chat"}``
    names a mode, but not one of the two this grammar defines, so it is
    a nested default like any other and must pass without a word.
    """
    literal = {"provider": "x", "mode": "chat"}
    reports: list[tuple[str, Any, str, str]] = []

    defaults = SkipDefaults.from_stage(
        {"llm": literal},
        SKIP_DEFAULT_OVERWRITE,
        on_invalid=lambda *report: reports.append(report),
    )

    assert defaults.entries["llm"].value == literal
    assert reports == [], f"a nested default with a foreign 'mode' was reported: {reports}"


# ---------------------------------------------------------------------------
# 11. A near-miss annotation is reported
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_misspelled_value_key_beside_a_mode_is_reported(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``{values: false, mode: fill}`` is a broken annotation, not a value.

    The typo is silent and the failure is the opposite of what the author
    asked for: the whole mapping lands as the value, which is *truthy*
    where they wrote ``false``, so the branch the skip was meant to leave
    stays armed. This grammar rejects ``{value: "", label: "Email"}`` as
    an annotation precisely so ``label`` is not dropped in silence; the
    mirror case has to say something too.
    """
    config = _wizard(_stage(skip_default={"kb_enabled": {"values": False, "mode": "fill"}}))
    navigator = _navigator_for(config)
    state = WizardState(current_stage="configure", data={})

    with caplog.at_level(logging.WARNING):
        await navigator.navigate_skip(state)

    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("kb_enabled" in message for message in warnings), (
        f"a mapping naming a mode but no value was stored in silence: {warnings}"
    )
    assert state.data["kb_enabled"] == {"values": False, "mode": "fill"}, (
        "reporting must not change what is written -- a nested default whose "
        "'mode' happens to name one of ours is still the value"
    )


# ---------------------------------------------------------------------------
# 12. Which default an unreadable mode falls back to
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_unreadable_per_key_mode_falls_back_to_the_block_mode() -> None:
    """The fallback is the block's mode, not ``overwrite``.

    A pin rather than a reproduction: the code already does this and the
    docstring said "the documented default", which reads as
    ``overwrite``. Nothing could tell the two apart, because the only
    test of a bad mode used a block with no mode of its own -- where
    both readings give the same answer. They give opposite answers here,
    on the key an author cared enough about to annotate.
    """
    config = _wizard(
        _stage(
            skip_default={"kb_enabled": {"value": False, "mode": "fil"}},
            skip_default_mode=SKIP_DEFAULT_FILL,
        )
    )
    navigator = _navigator_for(config)
    state = WizardState(current_stage="configure", data={"kb_enabled": True})

    await navigator.navigate_skip(state)

    assert state.data["kb_enabled"] is True, (
        "a typo in one key's mode fell back past the block's own mode to "
        "'overwrite', destroying the value the block said to preserve"
    )


@pytest.mark.asyncio
async def test_a_blank_mode_falls_back_without_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``mode:`` left empty is unset, which is what a YAML null says.

    ``_stage_field`` was taught this in the same change -- an authored
    ``null`` reads as unset rather than as a wrong-typed value -- and the
    rule has to hold one level down too, or the author who leaves a mode
    blank to mean "use the block's" is warned for saying so.
    """
    config = _wizard(
        _stage(
            skip_default={"kb_enabled": {"value": False, "mode": None}},
            skip_default_mode=SKIP_DEFAULT_FILL,
        )
    )
    navigator = _navigator_for(config)
    state = WizardState(current_stage="configure", data={"kb_enabled": True})

    with caplog.at_level(logging.WARNING):
        await navigator.navigate_skip(state)

    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert warnings == [], f"a deliberately blank mode was reported as unreadable: {warnings}"
    assert state.data["kb_enabled"] is True, "the blank mode did not fall back to the block's"


@pytest.mark.asyncio
async def test_an_unknown_block_mode_is_reported_and_falls_back(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The block-level mode gets the same check the per-key one does."""
    config = _wizard(_stage(skip_default={"kb_enabled": False}, skip_default_mode="fil"))
    navigator = _navigator_for(config)
    state = WizardState(current_stage="configure", data={"kb_enabled": True})

    with caplog.at_level(logging.WARNING):
        await navigator.navigate_skip(state)

    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("skip_default_mode" in message for message in warnings), (
        f"an unreadable block mode was accepted in silence: {warnings}"
    )
    assert state.data["kb_enabled"] is False, "the block mode did not fall back to overwrite"


@pytest.mark.asyncio
async def test_a_non_string_block_mode_is_reported(caplog: pytest.LogCaptureFixture) -> None:
    """``skip_default_mode: 3`` is caught by the field's type contract."""
    config = _wizard(_stage(skip_default={"kb_enabled": False}, skip_default_mode=3))
    navigator = _navigator_for(config)
    state = WizardState(current_stage="configure", data={"kb_enabled": True})

    with caplog.at_level(logging.WARNING):
        await navigator.navigate_skip(state)

    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("skip_default_mode" in message for message in warnings), (
        f"a wrong-typed block mode was accepted in silence: {warnings}"
    )
    assert state.data["kb_enabled"] is False, "the block mode did not fall back to overwrite"


# ---------------------------------------------------------------------------
# 13. ``from_dict`` is not the constructor for an authored block
# ---------------------------------------------------------------------------


def test_from_dict_rejects_an_authored_block() -> None:
    """The authored shape and the dataclass shape are different shapes.

    ``from_dict`` is the idiomatic entry point everywhere else in this
    codebase, so a consumer reaches for it with the block they have --
    and every key of that block is an unknown field, which
    ``StructuredConfig`` ignores by default. The result was an empty
    ``SkipDefaults`` that applied nothing, with no error and no log.
    """
    with pytest.raises(ValueError, match="entries"):
        SkipDefaults.from_dict({"kb_enabled": False})


def test_from_dict_still_accepts_what_to_dict_produces() -> None:
    """Rejecting the authored shape must not break the round trip."""
    original = SkipDefaults.from_stage(
        {"a": {"value": 1, "mode": SKIP_DEFAULT_FILL}, "b": 2},
        SKIP_DEFAULT_OVERWRITE,
    )

    assert SkipDefaults.from_dict(original.to_dict()) == original


# ---------------------------------------------------------------------------
# 14. A nested default belongs to the config, not to one conversation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_nested_default_is_not_aliased_into_the_data() -> None:
    """Writing a mutable default hands out the config's own object.

    ``_StageField.extract`` already copies mutable list defaults for this
    reason. Without the same care here, a transform doing
    ``data["prefs"]["theme"] = ...`` edits the loaded stage metadata, and
    every later conversation on that FSM starts from the edit.
    """
    config = _wizard(_stage(skip_default={"prefs": {"theme": "dark"}}))
    wizard_fsm = WizardConfigLoader().load_from_dict(config)
    navigator = WizardReasoning(wizard_fsm=wizard_fsm, strict_validation=False)._navigator

    state = WizardState(current_stage="configure", data={})
    await navigator.navigate_skip(state)
    state.data["prefs"]["theme"] = "light"

    assert wizard_fsm.get_skip_defaults("configure").entries["prefs"].value == {"theme": "dark"}, (
        "one conversation's edit to a nested default reached the loaded "
        "config: the value written was the config's own object, not a copy"
    )


# ---------------------------------------------------------------------------
# 15. The grammar is reachable by name
# ---------------------------------------------------------------------------


def test_the_grammar_is_exported_from_the_package() -> None:
    """``get_skip_defaults`` is public, so its return type has to be.

    Without the export a consumer asserting on a mode has the bare
    strings ``"fill"``/``"overwrite"`` and a reach into a private module
    as the only options -- while ``NavigationConfig``, the class this one
    is modelled on, is exported from right here.
    """
    from dataknobs_bots import reasoning

    for name in (
        "SkipDefaults",
        "SkipDefaultEntry",
        "SKIP_DEFAULT_FILL",
        "SKIP_DEFAULT_OVERWRITE",
    ):
        assert hasattr(reasoning, name), f"{name} is not reachable from dataknobs_bots.reasoning"
        assert name in reasoning.__all__, f"{name} is missing from reasoning.__all__"
