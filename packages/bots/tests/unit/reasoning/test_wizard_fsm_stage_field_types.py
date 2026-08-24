"""What a stage accessor returns when the config holds the wrong type.

``WizardFSM``'s stage accessors declare concrete return types -- ``str``,
``bool``, ``list[str]`` -- and every caller relies on them. Nothing between
the YAML and the accessor enforces any of it: ``_StageField.extract`` copies
the authored value out of the stage dict without coercing it, so a stage
written ``can_skip: "no"`` produced a *truthy string* from a method
declared ``-> bool``, and the stage the author marked unskippable was
skippable.

Passing the wrong type through does not make the error visible -- it moves
it somewhere the config is no longer in view. These accessors return the
field's documented default instead, and say so.

Loading such a config used to raise before any accessor could be reached:
two of the loader's *warning* heuristics hand an authored value straight to
a regex, so a non-string prompt or condition took the load down with a
``TypeError`` out of ``re``. A check that exists to advise about a config
must not be the thing that refuses it.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader

#: (accessor name, stage field, authored-but-wrong value, expected default)
_WRONG_TYPES: list[tuple[str, str, Any, Any]] = [
    ("get_stage_prompt", "prompt", 42, ""),
    ("get_stage_tools", "tools", "search", []),
    ("get_stage_suggestions", "suggestions", "yes", []),
    ("can_skip", "can_skip", "no", False),
    ("can_go_back", "can_go_back", "no", True),
    ("is_start_stage", "is_start", "yes", False),
    ("is_end_stage", "is_end", "yes", False),
]


def _fsm_with(stage_field: str, value: Any) -> Any:
    """A one-stage wizard whose ``stage_field`` holds an ill-typed value."""
    stage: dict[str, Any] = {"name": "only", "is_start": True, "prompt": "Hello."}
    stage[stage_field] = value
    if stage_field != "is_end":
        stage["is_end"] = True
    return WizardConfigLoader().load_from_dict(
        {"name": "wrong-types", "version": "1.0", "stages": [stage]}
    )


@pytest.mark.parametrize(
    ("accessor", "stage_field", "wrong_value", "expected"),
    _WRONG_TYPES,
    ids=[row[1] for row in _WRONG_TYPES],
)
def test_a_stage_accessor_returns_its_declared_type(
    caplog: pytest.LogCaptureFixture,
    accessor: str,
    stage_field: str,
    wrong_value: Any,
    expected: Any,
) -> None:
    """The declared type comes back, and the substitution is reported."""
    fsm = _fsm_with(stage_field, wrong_value)
    try:
        with caplog.at_level(logging.WARNING, logger="dataknobs_bots.reasoning.wizard_fsm"):
            result = getattr(fsm, accessor)("only")
    finally:
        fsm.close()

    assert result == expected
    assert isinstance(result, type(expected)), (
        f"{accessor}() returned {type(result).__name__}, not {type(expected).__name__}"
    )
    messages = " ".join(record.getMessage() for record in caplog.records)
    assert stage_field in messages and "only" in messages, (
        f"the substitution was silent; nothing named the stage and field: {messages!r}"
    )


def test_a_correctly_typed_value_is_returned_untouched(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Anti-overreach: a well-typed config is not warned about or replaced."""
    fsm = _fsm_with("can_skip", True)
    try:
        with caplog.at_level(logging.WARNING, logger="dataknobs_bots.reasoning.wizard_fsm"):
            assert fsm.can_skip("only") is True
            assert fsm.get_stage_prompt("only") == "Hello."
    finally:
        fsm.close()

    assert not caplog.records, f"a valid stage was warned about: {caplog.records}"


def test_a_transition_condition_that_is_not_a_string_is_not_reported_as_one() -> None:
    """``get_transition_condition`` feeds an observability record only.

    A non-string here would be written into a transition record's
    ``condition_evaluated`` field and read back as the expression that
    fired. "Nothing recorded" is the honest answer.
    """
    fsm = WizardConfigLoader().load_from_dict(
        {
            "name": "bad-condition",
            "version": "1.0",
            "stages": [
                {
                    "name": "start",
                    "is_start": True,
                    "prompt": "Go.",
                    "transitions": [{"target": "end", "condition": True}],
                },
                {"name": "end", "is_end": True, "prompt": "Done."},
            ],
        }
    )
    try:
        assert fsm.get_transition_condition("start", "end") is None
    finally:
        fsm.close()


def test_a_registry_entry_that_is_not_callable_resolves_to_none() -> None:
    """``resolve_function`` hands its result to a caller that calls it.

    Returning a non-callable defers the failure to the call site, where
    the routing-transform name is no longer in view. ``None`` is the
    answer the caller already handles -- it logs the name it could not
    resolve.
    """
    fsm = WizardConfigLoader().load_from_dict(
        {
            "name": "bad-function",
            "version": "1.0",
            "stages": [{"name": "only", "is_start": True, "is_end": True, "prompt": "Hi."}],
        }
    )
    try:
        registry = fsm._fsm.fsm.function_registry
        registry.functions["not_a_function"] = "just a string"

        assert fsm.resolve_function("not_a_function") is None
        assert fsm.resolve_function("absent_entirely") is None
    finally:
        fsm.close()


def test_a_config_with_ill_typed_text_fields_still_loads() -> None:
    """The warning heuristics advise; they do not refuse.

    ``_PYTHON_FORMAT_PATTERN`` over the template fields and
    ``_ENGLISH_CONDITION_PATTERNS`` over the conditions both search an
    authored value directly, so an int prompt or a bool condition reached
    ``re`` and raised ``TypeError`` -- from a block whose entire purpose is
    to log advice.
    """
    fsm = WizardConfigLoader().load_from_dict(
        {
            "name": "ill-typed",
            "version": "1.0",
            "stages": [
                {
                    "name": "start",
                    "is_start": True,
                    "prompt": 42,
                    "response_template": ["not", "a", "string"],
                    "transitions": [{"target": "end", "condition": True}],
                },
                {"name": "end", "is_end": True, "prompt": "Done."},
            ],
        }
    )
    try:
        assert fsm.get_stage_prompt("start") == ""
        assert fsm.get_transition_condition("start", "end") is None
    finally:
        fsm.close()
