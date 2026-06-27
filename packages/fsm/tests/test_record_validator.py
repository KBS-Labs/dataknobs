"""Unit tests for the shared record-validation builder.

``build_record_validator`` normalizes any of three validation-spec forms — a
friendly dict schema, a library ``IValidationFunction``, or a callable
predicate — into the ``(record, context) -> bool`` arc-condition gate the FSM
engine invokes. Both the ETL and file-processing patterns build their
``validate`` gate through it, so its branch behavior is pinned here once.
"""

from __future__ import annotations

import pytest

from dataknobs_fsm.functions.library.validators import (
    RangeValidator,
    build_record_validator,
)


# --------------------------------------------------------------------------
# Friendly dict-schema branch (the config-authored, serializable form).
# --------------------------------------------------------------------------


def test_dict_schema_required_and_type() -> None:
    check = build_record_validator(
        {"name": {"required": True, "type": "str"}, "active": True}
    )
    assert check({"name": "alice", "active": True}, None) is True
    assert check({"active": True}, None) is False  # missing required name
    assert check({"name": 5, "active": True}, None) is False  # wrong type
    assert check({"name": "alice"}, None) is False  # literal-True means present


def test_dict_schema_min_max_pattern() -> None:
    check = build_record_validator(
        {
            "age": {"type": "int", "min": 0, "max": 120},
            "email": {"pattern": r"^[^@]+@[^@]+\.[^@]+$"},
        }
    )
    assert check({"age": 30, "email": "a@b.co"}, None) is True
    assert check({"age": -1, "email": "a@b.co"}, None) is False
    assert check({"age": 200, "email": "a@b.co"}, None) is False
    assert check({"age": 30, "email": "not-an-email"}, None) is False


def test_dict_schema_present_value_constraints_are_type_safe() -> None:
    """A present ``min``/``max`` value that is non-numeric rejects, never raises.

    Regression for the promoted ``_make_validator`` quirk: a non-numeric value
    used to raise ``TypeError`` (``"abc" >= 18``). It is now a clean reject —
    bad data diverts to the reject terminal, never surfacing as a pipeline
    error.
    """
    check = build_record_validator({"age": {"min": 18}})
    assert check({"age": 30}, None) is True
    assert check({"age": 10}, None) is False
    assert check({"age": "old"}, None) is False  # non-numeric: reject, not TypeError
    assert check({"age": None}, None) is False


def test_dict_schema_presence_and_value_are_independent() -> None:
    """``required`` governs absence; ``min``/``max``/``type`` apply when present.

    The promoted ``_make_validator`` defaulted a missing numeric field to ``0``,
    so a ``min`` bound silently depended on whether ``0`` happened to satisfy it.
    Presence is now decoupled: an absent field passes unless ``required`` (so an
    optional bounded field is "if present, must satisfy the bound"); a
    ``required`` field must be present *and* satisfy the bound.
    """
    optional = build_record_validator({"score": {"min": 0}})
    assert optional({}, None) is True  # absent + not required → passes
    assert optional({"score": 5}, None) is True
    assert optional({"score": -1}, None) is False  # present → bound applies

    required = build_record_validator({"score": {"required": True, "min": 0}})
    assert required({}, None) is False  # absent + required → rejects
    assert required({"score": 5}, None) is True


# --------------------------------------------------------------------------
# IValidationFunction branch — the raise contract becomes a boolean gate.
# --------------------------------------------------------------------------


def test_validation_function_branch_pass_and_reject() -> None:
    check = build_record_validator(RangeValidator({"age": {"min": 18}}))
    assert check({"age": 30}, None) is True
    assert check({"age": 10}, None) is False  # FSMValidationError -> False


# --------------------------------------------------------------------------
# Callable branch — arity and async normalization to (record, context).
# --------------------------------------------------------------------------


def test_callable_one_arg_predicate() -> None:
    check = build_record_validator(lambda r: r.get("age", 0) >= 18)
    # The engine always calls with (data, context); a one-arg predicate must
    # still be invoked with just the record.
    assert check({"age": 30}, None) is True
    assert check({"age": 10}, None) is False


def test_callable_two_arg_predicate_receives_context() -> None:
    seen: dict = {}

    def predicate(record, context) -> bool:
        seen["context"] = context
        return record.get("ok", False)

    check = build_record_validator(predicate)
    assert check({"ok": True}, {"marker": 1}) is True
    assert seen["context"] == {"marker": 1}


@pytest.mark.asyncio
async def test_callable_async_predicate_is_awaitable() -> None:
    import inspect

    async def predicate(record, context=None) -> bool:
        return record.get("age", 0) >= 18

    check = build_record_validator(predicate)
    # An async input yields a coroutine function so the engine awaits it.
    assert inspect.iscoroutinefunction(check)
    assert await check({"age": 30}, None) is True
    assert await check({"age": 10}, None) is False


# --------------------------------------------------------------------------
# Unsupported spec -> TypeError (fail loud, not silently pass everything).
# --------------------------------------------------------------------------


def test_unsupported_spec_raises_type_error() -> None:
    with pytest.raises(TypeError):
        build_record_validator(42)  # type: ignore[arg-type]
