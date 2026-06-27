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
