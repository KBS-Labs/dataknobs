"""One schema type, reachable by both routes that claim to produce it.

``StateDefinition.schema`` is declared ``StateSchema | None`` and
``validate_data`` is declared to return ``tuple[bool, list[str]]``. Neither
was true of a config-built FSM: the builder minted a *different* class per
call, function-local and un-importable, whose ``validate`` returned an
anonymous ``type("Result", (), {...})()`` carrying ``.valid`` and ``.errors``.
So the declared type was never the built type, and the documented return of a
public method raised ``TypeError`` on every FSM that configuration produces.

The two validators also disagreed about what validation *is* --- one checked a
``Field`` list, the other a JSON Schema mapping --- and only the second was
reachable from configuration. This file pins the surviving semantics against
the type the declaration always named.

Real builds throughout: the subject is what the builder produces, so a
hand-constructed schema would be testing the half that already worked.
"""

from __future__ import annotations

from typing import Any

import pytest
from dataknobs_data import Record

from dataknobs_fsm.api.async_simple import AsyncSimpleFSM
from dataknobs_fsm.config.builder import FSMBuilder
from dataknobs_fsm.core.state import StateSchema


def _config_with_schema(schema: dict[str, Any] | None = None) -> dict[str, Any]:
    """An FSM whose start state carries a ``data_schema`` block."""
    start: dict[str, Any] = {"name": "start", "is_start": True}
    if schema is not None:
        start["schema"] = schema
    return {
        "name": "schema_fsm",
        "main_network": "main",
        "networks": [
            {
                "name": "main",
                "states": [start, {"name": "end", "is_end": True}],
                "arcs": [{"from": "start", "to": "end", "name": "go"}],
            }
        ],
    }


_OBJECT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"n": {"type": "integer"}, "s": {"type": "string"}},
    "required": ["n"],
}


def _built_start_state(schema: dict[str, Any] | None = _OBJECT_SCHEMA) -> Any:
    """The start state of a real, config-built FSM."""
    from dataknobs_fsm.config.loader import ConfigLoader

    config = ConfigLoader().load_from_dict(_config_with_schema(schema))
    return FSMBuilder().build(config).get_start_state()


# --------------------------------------------------------------------------- #
# The declared contract, on the states production actually has
# --------------------------------------------------------------------------- #


def test_validate_data_returns_a_pair_for_a_config_built_state() -> None:
    """The documented return of a public method, on the only states production has.

    ``validate_data`` says it returns ``(bool, list[str])`` and every test of
    it passed --- because every test of it built the schema by hand. Through
    the builder it raised ``TypeError: cannot unpack non-iterable Result
    object``, which is the whole item in one line.
    """
    state = _built_start_state()

    is_valid, errors = state.validate_data({"n": 1})

    assert is_valid is True
    assert errors == []


def test_validate_data_names_a_missing_required_field() -> None:
    """The pair carries the message, not merely the flag."""
    state = _built_start_state()

    is_valid, errors = state.validate_data({"s": "x"})

    assert is_valid is False
    assert errors == ["Required field 'n' is missing"]


def test_the_schema_a_builder_produces_is_the_declared_type() -> None:
    """``StateDefinition.schema`` is what it says it is.

    Impossible before: the builder's product was a class defined inside a
    method body, so no caller could name it and ``isinstance`` had nothing to
    ask about. This is the assertion that stops the two shapes reappearing.
    """
    assert isinstance(_built_start_state().schema, StateSchema)


def test_a_state_with_no_schema_block_still_validates_anything() -> None:
    """The ``None`` branch of ``validate_data`` is unchanged."""
    state = _built_start_state(schema=None)

    assert state.schema is None
    assert state.validate_data({"anything": object()}) == (True, [])


# --------------------------------------------------------------------------- #
# The async surface returns exactly what it returned before
# --------------------------------------------------------------------------- #


async def test_async_validate_returns_the_same_dict_it_did_before() -> None:
    """All four verdicts, unchanged in shape and content.

    ``validate`` used to read ``.valid``/``.errors`` off an anonymous object
    and rebuild a dict; it now unpacks a pair. The dict a caller sees is
    identical, which is the claim that lets the type change ship on its own.
    """
    async with AsyncSimpleFSM(_config_with_schema(_OBJECT_SCHEMA)) as fsm:
        assert await fsm.validate({"n": 1}) == {"valid": True, "errors": []}
        assert await fsm.validate({"s": "x"}) == {
            "valid": False,
            "errors": ["Required field 'n' is missing"],
        }
        assert await fsm.validate({"n": "not an int"}) == {
            "valid": False,
            "errors": ["Field 'n' has wrong type"],
        }

    async with AsyncSimpleFSM(_config_with_schema(None)) as fsm:
        assert await fsm.validate({"anything": 1}) == {"valid": True, "errors": []}


async def test_async_validate_accepts_a_record_and_a_dict_alike() -> None:
    """The deleted ``Record`` round-trip changed nothing.

    The method used to wrap a ``dict`` in a ``Record`` and hand it to a
    validator whose first act was to unwrap it again. ``StateSchema.validate``
    takes either, so the conversion is gone rather than reversed --- and both
    spellings must still reach the same verdict.
    """
    async with AsyncSimpleFSM(_config_with_schema(_OBJECT_SCHEMA)) as fsm:
        assert await fsm.validate({"n": 1}) == await fsm.validate(Record({"n": 1}))
        assert await fsm.validate({"s": "x"}) == await fsm.validate(Record({"s": "x"}))


async def test_validate_raises_a_named_error_when_there_is_no_start_state() -> None:
    """A machine that validated nothing must not report ``valid``.

    ``get_start_state()`` is declared ``StateDefinition | None`` and was
    dereferenced unguarded, so this case raised ``AttributeError: 'NoneType'
    object has no attribute 'schema'`` --- an answer that names neither the
    FSM nor the problem.

    Reached by emptying the built FSM rather than by configuring it away:
    config validation refuses a start-less network outright ("Network must
    have at least one start state"), so this guards the contract the return
    type declares rather than a state configuration can currently produce.
    """
    async with AsyncSimpleFSM(_config_with_schema(_OBJECT_SCHEMA)) as fsm:
        fsm._fsm.networks.clear()
        assert fsm._fsm.get_start_state() is None

        with pytest.raises(ValueError, match="no start state"):
            await fsm.validate({"n": 1})


# --------------------------------------------------------------------------- #
# Rules carried across unchanged, including the ones that are wrong
# --------------------------------------------------------------------------- #


def test_a_non_object_schema_passes_everything() -> None:
    """The pass-through rule survives the move out of the builder.

    A definition whose top level is not ``object`` carries no ``properties``
    to check against, so everything passes. Kept in the validator rather than
    moved to the builder --- returning ``None`` there would put half of one
    decision in each file.
    """
    assert StateSchema({"type": "array"}).validate({"anything": 1}) == (True, [])
    assert StateSchema({}).validate({"anything": 1}) == (True, [])


def test_bool_is_still_accepted_for_an_integer_field() -> None:
    """A known-wrong verdict, pinned deliberately.

    ``isinstance(True, int)`` is true in Python, so ``True`` satisfies an
    ``integer`` field. That is wrong and is preserved: correcting fidelity
    and unifying the type in one change would leave no way to tell which
    change caused a behaviour difference. The follow-up that fixes this
    inverts this test.
    """
    schema = StateSchema({"type": "object", "properties": {"n": {"type": "integer"}}})

    assert schema.validate({"n": True}) == (True, [])


def test_additional_properties_false_is_still_ignored() -> None:
    """The second known-wrong verdict, pinned for the same reason.

    ``additionalProperties: false`` is accepted in configuration and never
    read. The old body had an ``allow_extra_fields`` flag that would have
    expressed it, defaulted to ``True``, and nothing ever mapped the keyword
    onto it --- so the flag is gone rather than kept as a knob no config
    could reach.
    """
    schema = StateSchema(
        {"type": "object", "properties": {"n": {"type": "integer"}}, "additionalProperties": False}
    )

    assert schema.validate({"n": 1, "unexpected": "kept"}) == (True, [])


def test_an_unrecognised_type_keyword_passes() -> None:
    """A vocabulary this validator does not implement is not a failure."""
    schema = StateSchema({"type": "object", "properties": {"x": {"type": "widget"}}})

    assert schema.validate({"x": object()}) == (True, [])


def test_a_non_mapping_is_reported_by_type_name() -> None:
    """The rejection message is unchanged, including the type name it names."""
    assert StateSchema({"type": "object"}).validate("not a mapping") == (
        False,
        ["Expected object or Record, got str"],
    )
