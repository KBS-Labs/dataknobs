"""Config-authored ``builtin`` / ``custom`` function references — resolve + run.

Reproduce-first coverage for the documented dict function-reference forms
(``{type: builtin, ...}`` / ``{type: custom, module, name}``). These build a real
FSM from config through the public ``SimpleFSM`` path and run a record through
it — establishing that a configured built-in/custom *function* actually resolves
to a working transform/validator (not merely that the schema accepts the dict).

Real constructs only: real library functions
(``transformers.map_fields`` / ``transformers.FieldMapper`` /
``validators.RequiredFieldsValidator`` / ``validators.range_check``), a real
importable custom module (``tests.custom_fns_fixture``), and the real
builder/loader resolution path. No mocks.

Boundaries pinned here:
- the dict form is the supported way to reference ``builtin``/``custom``
  functions (they materialize with ``params`` and run);
- the bare-string state-sugar shorthand resolves only to ``registered`` /
  ``inline`` — never silently to a builtin (so the shorthand limit is a tested,
  deliberate boundary, not a surprise).
"""

from __future__ import annotations

import pytest

from dataknobs_fsm.api.simple import SimpleFSM
from dataknobs_fsm.config.loader import ConfigLoader


def _single_state_config(state: dict) -> dict:
    """Wrap the given function refs on a start state (arc'd to an end state).

    A ``start -> end`` shape is used rather than a lone start+end state because
    SimpleFSM does not run a single combined start/end state — the two-state
    form is the supported minimal pipeline.
    """
    start = {"name": "start", "is_start": True, "arcs": [{"target": "end"}], **state}
    return {
        "name": "fn_ref_fsm",
        "main_network": "main",
        "networks": [
            {
                "name": "main",
                "states": [start, {"name": "end", "is_end": True}],
            }
        ],
    }


def _run(config: dict, record: dict) -> dict:
    fsm = SimpleFSM(config)
    try:
        return fsm.process(record)
    finally:
        fsm.close()


# --------------------------------------------------------------------------- #
# builtin transforms — factory form and class form
# --------------------------------------------------------------------------- #

def test_builtin_function_reference_factory_resolves_and_runs() -> None:
    """``{type: builtin, name: "transformers.map_fields", params: {...}}`` runs.

    ``map_fields`` is a *factory* returning a ``FieldMapper``; ``params`` configure
    it. The resolved transform must rename ``a`` -> ``b`` and copy the unmapped
    ``keep`` field through.
    """
    config = _single_state_config(
        {
            "transforms": [
                {
                    "type": "builtin",
                    "name": "transformers.map_fields",
                    "params": {"mapping": {"a": "b"}},
                }
            ]
        }
    )
    result = _run(config, {"a": 1, "keep": 2})

    assert result["success"], f"builtin factory transform did not run: {result}"
    data = result["data"]
    assert data.get("b") == 1, f"field 'a' was not mapped to 'b': {data}"
    assert data.get("keep") == 2, f"unmapped field was dropped: {data}"
    assert "a" not in data, f"mapped source field 'a' should be gone: {data}"


def test_builtin_function_reference_class_form_resolves_and_runs() -> None:
    """``{type: builtin, name: "transformers.FieldMapper", params: {field_map: ...}}``.

    Same effect as the factory form, but referencing the *class* directly with
    its constructor's keyword (``field_map``) — proving the class shape resolves
    distinctly from the factory shape.
    """
    config = _single_state_config(
        {
            "transforms": [
                {
                    "type": "builtin",
                    "name": "transformers.FieldMapper",
                    "params": {"field_map": {"a": "b"}},
                }
            ]
        }
    )
    result = _run(config, {"a": 5, "keep": 9})

    assert result["success"], f"builtin class transform did not run: {result}"
    data = result["data"]
    assert data.get("b") == 5, f"class-form mapping did not apply: {data}"
    assert data.get("keep") == 9, f"unmapped field was dropped: {data}"


# --------------------------------------------------------------------------- #
# builtin validators (as pre_validators, which gate) — class form + kwargs factory
# --------------------------------------------------------------------------- #

def test_builtin_validator_reference_gates_via_pre_validators() -> None:
    """A builtin class validator authored as a ``pre_validators:`` ref gates entry.

    ``RequiredFieldsValidator(fields=[...])`` returns ``False`` when a required
    field is missing; the pre-validation phase fails the record. A record with
    the field passes.
    """
    config = _single_state_config(
        {
            "pre_validators": [
                {
                    "type": "builtin",
                    "name": "validators.RequiredFieldsValidator",
                    "params": {"fields": ["a"]},
                }
            ]
        }
    )

    ok = _run(config, {"a": 1})
    assert ok["success"], f"valid record was rejected by the builtin validator: {ok}"

    bad = _run(config, {"x": 1})
    assert not bad["success"], (
        "a record missing the required field should be failed by the builtin "
        f"pre-validator, but it succeeded: {bad}"
    )


def test_builtin_validator_kwargs_factory_reference_gates() -> None:
    """A ``**kwargs`` builtin factory (``validators.range_check``) resolves + gates.

    ``range_check(**field_ranges)`` returns a ``RangeValidator``; proves the
    adapter generalizes beyond the two headline transforms to a kwargs-factory
    validator.
    """
    config = _single_state_config(
        {
            "pre_validators": [
                {
                    "type": "builtin",
                    "name": "validators.range_check",
                    "params": {"age": {"min": 0, "max": 150}},
                }
            ]
        }
    )

    ok = _run(config, {"age": 30})
    assert ok["success"], f"in-range record was rejected: {ok}"

    bad = _run(config, {"age": 999})
    assert not bad["success"], f"out-of-range record should be failed: {bad}"


# --------------------------------------------------------------------------- #
# custom function reference — real importable module
# --------------------------------------------------------------------------- #

def test_custom_function_reference_imports_and_runs() -> None:
    """``{type: custom, module: "tests.custom_fns_fixture", name: "AddMarker"}``.

    The custom class is imported, constructed from ``params``, and run; its marker
    must land on the record.
    """
    config = _single_state_config(
        {
            "transforms": [
                {
                    "type": "custom",
                    "module": "tests.custom_fns_fixture",
                    "name": "AddMarker",
                    "params": {"key": "marked", "value": "yes"},
                }
            ]
        }
    )
    result = _run(config, {"id": 1})

    assert result["success"], f"custom transform did not run: {result}"
    assert result["data"].get("marked") == "yes", (
        f"custom AddMarker transform did not stamp its marker: {result['data']}"
    )


# --------------------------------------------------------------------------- #
# loud failure modes
# --------------------------------------------------------------------------- #

def test_builtin_missing_name_is_loud() -> None:
    """An unknown builtin name raises a clear ``ValueError`` at build time.

    Pins G3's failure mode: the headline doc example previously named a
    non-existent ``validators.validate_json``; a missing builtin must fail loudly,
    not silently no-op.
    """
    config = _single_state_config(
        {
            "transforms": [
                {"type": "builtin", "name": "validators.does_not_exist"}
            ]
        }
    )
    with pytest.raises(ValueError, match="Built-in function not found"):
        SimpleFSM(config)


def test_custom_missing_module_rejected_by_schema() -> None:
    """``{type: custom, name: "x"}`` (no ``module``) is rejected by the schema."""
    config = _single_state_config(
        {"transforms": [{"type": "custom", "name": "x"}]}
    )
    with pytest.raises(Exception, match=r"Custom functions require"):
        SimpleFSM(config)


# --------------------------------------------------------------------------- #
# bare-string shorthand boundary (G1) — registered / inline only
# --------------------------------------------------------------------------- #

def test_bare_string_shorthand_resolves_registered_or_inline_only() -> None:
    """The state-sugar bare string maps only to ``registered`` or ``inline``.

    A pre-registered name -> ``registered``; any other bare string -> ``inline``;
    a ``validators.<Name>`` string is NOT silently promoted to ``builtin`` (it is
    treated as inline code). Documents the deliberate shorthand limit so a future
    shorthand extension is a tested change, not an accidental behavior shift.
    """
    loader = ConfigLoader()
    loader.add_registered_function("known_fn")

    assert loader._convert_to_function_reference("known_fn") == {
        "type": "registered",
        "name": "known_fn",
    }

    arbitrary = loader._convert_to_function_reference("data['x'] = 1")
    assert arbitrary["type"] == "inline"

    builtin_name = loader._convert_to_function_reference("validators.RequiredFieldsValidator")
    assert builtin_name["type"] == "inline", (
        "a bare 'validators.<Name>' string must NOT be auto-promoted to a builtin "
        f"reference by the shorthand; got {builtin_name}"
    )


# --------------------------------------------------------------------------- #
# doc-truth regression guard (G3): the corrected guide example must run
# --------------------------------------------------------------------------- #

def test_documented_config_guide_builtin_example_runs() -> None:
    """The corrected FSM_CONFIG_GUIDE builtin example builds and runs.

    Mirrors the exact form documented in ``FSM_CONFIG_GUIDE.md`` so a future
    edit that re-breaks the headline example fails here.
    """
    config = _single_state_config(
        {
            "transforms": [
                {
                    "type": "builtin",
                    "name": "transformers.map_fields",
                    "params": {"mapping": {"source_id": "id"}},
                }
            ]
        }
    )
    result = _run(config, {"source_id": "abc", "payload": 1})

    assert result["success"], f"documented builtin example failed to run: {result}"
    assert result["data"].get("id") == "abc", (
        f"documented builtin example did not apply its mapping: {result['data']}"
    )
