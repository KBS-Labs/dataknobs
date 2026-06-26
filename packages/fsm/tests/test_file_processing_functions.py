"""Unit tests for the FileProcessor function adapters and wiring.

These cover the registered functions the pattern wires into its FSM — the
``_FileTransform`` / ``_FileAggregator`` state transforms and the
``_make_filter`` / ``_make_validator`` arc conditions — plus
``_build_custom_functions`` (which only exposes functions for configured
stages). End-to-end behavior across processing modes lives in
``test_file_processing_flow.py``; this file isolates the building blocks.

They replace the former code-generation tests, which asserted the inline
``_get_*_code`` string generators — that design never executed (the
generated code referenced registered functions by bare name, which the
inline ``eval`` scope could not resolve), so those methods were removed in
favor of real registered ``ITransformFunction`` transforms and
``(data, context)`` condition callables.
"""

from __future__ import annotations

import pytest

from dataknobs_fsm.functions.base import TransformError
from dataknobs_fsm.patterns.file_processing import (
    FileProcessingConfig,
    FileProcessor,
    _FileAggregator,
    _FileTransform,
    _make_filter,
    _make_validator,
)


# --------------------------------------------------------------------------
# _FileTransform — applies map-style transformations in order.
# --------------------------------------------------------------------------


def test_file_transform_applies_in_order() -> None:
    transform = _FileTransform(
        [
            lambda r: {**r, "step1": True},
            lambda r: {**r, "step2": r["step1"]},
        ]
    )
    out = transform.transform({"value": 1})
    assert out == {"value": 1, "step1": True, "step2": True}


def test_file_transform_empty_is_identity() -> None:
    assert _FileTransform([]).transform({"a": 1}) == {"a": 1}


def test_file_transform_non_dict_return_raises() -> None:
    transform = _FileTransform([lambda r: None])
    with pytest.raises(TransformError, match="must return a dict"):
        transform.transform({"a": 1})


# --------------------------------------------------------------------------
# _FileAggregator — reduces a record into a per-record summary dict.
# --------------------------------------------------------------------------


def test_file_aggregator_builds_summary() -> None:
    aggregator = _FileAggregator(
        {
            "total": lambda r: sum(r.get("values", [])),
            "count": lambda r: len(r.get("items", [])),
        }
    )
    out = aggregator.transform({"values": [1, 2, 3], "items": ["a", "b"]})
    assert out == {"total": 6, "count": 2}


# --------------------------------------------------------------------------
# _make_filter — arc condition; record passes iff all filters pass. The
# callable must accept (data, context) so the engine feeds it the raw dict.
# --------------------------------------------------------------------------


def test_make_filter_requires_all_predicates() -> None:
    filter_pass = _make_filter(
        [lambda r: r["value"] > 0, lambda r: r["status"] == "active"]
    )
    assert filter_pass({"value": 5, "status": "active"}, None) is True
    assert filter_pass({"value": 5, "status": "inactive"}, None) is False
    assert filter_pass({"value": -1, "status": "active"}, None) is False


def test_make_filter_accepts_context_kwarg() -> None:
    # The engine invokes arc conditions as (data, context); a one-arg
    # callable would be mis-detected as expecting a wrapped state object.
    filter_pass = _make_filter([lambda r: True])
    assert filter_pass({"x": 1}) is True  # context defaulted


# --------------------------------------------------------------------------
# _make_validator — arc condition built from a validation schema.
# --------------------------------------------------------------------------


def test_make_validator_required_and_type() -> None:
    check = _make_validator(
        {"name": {"required": True, "type": "str"}, "active": True}
    )
    assert check({"name": "alice", "active": True}, None) is True
    assert check({"active": True}, None) is False  # missing required name
    assert check({"name": 5, "active": True}, None) is False  # wrong type
    assert check({"name": "alice"}, None) is False  # missing required 'active'


def test_make_validator_min_max_pattern() -> None:
    check = _make_validator(
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
# _build_custom_functions — only configured stages are exposed by name.
# --------------------------------------------------------------------------


def test_custom_functions_passthrough_is_empty() -> None:
    proc = FileProcessor(FileProcessingConfig(input_path="x.json"))
    assert proc._build_custom_functions() == {}


def test_custom_functions_only_configured_stages() -> None:
    proc = FileProcessor(
        FileProcessingConfig(
            input_path="x.json",
            filters=[lambda r: True],
            transformations=[lambda r: r],
            aggregations={"n": lambda r: 1},
            validation_schema={"name": True},
        )
    )
    funcs = proc._build_custom_functions()
    assert set(funcs) == {"transform", "aggregate", "filter_pass", "validate_check"}
    assert isinstance(funcs["transform"], _FileTransform)
    assert isinstance(funcs["aggregate"], _FileAggregator)
    assert callable(funcs["filter_pass"])
    assert callable(funcs["validate_check"])


# --------------------------------------------------------------------------
# FSM wiring — the configured functions reach the built FSM's registry.
# --------------------------------------------------------------------------


def _registry_names(proc: FileProcessor) -> list[str]:
    return proc._fsm._fsm.function_registry.list_functions()


def test_fsm_registers_configured_functions() -> None:
    proc = FileProcessor(
        FileProcessingConfig(
            input_path="x.json",
            filters=[lambda r: True],
            transformations=[lambda r: r],
            validation_schema={"name": True},
        )
    )
    names = _registry_names(proc)
    assert "transform" in names
    assert "filter_pass" in names
    assert "validate_check" in names


def test_passthrough_fsm_builds_without_function_refs() -> None:
    # A passthrough config registers none of the pattern's own functions and
    # still builds a valid FSM (no state references a missing function). The
    # registry still carries the FSM's builtins, so assert the pattern's custom
    # names are absent rather than that the registry is empty.
    proc = FileProcessor(FileProcessingConfig(input_path="x.json"))
    assert proc._fsm is not None
    names = _registry_names(proc)
    for custom in ("transform", "aggregate", "filter_pass", "validate_check"):
        assert custom not in names
