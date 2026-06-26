"""Reproduce-first tests for the ETL pattern's function-wiring adoption.

The ETL pattern builds its database function *instances* into a top-level
``config['functions']`` dict that the Pydantic ``FSMConfig`` silently drops
(it has no ``functions`` field), and never passes those instances via
``AsyncSimpleFSM(config, custom_functions=...)``. As a result the built
FSM's ``function_registry`` never contains the pattern's own functions and
no state can invoke them — records traverse skeleton states doing nothing.

These tests pin the *wiring* contract — the pattern's functions reach the
built FSM's ``function_registry`` and are referenced by the state that
should run them — independent of resource injection and target persistence.
They FAIL against the unwired pattern and PASS once the instances are routed
through ``custom_functions=`` plus state-level ``functions`` blocks (the
idiom proven by ``examples/database_etl.py``).

Construction touches no filesystem (resource providers are created lazily
and ``_setup_resources`` logs construction errors today), so these tests
use throwaway paths and assert purely on the built FSM's structure.
"""

from __future__ import annotations

from typing import Any

from dataknobs_fsm.core.fsm import FSM as CoreFSM
from dataknobs_fsm.core.state import StateDefinition
from dataknobs_fsm.patterns.etl import DatabaseETL, ETLConfig, ETLMode
from dataknobs_fsm.patterns.file_processing import (
    FileProcessingConfig,
    FileProcessor,
    ProcessingMode,
    create_batch_file_processor,
)


def _core_fsm(pattern: Any) -> CoreFSM:
    """Reach the core FSM behind a pattern's ``AsyncSimpleFSM``.

    ``pattern._fsm`` is the ``AsyncSimpleFSM``; its ``_fsm`` is the core
    ``FSM`` carrying the populated ``function_registry`` and networks.
    """
    return pattern._fsm._fsm


def _state(core: CoreFSM, name: str) -> StateDefinition:
    for network in core.networks.values():
        if name in network.states:
            return network.states[name]
    raise KeyError(f"state {name!r} not found in any network")


# --------------------------------------------------------------------------
# ETL — the DB load function must reach the registry and be wired to `load`.
# --------------------------------------------------------------------------


def _etl() -> DatabaseETL:
    return DatabaseETL(
        ETLConfig(
            source_db={"type": "file", "path": "/tmp/dk_w1_src.json"},
            target_db={"type": "file", "path": "/tmp/dk_w1_tgt.json"},
            source_query=None,
            target_table="t",
            key_columns=["id"],
            mode=ETLMode.FULL_REFRESH,
        )
    )


def test_etl_registers_load_function() -> None:
    """The DatabaseUpsert 'load' instance must reach the FSM registry."""
    registry = _core_fsm(_etl()).function_registry
    load_fn = registry.get_function("load")
    assert load_fn is not None, (
        "DatabaseUpsert 'load' function was dropped — it never reached the "
        "built FSM's function_registry (top-level config['functions'] is "
        "silently discarded and custom_functions= was never passed)"
    )


def test_etl_load_state_references_load_transform() -> None:
    """The `load` state must reference the load transform to run it."""
    load_state = _state(_core_fsm(_etl()), "load")
    assert load_state.transform_functions, (
        "the ETL `load` state declares no transform — DatabaseUpsert is not "
        "wired as the load step, so a record reaching `load` does nothing"
    )


def test_etl_does_not_wire_fetch_as_per_record_step() -> None:
    """Per-record `extract` stays a passthrough (no fetch).

    Extraction is owned by ``run()._extract_batches``; wiring a per-record
    ``DatabaseFetch`` ('fetch all' per row) would be nonsensical. The
    `extract` start state must carry no transform.
    """
    extract_state = _state(_core_fsm(_etl()), "extract")
    assert not extract_state.transform_functions, (
        "the ETL `extract` state should be a passthrough — DatabaseFetch is "
        "repaired at the library layer, not wired as a per-record step"
    )


# --------------------------------------------------------------------------
# FileProcessor — enabled stages connect with no dead-ends; transforms are
# wired to their state; filtered records terminate in a non-emitting state.
# --------------------------------------------------------------------------


def _all_states(core: CoreFSM) -> dict[str, StateDefinition]:
    states: dict[str, StateDefinition] = {}
    for network in core.networks.values():
        states.update(network.states)
    return states


def _fp(**kwargs: Any) -> FileProcessor:
    return FileProcessor(FileProcessingConfig(input_path="x.json", **kwargs))


def test_fileprocessor_passthrough_has_no_dead_end() -> None:
    """A passthrough config flows read -> parse -> write -> complete.

    The dead-end bug: disabled stages (e.g. `filter`) were still on the path
    and had no outgoing arc, so every record stalled before `write`/`complete`.
    Now only enabled stages are wired, and no non-end state is a dead-end.
    """
    states = _all_states(_core_fsm(_fp()))
    assert set(states) == {"read", "parse", "write", "complete"}
    for name, state in states.items():
        if state.is_end_state():
            continue
        assert state.outgoing_arcs, (
            f"non-end state {name!r} has no outgoing arc (dead-end)"
        )


def test_fileprocessor_transform_state_references_function() -> None:
    """The `transform` state must reference the transform to run it."""
    transform_state = _state(_core_fsm(_fp(transformations=[lambda r: r])), "transform")
    assert transform_state.transform_functions, (
        "the FileProcessor `transform` state declares no transform — the "
        "configured transformation is not wired and would not run"
    )


def test_fileprocessor_filtered_terminal_is_non_emitting() -> None:
    """Filtered records terminate in a non-emitting end state.

    ``emit_output=False`` is what keeps filtered records out of the output in
    every mode (the streaming sink and the batch/whole writers both honor it).
    """
    filtered = _state(_core_fsm(_fp(filters=[lambda r: True])), "filtered")
    assert filtered.is_end_state()
    assert filtered.emit_output is False, (
        "filtered records must be excluded from output (emit_output=False)"
    )


def test_fileprocessor_emission_decided_by_emit_output_flag() -> None:
    """Batch/whole emission is resolved from the FSM's ``emit_output`` flag.

    The batch/whole accounting previously decided emission by a hardcoded
    ``final_state == 'complete'`` name comparison, diverging from the streaming
    path (which honors ``emit_output``). They now share one source of truth:
    ``FileProcessor._should_emit`` resolves the flag from the FSM, so all three
    modes apply the same exclusion policy. This FAILS pre-fix (``_should_emit``
    did not exist on ``FileProcessor``) and locks the unified mechanism using
    the real built FSM (no mock).
    """
    fp = _fp(
        filters=[lambda r: True],
        validation_schema={"id": {"required": True}},
    )
    assert fp._should_emit("complete") is True
    assert fp._should_emit("filtered") is False, (
        "the non-emitting 'filtered' terminal must be excluded via emit_output"
    )
    assert fp._should_emit("error") is False, (
        "the non-emitting 'error' terminal must be excluded via emit_output"
    )
    # An unknown / missing state defaults to emitting, matching the streaming
    # gate's fail-open default.
    assert fp._should_emit("nonexistent") is True
    assert fp._should_emit(None) is True


def test_create_batch_file_processor_constructs() -> None:
    """Pre-existing crash: the factory passed a non-existent ``batch_size``
    field to the frozen ``FileProcessingConfig`` (the field is ``chunk_size``),
    raising ``TypeError`` on every call. Constructing must succeed and the
    batch size must land on ``chunk_size``.
    """
    proc = create_batch_file_processor(
        input_paths=["a.txt"],
        output_path="out.txt",
        patterns=["*.txt"],
        batch_size=25,
    )
    assert isinstance(proc, FileProcessor)
    assert proc.config.chunk_size == 25
    assert proc.config.mode is ProcessingMode.BATCH
