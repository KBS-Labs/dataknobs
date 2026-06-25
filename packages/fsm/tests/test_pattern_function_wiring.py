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
