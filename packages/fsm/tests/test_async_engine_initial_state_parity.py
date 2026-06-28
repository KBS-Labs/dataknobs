"""Reproduce-first tests: batch/stream items enter their initial state at parity.

The async engine's single-record path enters the initial state through the
shared ``_enter_initial_state`` (pre-validators + resource allocation + initial
transforms). Batch (``_execute_batch``) and stream (``_execute_stream``) mode,
however, used to seed each fresh child/record context with a bare
``set_state(initial_state)`` and then call ``_execute_single`` directly — so an
item's **initial-state transform never ran**, its initial-state pre-validators
never fired, and its initial-state resources were never allocated. A start-state
transform that stamps the record (the common "normalize on entry" shape) was
silently skipped for every batch item and every streamed record.

These tests drive the engine's batch/stream data modes directly (the only paths
that reach ``_execute_batch`` / ``_execute_stream``) and assert the start-state
transform ran for each item/record, via a recorder side effect. They FAIL
against the bare-``set_state`` code (only the parent context's entry is
recorded, not the per-item/record ones) and PASS once both paths route child
entry through the shared ``_enter_and_execute_child`` → ``_enter_initial_state``.

The recorder assertions are **exact-count** (not subset): the start transform
must fire once per item/record and *not* on the parent aggregate context. The
``execute`` entry point used to call ``_enter_initial_state`` on the parent
context for *every* data mode, which on batch/stream both (a) ran the start
transform / pre-validators once spuriously on the aggregate context and (b)
allocated the start state's resources on a context that is never run through
``_execute_single`` — so they were never released (a leak on every batch/stream
run). The parent entry now happens only in single-record mode; the dedicated
resource-release tests below pin the no-leak half, and the exact-count recorder
assertions pin the no-spurious-run half.

Real constructs only: real FSM builds, a real registered start transform, and a
real in-memory database resource — no mocks.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import pytest

from dataknobs_fsm.api.async_simple import AsyncSimpleFSM
from dataknobs_fsm.core.context_factory import ContextFactory
from dataknobs_fsm.core.data_modes import DataHandlingMode
from dataknobs_fsm.core.modes import ProcessingMode
from dataknobs_fsm.execution.context import ExecutionContext
from dataknobs_fsm.streaming.core import StreamConfig, StreamContext


def _recording_fsm() -> tuple[AsyncSimpleFSM, list[Any]]:
    """FSM whose START state transform records each entered record's id.

    Returns the FSM plus the shared ``seen`` list the start transform appends
    to on every entry — so the test can assert it ran once per item/record.
    """
    seen: list[Any] = []

    def start_stamp(state: Any) -> dict[str, Any]:
        seen.append(state.data.get("id"))
        return {**state.data, "seen_start": True}

    config = {
        "name": "start_stamp_fsm",
        "main_network": "main",
        "networks": [
            {
                "name": "main",
                "states": [
                    {
                        "name": "start",
                        "is_start": True,
                        "functions": {"transform": "start_stamp"},
                    },
                    {"name": "end", "is_end": True},
                ],
                "arcs": [
                    {"from": "start", "to": "end", "name": "to_end"},
                ],
            }
        ],
    }
    fsm = AsyncSimpleFSM(
        config,
        data_mode=DataHandlingMode.COPY,
        custom_functions={"start_stamp": start_stamp},
    )
    return fsm, seen


@pytest.mark.asyncio
async def test_batch_items_run_initial_state_transform() -> None:
    """Every batch item runs the start-state transform (initial-entry parity).

    ``_execute_batch`` used to seed each child with a bare ``set_state`` and skip
    the initial-state entry, so the start transform never ran for batch items.
    """
    fsm, seen = _recording_fsm()
    items = [{"id": 1}, {"id": 2}, {"id": 3}]
    try:
        context = ContextFactory.create_batch_context(fsm._fsm, items)
        success, result = await fsm._async_engine.execute(context)
        assert success, f"Batch run did not complete cleanly: {result}"

        # Each item's result data carries the stamp...
        values = result["results"]
        assert len(values) == len(items)
        for value in values:
            assert value.get("seen_start") is True, (
                "A batch item did not run its start-state transform — "
                f"item result was {value!r}"
            )
    finally:
        await fsm.close()

    # ...and the start transform fired exactly once per item (ids 1-3), and NOT
    # on the parent aggregate context. A subset check (``{1,2,3} <= set(seen)``)
    # would pass even with a spurious parent entry (which appends the parent's
    # id-less data, e.g. ``None``); the exact multiset catches it.
    assert Counter(seen) == Counter([1, 2, 3]), (
        "Batch start-state entry was not at exact parity — expected one entry "
        f"per item and none on the parent aggregate, recorded {seen!r}"
    )


@pytest.mark.asyncio
async def test_stream_records_run_initial_state_transform() -> None:
    """Every streamed record runs the start-state transform (initial-entry parity).

    ``_execute_stream`` used to seed each record context with a bare
    ``set_state`` and skip the initial-state entry. The stream result carries
    only counts, so per-record entry is observed via the recorder side effect.
    """
    fsm, seen = _recording_fsm()
    records = [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}]
    try:
        context = ExecutionContext(
            data_mode=ProcessingMode.STREAM,
            transaction_mode=fsm._fsm.transaction_mode,
        )
        context.stream_context = StreamContext(config=StreamConfig())
        for i, record in enumerate(records):
            context.stream_context.add_data(
                record, chunk_id=f"chunk_{i}", is_last=(i == len(records) - 1)
            )

        success, result = await fsm._async_engine.execute(context)
        assert success, f"Stream run did not complete cleanly: {result}"
        assert result["records_processed"] == len(records)
        assert result["errors"] == []
    finally:
        await fsm.close()

    # Exact parity: one entry per record (ids 1-4) and none on the parent
    # aggregate context (see the batch test for why a subset check is too weak).
    assert Counter(seen) == Counter([1, 2, 3, 4]), (
        "Stream start-state entry was not at exact parity — expected one entry "
        f"per record and none on the parent aggregate, recorded {seen!r}"
    )


# --------------------------------------------------------------------------- #
# Resource release — the parent aggregate context must not strand start-state
# resources on batch/stream runs.
# --------------------------------------------------------------------------- #


def _resource_start_fsm() -> AsyncSimpleFSM:
    """FSM whose START state declares a real in-memory database resource.

    Entering ``start`` acquires ``scratch_db``; a clean run must release every
    acquisition. There is no transform, so the only acquisitions are the
    per-context start-state ones — making a post-run resource-manager scan an
    exact leak check.
    """
    config = {
        "name": "resource_start_fsm",
        "main_network": "main",
        "resources": [
            {
                "name": "scratch_db",
                "type": "async_database",
                "config": {"type": "memory"},
            },
        ],
        "networks": [
            {
                "name": "main",
                "states": [
                    {"name": "start", "is_start": True, "resources": ["scratch_db"]},
                    {"name": "end", "is_end": True},
                ],
                "arcs": [
                    {"from": "start", "to": "end", "name": "to_end"},
                ],
            }
        ],
    }
    return AsyncSimpleFSM(config, data_mode=DataHandlingMode.COPY)


def _assert_no_leaked_resources(manager: Any, *, mode: str) -> None:
    """Assert the resource manager holds no live resource after a run.

    Checked BEFORE ``close()`` so it observes the *run's own* release on every
    exit path — not the unconditional teardown ``cleanup()`` performs. A live
    entry means a context acquired ``scratch_db`` and never released it.
    """
    leaked_resources = dict(manager._resources)
    leaked_owners = {
        name: owners for name, owners in manager._resource_owners.items() if owners
    }
    assert not leaked_resources, (
        f"{mode} run leaked start-state resources: {leaked_resources!r} — the "
        "parent aggregate context entered its initial state but is never run "
        "through _execute_single, so its acquisition is never released"
    )
    assert not leaked_owners, (
        f"{mode} run leaked resource owners: {leaked_owners!r}"
    )


@pytest.mark.asyncio
async def test_batch_run_releases_start_state_resources() -> None:
    """A batch run releases every start-state resource it acquires.

    Each child item enters ``start`` (acquiring ``scratch_db``) and releases it
    in ``_execute_single``'s finally. The parent aggregate context must NOT
    enter ``start`` at all in batch mode — if it did, its acquisition would have
    no release path and leak on every batch run.
    """
    fsm = _resource_start_fsm()
    manager = fsm._resource_manager
    items = [{"id": 1}, {"id": 2}, {"id": 3}]
    try:
        context = ContextFactory.create_batch_context(fsm._fsm, items)
        # Wire the resource manager the way production does
        # (FSM.create_context / AsyncSimpleFSM.process), so start-state resource
        # acquisition is actually exercised on this context tree.
        context.resource_manager = manager
        success, result = await fsm._async_engine.execute(context)
        assert success, f"Batch run did not complete cleanly: {result}"
        _assert_no_leaked_resources(manager, mode="batch")
    finally:
        await fsm.close()


@pytest.mark.asyncio
async def test_stream_run_releases_start_state_resources() -> None:
    """A stream run releases every start-state resource it acquires.

    Stream counterpart of the batch test: each streamed record enters ``start``
    and releases ``scratch_db`` on exit; the parent aggregate context must not
    strand an acquisition by entering ``start`` itself.
    """
    fsm = _resource_start_fsm()
    manager = fsm._resource_manager
    records = [{"id": 1}, {"id": 2}, {"id": 3}]
    try:
        context = ExecutionContext(
            data_mode=ProcessingMode.STREAM,
            transaction_mode=fsm._fsm.transaction_mode,
        )
        context.resource_manager = manager
        context.stream_context = StreamContext(config=StreamConfig())
        for i, record in enumerate(records):
            context.stream_context.add_data(
                record, chunk_id=f"chunk_{i}", is_last=(i == len(records) - 1)
            )

        success, result = await fsm._async_engine.execute(context)
        assert success, f"Stream run did not complete cleanly: {result}"
        assert result["records_processed"] == len(records)
        _assert_no_leaked_resources(manager, mode="stream")
    finally:
        await fsm.close()
