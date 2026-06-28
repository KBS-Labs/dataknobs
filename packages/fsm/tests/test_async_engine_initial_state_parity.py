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

Real constructs only: a real FSM build and a real registered start transform —
no mocks.
"""

from __future__ import annotations

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

    # ...and the start transform fired once per item (ids 1-3 all entered).
    assert {1, 2, 3} <= set(seen), (
        "Not every batch item entered its initial state — recorded entries were "
        f"{seen!r}"
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

    assert {1, 2, 3, 4} <= set(seen), (
        "Not every streamed record entered its initial state — recorded entries "
        f"were {seen!r}"
    )
