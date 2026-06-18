"""Tests for ``ExecutionTracker``'s callback-registry composition.

``record`` fires the ``execution:record`` topic on the lazily
constructed ``execution_callbacks`` registry; the pre-existing
``record / query / get_stats / clear / __len__`` surface is unchanged.
"""

from __future__ import annotations

import pytest
from dataknobs_common.capabilities import Capability
from dataknobs_common.events import Event, InMemoryEventBus
from dataknobs_llm.tools.observability import (
    EXECUTION_RECORD_TOPIC,
    ExecutionHistoryQuery,
    ExecutionTracker,
    create_execution_record,
)


def _record(tool: str = "search", *, success: bool = True, error=None):
    return create_execution_record(
        tool_name=tool,
        parameters={"q": "x"},
        result=["r"] if success else None,
        duration_ms=12.0,
        success=success,
        error=error,
    )


def test_record_fires_execution_topic() -> None:
    tracker = ExecutionTracker(max_history=10)
    fired: list[dict] = []
    tracker.execution_callbacks.register(
        EXECUTION_RECORD_TOPIC, fired.append
    )

    for _ in range(5):
        tracker.record(_record())

    assert len(fired) == 5
    ev = fired[0]
    assert ev == {
        "tool_name": "search",
        "success": True,
        "duration_ms": 12.0,
        "error": None,
    }


def test_record_payload_carries_error_message() -> None:
    tracker = ExecutionTracker()
    fired: list[dict] = []
    tracker.execution_callbacks.register(
        EXECUTION_RECORD_TOPIC, fired.append
    )
    tracker.record(_record(success=False, error="boom"))
    assert fired[0]["success"] is False
    assert fired[0]["error"] == "boom"


def test_zero_overhead_when_no_callbacks() -> None:
    tracker = ExecutionTracker()
    tracker.record(_record())
    assert tracker._execution_callbacks is None


def test_public_surface_unchanged() -> None:
    """The record/query/get_stats/clear/__len__ surface is unchanged."""
    tracker = ExecutionTracker(max_history=100)
    for i in range(100):
        tracker.record(_record(tool="search" if i % 2 else "calc"))

    assert len(tracker) == 100
    search = tracker.query(ExecutionHistoryQuery(tool_name="search"))
    assert all(r.tool_name == "search" for r in search)
    stats = tracker.get_stats("search")
    assert stats.total_executions == len(search)
    assert stats.success_rate == 100.0
    tracker.clear()
    assert len(tracker) == 0


def test_max_history_eviction_preserved() -> None:
    tracker = ExecutionTracker(max_history=3)
    for i in range(5):
        rec = _record()
        rec.context_id = str(i)
        tracker.record(rec)
    # Oldest two evicted (FIFO pop(0)); newest three retained in order.
    ids = [r.context_id for r in tracker.query()]
    assert ids == ["2", "3", "4"]


def test_capability_advertised() -> None:
    tracker = ExecutionTracker()
    assert tracker.supports(Capability.EXECUTION_TRACKING)
    assert tracker.supports(Capability.CALLBACK_REGISTRY)


@pytest.mark.asyncio
async def test_record_async_fires_execution_topic() -> None:
    """``record_async`` records and fires, mirroring sync ``record``."""
    tracker = ExecutionTracker(max_history=10)
    fired: list[dict] = []
    tracker.execution_callbacks.register(
        EXECUTION_RECORD_TOPIC, fired.append
    )

    await tracker.record_async(_record())

    assert len(tracker) == 1
    assert fired == [
        {
            "tool_name": "search",
            "success": True,
            "duration_ms": 12.0,
            "error": None,
        }
    ]


@pytest.mark.asyncio
async def test_record_async_drives_event_bus_fanout_inside_loop() -> None:
    """The composed-fan-out path works from inside a running loop.

    Regression guard: ``record`` is always called from inside the
    running ``execute_tool`` loop, and a consumer following the
    documented ``also_publish_to`` composition would crash every tool
    execution with a ``TypeError`` if recording went through sync
    ``fire``. ``record_async`` fires through ``fire_async``, so the bus
    receives the event and nothing raises.
    """
    tracker = ExecutionTracker()
    bus = InMemoryEventBus()
    await bus.connect()
    received: list[Event] = []

    async def handler(event: Event) -> None:
        received.append(event)

    await bus.subscribe(EXECUTION_RECORD_TOPIC, handler)
    tracker.execution_callbacks.also_publish_to(bus)

    # Must NOT raise (this is the bug the guard protects against).
    await tracker.record_async(_record())

    assert len(received) == 1
    assert received[0].topic == EXECUTION_RECORD_TOPIC
    assert received[0].payload["tool_name"] == "search"


@pytest.mark.asyncio
async def test_sync_record_with_fanout_in_loop_rejects() -> None:
    """Sync ``record`` with composed fan-out inside a running loop is
    rejected by the substrate's fire-and-forget guard — the documented
    reason ``execute_tool`` and consumers must use ``record_async`` for
    the fan-out path.
    """
    tracker = ExecutionTracker()
    bus = InMemoryEventBus()
    await bus.connect()
    tracker.execution_callbacks.also_publish_to(bus)

    with pytest.raises(TypeError, match="fire_async"):
        tracker.record(_record())
