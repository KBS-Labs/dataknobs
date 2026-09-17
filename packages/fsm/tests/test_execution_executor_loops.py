# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""The synchronous FSM executors reach the engine on a loop the caller can name.

``BatchExecutor`` and ``StreamExecutor`` drive the one async engine through a
:class:`~dataknobs_common.SyncLoopBridge` scoped to the operation --- which
``98191919`` chose deliberately, so a discarded executor leaves no
process-lifetime daemon thread behind. That intent is about **thread
lifetime**, and it is not in question here; ``test_sync_entrypoint_bridge.py``
pins it and keeps pinning it.

What the scope does not settle is **loop identity**, and an FSM's async
resources outlive any one operation. ``AsyncDatabaseResourceAdapter`` opens its
``AsyncDatabase`` lazily on first use and says so in its ``release``: *"the
shared async database stays open across acquisitions ... so pooled connections
are not churned per acquire/release."* It is therefore bound to whichever loop
first touched it, and every later operation arrives on a different one. An
``asyncpg`` pool acquired that way is unusable from the second loop, with an
error that names the connection rather than the loop ---
``InterfaceError: cannot perform operation: another operation is in progress``.

Three shapes reach it, measured before the fix:

=========================================== =========
``execute_batches(batch_size=2)``, 6 items  3 loops
a second ``execute_batch`` on one executor  2 loops
a second ``execute_stream`` on one executor 2 loops
=========================================== =========

The first is the worst, because it is **one public call**: ``execute_batches``
loops over ``execute_batch``, and each of those opened a bridge of its own. It
is the same defect ``bulk_insert_dataframe`` had in ``dataknobs-data``, where
one insert ran each chunk on a different loop.

The other two cannot be fixed by scoping alone --- an executor that does not
own the FSM cannot own the loop its resources bound to --- so ``bridge=`` is
how the owner says which loop that is. It is also the answer to the fourth
shape, which is not about executors at all: ``SimpleFSM.process()`` runs on the
FSM's own long-lived bridge (``FSM.get_sync_bridge()``), so a consumer who
calls that and then builds a ``BatchExecutor`` over the same FSM has already
opened the resources on a loop the executor will not use. Passing
``fsm.get_sync_bridge()`` as ``bridge=`` is what makes those agree.

The loop-identity tests below pin all of it without a service, with a real
async transform that records the loop that ran it. They hold the loop
*objects*, not their ids: a bridge's loop is freed when its thread ends, and
CPython hands the next allocation the same address often enough that counting
``id()`` alone reports one loop where there were two.

``timeout=`` arrives with ``bridge=`` for the same reason it did on the Simple
API in ``98191919``: these calls block a synchronous caller for an operation
whose size it does not know, and a caller inside a ``def`` has no cancellation
of its own. It bounds the **operation**, not each item --- a per-item bound
would be no bound at all on the call the caller actually made.

Two pre-existing defects are pinned here too, because both are in the code
these changes open and neither could be left in place:

* ``StreamExecutor`` gated execution on ``fsm.name in fsm.networks``, which is
  false whenever an FSM is named anything other than its main network --- the
  ordinary case. Every record then took the "no FSM configured" path, passed
  through untouched, and was counted as successful.
* ``BatchExecutor._release_resources`` compared a ``ResourceStatus`` member to
  the string ``"allocated"``, so its body never ran and nothing ever returned
  to the pool. mypy had been reporting both halves of that
  (``comparison-overlap``, then ``unreachable``) for as long as the cell has
  been measured.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from dataknobs_common import SyncLoopBridge
from dataknobs_common.testing import assert_no_leaked_bridge_threads
from dataknobs_fsm.config.builder import FSMBuilder
from dataknobs_fsm.config.schema import (
    ArcConfig,
    FSMConfig,
    FunctionReference,
    NetworkConfig,
    StateConfig,
)
from dataknobs_fsm.execution.batch import BatchExecutor
from dataknobs_fsm.execution.context import ExecutionContext, ResourceStatus
from dataknobs_fsm.execution.stream import StreamExecutor, StreamPipeline
from dataknobs_fsm.streaming.core import IStreamSource, StreamChunk


class LoopRecorder:
    """Records the event loop that ran each pass through an FSM transform.

    Holds the loop objects rather than their ids. A bridge's loop is freed
    when its thread ends, so ``id()`` alone can be reused by the next one and
    report a single loop where there were several.
    """

    def __init__(self) -> None:
        self.loops: list[asyncio.AbstractEventLoop] = []

    async def transform(self, data: Any, context: Any) -> Any:
        """A real async transform whose only side effect is the witness."""
        self.loops.append(asyncio.get_running_loop())
        return data

    @property
    def calls(self) -> int:
        """How many times the transform ran."""
        return len(self.loops)

    @property
    def distinct_loops(self) -> int:
        """How many different event loops ran it."""
        return len({id(loop) for loop in self.loops})


class SlowRecorder(LoopRecorder):
    """A ``LoopRecorder`` whose transform also sleeps, for the timeout tests."""

    def __init__(self, delay: float) -> None:
        super().__init__()
        self.delay = delay

    async def transform(self, data: Any, context: Any) -> Any:
        await super().transform(data, context)
        await asyncio.sleep(self.delay)
        return data


def witnessed_fsm(recorder: LoopRecorder, *, name: str = "witness_fsm") -> Any:
    """A start→end FSM whose one arc runs ``recorder.transform``.

    The FSM is named differently from its network on purpose: that is the
    ordinary shape, and it is the shape ``StreamExecutor`` used to no-op on.
    """
    config = FSMConfig(
        name=name,
        main_network="main",
        networks=[
            NetworkConfig(
                name="main",
                states=[
                    StateConfig(
                        name="start",
                        is_start=True,
                        arcs=[
                            ArcConfig(
                                target="end",
                                transform=FunctionReference(type="registered", name="witness"),
                            )
                        ],
                    ),
                    StateConfig(name="end", is_end=True),
                ],
            )
        ],
    )
    builder = FSMBuilder()
    builder.register_function("witness", recorder.transform)
    return builder.build(config)


class ListSource(IStreamSource):
    """A real stream source over in-memory chunks."""

    def __init__(self, chunks: list[list[dict[str, Any]]]) -> None:
        self._chunks = chunks
        self._index = 0
        self.closed = False

    def read_chunk(self) -> StreamChunk | None:
        if self._index >= len(self._chunks):
            return None
        chunk = StreamChunk(
            data=self._chunks[self._index],
            sequence_number=self._index,
            is_last=(self._index == len(self._chunks) - 1),
        )
        self._index += 1
        return chunk

    def close(self) -> None:
        self.closed = True


def items(n: int) -> list[dict[str, int]]:
    """``n`` trivial records, small enough to read in a failure message."""
    return [{"id": i} for i in range(n)]


# --------------------------------------------------------------------------- #
# 1. Loop identity within one public call
# --------------------------------------------------------------------------- #


def test_one_batch_runs_every_item_on_one_loop() -> None:
    """``execute_batch`` already scopes one loop; this keeps it that way.

    Green before the change as well as after. It guards the half of
    ``98191919`` that was right, so a later scoping change cannot quietly
    reintroduce per-item churn.
    """
    recorder = LoopRecorder()
    BatchExecutor(fsm=witnessed_fsm(recorder), parallelism=1).execute_batch(items(6))

    assert recorder.calls == 6
    assert recorder.distinct_loops == 1


def test_one_parallel_batch_runs_every_item_on_one_loop() -> None:
    """The parallel path shares the operation's loop across its worker threads."""
    recorder = LoopRecorder()
    BatchExecutor(fsm=witnessed_fsm(recorder), parallelism=3).execute_batch(items(6))

    assert recorder.calls == 6
    assert recorder.distinct_loops == 1


def test_batching_runs_every_batch_on_one_loop() -> None:
    """``execute_batches`` is one call, so it is one loop.

    It loops over ``execute_batch``, and each of those opened a bridge of its
    own --- so six items at ``batch_size=2`` reached the engine on three
    different loops, from a single public call. An FSM resource opened during
    the first batch is unusable from the second.
    """
    recorder = LoopRecorder()
    executor = BatchExecutor(fsm=witnessed_fsm(recorder), parallelism=1, batch_size=2)

    executor.execute_batches(items(6))

    assert recorder.calls == 6
    assert recorder.distinct_loops == 1, (
        f"execute_batches ran its 3 batches on {recorder.distinct_loops} loops; "
        "one public call must reach the engine on one loop"
    )


def test_one_stream_runs_every_chunk_on_one_loop() -> None:
    """``execute_stream`` reaches the engine on one loop for the whole stream."""
    recorder = LoopRecorder()
    StreamExecutor(fsm=witnessed_fsm(recorder)).execute_stream(
        StreamPipeline(source=ListSource([[{"id": 1}, {"id": 2}], [{"id": 3}]]))
    )

    assert recorder.calls == 3
    assert recorder.distinct_loops == 1


# --------------------------------------------------------------------------- #
# 2. bridge= — the loop the caller already owns
# --------------------------------------------------------------------------- #


def test_the_caller_can_name_the_loop_a_batch_runs_on() -> None:
    """A supplied bridge is used as-is, so both batches meet the same resources."""
    recorder = LoopRecorder()
    executor_fsm = witnessed_fsm(recorder)

    with SyncLoopBridge() as bridge:
        executor = BatchExecutor(fsm=executor_fsm, parallelism=1, bridge=bridge)
        executor.execute_batch(items(2))
        executor.execute_batch(items(2))

    assert recorder.calls == 4
    assert recorder.distinct_loops == 1


def test_the_caller_can_name_the_loop_a_stream_runs_on() -> None:
    """The same for streams: two operations, one caller-owned loop."""
    recorder = LoopRecorder()
    executor_fsm = witnessed_fsm(recorder)

    with SyncLoopBridge() as bridge:
        executor = StreamExecutor(fsm=executor_fsm, bridge=bridge)
        executor.execute_stream(StreamPipeline(source=ListSource([[{"id": 1}]])))
        executor.execute_stream(StreamPipeline(source=ListSource([[{"id": 2}]])))

    assert recorder.calls == 2
    assert recorder.distinct_loops == 1


def test_a_batch_and_a_stream_share_a_supplied_loop() -> None:
    """Two executors over one FSM reach its resources on one loop.

    This is the shape ``bridge=`` exists for: the executors do not own the FSM,
    so neither can own the loop its resources bound to. Only the caller knows
    that both surfaces touch the same FSM.
    """
    recorder = LoopRecorder()
    shared_fsm = witnessed_fsm(recorder)

    with SyncLoopBridge() as bridge:
        BatchExecutor(fsm=shared_fsm, parallelism=1, bridge=bridge).execute_batch(items(2))
        StreamExecutor(fsm=shared_fsm, bridge=bridge).execute_stream(
            StreamPipeline(source=ListSource([[{"id": 9}]]))
        )

    assert recorder.calls == 3
    assert recorder.distinct_loops == 1


def test_an_executor_can_run_on_the_fsms_own_loop() -> None:
    """``fsm.get_sync_bridge()`` is a bridge like any other.

    ``SimpleFSM`` and ``AdvancedFSM.execute_step_sync`` already run there, so a
    consumer mixing those surfaces with an executor passes that bridge and the
    two agree about which loop the FSM's resources belong to.
    """
    recorder = LoopRecorder()
    shared_fsm = witnessed_fsm(recorder)
    try:
        shared_fsm.execute({"id": 0})
        BatchExecutor(
            fsm=shared_fsm, parallelism=1, bridge=shared_fsm.get_sync_bridge()
        ).execute_batch(items(2))
        BatchExecutor(
            fsm=shared_fsm, parallelism=1, bridge=shared_fsm.get_sync_bridge()
        ).execute_batch(items(2))
    finally:
        shared_fsm.close()

    assert recorder.calls == 5
    # ``FSM.execute`` is a throwaway loop of its own, so it is not counted here;
    # what must agree is the two executors sharing the FSM's bridge.
    assert recorder.distinct_loops == 2


def test_a_supplied_bridge_outlives_the_operation() -> None:
    """Nothing closes a bridge the caller owns.

    A wrapper that closed a bridge it was handed would tear the loop out from
    under whatever else the caller is running on it.
    """
    recorder = LoopRecorder()
    bridge = SyncLoopBridge()
    try:
        BatchExecutor(fsm=witnessed_fsm(recorder), parallelism=1, bridge=bridge).execute_batch(
            items(1)
        )
        assert not bridge.is_closed
        StreamExecutor(fsm=witnessed_fsm(recorder), bridge=bridge).execute_stream(
            StreamPipeline(source=ListSource([[{"id": 1}]]))
        )
        assert not bridge.is_closed
    finally:
        bridge.close()


def test_an_executor_without_a_bridge_still_leaves_no_thread() -> None:
    """The default stays a throwaway bridge, torn down with the operation.

    ``98191919``'s intent: a discarded executor must not leave a
    process-lifetime daemon thread behind. ``bridge=`` adds a way to say which
    loop; it does not change what happens when nobody says.
    """
    recorder = LoopRecorder()
    with assert_no_leaked_bridge_threads():
        BatchExecutor(fsm=witnessed_fsm(recorder), parallelism=2, batch_size=2).execute_batches(
            items(4)
        )
        StreamExecutor(fsm=witnessed_fsm(recorder)).execute_stream(
            StreamPipeline(source=ListSource([[{"id": 1}]]))
        )


# --------------------------------------------------------------------------- #
# 3. timeout= — the bound a blocked synchronous caller cannot otherwise get
# --------------------------------------------------------------------------- #


def test_a_batch_timeout_bounds_the_operation_not_each_item() -> None:
    """Six items of 0.2s under a 0.35s bound stop early, they do not take 1.2s.

    A per-item bound would let the operation run for the item count times the
    bound, which is no bound at all on the call the caller made.
    """
    recorder = SlowRecorder(delay=0.2)
    executor = BatchExecutor(fsm=witnessed_fsm(recorder), parallelism=1, timeout=0.35)

    started = time.monotonic()
    with pytest.raises(TimeoutError):
        executor.execute_batch(items(6))
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, f"the bound did not hold: {elapsed:.2f}s for a 0.35s timeout"
    assert recorder.calls < 6, "every item ran, so the deadline bounded nothing"


def test_a_batch_timeout_spans_the_batches_of_one_call() -> None:
    """``execute_batches`` spends one budget across all of its batches."""
    recorder = SlowRecorder(delay=0.2)
    executor = BatchExecutor(fsm=witnessed_fsm(recorder), parallelism=1, batch_size=2, timeout=0.35)

    started = time.monotonic()
    with pytest.raises(TimeoutError):
        executor.execute_batches(items(6))
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, f"the bound restarted per batch: {elapsed:.2f}s for a 0.35s timeout"


def test_a_parallel_batch_timeout_bounds_the_wall_clock() -> None:
    """Leaving the thread pool waits for it, and the wait is still bounded.

    ``ThreadPoolExecutor.__exit__`` shuts down with ``wait=True`` and does not
    cancel what is queued, so a deadline that only stopped *new* work would
    still block for every item already submitted. It does not, because a
    queued worker reaches the operation, finds the budget spent, and refuses
    without reaching the loop --- leaving only the items already in flight.
    """
    recorder = SlowRecorder(delay=0.4)
    executor = BatchExecutor(fsm=witnessed_fsm(recorder), parallelism=3, timeout=0.5)

    started = time.monotonic()
    with pytest.raises(TimeoutError):
        executor.execute_batch(items(12))
    elapsed = time.monotonic() - started

    assert elapsed < 1.2, (
        f"the pool shutdown outlasted the bound: {elapsed:.2f}s for a 0.5s timeout"
    )
    assert recorder.calls < 12, "every item ran, so the deadline bounded nothing"


def test_a_stream_timeout_bounds_the_operation() -> None:
    """The same bound, over a stream's records rather than a batch's items."""
    recorder = SlowRecorder(delay=0.2)
    executor = StreamExecutor(fsm=witnessed_fsm(recorder), timeout=0.35)
    pipeline = StreamPipeline(source=ListSource([[{"id": i} for i in range(6)]]))

    started = time.monotonic()
    with pytest.raises(TimeoutError):
        executor.execute_stream(pipeline)
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, f"the bound did not hold: {elapsed:.2f}s for a 0.35s timeout"


def test_no_timeout_means_the_operation_runs_to_completion() -> None:
    """The default is unbounded, so nothing changes for an existing caller."""
    recorder = SlowRecorder(delay=0.01)
    results = BatchExecutor(fsm=witnessed_fsm(recorder), parallelism=1).execute_batch(items(4))

    assert recorder.calls == 4
    assert all(result.success for result in results)


def test_a_stream_closes_its_source_when_the_timeout_fires() -> None:
    """Teardown still runs on the way out of a bounded stream.

    ``execute_stream`` closes the source and the sink in a ``finally``; a
    deadline that escaped past it would leak the source.
    """
    recorder = SlowRecorder(delay=0.2)
    source = ListSource([[{"id": i} for i in range(6)]])

    with pytest.raises(TimeoutError):
        StreamExecutor(fsm=witnessed_fsm(recorder), timeout=0.3).execute_stream(
            StreamPipeline(source=source)
        )

    assert source.closed, "the stream source was left open by a timed-out operation"


# --------------------------------------------------------------------------- #
# 4. Pre-existing: the stream executor ran the FSM only by coincidence of naming
# --------------------------------------------------------------------------- #


def test_the_stream_executor_runs_the_fsm_it_was_given() -> None:
    """A stream must execute the FSM, not pass its records through.

    ``_find_initial_state`` looked the network up by the **FSM's** name, so it
    found one only when the FSM happened to be named after its main network.
    Otherwise it returned ``None`` and ``_process_chunk`` took its "no FSM
    configured" branch: every record passed through untouched and was counted
    successful, with nothing raised and nothing logged.
    """
    recorder = LoopRecorder()
    stats = StreamExecutor(fsm=witnessed_fsm(recorder, name="not_the_network_name")).execute_stream(
        StreamPipeline(source=ListSource([[{"id": 1}, {"id": 2}]]))
    )

    assert recorder.calls == 2, (
        "the stream executor reported success without running the FSM — "
        "it resolved the main network by the FSM's name"
    )
    assert stats["total_processed"] == 2


def test_the_batch_and_stream_executors_agree_on_the_initial_state() -> None:
    """Both resolve the main network the way the FSM itself does.

    The lookup was written twice, identically and identically wrong. It was
    invisible in ``BatchExecutor`` because that path calls the engine whether
    or not the lookup found anything, and the engine resolves the start state
    for itself.
    """
    batch_recorder = LoopRecorder()
    stream_recorder = LoopRecorder()

    batch = BatchExecutor(fsm=witnessed_fsm(batch_recorder, name="alpha"), parallelism=1)
    stream = StreamExecutor(fsm=witnessed_fsm(stream_recorder, name="alpha"))

    assert batch._find_initial_state() == "start"
    assert stream._find_initial_state() == "start"


# --------------------------------------------------------------------------- #
# 5. Pre-existing: released resources never returned to the pool
# --------------------------------------------------------------------------- #


def test_a_released_resource_returns_to_the_pool() -> None:
    """``_release_resources`` must actually release.

    It compared ``allocation.status`` --- a ``ResourceStatus`` member --- to
    the string ``"allocated"``, which is never equal, so its whole body was
    unreachable: nothing went back to the pool, and the allocation stayed
    marked as held for the lifetime of the context. mypy reported both halves
    of this (``comparison-overlap`` on the test, ``unreachable`` on the body).
    """
    recorder = LoopRecorder()
    executor = BatchExecutor(fsm=witnessed_fsm(recorder), parallelism=1)

    context = ExecutionContext(resources={"db": 4})
    context.metadata["batch_info"] = {"batch_id": 0}
    executor._acquire_resources(context)
    assert context.allocate_resource("db", "conn-1")

    executor._release_resources(context)

    allocation = context.resources["db:conn-1"]
    assert allocation.status is ResourceStatus.AVAILABLE, (
        "the allocation is still marked held — _release_resources did not run"
    )
    assert executor._resource_pool.get("db") == ["conn-1"], (
        "the resource did not return to the pool"
    )
