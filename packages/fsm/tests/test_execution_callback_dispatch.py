"""The stream and batch executors judge their consumer callbacks correctly.

Three dispatches, two classes, one defect. ``AsyncStreamExecutor`` and
``AsyncBatchExecutor`` each branched on ``asyncio.iscoroutinefunction``, which
answers for *functions* and reports a callable **object** whose ``__call__``
is an ``async def`` as synchronous --- the shape anything holding state takes,
and the shape a consumer implementing one of this repository's own
``async def __call__`` protocols writes by default.

Misread that way, the object takes the synchronous arm and is handed to
``run_in_executor``, which calls it on a worker thread. Calling an async
callable only *constructs* a coroutine; nothing runs it, and the coroutine is
returned into the executor's future and dropped. Nothing raises.

**The sink is the expensive one.** ``progress.records_emitted`` is incremented
*before* the sink is dispatched, so a misread sink leaves the caller reporting
records it was never handed --- data loss with an accounting trail that says
otherwise. The two progress callbacks cost a silently missing notification.

The two ``_fire_progress_callback`` methods were byte-identical across the two
classes, which is why one defect appeared in two places. Both now delegate to
``run_callback_off_loop``, so the judgement is made once in
``dataknobs-common`` rather than spelled out per class.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from dataknobs_common.testing import assert_no_blocking, requires_blockbuster
from dataknobs_fsm.config.builder import FSMBuilder
from dataknobs_fsm.config.loader import ConfigLoader
from dataknobs_fsm.execution.async_batch import AsyncBatchExecutor
from dataknobs_fsm.execution.async_stream import AsyncStreamExecutor


def _build_fsm() -> Any:
    """A two-state FSM that every record traverses to a terminal."""
    config = {
        "name": "callback_dispatch_fsm",
        "main_network": "main",
        "networks": [
            {
                "name": "main",
                "states": [
                    {"name": "start", "is_start": True},
                    {"name": "end", "is_end": True},
                ],
                "arcs": [{"from": "start", "to": "end", "name": "finish"}],
            }
        ],
    }
    loader = ConfigLoader()
    builder = FSMBuilder()
    return builder.build(loader.load_from_dict(config))


@pytest.fixture
def fsm() -> Any:
    return _build_fsm()


class StatefulAsyncCallable:
    """An ``async def __call__`` on an object that remembers what it saw.

    The state is the point. A plain ``async def`` closure records the same
    thing and *is* classified correctly by ``iscoroutinefunction``; it is the
    object wrapper these sites misread.
    """

    def __init__(self) -> None:
        self.seen: list[Any] = []

    async def __call__(self, payload: Any) -> None:
        self.seen.append(payload)


class StatefulSyncCallable:
    """The synchronous twin, recording which thread ran it."""

    def __init__(self) -> None:
        self.seen: list[Any] = []
        self.threads: list[str] = []

    def __call__(self, payload: Any) -> None:
        self.seen.append(payload)
        self.threads.append(threading.current_thread().name)


class BlockingSyncCallable:
    """A synchronous callback that really blocks, to prove the offload."""

    def __init__(self, path: Any) -> None:
        self.path = path
        self.calls = 0

    def __call__(self, payload: Any) -> None:
        self.calls += 1
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(f"{self.calls}\n")


# --------------------------------------------------------------------- #
# The sink --- the site where a misread loses data
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_an_async_callable_object_sink_receives_the_records(fsm: Any) -> None:
    """The defect: counted as emitted, never handed over.

    Before the fix ``iscoroutinefunction`` reported the object synchronous,
    ``run_in_executor`` constructed its coroutine on a worker thread and
    dropped it, and ``sink.seen`` stayed empty --- while
    ``records_emitted`` had already counted every record.
    """
    sink = StatefulAsyncCallable()
    executor = AsyncStreamExecutor(fsm=fsm)

    result = await executor.execute_stream(
        source=[{"id": 1}, {"id": 2}, {"id": 3}], sink=sink, chunk_size=2
    )

    delivered = [record for chunk in sink.seen for record in chunk]
    assert delivered, "the sink was never handed the records it was credited with"
    assert len(delivered) == result.emitted, (
        "records_emitted counts what the sink received; a misread sink makes the "
        "caller report records that were dropped"
    )


@pytest.mark.asyncio
async def test_a_synchronous_sink_still_receives_the_records(fsm: Any) -> None:
    """The arm that already worked keeps working."""
    sink = StatefulSyncCallable()
    executor = AsyncStreamExecutor(fsm=fsm)

    result = await executor.execute_stream(source=[{"id": 1}, {"id": 2}], sink=sink, chunk_size=1)

    delivered = [record for chunk in sink.seen for record in chunk]
    assert len(delivered) == result.emitted


@pytest.mark.asyncio
async def test_a_synchronous_sink_runs_off_the_loop_thread(fsm: Any) -> None:
    """A sink writes somewhere, so it must not run on the event loop.

    A thread-identity proof rather than a blocking-detector one, so the claim
    holds whether or not ``blockbuster`` is installed.
    """
    sink = StatefulSyncCallable()
    executor = AsyncStreamExecutor(fsm=fsm)
    loop_thread = threading.current_thread().name

    await executor.execute_stream(source=[{"id": 1}], sink=sink, chunk_size=1)

    assert sink.threads, "the sink never ran"
    assert all(name != loop_thread for name in sink.threads)


@requires_blockbuster
@pytest.mark.asyncio
async def test_a_blocking_sink_does_not_stall_the_loop(fsm: Any, tmp_path: Any) -> None:
    """The non-functional half: the value is right either way."""
    sink = BlockingSyncCallable(tmp_path / "sink.log")
    executor = AsyncStreamExecutor(fsm=fsm)

    with assert_no_blocking():
        await executor.execute_stream(source=[{"id": 1}, {"id": 2}], sink=sink, chunk_size=1)

    assert sink.calls > 0


# --------------------------------------------------------------------- #
# The progress callbacks --- one defect, two classes
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_stream_progress_reaches_an_async_callable_object(fsm: Any) -> None:
    progress_callback = StatefulAsyncCallable()
    executor = AsyncStreamExecutor(fsm=fsm, progress_callback=progress_callback)

    await executor.execute_stream(source=[{"id": 1}, {"id": 2}], chunk_size=1)

    assert progress_callback.seen, "progress reporting stopped silently"


@pytest.mark.asyncio
async def test_batch_progress_reaches_an_async_callable_object(fsm: Any) -> None:
    """The sibling class, because the two bodies were identical."""
    progress_callback = StatefulAsyncCallable()
    executor = AsyncBatchExecutor(fsm=fsm, progress_callback=progress_callback, batch_size=2)

    await executor.execute_batch([{"id": 1}, {"id": 2}, {"id": 3}])

    assert progress_callback.seen, "progress reporting stopped silently"


@pytest.mark.parametrize("executor_name", ["stream", "batch"])
@pytest.mark.asyncio
async def test_a_synchronous_progress_callback_runs_off_the_loop_thread(
    fsm: Any, executor_name: str
) -> None:
    """Both classes offloaded before this change, and still do.

    Worth pinning rather than assuming: the fix replaced an explicit
    ``run_in_executor`` with ``run_callback_off_loop``, and a fix that
    quietly moved a consumer's callback back onto the loop would be a
    regression no outcome assertion could see.
    """
    progress_callback = StatefulSyncCallable()
    loop_thread = threading.current_thread().name

    if executor_name == "stream":
        executor: Any = AsyncStreamExecutor(fsm=fsm, progress_callback=progress_callback)
        await executor.execute_stream(source=[{"id": 1}, {"id": 2}], chunk_size=1)
    else:
        executor = AsyncBatchExecutor(fsm=fsm, progress_callback=progress_callback, batch_size=1)
        await executor.execute_batch([{"id": 1}, {"id": 2}])

    assert progress_callback.threads, "the progress callback never ran"
    assert all(name != loop_thread for name in progress_callback.threads)
