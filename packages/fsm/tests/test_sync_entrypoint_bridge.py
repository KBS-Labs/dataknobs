"""Reproduce-first tests for the synchronous FSM entry points on one engine.

The sync FSM APIs now run on the single ``AsyncExecutionEngine`` via an
async→sync bridge. These tests pin the consumer-visible behaviors of that
routing at a *synchronous* entry point:

* **No leaked bridge thread (finding #1).** The stateless one-shot sync
  surfaces — ``FSM.execute`` and the sync ``BatchExecutor`` / ``StreamExecutor``
  — scope a throwaway bridge to the operation and tear it down, leaving no
  process-lifetime daemon thread behind. (Explicit-lifecycle objects like
  ``SimpleFSM`` keep a shared bridge released by ``close()`` — tested elsewhere.)
* **Sync push arc enters the sub-network (finding #2).** The old standalone sync
  engine reconstructed plain arcs and flat-traversed push arcs; the unified
  engine pushes into the sub-network.
* **Async transform runs on a sync API (finding #2).** The old sync engine
  invoked an ``async def`` transform without awaiting it (discarding the
  coroutine); the unified engine awaits it.
* **process(timeout=) is bounded (finding #6).** The bridge timeout cancels the
  in-flight coroutine and returns, instead of blocking until it finishes anyway.

Real constructs only — real FSM builds, a real async transform, the real bridge.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from dataknobs_common import SyncLoopBridge
from dataknobs_common.testing import assert_no_leaked_bridge_threads
from dataknobs_fsm.api.simple import SimpleFSM
from dataknobs_fsm.config.builder import FSMBuilder
from dataknobs_fsm.config.schema import (
    ArcConfig,
    FSMConfig,
    FunctionReference,
    NetworkConfig,
    PushArcConfig,
    StateConfig,
)
from dataknobs_fsm.execution.batch import BatchExecutor
from dataknobs_fsm.execution.stream import StreamExecutor, StreamPipeline
from dataknobs_fsm.streaming.core import IStreamSource, StreamChunk


def _trivial_fsm() -> Any:
    """A minimal start→end FSM (no transforms, no resources)."""
    config = FSMConfig(
        name="trivial",
        main_network="main",
        networks=[
            NetworkConfig(
                name="main",
                states=[
                    StateConfig(name="start", is_start=True, arcs=[ArcConfig(target="end")]),
                    StateConfig(name="end", is_end=True),
                ],
            )
        ],
    )
    return FSMBuilder().build(config)


# --------------------------------------------------------------------------- #
# Finding #1 — one-shot sync surfaces leave no process-lifetime bridge thread
# --------------------------------------------------------------------------- #


def test_fsm_execute_leaves_no_bridge_thread() -> None:
    """``FSM.execute`` scopes a throwaway bridge and leaves no thread behind.

    A one-shot sync execute must scope its bridge to the call
    (``run_coro_sync``) rather than allocating the shared, process-lifetime
    one that repeated stepping uses.
    """
    fsm = _trivial_fsm()
    with assert_no_leaked_bridge_threads():
        fsm.execute({"id": 1})


def test_batch_executor_leaves_no_bridge_thread() -> None:
    """The sync ``BatchExecutor`` tears down its operation-scoped bridge.

    Its bridge belongs to the batch operation, not to the process.
    """
    fsm = _trivial_fsm()
    executor = BatchExecutor(fsm=fsm, parallelism=2)
    with assert_no_leaked_bridge_threads():
        executor.execute_batch([{"id": 1}, {"id": 2}, {"id": 3}])


class _ListSource(IStreamSource):
    """Minimal in-memory stream source over a list of record chunks."""

    def __init__(self, chunks: list[list[dict[str, Any]]]) -> None:
        self._chunks = chunks
        self._i = 0
        self.closed = False

    def read_chunk(self) -> StreamChunk | None:
        if self._i >= len(self._chunks):
            return None
        chunk = StreamChunk(
            data=self._chunks[self._i],
            chunk_id=self._i,
            is_last=(self._i == len(self._chunks) - 1),
        )
        self._i += 1
        return chunk

    def close(self) -> None:
        self.closed = True


def test_stream_executor_leaves_no_bridge_thread() -> None:
    """The sync ``StreamExecutor`` tears down its operation-scoped bridge.

    Its bridge belongs to the stream operation, not to the process.
    """
    fsm = _trivial_fsm()
    executor = StreamExecutor(fsm=fsm)
    pipeline = StreamPipeline(source=_ListSource([[{"id": 1}, {"id": 2}]]))
    with assert_no_leaked_bridge_threads():
        executor.execute_stream(pipeline)


# --------------------------------------------------------------------------- #
# Finding #2 — headline fixes are observable at a sync entry point
# --------------------------------------------------------------------------- #


def test_fsm_execute_enters_push_subnetwork() -> None:
    """A synchronous push arc enters its sub-network instead of flat-traversing.

    A transform on the sub-network's only arc (``s1`` → ``s2``) records its run;
    it fires only if the push actually entered the sub-network and traversed it.
    On the old sync engine (which reconstructed plain arcs) the push arc was
    flat-traversed and the sub-network was never entered, so the recorder stayed
    empty.
    """
    entered: list[str] = []

    def sub_mark(data: Any, context: Any) -> Any:
        entered.append("sub")
        return data

    config = FSMConfig(
        name="push_fsm",
        main_network="main",
        networks=[
            NetworkConfig(
                name="main",
                states=[
                    StateConfig(
                        name="start",
                        is_start=True,
                        arcs=[
                            PushArcConfig(
                                target="after",
                                target_network="sub",
                                return_state="after",
                            )
                        ],
                    ),
                    StateConfig(name="after", arcs=[ArcConfig(target="end")]),
                    StateConfig(name="end", is_end=True),
                ],
            ),
            NetworkConfig(
                name="sub",
                states=[
                    StateConfig(
                        name="s1",
                        is_start=True,
                        arcs=[
                            ArcConfig(
                                target="s2",
                                transform=FunctionReference(type="registered", name="sub_mark"),
                            )
                        ],
                    ),
                    StateConfig(name="s2", is_end=True),
                ],
            ),
        ],
    )
    builder = FSMBuilder()
    builder.register_function("sub_mark", sub_mark)
    fsm = builder.build(config)

    fsm.execute({"id": 1})

    assert entered == ["sub"], (
        "A synchronous push arc did not enter its sub-network — FSM.execute() "
        "flat-traversed instead of pushing into 'sub'"
    )


def test_fsm_execute_runs_async_transform() -> None:
    """An ``async def`` transform actually runs on the sync ``FSM.execute`` path.

    The old standalone sync engine invoked an async transform without awaiting
    it (the coroutine was created and discarded), so the recorder stayed empty.
    """
    ran: list[str] = []

    async def async_t(data: Any, context: Any) -> Any:
        ran.append("ran")
        if isinstance(data, dict):
            return {**data, "async_ran": True}
        return data

    config = FSMConfig(
        name="async_t_fsm",
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
                                transform=FunctionReference(type="registered", name="async_t"),
                            )
                        ],
                    ),
                    StateConfig(name="end", is_end=True),
                ],
            )
        ],
    )
    builder = FSMBuilder()
    builder.register_function("async_t", async_t)
    fsm = builder.build(config)

    fsm.execute({"id": 1})

    assert ran == ["ran"], (
        "An async def transform did not run on the sync FSM.execute() path — "
        "the unified engine must await async transforms"
    )


# --------------------------------------------------------------------------- #
# Finding #6 — process(timeout=) actually bounds the wait
# --------------------------------------------------------------------------- #


def test_simple_fsm_process_timeout_is_bounded() -> None:
    """``SimpleFSM.process(timeout=)`` bounds the wait via the bridge timeout.

    A slow async transform sleeps far longer than the timeout. The old
    ThreadPoolExecutor path blocked on ``shutdown(wait=True)`` until the
    coroutine finished anyway, so the timeout never bounded the wait; the bridge
    timeout cancels and returns promptly.
    """

    async def slow(data: Any, context: Any) -> Any:
        await asyncio.sleep(5)
        return data

    config = {
        "name": "slow_fsm",
        "main_network": "main",
        "networks": [
            {
                "name": "main",
                "states": [
                    {"name": "start", "is_start": True},
                    {"name": "end", "is_end": True},
                ],
                "arcs": [
                    {
                        "from": "start",
                        "to": "end",
                        "name": "go",
                        "transform": {"type": "registered", "name": "slow"},
                    }
                ],
            }
        ],
    }
    fsm = SimpleFSM(config, custom_functions={"slow": slow})
    try:
        start = time.monotonic()
        result = fsm.process({"id": 1}, timeout=0.2)
        elapsed = time.monotonic() - start

        assert elapsed < 2.0, (
            f"process(timeout=0.2) was not bounded — it took {elapsed:.2f}s "
            "(the slow transform sleeps 5s), so the timeout did not cancel it"
        )
        assert result["success"] is False
        assert "timeout" in (result.get("error") or "").lower(), (
            f"expected a timeout error, got: {result.get('error')!r}"
        )
    finally:
        fsm.close()


def test_simple_fsm_process_batch_timeout_is_bounded() -> None:
    """``SimpleFSM.process_batch(timeout=)`` bounds the wait via the bridge.

    The batch coroutine (slow async transform per item) is cancelled and
    `TimeoutError` is raised promptly, rather than the caller waiting for the
    whole batch to finish.
    """

    async def slow(data: Any, context: Any) -> Any:
        await asyncio.sleep(5)
        return data

    config = {
        "name": "slow_batch_fsm",
        "main_network": "main",
        "networks": [
            {
                "name": "main",
                "states": [
                    {"name": "start", "is_start": True},
                    {"name": "end", "is_end": True},
                ],
                "arcs": [
                    {
                        "from": "start",
                        "to": "end",
                        "name": "go",
                        "transform": {"type": "registered", "name": "slow"},
                    }
                ],
            }
        ],
    }
    fsm = SimpleFSM(config, custom_functions={"slow": slow})
    try:
        start = time.monotonic()
        with pytest.raises(TimeoutError):
            fsm.process_batch([{"id": 1}, {"id": 2}], timeout=0.3)
        elapsed = time.monotonic() - start
        assert elapsed < 3.0, f"process_batch(timeout=0.3) was not bounded — it took {elapsed:.2f}s"
    finally:
        fsm.close()


# --------------------------------------------------------------------------- #
# ``FSM.execute`` takes the same bridge= / timeout= pair as the executors
# --------------------------------------------------------------------------- #


def _loop_witness_fsm(loops: list[Any], *, delay: float = 0.0) -> Any:
    """A start→end FSM whose transform records the loop that ran it."""

    async def witness(data: Any, context: Any) -> Any:
        loops.append(asyncio.get_running_loop())
        if delay:
            await asyncio.sleep(delay)
        return data

    config = FSMConfig(
        name="loop_witness",
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
    builder.register_function("witness", witness)
    return builder.build(config)


def test_the_caller_can_name_the_loop_fsm_execute_runs_on() -> None:
    """``FSM.execute`` was the one-shot surface left without ``bridge=``.

    It drives the same engine the executors drive, so the same thing is true of
    it: an ``AsyncDatabaseResourceAdapter`` keeps its ``AsyncDatabase`` open
    across acquisitions and belongs to whichever loop opened it, and a
    ``SimpleFSM.process()`` on the same FSM has already opened it on the FSM's
    own bridge. Two ``execute`` calls ran on two throwaway loops, neither of
    them that one.

    ``98191919`` scoped this surface to a throwaway bridge so a one-shot
    execute leaves no process-lifetime thread. That is about thread *lifetime*
    and is unchanged: without ``bridge=`` the scoping is exactly what that
    commit left, which ``test_fsm_execute_leaves_no_bridge_thread`` still pins.
    """
    loops: list[Any] = []
    fsm = _loop_witness_fsm(loops)

    with SyncLoopBridge(thread_name="dk-test-owner") as bridge:
        fsm.execute({"id": 1}, bridge=bridge)
        fsm.execute({"id": 2}, bridge=bridge)

    assert len(loops) == 2, f"the transform ran {len(loops)} times, expected 2"
    assert len({id(loop) for loop in loops}) == 1, (
        "two FSM.execute calls given one bridge still ran on different loops"
    )


def test_fsm_execute_without_a_bridge_still_leaves_no_thread() -> None:
    """The opt-in does not change the default, which is the leak guard's claim."""
    loops: list[Any] = []
    fsm = _loop_witness_fsm(loops)

    with assert_no_leaked_bridge_threads():
        fsm.execute({"id": 1})
        fsm.execute({"id": 2})

    assert len({id(loop) for loop in loops}) == 2, (
        "the default is a throwaway loop per call; this test would not detect a "
        "regression to a shared one"
    )


def test_fsm_execute_timeout_is_bounded() -> None:
    """``timeout=`` bounds a call that blocks a caller inside a ``def``.

    ``98191919`` gave this bound to ``SimpleFSM.process`` and the two module
    helpers. ``FSM.execute`` blocks its caller exactly as they do and was the
    one one-shot surface it did not reach.

    It *reports* the expiry rather than raising it, which is what this surface
    does with every other failure: ``execute`` returns a result envelope and
    ``SimpleFSM.process(timeout=)`` --- the closest sibling --- converts a
    ``TimeoutError`` into an error result in so many words. The executors raise
    because they return lists and statistics, with no envelope to put it in.
    """
    loops: list[Any] = []
    fsm = _loop_witness_fsm(loops, delay=5.0)

    started = time.monotonic()
    result = fsm.execute({"id": 1}, timeout=0.2)
    elapsed = time.monotonic() - started

    assert elapsed < 2.0, f"the call ran {elapsed:.2f}s under a 0.2s bound"
    assert result["status"] == "error"
    assert "timeout" in result["error"].lower(), (
        f"the result does not say the operation timed out: {result['error']!r}"
    )
