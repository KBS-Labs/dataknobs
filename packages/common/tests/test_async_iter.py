"""Tests for the ``aiter_sync_in_thread`` thread-pump primitive.

The primitive drives a blocking sync iterator on a worker thread and
pumps its items to an async consumer across a bounded queue. These tests
exercise the four behaviors that make it safe: in-order draining, clean
teardown on abandoned iteration (thread joined + source ``close()``d),
exception propagation across the thread boundary, and backpressure
(bounded look-ahead under a slow consumer). No mocks — a plain Python
generator is the real collaborator.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator

import pytest

from dataknobs_common import aiter_sync_in_thread
from dataknobs_common.async_iter import _THREAD_NAME


def _pump_threads_alive() -> list[str]:
    """Names of any live pump threads (should be empty after teardown)."""
    return [t.name for t in threading.enumerate() if t.name == _THREAD_NAME]


async def test_happy_path_drains_all_items_in_order() -> None:
    def make_iter() -> Iterator[int]:
        return iter(range(5))

    got = [item async for item in aiter_sync_in_thread(make_iter)]

    assert got == [0, 1, 2, 3, 4]
    # Wait briefly for the joined producer thread to clear from the table.
    await asyncio.sleep(0)
    assert _pump_threads_alive() == []


async def test_teardown_on_abandonment_closes_source_and_thread() -> None:
    closed = {"flag": False}

    def make_iter() -> Iterator[int]:
        def gen() -> Iterator[int]:
            try:
                yield from range(100)
            finally:
                # Proxy for releasing a real file handle / decompressor.
                closed["flag"] = True

        return gen()

    agen = aiter_sync_in_thread(make_iter, max_buffer=4)
    first = None
    async for item in agen:
        first = item
        break
    # ``async for`` + ``break`` does NOT synchronously finalize an async
    # generator, so close it explicitly for a deterministic teardown.
    await agen.aclose()

    assert first == 0
    # Source generator's ``finally`` ran -> handle released.
    assert closed["flag"] is True
    # Producer thread was joined -> none left alive.
    assert _pump_threads_alive() == []


async def test_error_during_iteration_propagates() -> None:
    def make_iter() -> Iterator[int]:
        def gen() -> Iterator[int]:
            yield 1
            raise ValueError("mid-iteration boom")

        return gen()

    got: list[int] = []
    with pytest.raises(ValueError, match="mid-iteration boom"):
        async for item in aiter_sync_in_thread(make_iter):
            got.append(item)

    assert got == [1]
    await asyncio.sleep(0)
    assert _pump_threads_alive() == []


async def test_error_during_setup_propagates() -> None:
    def make_iter() -> Iterator[int]:
        # ``make_iter`` itself raises (e.g. a malformed source the
        # generator factory rejects before producing anything).
        raise RuntimeError("setup boom")

    with pytest.raises(RuntimeError, match="setup boom"):
        async for _ in aiter_sync_in_thread(make_iter):
            pass

    await asyncio.sleep(0)
    assert _pump_threads_alive() == []


@pytest.mark.parametrize("bad_buffer", [0, -1])
async def test_zero_or_negative_max_buffer_rejected(bad_buffer: int) -> None:
    # ``queue.Queue(maxsize=0)`` is unbounded and would silently defeat the
    # backpressure the primitive exists to provide; guard it up front.
    with pytest.raises(ValueError, match="max_buffer must be >= 1"):
        async for _ in aiter_sync_in_thread(lambda: iter(range(3)), max_buffer=bad_buffer):
            pass


async def test_many_concurrent_streams_do_not_starve_each_other() -> None:
    # A waiting consumer parks on an ``asyncio.Event`` (no executor polling),
    # so far more concurrent streams than the default thread-pool size can run
    # without deadlock. A polling design would wedge once the pool saturated.
    stream_count = 64
    per_stream = 20

    def make_iter() -> Iterator[int]:
        def gen() -> Iterator[int]:
            yield from range(per_stream)

        return gen()

    async def drain() -> list[int]:
        return [item async for item in aiter_sync_in_thread(make_iter, max_buffer=2)]

    results = await asyncio.gather(*(drain() for _ in range(stream_count)))

    assert all(r == list(range(per_stream)) for r in results)
    await asyncio.sleep(0)
    assert _pump_threads_alive() == []


async def test_backpressure_bounds_producer_lookahead() -> None:
    total = 50
    max_buffer = 2
    produced: list[int] = []

    def make_iter() -> Iterator[int]:
        def gen() -> Iterator[int]:
            for i in range(total):
                produced.append(i)
                yield i

        return gen()

    consumed = 0
    async for _ in aiter_sync_in_thread(make_iter, max_buffer=max_buffer):
        # Slow consumer: without backpressure the producer would race to
        # ``total`` immediately; with it, look-ahead stays bounded by the
        # queue (max_buffer) + the one in-flight item the producer holds
        # while blocked on ``put`` (+ scheduling slack).
        await asyncio.sleep(0.01)
        consumed += 1
        assert len(produced) <= consumed + max_buffer + 3

    assert consumed == total
    assert produced == list(range(total))
