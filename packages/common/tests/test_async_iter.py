"""Tests for the ``async_iter`` primitives.

The primitive drives a blocking sync iterator on a worker thread and
pumps its items to an async consumer across a bounded queue. These tests
exercise the four behaviors that make it safe: in-order draining, clean
teardown on abandoned iteration (thread joined + source ``close()``d),
exception propagation across the thread boundary, and backpressure
(bounded look-ahead under a slow consumer). No mocks — a plain Python
generator is the real collaborator.

``aclosing_iter`` closes an async iterator when its block ends, where the
iterator has a close to run. Its tests are at the foot of this module.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Iterator

import pytest

from dataknobs_common import aclosing_iter, aiter_sync_in_thread
from dataknobs_common.testing import (
    DK_AITER_PUMP_THREAD,
)


async def test_happy_path_drains_all_items_in_order(
    new_dk_daemon_threads: Callable[..., list[str]],
) -> None:
    def make_iter() -> Iterator[int]:
        return iter(range(5))

    got = [item async for item in aiter_sync_in_thread(make_iter)]

    assert got == [0, 1, 2, 3, 4]
    # Wait briefly for the joined producer thread to clear from the table.
    await asyncio.sleep(0)
    assert new_dk_daemon_threads(DK_AITER_PUMP_THREAD) == []


async def test_teardown_on_abandonment_closes_source_and_thread(
    new_dk_daemon_threads: Callable[..., list[str]],
) -> None:
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
    assert new_dk_daemon_threads(DK_AITER_PUMP_THREAD) == []


async def test_error_during_iteration_propagates(
    new_dk_daemon_threads: Callable[..., list[str]],
) -> None:
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
    assert new_dk_daemon_threads(DK_AITER_PUMP_THREAD) == []


async def test_error_during_setup_propagates(
    new_dk_daemon_threads: Callable[..., list[str]],
) -> None:
    def make_iter() -> Iterator[int]:
        # ``make_iter`` itself raises (e.g. a malformed source the
        # generator factory rejects before producing anything).
        raise RuntimeError("setup boom")

    with pytest.raises(RuntimeError, match="setup boom"):
        async for _ in aiter_sync_in_thread(make_iter):
            pass

    await asyncio.sleep(0)
    assert new_dk_daemon_threads(DK_AITER_PUMP_THREAD) == []


@pytest.mark.parametrize("bad_buffer", [0, -1])
async def test_zero_or_negative_max_buffer_rejected(bad_buffer: int) -> None:
    # ``queue.Queue(maxsize=0)`` is unbounded and would silently defeat the
    # backpressure the primitive exists to provide; guard it up front.
    with pytest.raises(ValueError, match="max_buffer must be >= 1"):
        async for _ in aiter_sync_in_thread(lambda: iter(range(3)), max_buffer=bad_buffer):
            pass


async def test_many_concurrent_streams_do_not_starve_each_other(
    new_dk_daemon_threads: Callable[..., list[str]],
) -> None:
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
    assert new_dk_daemon_threads(DK_AITER_PUMP_THREAD) == []


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


# --------------------------------------------------------------------------
# aclosing_iter
# --------------------------------------------------------------------------


async def test_aclosing_iter_closes_a_generator_the_block_walked_away_from() -> None:
    """The case the helper exists for, and the ordering is the assertion.

    An async generator a consumer abandons is left suspended at its
    ``yield``, and its ``finally`` runs when the interpreter finalizes it ---
    a later turn of the loop, unordered against whatever the consumer does
    next. Two shipped generators hold something across that gap:
    ``AsyncPostgresDatabase.stream_read`` yields from inside an acquired
    pool connection and an open transaction, and ``EntitySourceIndexSource``
    reports an all-empty stream from a ``finally``.

    So the release is asserted **at the close**. "Released eventually" is
    what a bare ``async for`` already does.
    """
    held = 0

    async def rows() -> AsyncIterator[int]:
        nonlocal held
        held += 1
        try:
            for value in range(10):
                yield value
        finally:
            held -= 1

    async with aclosing_iter(rows()) as values:
        async for _value in values:
            break
        assert held == 1, "still open while the block is running"

    assert held == 0


async def test_aclosing_iter_accepts_an_iterator_that_cannot_be_closed() -> None:
    """Which is why this is not :func:`contextlib.aclosing`.

    The protocols here declare ``AsyncIterator`` deliberately --- calling an
    ``async def`` generator function returns its iterator without awaiting,
    so the plain iterator is the signature a structural implementation
    actually has, and such an implementation need only carry ``__aiter__``
    and ``__anext__``. ``contextlib.aclosing`` refuses that type at the type
    checker and raises ``AttributeError`` at the **exit** if it gets one
    anyway --- replacing whatever the block was already failing with, which
    is the worst moment available.

    Nothing is lost by admitting it: an iterator with no ``aclose`` has no
    cleanup to run at a close, which is why it has no ``aclose``.
    """

    class _Counts:
        """An async iterator and nothing more --- no ``aclose`` anywhere."""

        def __init__(self) -> None:
            self.remaining = 3

        def __aiter__(self) -> _Counts:
            return self

        async def __anext__(self) -> int:
            if not self.remaining:
                raise StopAsyncIteration
            self.remaining -= 1
            return self.remaining

    assert not hasattr(_Counts(), "aclose")

    seen = []
    async with aclosing_iter(_Counts()) as values:
        async for value in values:
            seen.append(value)

    assert seen == [2, 1, 0]

    # And on the abandoning path, which is the one that reaches the close.
    async with aclosing_iter(_Counts()) as values:
        async for _value in values:
            break


async def test_aclosing_iter_closes_even_when_the_block_raises() -> None:
    """The failing consumer is the one that most needs the close to happen.

    A block that raises is exactly where a caller is about to decide what to
    do next, and deciding it while the source still holds a connection is
    the condition this exists to end.
    """
    closed = False

    async def rows() -> AsyncIterator[int]:
        nonlocal closed
        try:
            yield 1
            yield 2
        finally:
            closed = True

    with pytest.raises(RuntimeError, match="the consumer gave up"):
        async with aclosing_iter(rows()) as values:
            async for _value in values:
                raise RuntimeError("the consumer gave up")

    assert closed, "the close ran before the error left the block"
