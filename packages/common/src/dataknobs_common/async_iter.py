"""Drive a blocking sync iterator from async without blocking the loop.

An async surface that needs to consume a *lazy, blocking* synchronous
iterator faces a recurring problem: pulling each item runs the
iterator's next step — a file read, a database-cursor fetch, a paginated
SDK call — on the event loop, stalling it. The obvious offload,
``await asyncio.to_thread(list, iterator)``, drains the whole iterator on
a worker thread but **buffers every item in memory**, defeating the
streaming the lazy iterator existed to provide.

:func:`aiter_sync_in_thread` is the bridge for that case. It runs the
sync iterator on a worker thread and hands items to the async consumer
across a bounded queue, so:

* the iterator's setup *and* every step happen off the event loop;
* memory stays bounded — the producer blocks once ``max_buffer`` items
  are unconsumed (backpressure), so streaming is preserved rather than
  buffered whole;
* abandoned iteration (the async consumer ``break``s, raises, or is
  cancelled) signals the producer to stop, ``close()``s the source
  iterator (releasing its file handle / cursor / decompressor), and
  joins the worker thread — no leaked thread, no dangling handle;
* exceptions raised by the factory or during iteration propagate to the
  async consumer.

Use it whenever an ``async def`` must iterate a blocking generator it
cannot rewrite as async — a streaming file/format parser, a sync DB
cursor, a paginated third-party SDK iterator. When the source is *not*
lazy (a fully-materialized list, a single bounded read),
``asyncio.to_thread`` is simpler and sufficient; reach for this only when
look-ahead must stay bounded.
"""

from __future__ import annotations

import asyncio
import queue
import threading
from collections.abc import AsyncIterator, Callable, Iterator
from typing import TypeVar, cast

__all__ = ["aiter_sync_in_thread"]

T = TypeVar("T")

# Queue message kinds (producer thread -> async consumer).
_ITEM = 0
_DONE = 1
_ERR = 2

# Poll interval for the bounded ``put`` / ``get`` waits. Both sides re-check
# their stop/liveness condition this often so neither a full queue (producer)
# nor a stalled producer (consumer) can wedge teardown or cancellation past a
# bounded delay. Small enough to be imperceptible, large enough to avoid spin.
_POLL_SECONDS = 0.05

# Name applied to the producer thread so tests (and debuggers) can assert no
# pump thread is left alive after teardown.
_THREAD_NAME = "dk-aiter-sync-pump"


async def aiter_sync_in_thread(
    make_iter: Callable[[], Iterator[T]],
    *,
    max_buffer: int = 32,
) -> AsyncIterator[T]:
    """Drive a blocking sync iterator on a worker thread.

    ``make_iter`` is invoked **on the worker thread**, so any file
    ``open`` / gzip setup / connection it performs happens off the event
    loop. Items cross a bounded queue: the producer blocks once
    ``max_buffer`` items are unconsumed (backpressure — memory stays
    bounded, streaming preserved). On consumer teardown (``break`` /
    exception / cancellation) the producer is signalled to stop, the
    source iterator is ``close()``d (releasing its file handle), and the
    thread is joined — no leaked thread, no dangling handle. Exceptions
    raised by ``make_iter`` or during iteration propagate to the async
    consumer.

    The worker thread is a ``daemon`` so it can never block process exit;
    the in-flight blocking step of an abandoned iterator cannot be
    interrupted, so teardown waits for that one step to return (bounded
    and acceptable) before the thread winds down.

    Args:
        make_iter: Zero-arg factory returning the blocking iterator to
            drive. Called once, on the worker thread, so its setup is
            also off the event loop.
        max_buffer: Maximum number of unconsumed items held in the
            hand-off queue before the producer blocks. Higher trades
            memory for smoother throughput; the default suits
            chunk-sized items.

    Yields:
        Each item produced by the wrapped iterator, in order.
    """
    bridge: queue.Queue[tuple[int, object]] = queue.Queue(maxsize=max_buffer)
    stop = threading.Event()

    def _put(message: tuple[int, object]) -> None:
        """Put a message, bailing out promptly if the consumer stopped."""
        while not stop.is_set():
            try:
                bridge.put(message, timeout=_POLL_SECONDS)
                return
            except queue.Full:
                continue

    def _producer() -> None:
        iterator: Iterator[T] | None = None
        try:
            iterator = make_iter()
            for item in iterator:
                if stop.is_set():
                    return
                _put((_ITEM, item))
                if stop.is_set():
                    return
            _put((_DONE, None))
        except Exception as exc:  # forwarded to the consumer across the boundary
            _put((_ERR, exc))
        finally:
            close = getattr(iterator, "close", None)
            if callable(close):
                close()

    thread = threading.Thread(
        target=_producer, name=_THREAD_NAME, daemon=True
    )
    thread.start()
    loop = asyncio.get_running_loop()

    def _get() -> tuple[int, object] | None:
        try:
            return bridge.get(timeout=_POLL_SECONDS)
        except queue.Empty:
            return None

    try:
        while True:
            message = await loop.run_in_executor(None, _get)
            if message is None:
                # No item yet; loop so a cancellation can interrupt us at a
                # bounded interval rather than blocking forever on ``get``.
                continue
            kind, payload = message
            if kind == _ITEM:
                yield cast("T", payload)
            elif kind == _DONE:
                return
            else:  # _ERR
                raise cast("BaseException", payload)
    finally:
        stop.set()
        await loop.run_in_executor(None, thread.join)
