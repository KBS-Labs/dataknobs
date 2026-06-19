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

# Poll interval for the producer's bounded ``put``. When the queue is full the
# producer re-checks ``stop`` this often, so an abandoned consumer can never
# wedge the producer past a bounded delay. Small enough to be imperceptible,
# large enough to avoid spin. The consumer side does not poll: it parks on an
# ``asyncio.Event`` woken by the producer via ``call_soon_threadsafe``, so a
# waiting stream occupies no executor thread.
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
    and acceptable) before the thread winds down. The teardown join is
    shielded from cancellation, so a second cancellation arriving
    mid-teardown cannot abandon it; the daemon flag is the final backstop
    if even the shielded join is interrupted.

    The waiting consumer parks on an :class:`asyncio.Event` woken by the
    producer through :meth:`~asyncio.loop.call_soon_threadsafe` — it does
    **not** poll an executor thread, so running many concurrent streams
    does not consume the default thread-pool (only the one-time teardown
    join briefly does).

    Args:
        make_iter: Zero-arg factory returning the blocking iterator to
            drive. Called once, on the worker thread, so its setup is
            also off the event loop.
        max_buffer: Maximum number of unconsumed items held in the
            hand-off queue before the producer blocks. Higher trades
            memory for smoother throughput; the default suits
            chunk-sized items. Must be ``>= 1`` — ``0`` would make the
            queue unbounded and silently defeat backpressure.

    Yields:
        Each item produced by the wrapped iterator, in order.

    Raises:
        ValueError: If ``max_buffer`` is less than 1.
    """
    if max_buffer < 1:
        raise ValueError(f"max_buffer must be >= 1, got {max_buffer}")

    bridge: queue.Queue[tuple[int, object]] = queue.Queue(maxsize=max_buffer)
    stop = threading.Event()
    loop = asyncio.get_running_loop()
    # Woken (on the loop thread) whenever the producer hands off a message,
    # so the consumer can park without polling. ``asyncio.Event`` is not
    # thread-safe, so the producer only ever schedules ``_set_ready`` via
    # ``call_soon_threadsafe`` — it is never touched from the worker thread.
    item_ready = asyncio.Event()

    def _set_ready() -> None:
        item_ready.set()

    def _wake() -> None:
        """Schedule a consumer wake-up from the worker thread."""
        try:
            loop.call_soon_threadsafe(_set_ready)
        except RuntimeError:
            # Loop already closed during teardown; the consumer is gone.
            pass

    def _put(message: tuple[int, object]) -> None:
        """Put a message, bailing out promptly if the consumer stopped."""
        while not stop.is_set():
            try:
                bridge.put(message, timeout=_POLL_SECONDS)
            except queue.Full:
                continue
            _wake()
            return

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

    try:
        while True:
            try:
                kind, payload = bridge.get_nowait()
            except queue.Empty:
                # Clear, then re-check before parking: if the producer put an
                # item between the first miss and the clear, the second
                # ``get_nowait`` catches it; otherwise we park and the next
                # ``put`` schedules a wake-up *after* the clear, so it lands.
                # (Standard clear-then-recheck against a lost wake-up.)
                item_ready.clear()
                try:
                    kind, payload = bridge.get_nowait()
                except queue.Empty:
                    await item_ready.wait()
                    continue
            if kind == _ITEM:
                yield cast("T", payload)
            elif kind == _DONE:
                return
            else:  # _ERR
                raise cast("BaseException", payload)
    finally:
        stop.set()
        # Join off-loop so a long in-flight blocking step cannot stall the
        # loop. ``shield`` keeps the join future running to completion even if
        # this frame is cancelled again mid-teardown — note the frame may then
        # re-raise ``CancelledError`` and return before the join finishes, so
        # the daemon flag (not the shield) is the final backstop against a
        # leaked thread.
        await asyncio.shield(loop.run_in_executor(None, thread.join))
