"""Detect blocking I/O on the event loop in async tests.

An ``async def`` method promises to keep the event loop free while it
awaits. A synchronous, blocking transport invoked from inside it (a sync
``boto3`` client, ``open()``, ``time.sleep``, a blocking socket read)
breaks that promise: the loop stalls for the duration of the call and
every other task on it is starved. The defect is *non-functional* — the
method still returns the right value — so ordinary outcome assertions
never catch it.

This module turns that invisible defect into a deterministic, reproducible
test failure. :func:`assert_no_blocking` activates a runtime detector
(`blockbuster <https://pypi.org/project/blockbuster/>`_) that patches the
common blocking syscalls to raise when they run on a live event loop, so a
test wrapping the awaited operation::

    async def test_put_does_not_block(backend):
        with assert_no_blocking():
            await backend.put_file("kb", "doc.md", b"...")

FAILS against a backend that blocks the loop and PASSES once the backend
uses an async transport or offloads via ``asyncio.to_thread``. This is the
reproduce-first tool for async-correctness fixes: write the test, watch it
fail, fix the transport, watch it pass.

``blockbuster`` is a **dev/test-only** dependency — it is never imported at
runtime by shipped code. The imports here are lazy so that importing
``dataknobs_common.testing`` (and pytest's plugin discovery of this module)
never requires it. Consumers guarding their own async backends add
``blockbuster`` to their dev dependencies and get this construct for free.

Detection only fires while an event loop is running, so
:func:`assert_no_blocking` is meaningful inside an ``async`` test (or any
frame with a running loop). In a synchronous frame there is no loop and the
block is a no-op.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

__all__ = [
    "assert_no_blocking",
    "blocking_error_type",
    "is_blockbuster_available",
    "no_blocking",
    "requires_blockbuster",
]


def is_blockbuster_available() -> bool:
    """Return whether the ``blockbuster`` detector can be imported.

    Underlies the :data:`requires_blockbuster` skip marker; call it
    directly only when a custom skip condition is needed. To gate a test on
    the detector, prefer the ready-made marker::

        from dataknobs_common.testing import requires_blockbuster

        @requires_blockbuster
        async def test_put_does_not_block(backend): ...
    """
    import importlib.util

    return importlib.util.find_spec("blockbuster") is not None


def blocking_error_type() -> type[Exception]:
    """Return ``blockbuster``'s ``BlockingError`` exception type (lazy).

    Exposed so tests can ``pytest.raises(blocking_error_type())`` to assert
    that a deliberately-blocking call is detected, without importing
    ``blockbuster`` directly (keeping the dependency lazy).
    """
    from blockbuster import BlockingError

    return BlockingError


@contextmanager
def assert_no_blocking() -> Iterator[None]:
    """Fail if any blocking syscall runs on the event loop in this block.

    Activates the ``blockbuster`` runtime detector for the duration of the
    ``with`` block. Any patched blocking call (``time.sleep``, ``open``,
    blocking ``socket``/``os`` reads and writes, the sync ``sqlite3`` driver,
    etc.) executed while an event loop is running raises ``BlockingError``,
    surfacing as a test failure.

    Scope the block tightly around the awaited operation under test — wrap
    only the ``await`` call, not synchronous test setup (building a temp
    directory, constructing fixtures) which may legitimately block::

        async def test_search_does_not_block(store):
            with assert_no_blocking():
                await store.search(query_vector, top_k=5)

    Raises:
        RuntimeError: if ``blockbuster`` is not installed. Gate such tests
            with :func:`is_blockbuster_available` (or the ``no_blocking``
            fixture, which skips instead) so they fail loudly rather than
            silently pass without detection.
    """
    if not is_blockbuster_available():
        raise RuntimeError(
            "assert_no_blocking() requires the 'blockbuster' package, which is "
            "a dev/test-only dependency. Add it to your dev dependencies, or "
            "gate the test with dataknobs_common.testing.is_blockbuster_available()."
        )

    from blockbuster import BlockBuster

    # Drive activate/deactivate directly rather than via blockbuster_ctx():
    # that context manager calls deactivate() AFTER its `yield` with no
    # try/finally, so a BlockingError propagating out of the block (the
    # common case here — the detector firing IS the test signal) skips
    # deactivation and leaks the patched syscalls into later tests. The
    # try/finally below guarantees the global patches are always undone.
    detector = BlockBuster()
    detector.activate()
    try:
        yield
    finally:
        detector.deactivate()


try:
    import pytest

    requires_blockbuster = pytest.mark.skipif(
        not is_blockbuster_available(),
        reason="blockbuster not installed",
    )

    @pytest.fixture
    def no_blocking() -> Iterator[None]:
        """Pytest fixture wrapping the whole test in :func:`assert_no_blocking`.

        Request it by name to assert that an entire async test runs without
        blocking the event loop. Skips automatically when ``blockbuster`` is
        not installed (so consumers without the dev dependency are not failed
        spuriously). For precise scoping around a single awaited call, prefer
        the :func:`assert_no_blocking` context manager directly.
        """
        if not is_blockbuster_available():
            pytest.skip("blockbuster not installed; cannot assert no blocking I/O")
        with assert_no_blocking():
            yield

except ImportError:  # pragma: no cover - pytest is always present in test envs
    no_blocking = None  # type: ignore[assignment]
    requires_blockbuster = None  # type: ignore[assignment]
