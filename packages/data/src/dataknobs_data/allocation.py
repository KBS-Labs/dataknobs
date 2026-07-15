"""Race-free allocation of a new record under a caller-computed monotonic key.

A consumer allocating a new immutable record under a monotonic key — a version
(``<stem>-v{N}``), a sequence number, or any derived key — reads current state,
computes the next key, and ``create()``s a record under it. Because ``create()``
is an atomic insert that raises :class:`~dataknobs_data.DuplicateRecordError` on a
colliding id, two concurrent allocators that compute the same key no longer both
win with a silent double-write: the loser fails closed. Failing closed is safe,
but not *seamless* — a legitimate second allocator should land the next key
rather than error.

:func:`allocate` (and its synchronous twin :func:`allocate_sync`) close that
window with a bounded create-on-conflict loop: on a colliding id they re-run the
caller's ``build`` callable — a fresh read, next-key computation, and record
construction — and retry the insert, so each concurrent allocator lands a
distinct next key. The helper is key-agnostic: it never mints or mutates ids, so
it works for any monotonic scheme the caller expresses through ``build``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from .exceptions import DuplicateRecordError

if TYPE_CHECKING:
    from .database import AsyncDatabase, SyncDatabase
    from .records import Record


async def allocate(
    db: AsyncDatabase,
    *,
    build: Callable[[], Awaitable[Record]],
    max_attempts: int = 8,
) -> str:
    """Create a record under a caller-computed monotonic key, retrying on a
    colliding id so concurrent allocators each land a distinct next key.

    ``build`` performs a fresh read, computes the next key, and returns a
    :class:`~dataknobs_data.Record` with that key set as its id. It is re-run on
    every attempt, so each retry sees the winning writer's record and computes a
    fresh key. On a colliding id ``create`` raises
    :class:`~dataknobs_data.DuplicateRecordError`; this retries up to
    ``max_attempts`` times, then re-raises the last collision — bounded, never an
    infinite loop. A single uncontended allocation makes exactly one attempt,
    identical to a direct ``create``.

    Persistent exhaustion means either genuine contention beyond ``max_attempts``
    or a ``build`` that does not recompute a fresh key each attempt (a bug in the
    caller's closure). The helper retries immediately with no delay, so N
    concurrent allocators diverge in O(N) attempts; a consumer wanting
    backoff/jitter under a thundering herd can drive the same read-compute-create
    callable through ``dataknobs_common.retry.RetryExecutor`` instead.

    Args:
        db: The async database to allocate into.
        build: A zero-arg callable returning a fresh ``Record`` (id already set)
            each time it is awaited.
        max_attempts: Maximum create attempts before giving up. Must be >= 1.

    Returns:
        The id of the created record.

    Raises:
        ValueError: If ``max_attempts`` is less than 1.
        DuplicateRecordError: If every attempt collides (fail-closed after the
            bound).
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    last: DuplicateRecordError | None = None
    for _ in range(max_attempts):
        record = await build()
        try:
            return await db.create(record)
        except DuplicateRecordError as exc:
            last = exc
    assert last is not None  # loop ran at least once; last is always set here
    raise last


def allocate_sync(
    db: SyncDatabase,
    *,
    build: Callable[[], Record],
    max_attempts: int = 8,
) -> str:
    """Synchronous twin of :func:`allocate`.

    ``build`` is a plain (non-async) callable returning a fresh
    :class:`~dataknobs_data.Record` with its computed id set each time it is
    called. Semantics, bounds, and exhaustion behavior match :func:`allocate`.

    Args:
        db: The sync database to allocate into.
        build: A zero-arg callable returning a fresh ``Record`` (id already set).
        max_attempts: Maximum create attempts before giving up. Must be >= 1.

    Returns:
        The id of the created record.

    Raises:
        ValueError: If ``max_attempts`` is less than 1.
        DuplicateRecordError: If every attempt collides (fail-closed after the
            bound).
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    last: DuplicateRecordError | None = None
    for _ in range(max_attempts):
        record = build()
        try:
            return db.create(record)
        except DuplicateRecordError as exc:
            last = exc
    assert last is not None  # loop ran at least once; last is always set here
    raise last
