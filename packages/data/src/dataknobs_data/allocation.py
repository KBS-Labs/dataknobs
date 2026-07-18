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

Both twins compose the shared bounded-retry engine
(:class:`dataknobs_common.retry.RetryExecutor`) with a zero-delay ``FIXED``
config scoped to :class:`~dataknobs_data.DuplicateRecordError`, so the retry
policy lives in one place rather than a hand-rolled loop.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from dataknobs_common.retry import BackoffStrategy, RetryConfig, RetryExecutor

from .exceptions import DuplicateRecordError

if TYPE_CHECKING:
    from .database import AsyncDatabase, SyncDatabase
    from .records import Record

#: Default create-on-conflict bound. The k-th concurrent allocator on one stem
#: needs up to k attempts to diverge (see :func:`allocate`), so this is also the
#: peak same-stem contention the seamless guarantee covers out of the box. Sized
#: for modest bursts; heavier contention should raise it or use ``RetryExecutor``.
DEFAULT_MAX_ATTEMPTS = 16


def _conflict_retry_config(max_attempts: int) -> RetryConfig:
    """Zero-delay bounded retry scoped to id collisions.

    ``FIXED`` with ``initial_delay=0.0`` preserves allocate's immediate-retry
    loop, and ``retry_on_exceptions`` confines retries to
    :class:`~dataknobs_data.DuplicateRecordError` so any other create error
    propagates on the first attempt. Building this config also validates
    ``max_attempts`` (``RetryConfig`` rejects ``< 1`` in ``__post_init__``),
    so a non-positive bound fails loud here — before any ``build`` runs.
    """
    return RetryConfig(
        max_attempts=max_attempts,
        initial_delay=0.0,
        backoff_strategy=BackoffStrategy.FIXED,
        retry_on_exceptions=[DuplicateRecordError],
    )


async def allocate(
    db: AsyncDatabase,
    *,
    build: Callable[[], Awaitable[Record]],
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
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

    Contention bound. Each attempt targets ``highest existing key + 1``, so the
    k-th concurrent allocator on one stem needs up to k attempts to diverge. A
    burst of **more than** ``max_attempts`` allocators contending on the same
    stem can therefore drive the tail allocators to exhaust the bound and fail
    closed *even though free keys remain* — the seamless guarantee holds only up
    to ``max_attempts``-way same-stem contention. Size ``max_attempts`` to the
    peak concurrent allocation you expect on a single stem. (Persistent
    exhaustion below that peak instead means a ``build`` that does not recompute
    a fresh key each attempt — a bug in the caller's closure.)

    The helper retries immediately with no delay; a consumer facing a thundering
    herd, or wanting backoff/jitter, can drive the same read-compute-create
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

    async def _attempt() -> str:
        return await db.create(await build())

    return await RetryExecutor(_conflict_retry_config(max_attempts)).execute(_attempt)


def allocate_sync(
    db: SyncDatabase,
    *,
    build: Callable[[], Record],
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> str:
    """Synchronous twin of :func:`allocate`.

    ``build`` is a plain (non-async) callable returning a fresh
    :class:`~dataknobs_data.Record` with its computed id set each time it is
    called. Semantics, bounds, exhaustion behavior, and the same-stem contention
    bound all match :func:`allocate`.

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

    def _attempt() -> str:
        return db.create(build())

    return RetryExecutor(_conflict_retry_config(max_attempts)).execute_sync(_attempt)
