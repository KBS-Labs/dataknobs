"""Async-correctness tests for ``PyrateRateLimiter``'s blocking buckets.

``PyrateRateLimiter`` is one class serving four bucket backends. Three of
them (``sqlite`` / ``redis`` / ``postgres``) drive a *synchronous* bucket
transport (disk / socket I/O); the fourth (``memory``, the default) is pure
in-memory. Run on the event loop, the blocking backends stall it — both the
per-acquire bucket I/O and the lazy first-call bucket construction execute
inside the single ``Limiter.try_acquire`` call. The fix offloads that call
via :func:`asyncio.to_thread` **only** for the blocking backends; the memory
bucket stays on the loop (no thread-dispatch overhead on the hot path).

The **sqlite** reproduce tests are the centerpiece: they run service-free in
CI (a tmp file + ``blockbuster``) and turn the otherwise non-functional
blocking defect into a deterministic red→green test. The redis / postgres
reproduce tests cover the same defect over a real service and gate on it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dataknobs_common.ratelimit import RateLimit, RateLimiterConfig
from dataknobs_common.testing import (
    assert_no_blocking,
    requires_blockbuster,
    requires_package,
    requires_postgres,
    requires_redis,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from dataknobs_common.ratelimit.pyrate import PyrateRateLimiter

# asyncio_mode = auto (see pytest.ini) auto-detects async tests, so no
# blanket asyncio mark — it would spuriously warn on the sync gate tests.
pytestmark = requires_package("pyrate_limiter")


def _make_limiter(
    bucket_config: dict[str, object],
    *,
    limit: int = 100,
    interval: float = 60.0,
) -> PyrateRateLimiter:
    """Build a PyrateRateLimiter over the given bucket backend config."""
    from dataknobs_common.ratelimit.pyrate import PyrateRateLimiter

    config = RateLimiterConfig(
        default_rates=[RateLimit(limit=limit, interval=interval)],
    )
    return PyrateRateLimiter(config, bucket_config)


@pytest.fixture
async def track() -> Iterator[list[PyrateRateLimiter]]:
    """Register limiters for guaranteed close() on teardown.

    Append every limiter a test builds; the fixture closes them all,
    avoiding leaked sqlite handles / sockets across tests.
    """
    created: list[PyrateRateLimiter] = []
    yield created
    for limiter in created:
        await limiter.close()


@pytest.fixture
def sqlite_limiter(tmp_path, track) -> PyrateRateLimiter:
    """A real sqlite-bucket limiter over a tmp db file, closed on teardown."""
    limiter = _make_limiter(
        {"bucket": "sqlite", "sqlite": {"db_path": str(tmp_path / "rl.db")}}
    )
    track.append(limiter)
    return limiter


# ---------------------------------------------------------------------------
# Reproduce-first — sqlite (service-free, CI-deterministic; the centerpiece)
# ---------------------------------------------------------------------------


@requires_blockbuster
async def test_try_acquire_sqlite_does_not_block(sqlite_limiter) -> None:
    """First try_acquire over a sqlite bucket must not block the loop.

    FAILS pre-fix: the lazy ``SQLiteBucket.init_from_file`` (table create)
    plus the per-acquire ``put`` fire synchronous ``sqlite3`` on the loop.
    PASSES once the whole ``Limiter.try_acquire`` call is offloaded.
    """
    with assert_no_blocking():
        assert await sqlite_limiter.try_acquire("api") is True


@requires_blockbuster
async def test_try_acquire_sqlite_warm_bucket_does_not_block(
    sqlite_limiter,
) -> None:
    """A *second* try_acquire (bucket already built) also must not block.

    Proves the per-acquire bucket I/O — not just the lazy construction — is
    offloaded. The first call (outside the block) builds the bucket; the
    second (inside the block) exercises only the per-acquire ``put``.
    """
    await sqlite_limiter.try_acquire("api")  # build the bucket off-band
    with assert_no_blocking():
        assert await sqlite_limiter.try_acquire("api") is True


@requires_blockbuster
async def test_acquire_sqlite_does_not_block(sqlite_limiter) -> None:
    """acquire() inherits the fix — its poll funnels through try_acquire."""
    with assert_no_blocking():
        await sqlite_limiter.acquire("api")


# ---------------------------------------------------------------------------
# Reproduce-first — redis / postgres (gated on a real service)
# ---------------------------------------------------------------------------


@requires_blockbuster
@requires_redis
async def test_try_acquire_redis_does_not_block(track) -> None:
    """try_acquire over a real redis bucket must not block the loop.

    The sync ``redis.Redis`` client's socket I/O (and the lazy
    ``RedisBucket.init`` Lua-script load) runs inside ``Limiter.try_acquire``
    — offloaded for the redis backend.
    """
    pytest.importorskip("redis")
    limiter = _make_limiter(
        {"bucket": "redis", "redis": {"url": "redis://localhost:6379"}}
    )
    track.append(limiter)
    with assert_no_blocking():
        await limiter.try_acquire("api")


@requires_blockbuster
@requires_postgres
async def test_try_acquire_postgres_does_not_block(
    postgres_connection_params, track
) -> None:
    """try_acquire over a real postgres bucket must not block the loop.

    PostgresBucket requires a *sync* psycopg pool (``postgres.pool``). The
    sync pool's socket I/O runs inside ``Limiter.try_acquire`` — offloaded
    for the postgres backend. Needs ``psycopg[pool]`` (the sync driver — not
    a default dev dependency; common ships ``asyncpg`` for its own async
    postgres code) plus a running postgres, so it skips by default. The
    offload *gate* for the postgres backend is covered service-free by
    ``test_blocking_bucket_is_offloaded``; the sqlite reproduce test covers
    the full blocking-I/O path in CI.
    """
    psycopg_pool = pytest.importorskip("psycopg_pool")
    p = postgres_connection_params
    conninfo = (
        f"host={p['host']} port={p['port']} dbname={p['database']} "
        f"user={p['user']} password={p['password']}"
    )
    pool = psycopg_pool.ConnectionPool(conninfo, open=True)
    try:
        limiter = _make_limiter(
            {"bucket": "postgres", "postgres": {"pool": pool}}
        )
        track.append(limiter)
        with assert_no_blocking():
            await limiter.try_acquire("api")
    finally:
        pool.close()


# ---------------------------------------------------------------------------
# Negative control — memory stays on the loop and is NOT offloaded
# ---------------------------------------------------------------------------


def test_memory_bucket_not_offloaded() -> None:
    """The memory default must never be wrapped in to_thread (hot path)."""
    limiter = _make_limiter({"bucket": "memory"})
    assert limiter._offload is False


@pytest.mark.parametrize("bucket", ["sqlite", "redis", "postgres"])
def test_blocking_bucket_is_offloaded(bucket: str) -> None:
    """Every blocking bucket flips the offload gate on — service-free.

    The gate is decided from config at construction (no bucket I/O), so this
    pins the conditional-offload contract for all three blocking backends —
    including ``postgres``, whose full-I/O reproduce test requires a sync
    psycopg pool (not a default dev dependency) and otherwise skips.
    """
    limiter = _make_limiter({"bucket": bucket})
    assert limiter._offload is True


@requires_blockbuster
async def test_memory_bucket_does_not_block(track) -> None:
    """Memory bucket try_acquire never blocked — the fix must not perturb it.

    PASSES both before and after the fix (memory is loop-safe); guards
    against the offload accidentally being applied to the default backend.
    """
    limiter = _make_limiter({"bucket": "memory"})
    track.append(limiter)
    with assert_no_blocking():
        assert await limiter.try_acquire("api") is True


# ---------------------------------------------------------------------------
# Functional regression — the offload must preserve sqlite-backed semantics
# ---------------------------------------------------------------------------


async def test_sqlite_bucket_enforces_limit(tmp_path, track) -> None:
    """Offload preserves limit enforcement AND cross-instance persistence.

    N acquires under the limit succeed; the next is denied; and a fresh
    limiter over the same db file sees the exhausted count (the sqlite
    bucket is the shared store).
    """
    db_path = str(tmp_path / "enforce.db")
    bucket = {"bucket": "sqlite", "sqlite": {"db_path": db_path}}

    limiter = _make_limiter(bucket, limit=3, interval=600.0)
    track.append(limiter)
    for _ in range(3):
        assert await limiter.try_acquire("api") is True
    assert await limiter.try_acquire("api") is False

    # A fresh limiter over the same db file inherits the exhausted count.
    limiter2 = _make_limiter(bucket, limit=3, interval=600.0)
    track.append(limiter2)
    assert await limiter2.try_acquire("api") is False
