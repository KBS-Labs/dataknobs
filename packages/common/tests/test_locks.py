"""Tests for the distributed lock abstraction.

Real constructs only — no mocked/faked lock. A ``FakeLock`` appending
to a list has the same blindness as ``MagicMock``; the point of a lock
is its concurrency behaviour, so the real :class:`InProcessLock` is
exercised directly (it is also the documented testing construct).

``PostgresAdvisoryLock`` is exercised against a **real** Postgres with
two independent instances (simulating two replicas) under the shared
``@requires_real_postgres`` gate — never a fake; the
whole point of the backend is cross-process behaviour a list-appending
fake cannot show. The ``_key_to_bigint`` keyspace mapping is a pure
function and is tested without a database.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import pytest

from dataknobs_common import create_lock as create_lock_top_level
from dataknobs_common.exceptions import OperationError
from dataknobs_common.locks import (
    DistributedLock,
    InProcessLock,
    PostgresAdvisoryLock,
    create_lock,
    lock_backends,
)
from dataknobs_common.testing import requires_real_postgres


class TestInProcessLock:
    """Behavioural tests for the default single-process lock."""

    async def test_satisfies_protocol(self) -> None:
        """:class:`InProcessLock` is a runtime :class:`DistributedLock`."""
        lock = InProcessLock()
        assert isinstance(lock, DistributedLock)
        await lock.close()  # no-op, must not raise

    async def test_mutual_exclusion_same_key(self) -> None:
        """Two tasks holding the same key never overlap."""
        lock = InProcessLock()
        events: list[tuple[str, int]] = []

        async def worker(n: int) -> None:
            async with lock.hold("k") as got:
                assert got is True
                events.append(("enter", n))
                await asyncio.sleep(0.02)
                events.append(("exit", n))

        await asyncio.gather(worker(1), worker(2))

        # Strict non-overlap: each enter is immediately followed by its
        # own exit before the other task enters.
        assert events in (
            [("enter", 1), ("exit", 1), ("enter", 2), ("exit", 2)],
            [("enter", 2), ("exit", 2), ("enter", 1), ("exit", 1)],
        ), events
        await lock.close()

    async def test_different_keys_run_concurrently(self) -> None:
        """Distinct keys are independent — no false serialization."""
        lock = InProcessLock()
        in_flight = 0
        peak = 0

        async def worker(key: str) -> None:
            nonlocal in_flight, peak
            async with lock.hold(key):
                in_flight += 1
                peak = max(peak, in_flight)
                await asyncio.sleep(0.05)
                in_flight -= 1

        await asyncio.gather(worker("a"), worker("b"), worker("c"))
        assert peak == 3, f"expected full concurrency, peak={peak}"
        await lock.close()

    async def test_timeout_returns_false_when_contended(self) -> None:
        """A finite timeout on a held key returns ``False`` (no raise)."""
        lock = InProcessLock()
        assert await lock.acquire("busy") is True
        try:
            assert await lock.acquire("busy", timeout=0.01) is False
        finally:
            await lock.release("busy")
        # Uncontended acquire with a timeout returns True.
        assert await lock.acquire("busy", timeout=0.01) is True
        await lock.release("busy")
        await lock.close()

    async def test_timeout_none_waits_then_acquires(self) -> None:
        """``timeout=None`` blocks until the holder releases."""
        lock = InProcessLock()
        await lock.acquire("k")
        waiter = asyncio.create_task(lock.acquire("k", timeout=None))
        await asyncio.sleep(0.02)
        assert not waiter.done(), "must still be waiting while held"
        await lock.release("k")
        assert await asyncio.wait_for(waiter, timeout=1.0) is True
        await lock.release("k")
        await lock.close()

    async def test_release_unheld_key_is_noop(self) -> None:
        """Releasing a key that was never acquired must not raise."""
        lock = InProcessLock()
        await lock.release("never-acquired")  # no exception
        await lock.close()

    async def test_hold_releases_on_exception(self) -> None:
        """``hold()`` releases even when the body raises."""
        lock = InProcessLock()

        with pytest.raises(RuntimeError, match="boom"):
            async with lock.hold("k"):
                raise RuntimeError("boom")

        # Lock is free again — re-acquire without blocking.
        assert await lock.acquire("k", timeout=0.01) is True
        await lock.release("k")
        await lock.close()

    async def test_key_map_evicted_no_leak(self) -> None:
        """The key→lock map is empty after acquire/release cycles.

        Leak regression: the orchestrator's previous inline
        ``_domain_locks`` dict never evicted, so distinct domain ids
        grew it unbounded. Reference-counted eviction closes that.
        """
        lock = InProcessLock()
        for i in range(25):
            async with lock.hold(f"domain-{i}"):
                pass
        assert lock._locks == {}, lock._locks
        assert lock._refs == {}, lock._refs
        await lock.close()

    async def test_eviction_safe_under_contention(self) -> None:
        """Eviction must not lose the lock while a waiter is queued."""
        lock = InProcessLock()
        order: list[int] = []

        async def worker(n: int) -> None:
            async with lock.hold("shared"):
                order.append(n)
                await asyncio.sleep(0.01)

        await asyncio.gather(*(worker(i) for i in range(10)))
        # All ten ran (none lost to a mis-evicted lock) and serialized.
        assert sorted(order) == list(range(10))
        assert lock._locks == {} and lock._refs == {}
        await lock.close()

    async def test_timed_out_waiter_then_release_no_leak(self) -> None:
        """A timed-out waiter + a subsequent release fully evicts the key.

        Exercises the path where a waiter's ``acquire()`` finally (ref
        decrement) interleaves with ``release()``'s eviction check. The
        key map must end empty and a fresh acquirer must still get
        correct mutual exclusion (no orphaned lock object).
        """
        lock = InProcessLock()
        await lock.acquire("k")  # holder; refs back to 0, lk held

        waiter = asyncio.create_task(lock.acquire("k", timeout=0.01))
        await asyncio.sleep(0.03)  # waiter times out while "k" is held
        assert await waiter is False

        await lock.release("k")
        assert lock._locks == {} and lock._refs == {}, (
            lock._locks,
            lock._refs,
        )

        # Fresh acquirer on the same key still mutually excludes.
        assert await lock.acquire("k") is True
        assert await lock.acquire("k", timeout=0.01) is False
        await lock.release("k")
        assert lock._locks == {} and lock._refs == {}
        await lock.close()

    async def test_release_racing_timed_waiters_no_double_entry(
        self,
    ) -> None:
        """release() racing many timed/blocking acquirers on one key
        never admits two holders and never orphans the key map.

        Hammers the release()/acquire()-finally interleave directly:
        mutual exclusion (peak == 1) is the invariant the fragile
        eviction predicate must never break.
        """
        lock = InProcessLock()
        concurrent = 0
        peak = 0

        async def worker(use_timeout: bool) -> None:
            nonlocal concurrent, peak
            timeout = 0.5 if use_timeout else None
            if not await lock.acquire("shared", timeout=timeout):
                return  # generous timeout: should not happen, but safe
            try:
                concurrent += 1
                peak = max(peak, concurrent)
                await asyncio.sleep(0.005)
                concurrent -= 1
            finally:
                await lock.release("shared")

        await asyncio.gather(*(worker(i % 2 == 0) for i in range(30)))
        assert peak == 1, f"mutual exclusion violated: peak={peak}"
        assert lock._locks == {} and lock._refs == {}
        await lock.close()


class TestLockFactory:
    """Tests for ``create_lock`` and the ``lock_backends`` registry."""

    async def test_default_backend_is_in_process(self) -> None:
        """``create_lock({})`` and ``{"backend": "memory"}`` → InProcessLock."""
        assert isinstance(create_lock({}), InProcessLock)
        assert isinstance(create_lock({"backend": "memory"}), InProcessLock)

    def test_top_level_export_is_same_factory(self) -> None:
        """``dataknobs_common.create_lock`` re-exports the same callable."""
        assert create_lock_top_level is create_lock

    def test_builtin_backends_registered(self) -> None:
        """The registry contains exactly the two built-ins.

        ``memory`` (``InProcessLock``) and ``postgres``
        (``PostgresAdvisoryLock``). Consumer-registered backends are
        additive on top of these.
        """
        assert set(lock_backends.list_keys()) == {"memory", "postgres"}

    def test_postgres_backend_resolves_without_asyncpg_import(self) -> None:
        """``create_lock({"backend":"postgres", ...})`` builds the lock.

        The factory + ``PostgresAdvisoryLock.__init__`` only resolve a
        DSN (no connection, no asyncpg needed) — asyncpg is imported
        lazily inside ``acquire``/``release``. A bogus-but-parseable DSN
        is enough to prove the wiring without a live server.
        """
        lock = create_lock(
            {
                "backend": "postgres",
                "connection_string": "postgresql://u:p@localhost:5432/db",
            }
        )
        assert isinstance(lock, PostgresAdvisoryLock)

    def test_unknown_backend_lists_registered(self) -> None:
        """Unknown backend raises ``ValueError`` naming registered ones.

        Mirrors ``create_event_bus`` exactly — same resolution pattern,
        same message shape (the G1 ``get_optional`` correction).
        """
        with pytest.raises(ValueError) as exc:
            create_lock({"backend": "nope"})
        msg = str(exc.value)
        assert "Unknown lock backend: nope" in msg
        # Sorted, includes every built-in (the G1 ``get_optional``
        # correction shared with ``create_event_bus``).
        assert "Available backends: memory, postgres" in msg

    async def test_custom_backend_plugin(self) -> None:
        """A consumer-registered backend resolves through the factory."""

        class TaggingLock(InProcessLock):
            """Real lock subclass — not a mock — tagged for assertion."""

            backend_tag = "custom-test"

        def _factory(config: dict[str, Any]) -> DistributedLock:
            return TaggingLock()

        lock_backends.register("custom_test", _factory)
        try:
            lock = create_lock({"backend": "custom_test"})
            assert isinstance(lock, TaggingLock)
            assert lock.backend_tag == "custom-test"
            # Behaviourally a real lock.
            async with lock.hold("k") as got:
                assert got is True
            await lock.close()
        finally:
            lock_backends.unregister("custom_test")

    def test_reregister_without_overwrite_raises(self) -> None:
        """Re-registering a built-in name without ``allow_overwrite``
        raises — guards accidental clobber of the ``memory`` backend.
        """
        with pytest.raises(OperationError):
            lock_backends.register("memory", lambda config: InProcessLock())
        # Built-in still intact and usable.
        assert isinstance(create_lock({"backend": "memory"}), InProcessLock)


class TestPostgresKeyspace:
    """Pure-function tests for ``_key_to_bigint`` — no database.

    The mapping must be deterministic, process-independent, and land in
    the signed 64-bit range ``pg_advisory_lock`` accepts; that is the
    property the cross-replica guarantee rests on, so it is pinned
    without needing a server.
    """

    _SIGNED64_MIN = -(2**63)
    _SIGNED64_MAX = 2**63 - 1

    def test_deterministic(self) -> None:
        """Same key → same id, every call (blake2b, not random)."""
        a = PostgresAdvisoryLock._key_to_bigint("ingest:my-domain")
        b = PostgresAdvisoryLock._key_to_bigint("ingest:my-domain")
        assert a == b

    def test_within_signed_64_bit_range(self) -> None:
        """Every id fits a Postgres ``bigint`` (signed 64-bit)."""
        for key in ("", "a", "ingest:x", "ingest:" + "z" * 4096, "🔒"):
            v = PostgresAdvisoryLock._key_to_bigint(key)
            assert self._SIGNED64_MIN <= v <= self._SIGNED64_MAX, (key, v)

    def test_distinct_keys_distinct_ids(self) -> None:
        """Distinct keys do not collide across a realistic sample."""
        keys = [f"ingest:domain-{i}" for i in range(1000)]
        ids = {PostgresAdvisoryLock._key_to_bigint(k) for k in keys}
        assert len(ids) == len(keys)

    def test_stable_against_recomputed_blake2b(self) -> None:
        """Pin the exact algorithm so a DB upgrade can't shift keyspace.

        ``hashtext`` is not contractually stable across PG majors; this
        asserts the implementation is the documented
        ``blake2b(digest_size=8)`` big-endian, biased to signed.
        """
        import hashlib

        key = "ingest:my-domain"
        expected = (
            int.from_bytes(
                hashlib.blake2b(key.encode(), digest_size=8).digest(),
                "big",
                signed=False,
            )
            - (1 << 63)
        )
        assert PostgresAdvisoryLock._key_to_bigint(key) == expected


@pytest.fixture
def pg_dsn(
    ensure_postgres_ready: None,
    postgres_connection_params: dict[str, Any],
) -> str:
    """libpq URI for the shared test database (real Postgres).

    Depends on ``ensure_postgres_ready`` so the server is reachable and
    the test DB exists; advisory locks are cluster-global (no table
    needed). Mirrors ``postgres_fixtures._pg_conn_str``.
    """
    p = postgres_connection_params
    return (
        f"postgresql://{p['user']}:{p['password']}"
        f"@{p['host']}:{p['port']}/{p['database']}"
    )


class TestPostgresAdvisoryLock:
    """Real-Postgres cross-replica behaviour.

    Two **independent** ``PostgresAdvisoryLock`` instances on separate
    connections stand in for two replicas. This is the test that would
    have caught the original ``asyncio.Lock`` defect (a process-local
    lock lets both "replicas" proceed).
    """

    pytestmark = requires_real_postgres

    async def test_two_instances_contend_same_key(self, pg_dsn: str) -> None:
        """Instance A holds; B times out; A releases; B then acquires."""
        a = PostgresAdvisoryLock(connection_string=pg_dsn)
        b = PostgresAdvisoryLock(connection_string=pg_dsn)
        key = f"it:contend:{os.getpid()}"
        try:
            assert await a.acquire(key) is True
            # B (a separate "replica") cannot get it while A holds it.
            assert await b.acquire(key, timeout=0.5) is False
            await a.release(key)
            # Now B can.
            assert await b.acquire(key, timeout=3.0) is True
            await b.release(key)
        finally:
            await a.close()
            await b.close()

    async def test_distinct_keys_do_not_contend(self, pg_dsn: str) -> None:
        """Different keys are independent across instances."""
        a = PostgresAdvisoryLock(connection_string=pg_dsn)
        b = PostgresAdvisoryLock(connection_string=pg_dsn)
        suffix = os.getpid()
        try:
            assert await a.acquire(f"it:k1:{suffix}") is True
            # Different key on a different instance is not serialized.
            assert await b.acquire(f"it:k2:{suffix}", timeout=1.0) is True
        finally:
            await a.release(f"it:k1:{suffix}")
            await b.release(f"it:k2:{suffix}")
            await a.close()
            await b.close()

    async def test_connection_death_releases_lock(self, pg_dsn: str) -> None:
        """A crashed holder must not wedge the key (liveness).

        Session-scoped locks are released by Postgres when the holding
        session dies. Forcibly closing A's held connection simulates a
        replica crash; B must then be able to acquire.
        """
        a = PostgresAdvisoryLock(connection_string=pg_dsn)
        b = PostgresAdvisoryLock(connection_string=pg_dsn)
        key = f"it:death:{os.getpid()}"
        try:
            assert await a.acquire(key) is True
            # Kill A's underlying session out from under it.
            await a._held[key].close()
            # Postgres released the advisory lock with the dead session.
            assert await b.acquire(key, timeout=5.0) is True
            await b.release(key)
        finally:
            # A.close() must tolerate the already-closed connection.
            await a.close()
            await b.close()

    async def test_hold_releases_on_exception(self, pg_dsn: str) -> None:
        """``hold()`` releases the cross-replica lock even on error."""
        a = PostgresAdvisoryLock(connection_string=pg_dsn)
        key = f"it:hold-exc:{os.getpid()}"
        try:
            with pytest.raises(RuntimeError, match="boom"):
                async with a.hold(key) as got:
                    assert got is True
                    raise RuntimeError("boom")
            # Released — re-acquire without contention.
            assert await a.acquire(key, timeout=2.0) is True
            await a.release(key)
        finally:
            await a.close()

    async def test_create_lock_round_trip(self, pg_dsn: str) -> None:
        """Factory-built postgres lock acquires and releases for real."""
        lock = create_lock(
            {"backend": "postgres", "connection_string": pg_dsn}
        )
        assert isinstance(lock, PostgresAdvisoryLock)
        key = f"it:factory:{os.getpid()}"
        try:
            async with lock.hold(key) as got:
                assert got is True
        finally:
            await lock.close()

    async def test_close_releases_all_held(self, pg_dsn: str) -> None:
        """``close()`` frees every held lock (session teardown)."""
        a = PostgresAdvisoryLock(connection_string=pg_dsn)
        b = PostgresAdvisoryLock(connection_string=pg_dsn)
        k1 = f"it:close1:{os.getpid()}"
        k2 = f"it:close2:{os.getpid()}"
        try:
            assert await a.acquire(k1) is True
            assert await a.acquire(k2) is True
            await a.close()  # releases both via session close
            # A fresh "replica" can take both immediately.
            assert await b.acquire(k1, timeout=3.0) is True
            assert await b.acquire(k2, timeout=3.0) is True
        finally:
            await b.release(k1)
            await b.release(k2)
            await b.close()
