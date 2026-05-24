"""``PostgresAdvisoryLock`` — the cross-replica :class:`DistributedLock`.

A session-scoped ``pg_advisory_lock`` on a dedicated asyncpg connection
per held key. Mutual exclusion spans **every process pointing at the
same database**, which is exactly what a multi-replica
:class:`~dataknobs_bots.knowledge.orchestration.IngestOrchestrator`
needs and what the process-local :class:`InProcessLock` cannot provide.

Connection resolution is the *same* call
:class:`~dataknobs_common.events.postgres.PostgresEventBus` makes
(``normalize_postgres_connection_config``) so the two Postgres-backed
primitives resolve a DSN identically — ``connection_string``,
individual host/port/database/user/password keys, ``DATABASE_URL``, and
``POSTGRES_*`` env-var fallbacks all work the same way.

``asyncpg`` is an *optional* dependency (the existing ``postgres``
extra, shared with ``PostgresEventBus``) and is imported lazily inside
the methods, so importing this module — and ``dataknobs_common.locks``
— never requires asyncpg and the base install stays
``dependencies = []``.

Liveness vs. fencing:

* **Liveness (guaranteed).** The lock is session-scoped: if a holding
  replica crashes, its Postgres session dies and the lock is released
  automatically. A dead replica can never wedge a domain.
* **Fencing (explicitly NOT provided).** Advisory locks bound
  concurrency, not ordering. This is a correct, documented property —
  mutual exclusion fully meets the orchestrator's need. If a consumer
  ever needs fencing tokens that is a distinct higher-level
  abstraction, not a defect here.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING, Any, ClassVar

from dataknobs_common.structured_config import StructuredConfigConsumer

from .config import PostgresLockConfig
from .lock import _hold

if TYPE_CHECKING:
    from collections.abc import Mapping

    import asyncpg

    from dataknobs_common.structured_config import StructuredConfig

logger = logging.getLogger(__name__)

_CONNECT_TIMEOUT = 10.0
"""Bound on a single ``asyncpg.connect`` (seconds).

Mirrors ``PostgresEventBus``'s bounded connect — an unreachable or
wedged server must fail an ``acquire`` promptly rather than stall the
caller (and, transitively, ``close()``) indefinitely.
"""


class PostgresAdvisoryLock(StructuredConfigConsumer[PostgresLockConfig]):
    """Cross-replica :class:`DistributedLock` via ``pg_advisory_lock``.

    One dedicated asyncpg connection is held per currently-locked key
    (a small map keyed by lock key). The lock is **session-level**, not
    transaction-level: a critical section (e.g. an ingest) routinely
    outlives any single transaction, and a transaction-level
    ``pg_advisory_xact_lock`` would release at the first commit inside
    the section.

    The opaque string key is mapped to the signed 64-bit integer
    ``pg_advisory_lock`` requires via Python ``blake2b`` —
    deterministically and identically in every process — rather than
    Postgres ``hashtext``, whose algorithm is not contractually stable
    across major versions (a lock key must mean the same thing after a
    DB upgrade).

    Example:
        ```python
        from dataknobs_common.locks import create_lock

        lock = create_lock({
            "backend": "postgres",
            "connection_string": "postgresql://u:p@host/db",
        })
        async with lock.hold("ingest:my-domain") as acquired:
            if acquired:
                ...  # cross-replica critical section
        await lock.close()
        ```

    Requires:
        asyncpg: Async PostgreSQL driver
            (``pip install 'dataknobs-common[postgres]'``).
    """

    CONFIG_CLS: ClassVar[type[PostgresLockConfig]] = PostgresLockConfig

    def __init__(
        self,
        connection_string: str | None = None,
        *,
        config: PostgresLockConfig | Mapping[str, Any] | None = None,
    ) -> None:
        """Resolve and store the Postgres DSN.

        Three construction shapes are supported:

        - **Typed config** (recommended): pass a
          :class:`PostgresLockConfig` via ``config=``.
        - **Loose dict config**: pass a dict via ``config=`` (normalized
          through :func:`normalize_postgres_connection_config`, so every
          input shape is supported: individual host/port/... keys,
          ``DATABASE_URL``, ``POSTGRES_*`` env-var fallbacks). May be
          combined with the legacy ``connection_string`` positional —
          which takes precedence.
        - **Legacy positional**: pass ``connection_string`` directly.

        Mixing a typed :class:`PostgresLockConfig` with the legacy
        positional ``connection_string`` raises ``TypeError``.

        Args:
            connection_string: PostgreSQL connection string. Retained
                for the convenient positional shape — new callers should
                prefer ``config`` for the unified resolution (individual
                keys + env fallbacks). An explicit value here wins over
                any ``connection_string`` inside ``config``.
            config: Optional typed :class:`PostgresLockConfig` or dict
                accepted by ``normalize_postgres_connection_config`` —
                supports ``connection_string``, individual host/port/
                database/user/password keys, ``DATABASE_URL``, and
                ``POSTGRES_*`` env-var fallbacks. This is the *same*
                resolution path ``PostgresEventBus`` uses.

        Raises:
            ConfigurationError: If no Postgres connection is resolvable
                from ``config``, ``connection_string``, or env vars.
            TypeError: If a typed :class:`PostgresLockConfig` is passed
                alongside the legacy positional ``connection_string``.
        """
        if connection_string is not None:
            if isinstance(config, PostgresLockConfig):
                raise TypeError(
                    "PostgresAdvisoryLock: cannot mix typed "
                    "`PostgresLockConfig` with the legacy positional "
                    "`connection_string`."
                )
            merged: dict[str, Any] = dict(config or {})
            merged["connection_string"] = connection_string
            config = merged
        super().__init__(config=config)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any] | StructuredConfig,
    ) -> PostgresAdvisoryLock:
        """Construct from a config dict or typed config.

        Overrides
        :meth:`~dataknobs_common.structured_config.StructuredConfigConsumer.from_config`
        so the typed config is delivered via the keyword-only ``config=``
        slot rather than the legacy ``connection_string`` positional.
        Reuses the inherited ``_coerce_config`` guard so a config of the
        wrong ``StructuredConfig`` subclass raises a clear ``TypeError``.
        """
        return cls(config=cls._coerce_config(config))

    def _setup(self) -> None:
        self._dsn: str = self._config.connection_string
        # One held asyncpg connection per currently-locked key.
        self._held: dict[str, asyncpg.Connection] = {}
        self._guard = asyncio.Lock()  # protects ``_held``

    @staticmethod
    def _key_to_bigint(key: str) -> int:
        """Map an opaque key to the signed 64-bit ``pg_advisory_lock`` id.

        ``blake2b`` (not Postgres ``hashtext``) so the mapping is
        process-independent and stable across Postgres major versions —
        a lock key must keep meaning the same thing after a DB upgrade.
        Collisions are astronomically unlikely for realistic keyspaces
        and would only over-serialize (never under-serialize), so the
        safety property is preserved even in that case.
        """
        digest = hashlib.blake2b(key.encode(), digest_size=8).digest()
        unsigned = int.from_bytes(digest, "big", signed=False)
        return unsigned - (1 << 63)  # → signed 64-bit, stable forever

    async def acquire(
        self, key: str, *, timeout: float | None = None
    ) -> bool:
        """Acquire ``key`` cross-replica. See :meth:`DistributedLock.acquire`.

        Opens a dedicated session connection and takes a session-level
        ``pg_advisory_lock``. ``timeout=None`` blocks until granted;
        a finite ``timeout`` tries the non-blocking
        ``pg_try_advisory_lock`` fast path, then a bounded wait on the
        blocking form, returning ``False`` if it elapses.

        ``timeout`` bounds only the lock-wait phase. Establishing the
        dedicated connection is separately bounded by the fixed
        ``_CONNECT_TIMEOUT`` and is not deducted from ``timeout``, so
        worst-case wall time for a finite ``timeout`` is
        ``_CONNECT_TIMEOUT + timeout``.
        """
        import asyncpg

        lock_id = self._key_to_bigint(key)
        conn = await asyncpg.connect(self._dsn, timeout=_CONNECT_TIMEOUT)
        try:
            if timeout is None:
                await conn.execute("SELECT pg_advisory_lock($1)", lock_id)
                got = True
            else:
                got = bool(
                    await conn.fetchval(
                        "SELECT pg_try_advisory_lock($1)", lock_id
                    )
                )
                if not got:
                    try:
                        await asyncio.wait_for(
                            conn.execute(
                                "SELECT pg_advisory_lock($1)", lock_id
                            ),
                            timeout,
                        )
                        got = True
                    except (TimeoutError, asyncio.TimeoutError):
                        got = False
            if not got:
                # Closing the connection is deliberate: it tears down
                # the abandoned backend ``pg_advisory_lock`` waiter
                # rather than leaving it racing for the lock after we
                # have already given up.
                await conn.close()
                return False
            async with self._guard:
                self._held[key] = conn
            return True
        except BaseException:
            # Any failure (incl. cancellation) before the connection is
            # registered must not leak it.
            await conn.close()
            raise

    async def release(self, key: str) -> None:
        """Release ``key``. No-op if not held by this instance.

        Issues ``pg_advisory_unlock`` then closes the dedicated
        connection. Closing the connection also frees the lock at the
        Postgres side even if the explicit unlock failed (session
        scope), so the lock can never be left stuck.
        """
        async with self._guard:
            conn = self._held.pop(key, None)
        if conn is None:
            return
        try:
            await conn.execute(
                "SELECT pg_advisory_unlock($1)", self._key_to_bigint(key)
            )
        except Exception:
            logger.warning(
                "pg_advisory_unlock failed for key=%s; closing the "
                "connection releases the session-scoped lock anyway",
                key,
                exc_info=True,
            )
        finally:
            await conn.close()

    def hold(
        self, key: str, *, timeout: float | None = None
    ) -> AbstractAsyncContextManager[bool]:
        """Async CM wrapping acquire/release. See the protocol."""
        return _hold(self, key, timeout)

    async def close(self) -> None:
        """Close every held connection (releasing every held lock).

        Idempotent. Connection close is session teardown, so every
        outstanding advisory lock is released by Postgres regardless of
        whether an explicit unlock ran.
        """
        async with self._guard:
            conns = list(self._held.values())
            self._held = {}
        for conn in conns:
            try:
                await conn.close()
            except Exception:
                logger.warning(
                    "Error closing a held advisory-lock connection",
                    exc_info=True,
                )
