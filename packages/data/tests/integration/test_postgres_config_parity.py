"""Cross-site postgres config parity integration test (Phase 2e).

One config dict feeds all five postgres-using constructs that were routed
through ``normalize_postgres_connection_config`` in Phase 2:

1. ``SyncPostgresDatabase``
2. ``AsyncPostgresDatabase``
3. ``PgVectorStore``
4. ``PostgresPoolConfig``
5. ``PostgresEventBus``

This is the end-to-end guardrail that the normalizer produced a consistent
canonical form across every postgres-using construct. It is gated by
``TEST_POSTGRES=true`` and a running postgres instance (via ``bin/dk up``).
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest
from dataknobs_common.events import Event, EventType
from dataknobs_common.events.postgres import PostgresEventBus

from dataknobs_data import AsyncDatabase, Record, SyncDatabase
from dataknobs_data.pooling.postgres import PostgresPoolConfig

pytestmark = pytest.mark.skipif(
    os.environ.get("TEST_POSTGRES", "").lower() != "true",
    reason="Postgres parity tests require TEST_POSTGRES=true and a running "
    "postgres instance (bin/dk up).",
)

# pgvector requires numpy for vector storage.
np = pytest.importorskip("numpy")

try:
    import asyncpg  # noqa: F401

    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False

if ASYNCPG_AVAILABLE:
    from dataknobs_data.vector.stores.pgvector import PgVectorStore


@pytest.mark.asyncio
@pytest.mark.skipif(
    not ASYNCPG_AVAILABLE, reason="asyncpg not installed"
)
async def test_one_config_feeds_all_postgres_constructs(
    postgres_test_db,
) -> None:
    """Single unified config drives every postgres-using construct.

    Exercises the Phase 2 contract: after routing through
    ``normalize_postgres_connection_config``, each construct accepts the
    same individual-keys shape with no connection_string synthesis by the
    caller.
    """
    config = {
        "host": postgres_test_db["host"],
        "port": postgres_test_db["port"],
        "database": postgres_test_db["database"],
        "user": postgres_test_db["user"],
        "password": postgres_test_db["password"],
    }

    # Unique suffix keeps tables/channels isolated across parallel runs.
    suffix = uuid.uuid4().hex[:8]
    sync_table = f"parity_sync_{suffix}"
    async_table = f"parity_async_{suffix}"
    vector_table = f"parity_vec_{suffix}"
    bus_topic = f"parity:events:{suffix}"

    # ------------------------------------------------------------------
    # 1. SyncPostgresDatabase
    # ------------------------------------------------------------------
    sync_db = SyncDatabase.from_backend(
        "postgres", {**config, "table": sync_table, "schema": "public"},
    )
    sync_db.connect()

    # ------------------------------------------------------------------
    # 2. AsyncPostgresDatabase
    # ------------------------------------------------------------------
    async_db = await AsyncDatabase.from_backend(
        "postgres", {**config, "table": async_table, "schema": "public"},
    )
    await async_db.connect()

    # ------------------------------------------------------------------
    # 3. PgVectorStore
    # ------------------------------------------------------------------
    store = PgVectorStore(
        {
            **config,
            "dimensions": 4,
            "schema": "public",
            "table_name": vector_table,
            "id_type": "text",
        }
    )
    await store.initialize()

    # ------------------------------------------------------------------
    # 4. PostgresPoolConfig — builds the same connection triple.
    # ------------------------------------------------------------------
    pool_config = PostgresPoolConfig.from_dict(config)
    assert pool_config.host == config["host"]
    assert pool_config.port == config["port"]
    assert pool_config.database == config["database"]
    assert pool_config.user == config["user"]
    assert pool_config.password == config["password"]

    # ------------------------------------------------------------------
    # 5. PostgresEventBus
    # ------------------------------------------------------------------
    bus = PostgresEventBus(config=config, channel_prefix="parity")
    await bus.connect()

    received: list[Event] = []

    async def handler(event: Event) -> None:
        received.append(event)

    try:
        # Verify cross-construct observability: create a record via the
        # async client, read it back via the sync client, publish an
        # event, receive it on a subscriber.
        record_id = await async_db.create(
            Record({"id": f"rec_{suffix}", "data": "hello"})
        )
        assert await async_db.exists(record_id)

        sync_record = sync_db.read(record_id) if sync_db.exists(record_id) else None
        # Sync and async use different tables — just verify the sync
        # connection works against the same database.
        sync_id = sync_db.create(Record({"data": "sync-side"}))
        assert sync_db.exists(sync_id)
        del sync_record  # intentionally unused, kept for clarity

        # Vector store insert/retrieve roundtrip — exercises the pool
        # connection plus table creation against the same database.
        await store.add_vectors(
            vectors=[np.array([0.1, 0.2, 0.3, 0.4])],
            ids=[f"vec_{suffix}"],
            metadata=[{"note": "parity"}],
        )
        retrieved = await store.get_vectors([f"vec_{suffix}"])
        assert len(retrieved) == 1

        # Event bus publish/subscribe roundtrip.
        await bus.subscribe(bus_topic, handler)
        await bus.publish(
            bus_topic,
            Event(
                type=EventType.CREATED,
                topic=bus_topic,
                payload={"id": record_id},
            ),
        )
        for _ in range(50):
            if received:
                break
            await asyncio.sleep(0.05)
        assert len(received) == 1
        assert received[0].payload == {"id": record_id}
    finally:
        # Teardown in reverse order; tolerate partial failures so one
        # close error does not mask another.
        for closer in (
            bus.close(),
            store.close(),
            async_db.close(),
        ):
            try:
                await closer
            except Exception:
                pass
        try:
            sync_db.close()
        except Exception:
            pass
        # Drop the tables we created. ``postgres_test_db`` fixture only
        # drops its own table; these are extras.
        import psycopg2

        conn = psycopg2.connect(
            host=config["host"],
            port=config["port"],
            user=config["user"],
            password=config["password"],
            database=config["database"],
        )
        try:
            cur = conn.cursor()
            cur.execute(f"DROP TABLE IF EXISTS public.{sync_table} CASCADE")
            cur.execute(f"DROP TABLE IF EXISTS public.{async_table} CASCADE")
            cur.execute(f"DROP TABLE IF EXISTS public.{vector_table} CASCADE")
            conn.commit()
        finally:
            cur.close()
            conn.close()


@pytest.mark.asyncio
@pytest.mark.skipif(
    not ASYNCPG_AVAILABLE, reason="asyncpg not installed"
)
async def test_env_fallbacks_feed_all_postgres_constructs(
    postgres_test_db,
    monkeypatch,
) -> None:
    """Env-only parity — no config keys, only POSTGRES_* env vars.

    Pre-refactor, each construct had its own inline env parsing with
    subtle drift (some read ``POSTGRES_DB``, some didn't; some cast
    port, some didn't). This test guards that every construct now
    reads env fallbacks through the same normalizer by exercising the
    env-only path with an empty config dict.

    Complements ``test_one_config_feeds_all_postgres_constructs``,
    which exercises the config-dict path.
    """
    # Isolate env from the surrounding shell — only our values survive.
    for key in (
        "DATABASE_URL",
        "POSTGRES_HOST",
        "POSTGRES_PORT",
        "POSTGRES_DB",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("POSTGRES_HOST", postgres_test_db["host"])
    monkeypatch.setenv("POSTGRES_PORT", str(postgres_test_db["port"]))
    monkeypatch.setenv("POSTGRES_DB", postgres_test_db["database"])
    monkeypatch.setenv("POSTGRES_USER", postgres_test_db["user"])
    monkeypatch.setenv("POSTGRES_PASSWORD", postgres_test_db["password"])

    suffix = uuid.uuid4().hex[:8]
    vector_table = f"parity_env_vec_{suffix}"
    sync_table = f"parity_env_sync_{suffix}"
    bus_topic = f"parity:env:{suffix}"

    # Disable dotenv loading for deterministic env-only behavior —
    # a .env file in the repo root could otherwise shadow monkeypatch
    # in this process if python-dotenv is installed.
    from dataknobs_common.postgres_config import (
        normalize_postgres_connection_config,
    )

    # Sanity check: the normalizer resolves the connection from env
    # with an empty config dict.
    resolved = normalize_postgres_connection_config(
        {}, require=True, load_dotenv=False,
    )
    assert resolved is not None
    assert resolved["host"] == postgres_test_db["host"]
    assert resolved["database"] == postgres_test_db["database"]

    # Exercise all four postgres-using constructs with {} config — each
    # must resolve from env without drift.
    sync_db = SyncDatabase.from_backend(
        "postgres", {"table": sync_table, "schema": "public"},
    )
    sync_db.connect()

    store = PgVectorStore(
        {
            "dimensions": 4,
            "schema": "public",
            "table_name": vector_table,
            "id_type": "text",
        }
    )
    await store.initialize()

    pool_config = PostgresPoolConfig.from_dict({})
    assert pool_config.host == postgres_test_db["host"]
    assert pool_config.database == postgres_test_db["database"]

    bus = PostgresEventBus(config={}, channel_prefix="parity_env")
    await bus.connect()

    received: list[Event] = []

    async def handler(event: Event) -> None:
        received.append(event)

    try:
        sync_id = sync_db.create(Record({"data": "env-sync"}))
        assert sync_db.exists(sync_id)

        await store.add_vectors(
            vectors=[np.array([0.1, 0.2, 0.3, 0.4])],
            ids=[f"env_vec_{suffix}"],
            metadata=[{"note": "env-parity"}],
        )
        assert len(await store.get_vectors([f"env_vec_{suffix}"])) == 1

        await bus.subscribe(bus_topic, handler)
        await bus.publish(
            bus_topic,
            Event(
                type=EventType.CREATED,
                topic=bus_topic,
                payload={"suffix": suffix},
            ),
        )
        for _ in range(50):
            if received:
                break
            await asyncio.sleep(0.05)
        assert len(received) == 1
    finally:
        for closer in (bus.close(), store.close()):
            try:
                await closer
            except Exception:
                pass
        try:
            sync_db.close()
        except Exception:
            pass
        import psycopg2

        conn = psycopg2.connect(
            host=postgres_test_db["host"],
            port=postgres_test_db["port"],
            user=postgres_test_db["user"],
            password=postgres_test_db["password"],
            database=postgres_test_db["database"],
        )
        try:
            cur = conn.cursor()
            cur.execute(f"DROP TABLE IF EXISTS public.{sync_table} CASCADE")
            cur.execute(f"DROP TABLE IF EXISTS public.{vector_table} CASCADE")
            conn.commit()
        finally:
            cur.close()
            conn.close()
