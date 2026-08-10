"""PostgreSQL-backed behavioral tests for the user-state coordinator.

Exercises the real CAS conflict and tenant-scoping paths against a live
PostgreSQL backend. Skips gracefully when PostgreSQL is unavailable
(``@requires_postgres``); the ``make_postgres_test_db`` fixture is provided by
the shared ``dataknobs_common_postgres`` pytest11 plugin and drops the
per-test table on teardown.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest

from dataknobs_data.user import AsyncUserStateStore, UserStateStoreConfig
from dataknobs_common.exceptions import ConcurrencyError
from dataknobs_common.tenancy import BoundTenantContext
from dataknobs_common.testing import requires_postgres
from dataknobs_data.backends.postgres import AsyncPostgresDatabase

pytestmark = requires_postgres

_CONFIG = {
    "backend": "postgres",
    "namespace": "acme",
    "sections": [
        {"name": "preferences", "kind": "document"},
        {"name": "alerts", "kind": "collection"},
    ],
}


@pytest.fixture
async def pg_db(make_postgres_test_db) -> AsyncGenerator[AsyncPostgresDatabase, None]:
    for pg in make_postgres_test_db("test_user_state_"):
        db = AsyncPostgresDatabase(
            {
                "host": pg["host"],
                "port": pg["port"],
                "database": pg["database"],
                "user": pg["user"],
                "password": pg["password"],
                "table": pg["table"],
            }
        )
        await db.connect()
        try:
            yield db
        finally:
            await db.close()


async def test_postgres_cas_conflict(pg_db: AsyncPostgresDatabase) -> None:
    cfg = UserStateStoreConfig.from_dict(_CONFIG)
    store = AsyncUserStateStore.from_components(cfg, db=pg_db)

    await store.put_document("u1", "preferences", {"theme": "dark"})
    token = await store.document_version("u1", "preferences")
    await store.put_document("u1", "preferences", {"theme": "light"}, expected_version=token)
    with pytest.raises(ConcurrencyError):
        await store.put_document("u1", "preferences", {"theme": "blue"}, expected_version=token)
    await store.close()  # injected db closed by the fixture


async def test_postgres_tenant_isolation(pg_db: AsyncPostgresDatabase) -> None:
    cfg = UserStateStoreConfig.from_dict(_CONFIG)
    t1 = AsyncUserStateStore.from_components(cfg, db=pg_db, tenant=BoundTenantContext("t1", "acme"))
    t2 = AsyncUserStateStore.from_components(cfg, db=pg_db, tenant=BoundTenantContext("t2", "acme"))
    await t1.add_record("u", "alerts", {"text": "t1"})
    await t2.add_record("u", "alerts", {"text": "t2"})

    assert [r.get_value("text") for r in await t1.query("u", "alerts")] == ["t1"]
    assert [r.get_value("text") for r in await t2.query("u", "alerts")] == ["t2"]
    await t1.close()
    await t2.close()
