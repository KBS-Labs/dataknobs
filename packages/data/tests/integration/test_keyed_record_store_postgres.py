"""Postgres integration tests for ``AsyncKeyedRecordStore``.

These exercise the ``metadata.X`` field-path routing that the SQL
backends implement via the JSONB ``metadata`` column.  Without this
test, a regression that broke metadata-column routing on the Postgres
backend would only surface in a downstream consumer (e.g. EduBot's
multi-tenant filter).
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

import pytest
from dataknobs_common.testing import requires_postgres

from dataknobs_data import AsyncKeyedRecordStore, Record
from dataknobs_data.backends.postgres import AsyncPostgresDatabase

pytestmark = requires_postgres


@dataclass
class Bot:
    bot_id: str
    config: dict[str, Any]
    status: str = "active"
    tenant_id: str | None = None
    audit: dict[str, Any] = field(default_factory=dict)


def _bot_to_columns(bot: Bot) -> tuple[dict[str, Any], dict[str, Any]]:
    data = {
        "bot_id": bot.bot_id,
        "config": bot.config,
        "status": bot.status,
    }
    metadata: dict[str, Any] = {}
    if bot.tenant_id is not None:
        metadata["tenant_id"] = bot.tenant_id
    if bot.audit:
        metadata["audit"] = bot.audit
    return data, metadata


def _bot_from_record(record: Record) -> Bot:
    return Bot(
        bot_id=record.get_value("bot_id"),
        config=record.get_value("config") or {},
        status=record.get_value("status") or "active",
        tenant_id=record.metadata.get("tenant_id"),
        audit=dict(record.metadata.get("audit") or {}),
    )


@pytest.fixture
async def async_postgres_store(make_postgres_test_db) -> AsyncGenerator[
    AsyncKeyedRecordStore[Bot], None
]:
    """Per-test Postgres-backed ``AsyncKeyedRecordStore[Bot]``.

    Uses a ``test_keyed_store_`` table prefix (instead of the generic
    package-level ``test_records_``) so a leaked table is traceable
    back to this module on inspection.
    """
    for pg in make_postgres_test_db("test_keyed_store_"):
        db = AsyncPostgresDatabase(pg)
        await db.connect()
        store = AsyncKeyedRecordStore[Bot](
            db,
            serializer=_bot_to_columns,
            deserializer=_bot_from_record,
        )
        try:
            yield store
        finally:
            await db.close()


class TestKeyedRecordStorePostgres:
    @pytest.mark.asyncio
    async def test_metadata_lands_in_metadata_column(self, async_postgres_store):
        """The structural prevention contract end-to-end on Postgres."""
        await async_postgres_store.put(
            "alpha",
            Bot("alpha", {"llm": "ollama"}, tenant_id="acme", audit={"by": "alice"}),
        )

        raw = await async_postgres_store.db.read("alpha")
        assert raw is not None
        assert raw.metadata == {
            "tenant_id": "acme",
            "audit": {"by": "alice"},
        }
        # And the value round-trips through deserializer.
        loaded = await async_postgres_store.get("alpha")
        assert loaded is not None
        assert loaded.tenant_id == "acme"
        assert loaded.audit == {"by": "alice"}

    @pytest.mark.asyncio
    async def test_filter_metadata_routes_to_jsonb_column(self, async_postgres_store):
        """Reproduces EduBot I-9 Cut D: tenant-scoped multi-tenant list.

        Before ``KeyedRecordStore`` existed, registries built
        ``Record(data={...})`` without ``metadata=`` and this filter
        returned an empty list — every record had an empty
        ``metadata`` column.  The store closes that defect class.
        """
        await async_postgres_store.put("a", Bot("a", {}, tenant_id="t1"))
        await async_postgres_store.put("b", Bot("b", {}, tenant_id="t2"))
        await async_postgres_store.put("c", Bot("c", {}, tenant_id="t1"))

        t1_bots = await async_postgres_store.list(filter_metadata={"tenant_id": "t1"})
        assert {b.bot_id for b in t1_bots} == {"a", "c"}

        t2_bots = await async_postgres_store.list(filter_metadata={"tenant_id": "t2"})
        assert {b.bot_id for b in t2_bots} == {"b"}

    @pytest.mark.asyncio
    async def test_combined_filters_postgres(self, async_postgres_store):
        await async_postgres_store.put(
            "a", Bot("a", {}, status="active", tenant_id="t1")
        )
        await async_postgres_store.put(
            "b", Bot("b", {}, status="inactive", tenant_id="t1")
        )
        await async_postgres_store.put(
            "c", Bot("c", {}, status="active", tenant_id="t2")
        )

        active_t1 = await async_postgres_store.list(
            filter_data={"status": "active"},
            filter_metadata={"tenant_id": "t1"},
        )
        assert {b.bot_id for b in active_t1} == {"a"}

    @pytest.mark.asyncio
    async def test_count_with_metadata_filter(self, async_postgres_store):
        await async_postgres_store.put("a", Bot("a", {}, tenant_id="t1"))
        await async_postgres_store.put("b", Bot("b", {}, tenant_id="t2"))
        await async_postgres_store.put("c", Bot("c", {}, tenant_id="t1"))

        assert await async_postgres_store.count(filter_metadata={"tenant_id": "t1"}) == 2
        assert await async_postgres_store.count(filter_metadata={"tenant_id": "t2"}) == 1
        assert await async_postgres_store.count() == 3
