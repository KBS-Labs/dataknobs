"""Tests for ``AsyncKeyedRecordStore`` and ``SyncKeyedRecordStore``.

These tests focus on the contract that the store is meant to enforce:

* The serializer's ``(data, metadata)`` tuple round-trips through the
  underlying database with the metadata channel preserved.
* ``filter_data`` and ``filter_metadata`` route to the correct columns
  via the existing ``metadata.X`` field-path convention.
* ``count()`` routes through ``AsyncDatabase.count(query)`` so that any
  future pushdown benefits consumers transparently.

Tests run against ``AsyncMemoryDatabase`` / ``SyncMemoryDatabase``;
Postgres integration coverage lives in
``tests/integration/test_keyed_record_store_postgres.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from dataknobs_data import (
    AsyncKeyedRecordStore,
    Record,
    SyncKeyedRecordStore,
)
from dataknobs_data.backends.memory import AsyncMemoryDatabase, SyncMemoryDatabase
from dataknobs_data.query import SortOrder, SortSpec


@dataclass
class Bot:
    """Small typed value used to exercise the generic store."""

    bot_id: str
    config: dict[str, Any]
    status: str = "active"
    tenant_id: str | None = None
    audit: dict[str, Any] = field(default_factory=dict)


def _bot_to_columns(bot: Bot) -> tuple[dict[str, Any], dict[str, Any]]:
    """Serializer: identity fields go into ``data``; cross-cutting into ``metadata``."""
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
    """Deserializer: pulls identity fields from ``data`` and cross-cutting from ``metadata``."""
    return Bot(
        bot_id=record.get_value("bot_id"),
        config=record.get_value("config") or {},
        status=record.get_value("status") or "active",
        tenant_id=record.metadata.get("tenant_id"),
        audit=dict(record.metadata.get("audit") or {}),
    )


# ---------------------------------------------------------------------------
# Async surface
# ---------------------------------------------------------------------------


@pytest.fixture
async def async_store() -> AsyncKeyedRecordStore[Bot]:
    db = AsyncMemoryDatabase()
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


class TestAsyncKeyedRecordStoreCRUD:
    @pytest.mark.asyncio
    async def test_put_get_round_trip(self, async_store):
        bot = Bot("alpha", {"llm": "ollama"}, tenant_id="t1", audit={"by": "alice"})
        await async_store.put("alpha", bot)

        loaded = await async_store.get("alpha")
        assert loaded == bot

    @pytest.mark.asyncio
    async def test_get_missing_returns_none(self, async_store):
        assert await async_store.get("nope") is None

    @pytest.mark.asyncio
    async def test_exists(self, async_store):
        await async_store.put("alpha", Bot("alpha", {}))
        assert await async_store.exists("alpha") is True
        assert await async_store.exists("beta") is False

    @pytest.mark.asyncio
    async def test_delete(self, async_store):
        await async_store.put("alpha", Bot("alpha", {}))
        assert await async_store.delete("alpha") is True
        assert await async_store.get("alpha") is None
        # Idempotent delete returns False on missing.
        assert await async_store.delete("alpha") is False

    @pytest.mark.asyncio
    async def test_put_overwrite(self, async_store):
        await async_store.put("alpha", Bot("alpha", {}, status="active"))
        await async_store.put("alpha", Bot("alpha", {}, status="inactive"))
        loaded = await async_store.get("alpha")
        assert loaded is not None
        assert loaded.status == "inactive"


class TestAsyncKeyedRecordStoreMetadataPreservation:
    """The defect class this abstraction exists to prevent.

    A consumer building ``Record(data={...})`` inline easily forgets to
    pass ``metadata=`` and silently drops cross-cutting fields.  The
    store's ``(data, metadata)`` tuple serializer makes that mistake
    impossible at the type level.
    """

    @pytest.mark.asyncio
    async def test_metadata_channel_preserved(self, async_store):
        bot = Bot(
            "alpha",
            {"llm": "ollama"},
            tenant_id="acme",
            audit={"created_by": "alice", "request_id": "req-1"},
        )
        await async_store.put("alpha", bot)

        # Reach through the store to the raw record: metadata should be
        # populated, not the data column.
        raw = await async_store.db.read("alpha")
        assert raw is not None
        assert raw.metadata == {
            "tenant_id": "acme",
            "audit": {"created_by": "alice", "request_id": "req-1"},
        }
        # Identity fields landed in data, not metadata.
        assert raw.get_value("bot_id") == "alpha"
        assert raw.get_value("status") == "active"
        assert "tenant_id" not in raw.fields

    @pytest.mark.asyncio
    async def test_serializer_dict_inputs_are_copied(self, async_store):
        """Mutating the dicts after put must not corrupt stored state."""
        audit = {"by": "alice"}
        bot = Bot("alpha", {"llm": "ollama"}, tenant_id="t1", audit=audit)
        await async_store.put("alpha", bot)

        # Mutate the audit dict the test still holds a reference to.
        audit["by"] = "MUTATED"

        loaded = await async_store.get("alpha")
        assert loaded is not None
        assert loaded.audit == {"by": "alice"}


class TestAsyncKeyedRecordStoreBatch:
    @pytest.mark.asyncio
    async def test_put_batch_get_batch_round_trip(self, async_store):
        bots = {
            "a": Bot("a", {"x": 1}, tenant_id="t1"),
            "b": Bot("b", {"x": 2}, tenant_id="t2"),
            "c": Bot("c", {"x": 3}, tenant_id="t1"),
        }
        await async_store.put_batch(bots)

        got = await async_store.get_batch(["a", "b", "c"])
        assert got == [bots["a"], bots["b"], bots["c"]]

    @pytest.mark.asyncio
    async def test_get_batch_includes_misses(self, async_store):
        await async_store.put("a", Bot("a", {}))
        got = await async_store.get_batch(["a", "missing"])
        assert got[0] is not None
        assert got[0].bot_id == "a"
        assert got[1] is None

    @pytest.mark.asyncio
    async def test_delete_batch_returns_actual_deletion_count(self, async_store):
        await async_store.put("a", Bot("a", {}))
        await async_store.put("b", Bot("b", {}))
        count = await async_store.delete_batch(["a", "b", "missing"])
        assert count == 2


class TestAsyncKeyedRecordStoreListAndCount:
    async def _seed(self, store: AsyncKeyedRecordStore[Bot]) -> None:
        await store.put("a", Bot("a", {}, status="active", tenant_id="t1"))
        await store.put("b", Bot("b", {}, status="active", tenant_id="t2"))
        await store.put("c", Bot("c", {}, status="active", tenant_id="t1"))
        await store.put("d", Bot("d", {}, status="inactive", tenant_id="t1"))

    @pytest.mark.asyncio
    async def test_list_no_filters_returns_all(self, async_store):
        await self._seed(async_store)
        result = await async_store.list()
        assert {b.bot_id for b in result} == {"a", "b", "c", "d"}

    @pytest.mark.asyncio
    async def test_list_filter_data_only(self, async_store):
        await self._seed(async_store)
        result = await async_store.list(filter_data={"status": "active"})
        assert {b.bot_id for b in result} == {"a", "b", "c"}

    @pytest.mark.asyncio
    async def test_list_filter_metadata_only(self, async_store):
        """Multi-tenant filter using metadata only (no data-column predicate)."""
        await self._seed(async_store)
        result = await async_store.list(filter_metadata={"tenant_id": "t1"})
        assert {b.bot_id for b in result} == {"a", "c", "d"}

    @pytest.mark.asyncio
    async def test_list_combined_filters_are_anded(self, async_store):
        await self._seed(async_store)
        result = await async_store.list(
            filter_data={"status": "active"},
            filter_metadata={"tenant_id": "t1"},
        )
        assert {b.bot_id for b in result} == {"a", "c"}

    @pytest.mark.asyncio
    async def test_list_empty_dicts_equivalent_to_none(self, async_store):
        await self._seed(async_store)
        a = await async_store.list()
        b = await async_store.list(filter_data={}, filter_metadata={})
        assert {x.bot_id for x in a} == {x.bot_id for x in b}

    @pytest.mark.asyncio
    async def test_list_sort_limit_offset(self, async_store):
        await self._seed(async_store)
        result = await async_store.list(
            filter_data={"status": "active"},
            sort=[SortSpec(field="bot_id", order=SortOrder.DESC)],
            limit=2,
            offset=1,
        )
        assert [b.bot_id for b in result] == ["b", "a"]

    @pytest.mark.asyncio
    async def test_count_no_filter(self, async_store):
        await self._seed(async_store)
        assert await async_store.count() == 4

    @pytest.mark.asyncio
    async def test_count_filter_data(self, async_store):
        await self._seed(async_store)
        assert await async_store.count(filter_data={"status": "active"}) == 3

    @pytest.mark.asyncio
    async def test_count_filter_metadata(self, async_store):
        await self._seed(async_store)
        assert await async_store.count(filter_metadata={"tenant_id": "t1"}) == 3

    @pytest.mark.asyncio
    async def test_count_combined_filters(self, async_store):
        await self._seed(async_store)
        assert (
            await async_store.count(
                filter_data={"status": "active"},
                filter_metadata={"tenant_id": "t1"},
            )
            == 2
        )


class TestAsyncKeyedRecordStoreClear:
    @pytest.mark.asyncio
    async def test_clear_empty_store_returns_zero(self, async_store):
        assert await async_store.clear() == 0
        assert await async_store.count() == 0

    @pytest.mark.asyncio
    async def test_clear_returns_deletion_count_and_empties_store(self, async_store):
        await async_store.put("a", Bot("a", {}, tenant_id="t1"))
        await async_store.put("b", Bot("b", {}, tenant_id="t2"))
        await async_store.put("c", Bot("c", {}, tenant_id="t1"))

        deleted = await async_store.clear()

        assert deleted == 3
        assert await async_store.count() == 0
        assert await async_store.list() == []

    @pytest.mark.asyncio
    async def test_clear_removes_records_with_metadata(self, async_store):
        """Records carrying metadata round-trip through ``clear`` correctly."""
        await async_store.put(
            "a", Bot("a", {}, tenant_id="t1", audit={"by": "alice"})
        )
        await async_store.put(
            "b", Bot("b", {}, tenant_id="t2", audit={"by": "bob"})
        )

        await async_store.clear()

        assert await async_store.get("a") is None
        assert await async_store.get("b") is None
        # Metadata-filtered queries also see an empty store.
        assert await async_store.list(filter_metadata={"tenant_id": "t1"}) == []


class TestAsyncKeyedRecordStoreStream:
    @pytest.mark.asyncio
    async def test_stream_all(self, async_store):
        await async_store.put("a", Bot("a", {}, tenant_id="t1"))
        await async_store.put("b", Bot("b", {}, tenant_id="t2"))

        seen = [b async for b in async_store.stream()]
        assert {b.bot_id for b in seen} == {"a", "b"}

    @pytest.mark.asyncio
    async def test_stream_filtered_by_metadata(self, async_store):
        await async_store.put("a", Bot("a", {}, tenant_id="t1"))
        await async_store.put("b", Bot("b", {}, tenant_id="t2"))
        await async_store.put("c", Bot("c", {}, tenant_id="t1"))

        seen = [b async for b in async_store.stream(filter_metadata={"tenant_id": "t1"})]
        assert {b.bot_id for b in seen} == {"a", "c"}


# ---------------------------------------------------------------------------
# Sync surface mirrors
# ---------------------------------------------------------------------------


@pytest.fixture
def sync_store() -> SyncKeyedRecordStore[Bot]:
    db = SyncMemoryDatabase()
    db.connect()
    store = SyncKeyedRecordStore[Bot](
        db,
        serializer=_bot_to_columns,
        deserializer=_bot_from_record,
    )
    try:
        yield store
    finally:
        db.close()


class TestSyncKeyedRecordStore:
    def test_put_get_round_trip(self, sync_store):
        bot = Bot("alpha", {"llm": "ollama"}, tenant_id="t1")
        sync_store.put("alpha", bot)
        assert sync_store.get("alpha") == bot

    def test_metadata_channel_preserved(self, sync_store):
        sync_store.put("alpha", Bot("alpha", {}, tenant_id="acme"))
        raw = sync_store.db.read("alpha")
        assert raw is not None
        assert raw.metadata == {"tenant_id": "acme"}

    def test_list_filter_metadata(self, sync_store):
        sync_store.put("a", Bot("a", {}, tenant_id="t1"))
        sync_store.put("b", Bot("b", {}, tenant_id="t2"))
        sync_store.put("c", Bot("c", {}, tenant_id="t1"))

        result = sync_store.list(filter_metadata={"tenant_id": "t1"})
        assert {b.bot_id for b in result} == {"a", "c"}

    def test_count_combined_filters(self, sync_store):
        sync_store.put("a", Bot("a", {}, status="active", tenant_id="t1"))
        sync_store.put("b", Bot("b", {}, status="inactive", tenant_id="t1"))
        sync_store.put("c", Bot("c", {}, status="active", tenant_id="t2"))

        assert (
            sync_store.count(
                filter_data={"status": "active"},
                filter_metadata={"tenant_id": "t1"},
            )
            == 1
        )

    def test_batch_round_trip(self, sync_store):
        bots = {"a": Bot("a", {}), "b": Bot("b", {})}
        sync_store.put_batch(bots)
        assert sync_store.get_batch(["a", "b", "missing"]) == [bots["a"], bots["b"], None]
        assert sync_store.delete_batch(["a", "b", "missing"]) == 2

    def test_clear(self, sync_store):
        sync_store.put("a", Bot("a", {}, tenant_id="t1"))
        sync_store.put("b", Bot("b", {}))
        assert sync_store.clear() == 2
        assert sync_store.count() == 0
        assert sync_store.get("a") is None
