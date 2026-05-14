"""Postgres integration tests for :class:`DataKnobsRegistryAdapter`.

Pin the metadata-column routing contract against a real PostgreSQL
backend.  The ``metadata.X`` field-path convention must be pushed into
the JSONB ``metadata`` column by the SQL backend (rather than scanning
every row in Python), so the canonical multi-tenant filter used by
downstream consumers works at scale.

These tests would have **failed** against earlier adapter implementations
that built ``Record(data=...)`` inline and never populated the metadata
column.  With the keyed-store composition, the
``AsyncKeyedRecordStore[Registration]`` serializer signature forces the
metadata channel into the JSONB column by construction.

Skipped automatically when PostgreSQL is unavailable via
``@requires_postgres`` (from the shared ``dataknobs_common.testing``
pytest11 plugin).  The plugin also provides ``make_postgres_test_db``
which gives a unique table per test and drops it on teardown.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest

from dataknobs_bots.registry import DataKnobsRegistryAdapter
from dataknobs_common.testing import requires_postgres
from dataknobs_data.backends.postgres import AsyncPostgresDatabase

pytestmark = requires_postgres


@pytest.fixture
async def postgres_registry_adapter(
    make_postgres_test_db,
) -> AsyncGenerator[DataKnobsRegistryAdapter, None]:
    """Yield a fresh ``DataKnobsRegistryAdapter`` backed by a unique PG table.

    Injects the database explicitly so the adapter does NOT own its
    lifecycle — the ``make_postgres_test_db`` fixture handles the per-
    test table drop on teardown.  This matches the production pattern
    where the database connection is shared across multiple consumers.
    """
    for pg in make_postgres_test_db("test_bot_registry_"):
        db = AsyncPostgresDatabase({
            "host": pg["host"],
            "port": pg["port"],
            "database": pg["database"],
            "user": pg["user"],
            "password": pg["password"],
            "table": pg["table"],
        })
        await db.connect()
        adapter = DataKnobsRegistryAdapter(database=db)
        await adapter.initialize()
        try:
            yield adapter
        finally:
            await adapter.close()
            await db.close()


class TestDataKnobsRegistryAdapterPostgres:
    """End-to-end metadata-column routing on PostgreSQL."""

    @pytest.mark.asyncio
    async def test_multi_tenant_filter_metadata(
        self, postgres_registry_adapter: DataKnobsRegistryAdapter,
    ) -> None:
        """Reproducer for the canonical multi-tenant filter case.

        Three bots, two tenants: ``list_active`` with ``filter_metadata``
        returns exactly the matching tenant's bots — and the filter is
        pushed into the JSONB column, not evaluated in Python.

        Against earlier adapter implementations this test would have
        returned an empty list for any ``filter_metadata`` query
        because every row's metadata column was empty.
        """
        await postgres_registry_adapter.register(
            "bot-a", {"llm": "echo"}, metadata={"tenant_id": "t1"}
        )
        await postgres_registry_adapter.register(
            "bot-b", {"llm": "echo"}, metadata={"tenant_id": "t2"}
        )
        await postgres_registry_adapter.register(
            "bot-c", {"llm": "echo"}, metadata={"tenant_id": "t1"}
        )

        t1_bots = await postgres_registry_adapter.list_active(
            filter_metadata={"tenant_id": "t1"}
        )
        assert {b.bot_id for b in t1_bots} == {"bot-a", "bot-c"}

        t2_bots = await postgres_registry_adapter.list_active(
            filter_metadata={"tenant_id": "t2"}
        )
        assert {b.bot_id for b in t2_bots} == {"bot-b"}

    @pytest.mark.asyncio
    async def test_metadata_round_trip_postgres(
        self, postgres_registry_adapter: DataKnobsRegistryAdapter,
    ) -> None:
        """``register(..., metadata=...)`` round-trips via the JSONB column.

        ``get`` reads via the deserializer, which reads from
        ``record.metadata`` — so this test will only pass if the
        metadata channel was populated on the write side.
        """
        await postgres_registry_adapter.register(
            "round-trip-bot",
            {"llm": "echo"},
            metadata={
                "tenant_id": "acme",
                "audit": {"created_by": "alice"},
                "correlation_id": "corr-42",
            },
        )

        retrieved = await postgres_registry_adapter.get("round-trip-bot")
        assert retrieved is not None
        assert retrieved.metadata == {
            "tenant_id": "acme",
            "audit": {"created_by": "alice"},
            "correlation_id": "corr-42",
        }

    @pytest.mark.asyncio
    async def test_combined_status_and_metadata_filter(
        self, postgres_registry_adapter: DataKnobsRegistryAdapter,
    ) -> None:
        """``list_active`` AND-combines ``status='active'`` with ``filter_metadata``.

        The data-column filter and the metadata-column filter must both
        be pushed into the SQL WHERE clause and combined.  Inactive
        rows matching the metadata filter must NOT be returned.
        """
        await postgres_registry_adapter.register(
            "active-acme", {}, metadata={"tenant_id": "acme"}
        )
        await postgres_registry_adapter.register(
            "inactive-acme",
            {},
            status="inactive",
            metadata={"tenant_id": "acme"},
        )
        await postgres_registry_adapter.register(
            "active-globex", {}, metadata={"tenant_id": "globex"}
        )

        active_acme = await postgres_registry_adapter.list_active(
            filter_metadata={"tenant_id": "acme"}
        )
        assert {b.bot_id for b in active_acme} == {"active-acme"}

    @pytest.mark.asyncio
    async def test_count_with_metadata_filter(
        self, postgres_registry_adapter: DataKnobsRegistryAdapter,
    ) -> None:
        """``count(filter_metadata=...)`` returns the matching active row count.

        Routes through ``AsyncKeyedRecordStore.count`` which dispatches
        to the backend's ``count(query)`` method, so when Postgres ships
        a ``SELECT COUNT(*) WHERE`` pushdown, every consumer benefits
        without code changes.
        """
        await postgres_registry_adapter.register(
            "a", {}, metadata={"tenant_id": "t1"}
        )
        await postgres_registry_adapter.register(
            "b", {}, metadata={"tenant_id": "t1"}
        )
        await postgres_registry_adapter.register(
            "c", {}, metadata={"tenant_id": "t2"}
        )
        await postgres_registry_adapter.register(
            "d",
            {},
            status="inactive",
            metadata={"tenant_id": "t1"},
        )

        assert (
            await postgres_registry_adapter.count(filter_metadata={"tenant_id": "t1"})
            == 2
        )
        assert (
            await postgres_registry_adapter.count(filter_metadata={"tenant_id": "t2"})
            == 1
        )
        assert await postgres_registry_adapter.count() == 3

    @pytest.mark.asyncio
    async def test_empty_filter_metadata_is_no_filter_postgres(
        self, postgres_registry_adapter: DataKnobsRegistryAdapter,
    ) -> None:
        """``filter_metadata={}`` is equivalent to ``filter_metadata=None`` on PG too."""
        await postgres_registry_adapter.register(
            "a", {}, metadata={"tenant_id": "t1"}
        )
        await postgres_registry_adapter.register(
            "b", {}, metadata={"tenant_id": "t2"}
        )

        with_empty = await postgres_registry_adapter.list_active(filter_metadata={})
        with_none = await postgres_registry_adapter.list_active()
        assert {r.bot_id for r in with_empty} == {r.bot_id for r in with_none}
        assert len(with_empty) == 2

    @pytest.mark.asyncio
    async def test_metadata_persisted_to_jsonb_column(
        self, postgres_registry_adapter: DataKnobsRegistryAdapter,
    ) -> None:
        """Read the raw record back: metadata is in ``record.metadata``, not ``record.data``.

        This is the structural pin that the migration's
        ``AsyncKeyedRecordStore[Registration]`` enforces.  A regression
        that routed metadata into the data column would break this
        assertion immediately.
        """
        await postgres_registry_adapter.register(
            "struct-bot",
            {"llm": "echo"},
            metadata={"tenant_id": "acme", "correlation_id": "corr-1"},
        )

        raw = await postgres_registry_adapter._db.read("struct-bot")
        assert raw is not None
        assert raw.metadata == {
            "tenant_id": "acme",
            "correlation_id": "corr-1",
        }
        # Structural fields stay in data.
        assert raw.data["bot_id"] == "struct-bot"
        assert raw.data["status"] == "active"
        # Metadata fields are NOT duplicated into the data column.
        assert "tenant_id" not in raw.data
        assert "correlation_id" not in raw.data
