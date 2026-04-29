"""Tests for auto_create_table opt-out on DuckDB backends."""

import pytest

from dataknobs_data import Record
from dataknobs_data.backends.duckdb import AsyncDuckDBDatabase, SyncDuckDBDatabase


class TestSyncDuckDBAutoCreateTable:
    def test_default_creates_table(self):
        """auto_create_table defaults to True; connect() creates the table."""
        db = SyncDuckDBDatabase({"path": ":memory:", "table": "records"})
        db.connect()
        record_id = db.create(Record({"x": 1}))
        assert record_id
        db.close()

    def test_disabled_raises_when_table_missing(self, tmp_path):
        """auto_create_table=False raises clearly when table is absent."""
        path = str(tmp_path / "missing.duckdb")
        db = SyncDuckDBDatabase({
            "path": path,
            "table": "records",
            "auto_create_table": False,
        })
        with pytest.raises(RuntimeError, match="auto_create_table is disabled"):
            db.connect()

    def test_disabled_succeeds_when_table_present(self, tmp_path):
        """auto_create_table=False is the happy path when the table exists."""
        path = str(tmp_path / "present.duckdb")
        bootstrap = SyncDuckDBDatabase({"path": path, "table": "records"})
        bootstrap.connect()
        bootstrap.close()

        db = SyncDuckDBDatabase({
            "path": path,
            "table": "records",
            "auto_create_table": False,
        })
        db.connect()
        record_id = db.create(Record({"x": 1}))
        assert record_id
        db.close()

    def test_string_false_coerced(self, tmp_path):
        """YAML/env-delivered 'false' string is correctly coerced."""
        path = str(tmp_path / "string.duckdb")
        db = SyncDuckDBDatabase({
            "path": path,
            "table": "records",
            "auto_create_table": "false",
        })
        with pytest.raises(RuntimeError, match="auto_create_table is disabled"):
            db.connect()


class TestAsyncDuckDBAutoCreateTable:
    @pytest.mark.asyncio
    async def test_default_creates_table(self):
        """auto_create_table defaults to True; connect() creates the table."""
        db = AsyncDuckDBDatabase({"path": ":memory:", "table": "records"})
        await db.connect()
        record_id = await db.create(Record({"x": 1}))
        assert record_id
        await db.close()

    @pytest.mark.asyncio
    async def test_disabled_raises_when_table_missing(self, tmp_path):
        """auto_create_table=False raises clearly when table is absent."""
        path = str(tmp_path / "missing_async.duckdb")
        db = AsyncDuckDBDatabase({
            "path": path,
            "table": "records",
            "auto_create_table": False,
        })
        with pytest.raises(RuntimeError, match="auto_create_table is disabled"):
            await db.connect()

    @pytest.mark.asyncio
    async def test_disabled_succeeds_when_table_present(self, tmp_path):
        """auto_create_table=False is the happy path when the table exists."""
        path = str(tmp_path / "present_async.duckdb")
        bootstrap = AsyncDuckDBDatabase({"path": path, "table": "records"})
        await bootstrap.connect()
        await bootstrap.close()

        db = AsyncDuckDBDatabase({
            "path": path,
            "table": "records",
            "auto_create_table": False,
        })
        await db.connect()
        record_id = await db.create(Record({"x": 1}))
        assert record_id
        await db.close()

    @pytest.mark.asyncio
    async def test_string_false_coerced(self, tmp_path):
        """YAML/env-delivered 'false' string is correctly coerced."""
        path = str(tmp_path / "string_async.duckdb")
        db = AsyncDuckDBDatabase({
            "path": path,
            "table": "records",
            "auto_create_table": "false",
        })
        with pytest.raises(RuntimeError, match="auto_create_table is disabled"):
            await db.connect()
