"""Tests for auto_create_table opt-out on SQLite backends."""

import pytest

from dataknobs_data import Record
from dataknobs_data.backends.sqlite import SyncSQLiteDatabase
from dataknobs_data.backends.sqlite_async import AsyncSQLiteDatabase


class TestSyncSQLiteAutoCreateTable:
    def test_default_creates_table(self):
        """auto_create_table defaults to True; connect() creates the table."""
        db = SyncSQLiteDatabase({"path": ":memory:", "table": "records"})
        db.connect()
        record_id = db.create(Record({"x": 1}))
        assert record_id
        db.close()

    def test_disabled_raises_when_table_missing(self, tmp_path):
        """auto_create_table=False raises clearly when table is absent."""
        path = str(tmp_path / "missing.db")
        db = SyncSQLiteDatabase({
            "path": path,
            "table": "records",
            "auto_create_table": False,
        })
        with pytest.raises(RuntimeError, match="auto_create_table is disabled"):
            db.connect()

    def test_disabled_succeeds_when_table_present(self, tmp_path):
        """auto_create_table=False is the happy path when the table exists."""
        path = str(tmp_path / "present.db")
        bootstrap = SyncSQLiteDatabase({"path": path, "table": "records"})
        bootstrap.connect()
        bootstrap.close()

        db = SyncSQLiteDatabase({
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
        path = str(tmp_path / "string.db")
        db = SyncSQLiteDatabase({
            "path": path,
            "table": "records",
            "auto_create_table": "false",
        })
        with pytest.raises(RuntimeError, match="auto_create_table is disabled"):
            db.connect()


class TestAsyncSQLiteAutoCreateTable:
    @pytest.mark.asyncio
    async def test_default_creates_table(self):
        """auto_create_table defaults to True; connect() creates the table."""
        db = AsyncSQLiteDatabase({"path": ":memory:", "table": "records"})
        await db.connect()
        record_id = await db.create(Record({"x": 1}))
        assert record_id
        await db.close()

    @pytest.mark.asyncio
    async def test_disabled_raises_when_table_missing(self, tmp_path):
        """auto_create_table=False raises clearly when table is absent."""
        path = str(tmp_path / "missing_async.db")
        db = AsyncSQLiteDatabase({
            "path": path,
            "table": "records",
            "auto_create_table": False,
        })
        with pytest.raises(RuntimeError, match="auto_create_table is disabled"):
            await db.connect()

    @pytest.mark.asyncio
    async def test_disabled_succeeds_when_table_present(self, tmp_path):
        """auto_create_table=False is the happy path when the table exists."""
        path = str(tmp_path / "present_async.db")
        bootstrap = AsyncSQLiteDatabase({"path": path, "table": "records"})
        await bootstrap.connect()
        await bootstrap.close()

        db = AsyncSQLiteDatabase({
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
        path = str(tmp_path / "string_async.db")
        db = AsyncSQLiteDatabase({
            "path": path,
            "table": "records",
            "auto_create_table": "false",
        })
        with pytest.raises(RuntimeError, match="auto_create_table is disabled"):
            await db.connect()
