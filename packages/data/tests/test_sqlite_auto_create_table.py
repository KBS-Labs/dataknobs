"""Tests for auto_create_table config option on SQLite backends.

Runs in-process — no external services required.
"""

from __future__ import annotations

import pytest

from dataknobs_data import Record
from dataknobs_data.backends.sqlite import SyncSQLiteDatabase
from dataknobs_data.backends.sqlite_async import AsyncSQLiteDatabase


class TestSyncSQLiteAutoCreateTable:
    def test_default_creates_table(self):
        """auto_create_table defaults to True; connect() creates the table."""
        db = SyncSQLiteDatabase({"path": ":memory:", "table": "records"})
        db.connect()
        db.create(Record({"x": 1}))
        db.close()

    def test_disabled_raises_when_table_missing(self, tmp_path):
        """auto_create_table=False raises clearly when the table is absent."""
        path = str(tmp_path / "missing.db")
        db = SyncSQLiteDatabase({
            "path": path,
            "table": "records",
            "auto_create_table": False,
        })
        with pytest.raises(RuntimeError, match="auto_create_table is disabled"):
            db.connect()

    def test_disabled_succeeds_when_table_present(self, tmp_path):
        """auto_create_table=False is a happy path when the table exists."""
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
        db.create(Record({"x": 1}))
        db.close()

    def test_string_false_coerced(self, tmp_path):
        """YAML/env serialization can deliver 'false' as a string."""
        path = str(tmp_path / "string.db")
        db = SyncSQLiteDatabase({
            "path": path,
            "table": "records",
            "auto_create_table": "false",
        })
        with pytest.raises(RuntimeError, match="auto_create_table is disabled"):
            db.connect()

    def test_string_true_coerced(self):
        """String 'true' is accepted as a truthy value."""
        db = SyncSQLiteDatabase({
            "path": ":memory:",
            "table": "records",
            "auto_create_table": "true",
        })
        db.connect()
        db.create(Record({"x": 1}))
        db.close()


class TestAsyncSQLiteAutoCreateTable:
    @pytest.mark.asyncio
    async def test_default_creates_table(self):
        """auto_create_table defaults to True; connect() creates the table."""
        db = AsyncSQLiteDatabase({"path": ":memory:", "table": "records"})
        await db.connect()
        await db.create(Record({"x": 1}))
        await db.close()

    @pytest.mark.asyncio
    async def test_disabled_raises_when_table_missing(self, tmp_path):
        """auto_create_table=False raises clearly when the table is absent."""
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
        """auto_create_table=False is a happy path when the table exists."""
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
        await db.create(Record({"x": 1}))
        await db.close()

    @pytest.mark.asyncio
    async def test_string_false_coerced(self, tmp_path):
        """YAML/env serialization can deliver 'false' as a string."""
        path = str(tmp_path / "string_async.db")
        db = AsyncSQLiteDatabase({
            "path": path,
            "table": "records",
            "auto_create_table": "false",
        })
        with pytest.raises(RuntimeError, match="auto_create_table is disabled"):
            await db.connect()

    @pytest.mark.asyncio
    async def test_string_true_coerced(self):
        """String 'true' is accepted as a truthy value."""
        db = AsyncSQLiteDatabase({
            "path": ":memory:",
            "table": "records",
            "auto_create_table": "true",
        })
        await db.connect()
        await db.create(Record({"x": 1}))
        await db.close()
