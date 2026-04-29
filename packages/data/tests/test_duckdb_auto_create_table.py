"""Tests for auto_create_table config option on DuckDB backends.

Runs in-process — no external services required.
"""

from __future__ import annotations

import pytest

from dataknobs_data import Record
from dataknobs_data.backends.duckdb import AsyncDuckDBDatabase, SyncDuckDBDatabase


class TestSyncDuckDBAutoCreateTable:
    def test_default_creates_table(self):
        """auto_create_table defaults to True; connect() creates the table."""
        db = SyncDuckDBDatabase({"path": ":memory:", "table": "records"})
        db.connect()
        db.create(Record({"x": 1}))
        db.close()

    def test_disabled_raises_when_table_missing(self, tmp_path):
        """auto_create_table=False raises clearly when the table is absent."""
        path = str(tmp_path / "missing.duckdb")
        db = SyncDuckDBDatabase({
            "path": path,
            "table": "records",
            "auto_create_table": False,
        })
        with pytest.raises(RuntimeError, match="auto_create_table is disabled"):
            db.connect()

    def test_disabled_succeeds_when_table_present(self, tmp_path):
        """auto_create_table=False is a happy path when the table exists."""
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
        db.create(Record({"x": 1}))
        db.close()

    def test_string_false_coerced(self, tmp_path):
        """YAML/env serialization can deliver 'false' as a string."""
        path = str(tmp_path / "string.duckdb")
        db = SyncDuckDBDatabase({
            "path": path,
            "table": "records",
            "auto_create_table": "false",
        })
        with pytest.raises(RuntimeError, match="auto_create_table is disabled"):
            db.connect()

    def test_string_true_coerced(self):
        """String 'true' is accepted as a truthy value."""
        db = SyncDuckDBDatabase({
            "path": ":memory:",
            "table": "records",
            "auto_create_table": "true",
        })
        db.connect()
        db.create(Record({"x": 1}))
        db.close()

    def test_read_only_skips_check(self, tmp_path):
        """read_only=True skips both DDL and existence check."""
        path = str(tmp_path / "readonly.duckdb")
        bootstrap = SyncDuckDBDatabase({"path": path, "table": "records"})
        bootstrap.connect()
        bootstrap.close()

        db = SyncDuckDBDatabase({
            "path": path,
            "table": "records",
            "auto_create_table": False,
            "read_only": True,
        })
        db.connect()
        db.close()


class TestAsyncDuckDBAutoCreateTable:
    @pytest.mark.asyncio
    async def test_default_creates_table(self):
        """auto_create_table defaults to True; connect() creates the table."""
        db = AsyncDuckDBDatabase({"path": ":memory:", "table": "records"})
        await db.connect()
        await db.create(Record({"x": 1}))
        await db.close()

    @pytest.mark.asyncio
    async def test_disabled_raises_when_table_missing(self, tmp_path):
        """auto_create_table=False raises clearly when the table is absent."""
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
        """auto_create_table=False is a happy path when the table exists."""
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
        await db.create(Record({"x": 1}))
        await db.close()

    @pytest.mark.asyncio
    async def test_string_false_coerced(self, tmp_path):
        """YAML/env serialization can deliver 'false' as a string."""
        path = str(tmp_path / "string_async.duckdb")
        db = AsyncDuckDBDatabase({
            "path": path,
            "table": "records",
            "auto_create_table": "false",
        })
        with pytest.raises(RuntimeError, match="auto_create_table is disabled"):
            await db.connect()

    @pytest.mark.asyncio
    async def test_string_true_coerced(self):
        """String 'true' is accepted as a truthy value."""
        db = AsyncDuckDBDatabase({
            "path": ":memory:",
            "table": "records",
            "auto_create_table": "true",
        })
        await db.connect()
        await db.create(Record({"x": 1}))
        await db.close()

    @pytest.mark.asyncio
    async def test_read_only_skips_check(self, tmp_path):
        """read_only=True skips both DDL and existence check."""
        path = str(tmp_path / "readonly_async.duckdb")
        bootstrap = AsyncDuckDBDatabase({"path": path, "table": "records"})
        await bootstrap.connect()
        await bootstrap.close()

        db = AsyncDuckDBDatabase({
            "path": path,
            "table": "records",
            "auto_create_table": False,
            "read_only": True,
        })
        await db.connect()
        await db.close()
