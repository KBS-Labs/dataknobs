"""Integration tests for auto_create_table opt-out on Postgres backends.

Requires a running PostgreSQL instance. Fixtures are provided by the
``dataknobs_common_postgres`` pytest11 plugin (``postgres_test_db`` from
the local conftest wraps ``make_postgres_test_db``).
"""

import pytest
from dataknobs_common.testing import requires_postgres

from dataknobs_data import Record
from dataknobs_data.backends.postgres import AsyncPostgresDatabase, SyncPostgresDatabase

pytestmark = requires_postgres


class TestAsyncPostgresAutoCreateTable:
    @pytest.mark.asyncio
    async def test_default_creates_table(self, postgres_test_db):
        """auto_create_table defaults to True; connect() creates the table."""
        db = AsyncPostgresDatabase(postgres_test_db)
        try:
            await db.connect()
            record_id = await db.create(Record({"x": 1}))
            assert record_id
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_disabled_raises_when_table_missing(self, postgres_test_db):
        """auto_create_table=False raises clearly when the table is absent."""
        config = {**postgres_test_db, "auto_create_table": False}
        db = AsyncPostgresDatabase(config)
        try:
            with pytest.raises(RuntimeError, match="auto_create_table is disabled"):
                await db.connect()
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_disabled_succeeds_when_table_present(self, postgres_test_db):
        """auto_create_table=False is the happy path when the table exists."""
        # Pre-create the table using the default (auto_create_table=True).
        bootstrap = AsyncPostgresDatabase(postgres_test_db)
        await bootstrap.connect()
        await bootstrap.close()

        # Now reconnect with the flag off — table already exists.
        config = {**postgres_test_db, "auto_create_table": False}
        db = AsyncPostgresDatabase(config)
        try:
            await db.connect()
            record_id = await db.create(Record({"x": 1}))
            assert record_id
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_string_false_coerced(self, postgres_test_db):
        """YAML/env-delivered 'false' string is correctly coerced."""
        config = {**postgres_test_db, "auto_create_table": "false"}
        db = AsyncPostgresDatabase(config)
        try:
            with pytest.raises(RuntimeError, match="auto_create_table is disabled"):
                await db.connect()
        finally:
            await db.close()


class TestSyncPostgresAutoCreateTable:
    def test_default_creates_table(self, postgres_test_db):
        """auto_create_table defaults to True; connect() creates the table."""
        db = SyncPostgresDatabase(postgres_test_db)
        try:
            db.connect()
            record_id = db.create(Record({"x": 1}))
            assert record_id
        finally:
            db.close()

    def test_disabled_raises_when_table_missing(self, postgres_test_db):
        """auto_create_table=False raises clearly when the table is absent."""
        config = {**postgres_test_db, "auto_create_table": False}
        db = SyncPostgresDatabase(config)
        with pytest.raises(RuntimeError, match="auto_create_table is disabled"):
            db.connect()

    def test_disabled_succeeds_when_table_present(self, postgres_test_db):
        """auto_create_table=False is the happy path when the table exists."""
        bootstrap = SyncPostgresDatabase(postgres_test_db)
        bootstrap.connect()
        bootstrap.close()

        config = {**postgres_test_db, "auto_create_table": False}
        db = SyncPostgresDatabase(config)
        try:
            db.connect()
            record_id = db.create(Record({"x": 1}))
            assert record_id
        finally:
            db.close()

    def test_string_false_coerced(self, postgres_test_db):
        """YAML/env-delivered 'false' string is correctly coerced."""
        config = {**postgres_test_db, "auto_create_table": "false"}
        db = SyncPostgresDatabase(config)
        with pytest.raises(RuntimeError, match="auto_create_table is disabled"):
            db.connect()
