"""Integration tests for auto_create_table config option on PostgreSQL backends.

Requires a running PostgreSQL instance and TEST_POSTGRES=true.
"""

from __future__ import annotations

import os
import uuid

import pytest

from dataknobs_data import Record
from dataknobs_data.backends.postgres import AsyncPostgresDatabase, SyncPostgresDatabase

pytestmark = pytest.mark.skipif(
    not os.environ.get("TEST_POSTGRES", "").lower() == "true",
    reason="PostgreSQL tests require TEST_POSTGRES=true and a running PostgreSQL instance",
)


class TestAsyncPostgresAutoCreateTable:
    @pytest.mark.asyncio
    async def test_default_creates_table(self, postgres_test_db):
        """auto_create_table defaults to True; connect() creates the table."""
        config = postgres_test_db.copy()
        config["table"] = f"test_auto_{uuid.uuid4().hex[:8]}"
        db = AsyncPostgresDatabase(config)
        try:
            await db.connect()
            await db.create(Record({"x": 1}))
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_disabled_raises_when_table_missing(self, postgres_test_db):
        """auto_create_table=False raises clearly when the table is absent."""
        config = postgres_test_db.copy()
        config["table"] = f"test_auto_{uuid.uuid4().hex[:8]}"
        config["auto_create_table"] = False
        db = AsyncPostgresDatabase(config)
        try:
            with pytest.raises(RuntimeError, match="auto_create_table is disabled"):
                await db.connect()
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_disabled_succeeds_when_table_present(self, postgres_test_db):
        """auto_create_table=False is a happy path when the table exists."""
        table_name = f"test_auto_{uuid.uuid4().hex[:8]}"

        # Pre-create the table via a separate connection.
        bootstrap_config = postgres_test_db.copy()
        bootstrap_config["table"] = table_name
        bootstrap = AsyncPostgresDatabase(bootstrap_config)
        await bootstrap.connect()
        await bootstrap.close()

        config = postgres_test_db.copy()
        config["table"] = table_name
        config["auto_create_table"] = False
        db = AsyncPostgresDatabase(config)
        try:
            await db.connect()
            await db.create(Record({"x": 1}))
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_string_false_coerced(self, postgres_test_db):
        """YAML/env serialization can deliver 'false' as a string."""
        config = postgres_test_db.copy()
        config["table"] = f"test_auto_{uuid.uuid4().hex[:8]}"
        config["auto_create_table"] = "false"
        db = AsyncPostgresDatabase(config)
        try:
            with pytest.raises(RuntimeError, match="auto_create_table is disabled"):
                await db.connect()
        finally:
            await db.close()


class TestSyncPostgresAutoCreateTable:
    def test_default_creates_table(self, postgres_test_db):
        """auto_create_table defaults to True; connect() creates the table."""
        config = postgres_test_db.copy()
        config["table"] = f"test_auto_{uuid.uuid4().hex[:8]}"
        db = SyncPostgresDatabase(config)
        try:
            db.connect()
            db.create(Record({"x": 1}))
        finally:
            db.close()

    def test_disabled_raises_when_table_missing(self, postgres_test_db):
        """auto_create_table=False raises clearly when the table is absent."""
        config = postgres_test_db.copy()
        config["table"] = f"test_auto_{uuid.uuid4().hex[:8]}"
        config["auto_create_table"] = False
        db = SyncPostgresDatabase(config)
        with pytest.raises(RuntimeError, match="auto_create_table is disabled"):
            db.connect()

    def test_disabled_succeeds_when_table_present(self, postgres_test_db):
        """auto_create_table=False is a happy path when the table exists."""
        table_name = f"test_auto_{uuid.uuid4().hex[:8]}"

        # Pre-create the table via a separate connection.
        bootstrap_config = postgres_test_db.copy()
        bootstrap_config["table"] = table_name
        bootstrap = SyncPostgresDatabase(bootstrap_config)
        bootstrap.connect()
        bootstrap.close()

        config = postgres_test_db.copy()
        config["table"] = table_name
        config["auto_create_table"] = False
        db = SyncPostgresDatabase(config)
        try:
            db.connect()
            db.create(Record({"x": 1}))
        finally:
            db.close()

    def test_string_false_coerced(self, postgres_test_db):
        """YAML/env serialization can deliver 'false' as a string."""
        config = postgres_test_db.copy()
        config["table"] = f"test_auto_{uuid.uuid4().hex[:8]}"
        config["auto_create_table"] = "false"
        db = SyncPostgresDatabase(config)
        with pytest.raises(RuntimeError, match="auto_create_table is disabled"):
            db.connect()
