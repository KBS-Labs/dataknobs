"""Integration tests for PostgreSQL auto-create database feature.

Requires a running PostgreSQL instance and TEST_POSTGRES=true.
"""

import os
import uuid

import psycopg2
import pytest
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

from dataknobs_data import Record
from dataknobs_data.backends.postgres import AsyncPostgresDatabase, SyncPostgresDatabase

pytestmark = pytest.mark.skipif(
    not os.environ.get("TEST_POSTGRES", "").lower() == "true",
    reason="PostgreSQL tests require TEST_POSTGRES=true and a running PostgreSQL instance",
)


def _drop_database(params: dict, name: str) -> None:
    """Drop a database, terminating active connections first."""
    conn = psycopg2.connect(
        host=params["host"],
        port=params["port"],
        user=params["user"],
        password=params["password"],
        database="postgres",
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
            "WHERE datname = %s AND pid <> pg_backend_pid()",
            (name,),
        )
        cursor.execute(f'DROP DATABASE IF EXISTS "{name}"')
    finally:
        cursor.close()
        conn.close()


def _database_exists(params: dict, name: str) -> bool:
    """Check whether a database exists."""
    conn = psycopg2.connect(
        host=params["host"],
        port=params["port"],
        user=params["user"],
        password=params["password"],
        database="postgres",
    )
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,))
        result = cursor.fetchone()
        cursor.close()
        return result is not None
    finally:
        conn.close()


def _pre_create_database(params: dict, name: str) -> None:
    """Create a database for testing the 'already exists' path."""
    conn = psycopg2.connect(
        host=params["host"],
        port=params["port"],
        user=params["user"],
        password=params["password"],
        database="postgres",
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()
    try:
        cursor.execute(f'CREATE DATABASE "{name}"')
    finally:
        cursor.close()
        conn.close()


@pytest.fixture
def unique_db_name(postgres_connection_params):
    """Generate a unique test database name and clean up after test."""
    name = f"dk_test_{uuid.uuid4().hex[:8]}"
    yield name
    _drop_database(postgres_connection_params, name)


class TestAsyncEnsureDatabase:
    """Async backend auto-create database tests."""

    @pytest.mark.asyncio
    async def test_creates_database_if_missing(
        self, unique_db_name: str, postgres_connection_params: dict,
    ) -> None:
        assert not _database_exists(postgres_connection_params, unique_db_name)

        db = AsyncPostgresDatabase(config={
            **postgres_connection_params,
            "database": unique_db_name,
            "table": "test_records",
        })

        await db.connect()
        try:
            assert _database_exists(postgres_connection_params, unique_db_name)

            # Verify we can actually use the database
            record = Record({"key": "value"})
            record_id = await db.create(record)
            assert record_id is not None
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_skips_existing_database(
        self, unique_db_name: str, postgres_connection_params: dict,
    ) -> None:
        """No error when database already exists."""
        _pre_create_database(postgres_connection_params, unique_db_name)
        assert _database_exists(postgres_connection_params, unique_db_name)

        db = AsyncPostgresDatabase(config={
            **postgres_connection_params,
            "database": unique_db_name,
            "table": "test_records",
        })

        await db.connect()
        await db.close()

    @pytest.mark.asyncio
    async def test_disabled_fails_on_missing_database(
        self, postgres_connection_params: dict,
    ) -> None:
        """With ensure_database=False, missing database raises."""
        missing_name = f"dk_noexist_{uuid.uuid4().hex[:8]}"
        assert not _database_exists(postgres_connection_params, missing_name)

        db = AsyncPostgresDatabase(config={
            **postgres_connection_params,
            "database": missing_name,
            "table": "test_records",
            "ensure_database": False,
        })

        with pytest.raises(Exception):
            await db.connect()

    @pytest.mark.asyncio
    async def test_system_database_not_created(
        self, postgres_connection_params: dict,
    ) -> None:
        """System databases (postgres, template0, template1) are skipped."""
        db = AsyncPostgresDatabase(config={
            **postgres_connection_params,
            "database": "postgres",
            "table": "test_system_skip",
        })

        # connect() should succeed — _ensure_database skips 'postgres'
        await db.connect()
        await db.close()


class TestSyncEnsureDatabase:
    """Sync backend auto-create database tests."""

    def test_creates_database_if_missing(
        self, unique_db_name: str, postgres_connection_params: dict,
    ) -> None:
        assert not _database_exists(postgres_connection_params, unique_db_name)

        db = SyncPostgresDatabase(config={
            **postgres_connection_params,
            "database": unique_db_name,
            "table": "test_records",
        })

        db.connect()
        try:
            assert _database_exists(postgres_connection_params, unique_db_name)

            record = Record({"key": "value"})
            record_id = db.create(record)
            assert record_id is not None
        finally:
            db.close()

    def test_skips_existing_database(
        self, unique_db_name: str, postgres_connection_params: dict,
    ) -> None:
        """No error when database already exists."""
        _pre_create_database(postgres_connection_params, unique_db_name)

        db = SyncPostgresDatabase(config={
            **postgres_connection_params,
            "database": unique_db_name,
            "table": "test_records",
        })

        db.connect()
        db.close()

    def test_disabled_fails_on_missing_database(
        self, postgres_connection_params: dict,
    ) -> None:
        """With ensure_database=False, missing database raises."""
        missing_name = f"dk_noexist_{uuid.uuid4().hex[:8]}"
        assert not _database_exists(postgres_connection_params, missing_name)

        db = SyncPostgresDatabase(config={
            **postgres_connection_params,
            "database": missing_name,
            "table": "test_records",
            "ensure_database": False,
        })

        with pytest.raises(Exception):
            db.connect()

    def test_system_database_not_created(
        self, postgres_connection_params: dict,
    ) -> None:
        """System databases (postgres, template0, template1) are skipped."""
        db = SyncPostgresDatabase(config={
            **postgres_connection_params,
            "database": "postgres",
            "table": "test_system_skip",
        })

        db.connect()
        db.close()
