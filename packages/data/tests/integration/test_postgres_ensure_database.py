"""Integration tests for PostgreSQL auto-create database feature.

Requires a running PostgreSQL instance and TEST_POSTGRES=true.
"""

import asyncio
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


def _pg_params() -> dict[str, str | int]:
    """Connection params from env, matching the conftest pattern."""
    if os.path.exists("/.dockerenv") or os.environ.get("DOCKER_CONTAINER"):
        default_host = "postgres"
    else:
        default_host = "localhost"
    return {
        "host": os.environ.get("POSTGRES_HOST", default_host),
        "port": int(os.environ.get("POSTGRES_PORT", 5432)),
        "user": os.environ.get("POSTGRES_USER", "postgres"),
        "password": os.environ.get("POSTGRES_PASSWORD", "postgres"),
    }


def _drop_database(name: str) -> None:
    """Drop a database, terminating active connections first."""
    params = _pg_params()
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


def _database_exists(name: str) -> bool:
    """Check whether a database exists."""
    params = _pg_params()
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


@pytest.fixture
def unique_db_name():
    """Generate a unique test database name and clean up after test."""
    name = f"dk_test_{uuid.uuid4().hex[:8]}"
    yield name
    _drop_database(name)


class TestAsyncEnsureDatabase:
    """Async backend auto-create database tests."""

    def test_creates_database_if_missing(self, unique_db_name: str) -> None:
        assert not _database_exists(unique_db_name)

        params = _pg_params()
        db = AsyncPostgresDatabase(config={
            "host": params["host"],
            "port": params["port"],
            "user": params["user"],
            "password": params["password"],
            "database": unique_db_name,
            "table": "test_records",
        })

        asyncio.get_event_loop().run_until_complete(db.connect())
        try:
            assert _database_exists(unique_db_name)

            # Verify we can actually use the database
            record = Record({"key": "value"})
            record_id = asyncio.get_event_loop().run_until_complete(db.create(record))
            assert record_id is not None
        finally:
            asyncio.get_event_loop().run_until_complete(db.close())

    def test_skips_existing_database(self, unique_db_name: str) -> None:
        """No error when database already exists."""
        # Pre-create the database
        params = _pg_params()
        conn = psycopg2.connect(
            host=params["host"],
            port=params["port"],
            user=params["user"],
            password=params["password"],
            database="postgres",
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        cursor.execute(f'CREATE DATABASE "{unique_db_name}"')
        cursor.close()
        conn.close()

        assert _database_exists(unique_db_name)

        db = AsyncPostgresDatabase(config={
            "host": params["host"],
            "port": params["port"],
            "user": params["user"],
            "password": params["password"],
            "database": unique_db_name,
            "table": "test_records",
        })

        # Should succeed without error
        asyncio.get_event_loop().run_until_complete(db.connect())
        asyncio.get_event_loop().run_until_complete(db.close())

    def test_disabled_fails_on_missing_database(self) -> None:
        """With ensure_database=False, missing database raises."""
        missing_name = f"dk_noexist_{uuid.uuid4().hex[:8]}"
        assert not _database_exists(missing_name)

        params = _pg_params()
        db = AsyncPostgresDatabase(config={
            "host": params["host"],
            "port": params["port"],
            "user": params["user"],
            "password": params["password"],
            "database": missing_name,
            "table": "test_records",
            "ensure_database": False,
        })

        with pytest.raises(Exception):
            asyncio.get_event_loop().run_until_complete(db.connect())

    def test_system_database_not_created(self) -> None:
        """System databases (postgres, template0, template1) are skipped."""
        params = _pg_params()
        db = AsyncPostgresDatabase(config={
            "host": params["host"],
            "port": params["port"],
            "user": params["user"],
            "password": params["password"],
            "database": "postgres",
            "table": "test_system_skip",
        })

        # connect() should succeed — _ensure_database skips 'postgres'
        asyncio.get_event_loop().run_until_complete(db.connect())
        asyncio.get_event_loop().run_until_complete(db.close())


class TestSyncEnsureDatabase:
    """Sync backend auto-create database tests."""

    def test_creates_database_if_missing(self, unique_db_name: str) -> None:
        assert not _database_exists(unique_db_name)

        params = _pg_params()
        db = SyncPostgresDatabase(config={
            "host": params["host"],
            "port": params["port"],
            "user": params["user"],
            "password": params["password"],
            "database": unique_db_name,
            "table": "test_records",
        })

        db.connect()
        try:
            assert _database_exists(unique_db_name)

            record = Record({"key": "value"})
            record_id = db.create(record)
            assert record_id is not None
        finally:
            db.close()

    def test_skips_existing_database(self, unique_db_name: str) -> None:
        """No error when database already exists."""
        params = _pg_params()
        conn = psycopg2.connect(
            host=params["host"],
            port=params["port"],
            user=params["user"],
            password=params["password"],
            database="postgres",
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        cursor.execute(f'CREATE DATABASE "{unique_db_name}"')
        cursor.close()
        conn.close()

        db = SyncPostgresDatabase(config={
            "host": params["host"],
            "port": params["port"],
            "user": params["user"],
            "password": params["password"],
            "database": unique_db_name,
            "table": "test_records",
        })

        db.connect()
        db.close()

    def test_disabled_fails_on_missing_database(self) -> None:
        """With ensure_database=False, missing database raises."""
        missing_name = f"dk_noexist_{uuid.uuid4().hex[:8]}"
        assert not _database_exists(missing_name)

        params = _pg_params()
        db = SyncPostgresDatabase(config={
            "host": params["host"],
            "port": params["port"],
            "user": params["user"],
            "password": params["password"],
            "database": missing_name,
            "table": "test_records",
            "ensure_database": False,
        })

        with pytest.raises(Exception):
            db.connect()

    def test_system_database_not_created(self) -> None:
        """System databases (postgres, template0, template1) are skipped."""
        params = _pg_params()
        db = SyncPostgresDatabase(config={
            "host": params["host"],
            "port": params["port"],
            "user": params["user"],
            "password": params["password"],
            "database": "postgres",
            "table": "test_system_skip",
        })

        db.connect()
        db.close()
