"""Real tables in native columns, on every engine a native layout renders for.

A test names its tables (:class:`Table`) and gets an :class:`Engine` per
engine: SQLite and DuckDB in process, and PostgreSQL through both asyncpg and
psycopg2 in a scratch schema dropped afterwards. Each engine runs the
statements a builder renders and returns rows by column name.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pytest
from dataknobs_common.testing import requires_postgres

from dataknobs_data.backends.column_layout import ColumnLayout
from dataknobs_data.backends.sql_base import SQLQueryBuilder
from dataknobs_data.backends.sqlite_mixins import register_regexp

#: The engines, as ``pytest.param`` ids for a module-scoped ``engine`` fixture.
ENGINES = [
    "sqlite",
    "duckdb",
    pytest.param("asyncpg", marks=requires_postgres),
    pytest.param("psycopg2", marks=requires_postgres),
]


@dataclass(frozen=True)
class Table:
    """A table: each column's SQL type per dialect, and its rows in column order."""

    name: str
    columns: dict[str, dict[str, str]]
    rows: Sequence[Sequence[Any]]

    def ddl(self, dialect: str, qualified: str) -> str:
        columns = ", ".join(f'"{name}" {types[dialect]}' for name, types in self.columns.items())
        return f"CREATE TABLE {qualified} ({columns})"


@dataclass
class Engine:
    """One engine holding the tables, and how to run a statement on it."""

    name: str
    dialect: str
    param_style: str
    schema_name: str | None
    fetch: Callable[[str, list[Any]], list[dict[str, Any]]]

    def builder(self, table: str, layout: ColumnLayout) -> SQLQueryBuilder:
        return SQLQueryBuilder(
            table,
            schema_name=self.schema_name,
            dialect=self.dialect,
            param_style=self.param_style,
            layout=layout,
        )

    def count(self, sql: str, params: list[Any]) -> int:
        return int(next(iter(self.fetch(sql, params)[0].values())))


def _sqlite_value(value: Any) -> Any:
    """SQLite has no uuid, time or array type: it stores their text, a list as JSON."""
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, list):
        return json.dumps(value)
    return value


def _insert(table: Table, qualified: str, placeholder: str) -> str:
    return f"INSERT INTO {qualified} VALUES ({', '.join([placeholder] * len(table.columns))})"


@contextmanager
def _sqlite(tables: Sequence[Table]) -> Iterator[Engine]:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    # SQLite has no REGEXP function of its own; the backends register this one.
    register_regexp(conn)
    for table in tables:
        conn.execute(table.ddl("sqlite", f'"{table.name}"'))
        conn.executemany(
            _insert(table, f'"{table.name}"', "?"),
            [[_sqlite_value(v) for v in row] for row in table.rows],
        )

    def fetch(sql: str, params: list[Any]) -> list[dict[str, Any]]:
        return [dict(row) for row in conn.execute(sql, params).fetchall()]

    try:
        yield Engine("sqlite", "sqlite", "qmark", None, fetch)
    finally:
        conn.close()


@contextmanager
def _duckdb(tables: Sequence[Table]) -> Iterator[Engine]:
    duckdb = pytest.importorskip("duckdb")
    conn = duckdb.connect()
    for table in tables:
        conn.execute(table.ddl("duckdb", f'"{table.name}"'))
        conn.executemany(_insert(table, f'"{table.name}"', "?"), [list(r) for r in table.rows])

    def fetch(sql: str, params: list[Any]) -> list[dict[str, Any]]:
        cursor = conn.execute(sql, params)
        names = [d[0] for d in cursor.description]
        return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]

    try:
        yield Engine("duckdb", "duckdb", "qmark", None, fetch)
    finally:
        conn.close()


def _psycopg2_connect(params: dict[str, Any]) -> Any:
    import psycopg2

    conn = psycopg2.connect(
        host=params["host"],
        port=params["port"],
        user=params["user"],
        password=params["password"],
        dbname=params["database"],
    )
    conn.autocommit = True
    return conn


@contextmanager
def _postgres_schema(params: dict[str, Any], tables: Sequence[Table]) -> Iterator[str]:
    """A scratch schema holding the tables, dropped afterwards."""
    schema = f"native_{uuid.uuid4().hex[:10]}"
    conn = _psycopg2_connect(params)
    with conn.cursor() as cur:
        cur.execute(f"CREATE SCHEMA {schema}")
        for table in tables:
            qualified = f'{schema}."{table.name}"'
            cur.execute(table.ddl("postgres", qualified))
            for row in table.rows:
                cur.execute(
                    _insert(table, qualified, "%s"),
                    [str(v) if isinstance(v, uuid.UUID) else v for v in row],
                )
    try:
        yield schema
    finally:
        with conn.cursor() as cur:
            cur.execute(f"DROP SCHEMA {schema} CASCADE")
        conn.close()


@contextmanager
def _asyncpg(params: dict[str, Any], tables: Sequence[Table]) -> Iterator[Engine]:
    asyncpg = pytest.importorskip("asyncpg")
    with _postgres_schema(params, tables) as schema:
        loop = asyncio.new_event_loop()
        conn = loop.run_until_complete(
            asyncpg.connect(
                host=params["host"],
                port=params["port"],
                user=params["user"],
                password=params["password"],
                database=params["database"],
            )
        )

        def fetch(sql: str, args: list[Any]) -> list[dict[str, Any]]:
            return [dict(r) for r in loop.run_until_complete(conn.fetch(sql, *args))]

        try:
            yield Engine("asyncpg", "postgres", "numeric", schema, fetch)
        finally:
            loop.run_until_complete(conn.close())
            loop.close()


@contextmanager
def _psycopg2(params: dict[str, Any], tables: Sequence[Table]) -> Iterator[Engine]:
    import psycopg2.extras

    with _postgres_schema(params, tables) as schema:
        conn = _psycopg2_connect(params)

        def fetch(sql: str, args: list[Any]) -> list[dict[str, Any]]:
            named = {f"p{i}": value for i, value in enumerate(args)}
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, named)
                return [dict(r) for r in cur.fetchall()]

        try:
            yield Engine("psycopg2", "postgres", "pyformat", schema, fetch)
        finally:
            conn.close()


def engine_for(request: pytest.FixtureRequest, tables: Sequence[Table]) -> Iterator[Engine]:
    """The engine ``request.param`` names, holding ``tables``. For a fixture to ``yield from``."""
    if request.param == "sqlite":
        manager = _sqlite(tables)
    elif request.param == "duckdb":
        manager = _duckdb(tables)
    else:
        request.getfixturevalue("ensure_postgres_ready")
        params = request.getfixturevalue("postgres_connection_params")
        manager = (_asyncpg if request.param == "asyncpg" else _psycopg2)(params, tables)
    with manager as engine:
        yield engine
