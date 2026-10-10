"""A helpdesk in a multi-tenant application, in real native columns, on every engine.

The data is authored for these tests. Two tenants, ``acme`` and ``zenith``,
share three tables: ``tickets``, ``orders``, and the ``categories`` tickets are
filed under. One tenant's reader must see only its own rows, which is what a
scope is for. Each table also holds a column the reader is never told about.

:func:`helpdesk_schema` creates the tables in a scratch Postgres schema dropped
afterwards, and :func:`helpdesk_files` writes them to a SQLite file and a DuckDB
file. :class:`Helpdesk` is one engine's copy, and makes the ``database:`` block
a reader of one table would write. :class:`Store` opens a backend over one
table, from configuration alone, as either twin, and gives both twins one
synchronous surface so a test is written once: the async twin runs on a loop of
its own.
"""

from __future__ import annotations

import asyncio
import sqlite3
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from _native_tables import Table, _insert, _postgres_schema, _sqlite_value
from dataknobs_common.testing import requires_postgres

from dataknobs_common.async_iter import aclosing_iter
from dataknobs_data.factory import AsyncDatabaseFactory, DatabaseFactory
from dataknobs_data.query import Query
from dataknobs_data.records import Record
from dataknobs_data.streaming import StreamConfig

ACME, ZENITH = "acme", "zenith"

#: The two classes of one engine, as the ``twin`` parameter names them.
TWINS = ["async", "sync"]

#: The engines holding the helpdesk, as a module-scoped ``helpdesk`` fixture's params.
ENGINES = ["sqlite", "duckdb", pytest.param("postgres", marks=requires_postgres)]


def _id(n: int) -> uuid.UUID:
    """A deterministic id, so assertions can name rows rather than rediscover them."""
    return uuid.UUID(int=n)


HARDWARE, LAPTOPS, PRINTERS, SOFTWARE = _id(0x101), _id(0x102), _id(0x103), _id(0x104)
BILLING = _id(0x201)
T1, T2, T3, T4 = _id(0x1001), _id(0x1002), _id(0x1003), _id(0x1004)
T5, T6 = _id(0x2001), _id(0x2002)
O1, O2 = _id(0x3001), _id(0x3002)


def _cols(**columns: str | tuple[str, str, str]) -> dict[str, dict[str, str]]:
    """Each column's SQL type: one for every engine, or ``(postgres, duckdb, sqlite)``.

    SQLite has no array or uuid type: it holds a list as its JSON text and a
    uuid as its string.
    """
    return {
        name: dict(
            zip(
                ("postgres", "duckdb", "sqlite"),
                (t, t, t) if isinstance(t, str) else t,
                strict=True,
            )
        )
        for name, t in columns.items()
    }


TEXT = ("TEXT", "VARCHAR", "TEXT")
UUID_KEY = ("UUID PRIMARY KEY", "UUID PRIMARY KEY", "TEXT PRIMARY KEY")
UUID_REF = ("UUID", "UUID", "TEXT")
TEXT_LIST = ("TEXT[] NOT NULL DEFAULT '{}'", "VARCHAR[] NOT NULL", "TEXT NOT NULL")


CATEGORIES = Table(
    "categories",
    _cols(
        id=UUID_KEY,
        tenant_id="TEXT NOT NULL",
        name="TEXT NOT NULL",
        parent_id=UUID_REF,
        aliases=TEXT_LIST,
        owner_email=TEXT,
    ),
    [
        (HARDWARE, ACME, "Hardware", None, ["hw"], "it@acme.example"),
        (LAPTOPS, ACME, "Laptops", HARDWARE, ["notebooks", "portables"], "it@acme.example"),
        (PRINTERS, ACME, "Printers", HARDWARE, [], "it@acme.example"),
        (SOFTWARE, ACME, "Software", None, ["apps"], "it@acme.example"),
        (BILLING, ZENITH, "Billing", None, ["invoices"], "finance@zenith.example"),
    ],
)

#: Priority 1 is the most urgent. ``internal_note`` is never declared.
TICKETS = Table(
    "tickets",
    _cols(
        id=UUID_KEY,
        tenant_id="TEXT NOT NULL",
        number="INTEGER NOT NULL",
        subject="TEXT NOT NULL",
        status="TEXT NOT NULL",
        priority="INTEGER NOT NULL",
        opened_at="TIMESTAMPTZ NOT NULL",
        category_id=UUID_REF,
        tags=TEXT_LIST,
        internal_note=TEXT,
    ),
    [
        (T1, ACME, 101, "Laptop will not boot", "open", 2,
         datetime(2026, 1, 5, 9, 0, tzinfo=UTC), LAPTOPS, ["boot", "hardware"], "n"),
        (T2, ACME, 102, "Printer jams on duplex", "closed", 3,
         datetime(2026, 1, 9, 14, 30, tzinfo=UTC), PRINTERS, [], "n"),
        (T3, ACME, 103, "Licence key rejected", "open", 4,
         datetime(2026, 2, 2, 8, 15, tzinfo=UTC), SOFTWARE, ["licensing"], "n"),
        (T4, ACME, 104, "Request a second monitor", "pending", 5,
         datetime(2026, 2, 20, 11, 0, tzinfo=UTC), None, [], "n"),
        (T5, ZENITH, 201, "Invoice total is wrong", "open", 1,
         datetime(2026, 1, 7, 10, 0, tzinfo=UTC), BILLING, [], "n"),
        (T6, ZENITH, 202, "Refund not received", "closed", 2,
         datetime(2026, 1, 12, 16, 45, tzinfo=UTC), BILLING, [], "n"),
    ],
)  # fmt: skip

ORDERS = Table(
    "orders",
    _cols(
        order_id=UUID_KEY,
        tenant_id="TEXT NOT NULL",
        placed_at="TIMESTAMPTZ NOT NULL",
        quantity="INTEGER NOT NULL",
    ),
    [
        (O1, ACME, datetime(2026, 1, 5, 9, 0, tzinfo=UTC), 3),
        (O2, ACME, datetime(2026, 2, 10, 15, 30, tzinfo=UTC), 5),
    ],
)

UUID_COLUMN = {"type": "string", "metadata": {"sql_type": "uuid"}}
ZONED_COLUMN = {"type": "datetime", "metadata": {"sql_type": "timestamptz"}}

#: The columns a reader of ``tickets`` is told about.
TICKET_FIELDS: dict[str, object] = {
    "id": UUID_COLUMN,
    "tenant_id": "string",
    "number": "integer",
    "subject": "string",
    "status": "string",
    "priority": "integer",
    "opened_at": ZONED_COLUMN,
    "category_id": UUID_COLUMN,
    "tags": "json",
}

ORDER_FIELDS: dict[str, object] = {
    "order_id": UUID_COLUMN,
    "tenant_id": "string",
    "placed_at": ZONED_COLUMN,
    "quantity": "integer",
}

CATEGORY_FIELDS: dict[str, object] = {
    "id": UUID_COLUMN,
    "tenant_id": "string",
    "name": "string",
    "parent_id": UUID_COLUMN,
    "aliases": "json",
}

ACME_ONLY = [{"field": "tenant_id", "operator": "=", "value": ACME}]


HELPDESK = (CATEGORIES, TICKETS, ORDERS)


@contextmanager
def helpdesk_schema(params: dict[str, Any]) -> Iterator[str]:
    """A scratch schema holding the populated helpdesk tables, dropped afterwards."""
    with _postgres_schema(params, HELPDESK) as schema:
        yield schema


def write_sqlite_file(path: Path, tables: tuple[Table, ...] = HELPDESK) -> Path:
    """A SQLite file holding ``tables``, written and closed: somebody else's file."""
    conn = sqlite3.connect(path)
    try:
        for table in tables:
            conn.execute(table.ddl("sqlite", f'"{table.name}"'))
            conn.executemany(
                _insert(table, f'"{table.name}"', "?"),
                [[_sqlite_value(v) for v in row] for row in table.rows],
            )
        conn.commit()
    finally:
        conn.close()
    return path


def write_duckdb_file(path: Path, tables: tuple[Table, ...] = HELPDESK) -> Path:
    """A DuckDB file holding ``tables``, written and closed: somebody else's file.

    Closed because DuckDB refuses, within one process, a read-only connection
    to a file a read-write connection holds open.
    """
    duckdb = pytest.importorskip("duckdb")
    conn = duckdb.connect(str(path))
    try:
        for table in tables:
            conn.execute(table.ddl("duckdb", f'"{table.name}"'))
            conn.executemany(_insert(table, f'"{table.name}"', "?"), [list(r) for r in table.rows])
    finally:
        conn.close()
    return path


@dataclass(frozen=True)
class Helpdesk:
    """One engine's copy of the helpdesk, and the ``database:`` block over one of its tables."""

    engine: str
    #: Connection keys: a Postgres server and scratch schema, or a file's path.
    location: dict[str, Any]

    def config(self, table: str, fields: dict[str, object], **overrides: object) -> dict[str, Any]:
        """The configuration a registry's ``database:`` block would carry for one table."""
        config: dict[str, Any] = {
            "backend": self.engine,
            **self.location,
            "table": table,
            "layout": "native",
            "id_column": "id",
            "schema": {"fields": fields},
            "scope": ACME_ONLY,
        }
        config.update(overrides)
        return config


@contextmanager
def helpdesk_on(request: pytest.FixtureRequest, engine: str) -> Iterator[Helpdesk]:
    """The helpdesk on ``engine``: a scratch Postgres schema, or files in a fresh directory."""
    if engine == "postgres":
        request.getfixturevalue("ensure_postgres_ready")
        params = request.getfixturevalue("postgres_connection_params")
        with helpdesk_schema(params) as schema:
            yield Helpdesk(engine, _postgres_location(params, schema))
        return
    directory = request.getfixturevalue("tmp_path_factory").mktemp(f"helpdesk-{engine}")
    if engine == "sqlite":
        path = write_sqlite_file(directory / "helpdesk.db")
    else:
        path = write_duckdb_file(directory / "helpdesk.duckdb")
    yield Helpdesk(engine, {"path": str(path)})


def _postgres_location(params: dict[str, Any], pg_schema: str) -> dict[str, Any]:
    return {
        "host": params["host"],
        "port": params["port"],
        "database": params["database"],
        "user": params["user"],
        "password": params["password"],
        "schema_name": pg_schema,
    }


def native_config(
    params: dict[str, Any],
    pg_schema: str,
    table: str,
    fields: dict[str, object],
    **overrides: object,
) -> dict[str, Any]:
    """The configuration a registry's ``database:`` block would carry for one table."""
    return Helpdesk("postgres", _postgres_location(params, pg_schema)).config(
        table, fields, **overrides
    )


class Store:
    """One twin over one table, behind one synchronous surface.

    The async twin runs on a loop owned by this object, so a test reads the
    same either way and neither twin's calls block a loop that anything else
    is on.
    """

    def __init__(self, twin: str, config: dict[str, Any]) -> None:
        self.twin = twin
        self._loop: asyncio.AbstractEventLoop | None = None
        if twin == "async":
            self._loop = asyncio.new_event_loop()
            try:
                self.db: Any = self._loop.run_until_complete(_create_async(config))
            except BaseException:
                self._close_loop()
                raise
        else:
            self.db = DatabaseFactory().create(**config)
            self.db.connect()

    def call(self, method: str, *args: Any) -> Any:
        result = getattr(self.db, method)(*args)
        if self._loop is not None:
            return self._loop.run_until_complete(result)
        return result

    def search(self, query: Any = None) -> list[Record]:
        return self.call("search", query if query is not None else Query())

    def read(self, record_id: str) -> Record | None:
        return self.call("read", record_id)

    def exists(self, record_id: str) -> bool:
        return self.call("exists", record_id)

    def count(self, query: Query | None = None) -> int:
        return self.call("count", query)

    def stream(
        self,
        query: Query,
        config: StreamConfig,
        *,
        write_after: int | None = None,
        write: Callable[[], object] | None = None,
    ) -> list[Record]:
        """The records ``stream_read`` returns, ``write()`` run once ``write_after`` are read."""
        if self._loop is None:
            seen: list[Record] = []
            for record in self.db.stream_read(query, config):
                seen.append(record)
                if write is not None and len(seen) == write_after:
                    write()
            return seen

        async def collect() -> list[Record]:
            seen: list[Record] = []
            async with aclosing_iter(self.db.stream_read(query, config)) as records:
                async for record in records:
                    seen.append(record)
                    if write is not None and len(seen) == write_after:
                        write()
            return seen

        return self._loop.run_until_complete(collect())

    def close(self) -> None:
        self.call("close")
        self._close_loop()

    def _close_loop(self) -> None:
        """End the loop as ``asyncio.run`` would, its default executor's threads included."""
        if self._loop is not None:
            self._loop.run_until_complete(self._loop.shutdown_default_executor())
            self._loop.close()


async def _create_async(config: dict[str, Any]) -> Any:
    db = AsyncDatabaseFactory().create(**config)
    await db.connect()
    return db


@contextmanager
def opened(twin: str, config: dict[str, Any]) -> Iterator[Store]:
    store = Store(twin, config)
    try:
        yield store
    finally:
        store.close()


def built(twin: str, config: dict[str, Any]) -> Any:
    """Construct the backend without connecting: what construction alone refuses."""
    factory: Callable[..., Any] = (
        AsyncDatabaseFactory().create if twin == "async" else DatabaseFactory().create
    )
    return factory(**config)


def ids(records: list[Record]) -> set[str]:
    return {str(record.storage_id) for record in records}


def named(*rows: object) -> set[str]:
    return {str(row) for row in rows}
