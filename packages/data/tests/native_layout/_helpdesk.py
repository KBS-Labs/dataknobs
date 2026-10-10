"""A helpdesk in a multi-tenant application, in real Postgres native columns.

The data is authored for these tests. Two tenants, ``acme`` and ``zenith``,
share three tables: ``tickets``, ``orders``, and the ``categories`` tickets are
filed under. One tenant's reader must see only its own rows, which is what a
scope is for. Each table also holds a column the reader is never told about.

:func:`helpdesk_schema` creates the tables in a scratch schema dropped
afterwards. :class:`Store` opens the Postgres backend over one of them, from
configuration alone, as either twin, and gives both twins one synchronous
surface so a test is written once: the async twin runs on a loop of its own.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

from _native_tables import Table, _postgres_schema

from dataknobs_common.async_iter import aclosing_iter
from dataknobs_data.factory import AsyncDatabaseFactory, DatabaseFactory
from dataknobs_data.query import Query
from dataknobs_data.records import Record
from dataknobs_data.streaming import StreamConfig

ACME, ZENITH = "acme", "zenith"

#: The two Postgres classes, as the ``twin`` parameter names them.
TWINS = ["async", "sync"]


def _id(n: int) -> uuid.UUID:
    """A deterministic id, so assertions can name rows rather than rediscover them."""
    return uuid.UUID(int=n)


HARDWARE, LAPTOPS, PRINTERS, SOFTWARE = _id(0x101), _id(0x102), _id(0x103), _id(0x104)
BILLING = _id(0x201)
T1, T2, T3, T4 = _id(0x1001), _id(0x1002), _id(0x1003), _id(0x1004)
T5, T6 = _id(0x2001), _id(0x2002)
O1, O2 = _id(0x3001), _id(0x3002)


def _pg(**columns: str) -> dict[str, dict[str, str]]:
    return {name: {"postgres": sql_type} for name, sql_type in columns.items()}


CATEGORIES = Table(
    "categories",
    _pg(
        id="UUID PRIMARY KEY",
        tenant_id="TEXT NOT NULL",
        name="TEXT NOT NULL",
        parent_id="UUID",
        aliases="TEXT[] NOT NULL DEFAULT '{}'",
        owner_email="TEXT",
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
    _pg(
        id="UUID PRIMARY KEY",
        tenant_id="TEXT NOT NULL",
        number="INTEGER NOT NULL",
        subject="TEXT NOT NULL",
        status="TEXT NOT NULL",
        priority="INTEGER NOT NULL",
        opened_at="TIMESTAMPTZ NOT NULL",
        category_id="UUID",
        tags="TEXT[] NOT NULL DEFAULT '{}'",
        internal_note="TEXT",
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
    _pg(
        order_id="UUID PRIMARY KEY",
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


@contextmanager
def helpdesk_schema(params: dict[str, Any]) -> Iterator[str]:
    """A scratch schema holding the populated helpdesk tables, dropped afterwards."""
    with _postgres_schema(params, [CATEGORIES, TICKETS, ORDERS]) as schema:
        yield schema


def native_config(
    params: dict[str, Any],
    pg_schema: str,
    table: str,
    fields: dict[str, object],
    **overrides: object,
) -> dict[str, Any]:
    """The configuration a registry's ``database:`` block would carry for one table."""
    config: dict[str, Any] = {
        "backend": "postgres",
        "host": params["host"],
        "port": params["port"],
        "database": params["database"],
        "user": params["user"],
        "password": params["password"],
        "table": table,
        "schema_name": pg_schema,
        "layout": "native",
        "id_column": "id",
        "schema": {"fields": fields},
        "scope": ACME_ONLY,
    }
    config.update(overrides)
    return config


class Store:
    """One Postgres twin over one table, behind one synchronous surface.

    The async twin runs on a loop owned by this object, so a test reads the
    same either way and neither twin's calls block a loop that anything else
    is on.
    """

    def __init__(self, twin: str, config: dict[str, Any]) -> None:
        self.twin = twin
        self._loop: asyncio.AbstractEventLoop | None = None
        if twin == "async":
            self._loop = asyncio.new_event_loop()
            self.db: Any = self._loop.run_until_complete(_create_async(config))
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

    def stream(self, query: Query, config: StreamConfig) -> list[Record]:
        if self._loop is None:
            return list(self.db.stream_read(query, config))

        async def collect() -> list[Record]:
            async with aclosing_iter(self.db.stream_read(query, config)) as records:
                return [record async for record in records]

        return self._loop.run_until_complete(collect())

    def close(self) -> None:
        self.call("close")
        if self._loop is not None:
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
