"""A Postgres backend reading a native table refuses every write, by name, before any I/O.

The census below classifies every public method of both twins as a read or a
write. A write is refused before a connection is touched: the backends here
are never connected, so a write that reached the database would fail on the
missing connection instead, and the assertion on the message tells the two
apart. A public method added later fails the census until it is classified.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

import pytest

from dataknobs_common.exceptions import OperationError
from dataknobs_data.backends.postgres import AsyncPostgresDatabase, SyncPostgresDatabase
from dataknobs_data.backends.postgres_mixins import NATIVE_REFUSED
from dataknobs_data.query import Query
from dataknobs_data.records import Record

TWINS = [AsyncPostgresDatabase, SyncPostgresDatabase]

#: What a native table answers: reads, and methods that touch no rows.
ANSWERED = frozenset(
    {
        # reads
        "all", "count", "exists", "get_version", "read", "read_batch", "search",
        "stream_read", "stream_transform", "has_vector_support",
        # lifecycle and configuration
        "close", "connect", "disconnect", "from_backend", "from_components", "from_config",
        "from_config_async", "add_field_schema", "set_schema", "with_schema",
        "accepted_components", "expected_components", "forwardable_components",
        "missing_components", "missing_from", "optional_components", "require_components",
        "set_component", "set_components",
        # capabilities and helpers that run no statement
        "instance_capabilities", "supported_capabilities", "supports", "supports_transactions",
        "get_create_table_sql", "get_table_exists_sql", "get_vector_extraction_sql",
        "handle_connection_error", "handle_query_error", "log_operation",
        "json_to_record", "record_to_json", "record_to_row", "row_to_record",
    }
)  # fmt: skip

CONFIG = {
    "layout": "native",
    "table": "tickets",
    "id_column": "id",
    "schema": {"fields": {"id": "string", "status": "string"}},
}


def _public(cls: type) -> set[str]:
    return {
        name
        for name, value in inspect.getmembers(cls)
        if not name.startswith("_") and callable(value) and not inspect.isclass(value)
    }


@pytest.mark.parametrize("cls", TWINS, ids=lambda c: c.__name__)
def test_every_public_method_is_a_read_or_a_refused_write(cls: type) -> None:
    public = _public(cls)
    unclassified = public - ANSWERED - NATIVE_REFUSED
    assert not unclassified, f"classify as answered or refused: {sorted(unclassified)}"
    assert not ANSWERED & NATIVE_REFUSED
    assert public | {"begin_transaction", "transaction"} >= NATIVE_REFUSED


def _arguments(method: Any) -> list[Any]:
    """A placeholder for each required argument; a refusal never reads them."""
    return [
        None
        for p in inspect.signature(method).parameters.values()
        if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]


def _call(method: Any) -> None:
    result = method(*_arguments(method))
    if inspect.isawaitable(result):
        asyncio.run(_await(result))
    elif inspect.isasyncgen(result):
        asyncio.run(result.__anext__())
    elif hasattr(result, "__aenter__"):
        asyncio.run(_enter(result))
    elif hasattr(result, "__enter__"):
        with result:
            pass
    elif inspect.isgenerator(result):
        next(result)


async def _await(result: Any) -> Any:
    return await result


async def _enter(manager: Any) -> None:
    async with manager:
        pass


@pytest.mark.parametrize("cls", TWINS, ids=lambda c: c.__name__)
def test_every_write_is_refused_before_any_io(cls: type) -> None:
    db = cls(CONFIG)
    for name in sorted(NATIVE_REFUSED & _public(cls)):
        with pytest.raises(OperationError, match=rf"{name}.*'tickets'.*read-only"):
            _call(getattr(db, name))


@pytest.mark.parametrize("cls", TWINS, ids=lambda c: c.__name__)
def test_bulk_embedding_never_reports_a_record_stored(cls: type) -> None:
    """Refused before the caller's embedding function is spent on records nothing may store."""
    stored: list[Any] = []
    embedded: list[list[str]] = []

    def embed(texts: list[str]) -> list[list[float]]:
        embedded.append(texts)
        return [[1.0] for _ in texts]

    db = cls(CONFIG)
    with pytest.raises(OperationError, match=r"bulk_embed_and_store.*read-only"):
        result = db.bulk_embed_and_store(
            [Record({"id": "a", "status": "open"})],
            "status",
            embedding_fn=embed,
            on_stored=stored.append,
        )
        if inspect.isawaitable(result):
            asyncio.run(_await(result))
    assert stored == []
    assert embedded == []


@pytest.mark.parametrize("cls", TWINS, ids=lambda c: c.__name__)
def test_the_json_layout_refuses_nothing(cls: type) -> None:
    """Writes are refused only where the layout cannot write."""
    db = cls({"table": "records"})
    method = db.create
    with pytest.raises(Exception) as caught:
        _call(method)
    assert not isinstance(caught.value, OperationError) or "read-only" not in str(caught.value)


@pytest.mark.parametrize("cls", TWINS, ids=lambda c: c.__name__)
def test_a_changed_schema_rebuilds_the_layout(cls: type) -> None:
    """The layout is the declared columns, so a new declaration is a new layout."""
    from dataknobs_common.exceptions import ValidationError
    from dataknobs_data.fields import FieldType
    from dataknobs_data.schema import DatabaseSchema, FieldSchema
    from dataknobs_data.query import Filter, Operator

    db = cls(CONFIG)
    with pytest.raises(ValidationError, match="no declared column"):
        db.query_builder.build_search_query(Query(filters=[Filter("priority", Operator.EQ, 1)]))

    db.add_field_schema(FieldSchema("priority", FieldType.INTEGER))
    db.query_builder.build_search_query(Query(filters=[Filter("priority", Operator.EQ, 1)]))

    with pytest.raises(ValidationError, match="id_column"):
        db.set_schema(DatabaseSchema.create(status=FieldType.STRING))
