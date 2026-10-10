"""A ``database`` grounded source reads a Postgres table it does not own, in place.

The table holds two tenants' cases and a column the source is never told
about. The source's options carry what a native Postgres config does --
``layout: native``, the key column, a scope -- and its ``schema:`` declares the
columns, which is what the backend reads them from. So the declaration has to
reach the backend when it is built: a native table's columns are its layout,
and a backend built without them has none.

Skipped automatically when PostgreSQL is unavailable via ``@requires_postgres``.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from dataknobs_bots.knowledge.sources.factory import _create_database_source
from dataknobs_bots.reasoning.grounded_config import GroundedSourceConfig
from dataknobs_common.testing import requires_postgres
from dataknobs_data.sources.base import RetrievalIntent

pytestmark = requires_postgres

WIDGET, GADGET, LEDGER = uuid.UUID(int=1), uuid.UUID(int=2), uuid.UUID(int=3)

ROWS = [
    (WIDGET, "acme", "Widget recall", "A widget was recalled.", "hardware", "n"),
    (GADGET, "acme", "Gadget refund", "A gadget was refunded.", "billing", "n"),
    (LEDGER, "zenith", "Widget ledger", "Zenith's widget ledger.", "billing", "n"),
]

#: The columns the source is told about: not ``internal_note``.
FIELDS = [
    {"name": "id", "type": "string", "sql_type": "uuid"},
    {"name": "tenant_id", "type": "string"},
    {"name": "title", "type": "string"},
    {"name": "summary", "type": "text"},
    {"name": "dept", "type": "string", "enum": ["hardware", "billing"]},
]


@pytest.fixture
def cases(
    make_postgres_test_db: Callable[[str], Iterator[dict[str, Any]]],
) -> Iterator[dict[str, Any]]:
    """A populated table this package did not create, dropped afterwards."""
    import psycopg2

    for pg in make_postgres_test_db("test_native_cases_"):
        params = {k: pg[k] for k in ("host", "port", "user", "password", "database")}
        conn = psycopg2.connect(**params)
        try:
            with conn, conn.cursor() as cursor:
                cursor.execute(
                    f"CREATE TABLE public.{pg['table']} ("
                    "id UUID PRIMARY KEY, tenant_id TEXT NOT NULL, title TEXT NOT NULL, "
                    "summary TEXT NOT NULL, dept TEXT NOT NULL, internal_note TEXT)"
                )
                cursor.executemany(
                    f"INSERT INTO public.{pg['table']} VALUES (%s, %s, %s, %s, %s, %s)",
                    [(str(row[0]), *row[1:]) for row in ROWS],
                )
        finally:
            conn.close()
        yield {**params, "table": pg["table"]}


def _config(cases: dict[str, Any], **overrides: Any) -> GroundedSourceConfig:
    options: dict[str, Any] = {
        "backend": "postgres",
        **cases,
        "layout": "native",
        "id_column": "id",
        "scope": [{"field": "tenant_id", "operator": "=", "value": "acme"}],
        "content_field": "summary",
        "text_search_fields": ["title", "summary"],
        "schema": {"fields": FIELDS},
        **overrides,
    }
    return GroundedSourceConfig(name="case_studies", source_type="database", options=options)


async def test_a_source_reads_one_tenants_rows_from_a_native_table(cases: dict[str, Any]) -> None:
    source = await _create_database_source(_config(cases))
    try:
        found = await source.query(RetrievalIntent(text_queries=["Widget"]))
        billing = await source.query(RetrievalIntent(filters={"case_studies": {"dept": "billing"}}))
    finally:
        await source.close()

    assert [r.content for r in found] == ["A widget was recalled."], "zenith's row is out of scope"
    assert [r.content for r in billing] == ["A gadget was refunded."]


async def test_the_filter_schema_is_the_declared_columns(cases: dict[str, Any]) -> None:
    """The extractor is offered what was declared, and the enum, and no undeclared column."""
    source = await _create_database_source(_config(cases))
    await source.close()

    fields = source.get_schema().fields
    assert "internal_note" not in fields
    assert fields["dept"]["enum"] == ["hardware", "billing"]


async def test_a_native_source_over_a_missing_table_creates_none(cases: dict[str, Any]) -> None:
    """Building the source connects it, and a native table that is not there is refused, not made."""
    import psycopg2

    absent = f"{cases['table']}_absent"
    with pytest.raises(RuntimeError, match=absent):
        await _create_database_source(_config(cases, table=absent))

    params = {k: cases[k] for k in ("host", "port", "user", "password", "database")}
    conn = psycopg2.connect(**params)
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT to_regclass(%s)", (f"public.{absent}",))
            assert cursor.fetchone() == (None,)
    finally:
        conn.close()
