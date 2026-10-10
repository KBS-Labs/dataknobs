"""What only Postgres has, when its backend reads a table it does not own.

A column grant, a materialized view, a schema the role cannot use. What every
engine shares is ``test_native_store.py``, which runs over these twins too.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from _helpdesk import (
    T1,
    T2,
    T3,
    T4,
    TICKET_FIELDS,
    TWINS,
    helpdesk_schema,
    ids,
    named,
    native_config,
    opened,
)

from dataknobs_common.testing import requires_postgres
from dataknobs_data.query import Filter, Operator, Query
from dataknobs_data.streaming import StreamConfig

pytestmark = requires_postgres


@pytest.fixture(scope="module")
def pg(
    ensure_postgres_ready: None, postgres_connection_params: dict[str, Any]
) -> Iterator[tuple[dict[str, Any], str]]:
    with helpdesk_schema(postgres_connection_params) as schema:
        yield postgres_connection_params, schema


@pytest.fixture(params=TWINS)
def twin(request: pytest.FixtureRequest) -> str:
    return str(request.param)


def tickets(pg: tuple[dict[str, Any], str], **overrides: object) -> dict[str, Any]:
    """One tenant's view of the shared ``tickets`` table."""
    params, schema = pg
    return native_config(params, schema, "tickets", TICKET_FIELDS, **overrides)


def test_the_json_layout_still_cannot_read_this_table(
    pg: tuple[dict[str, Any], str], twin: str
) -> None:
    """Without ``layout: native`` the backend reads ``data->>'status'``, which is not there.

    The default is unchanged: a table this package did not create is read only
    when the configuration says how.
    """
    config = json_layout(tickets(pg), auto_create_table=False)
    with opened(twin, config) as db, pytest.raises(Exception, match='column "data" does not exist'):
        db.search(Query(filters=[Filter("status", Operator.EQ, "open")]))


def test_a_reader_granted_only_the_declared_columns_can_read(
    pg: tuple[dict[str, Any], str], twin: str
) -> None:
    """The store selects the columns it declares, never ``*``.

    An owner sharing a table often grants ``SELECT`` on some of its columns and
    not others. The reader here may not see ``internal_note``, so ``SELECT *``
    would be refused, while the declared columns read normally.
    """
    psycopg2 = pytest.importorskip("psycopg2")
    params, schema = pg
    admin = psycopg2.connect(
        host=params["host"], port=params["port"], user=params["user"],
        password=params["password"], dbname=params["database"],
    )  # fmt: skip
    admin.autocommit = True
    with admin.cursor() as cur:
        cur.execute("SELECT rolcreaterole OR rolsuper FROM pg_roles WHERE rolname = current_user")
        if not cur.fetchone()[0]:
            admin.close()
            pytest.skip("the connecting user cannot create a role to grant columns to")
    role, password = f"native_reader_{uuid.uuid4().hex[:8]}", uuid.uuid4().hex
    columns = ", ".join(f'"{name}"' for name in TICKET_FIELDS)
    try:
        with admin.cursor() as cur:
            cur.execute(f"CREATE ROLE {role} LOGIN PASSWORD %s", [password])
            cur.execute(f'GRANT USAGE ON SCHEMA "{schema}" TO {role}')
            cur.execute(f'GRANT SELECT ({columns}) ON "{schema}".tickets TO {role}')
        with opened(twin, tickets(pg, user=role, password=password)) as db:
            assert db.count() == 4
            assert ids(db.search(Query(filters=[Filter("status", Operator.EQ, "open")]))) == (
                named(T1, T3)
            )
            assert ids(db.stream(Query(), StreamConfig(batch_size=2))) == named(T1, T2, T3, T4)
        reader = psycopg2.connect(
            host=params["host"], port=params["port"], user=role, password=password,
            dbname=params["database"],
        )  # fmt: skip
        try:
            with reader.cursor() as cur, pytest.raises(psycopg2.errors.InsufficientPrivilege):
                cur.execute(f'SELECT * FROM "{schema}".tickets')
        finally:
            reader.close()
    finally:
        with admin.cursor() as cur:
            cur.execute(f"DROP OWNED BY {role}")
            cur.execute(f"DROP ROLE {role}")
        admin.close()


def test_a_materialized_view_is_read_in_place(pg: tuple[dict[str, Any], str], twin: str) -> None:
    """Bug: connecting looked the relation up in ``information_schema.tables``,
    which lists tables and views but not materialized views, so a materialized
    view -- a plausible thing to read in place -- was refused as missing.
    """
    psycopg2 = pytest.importorskip("psycopg2")
    params, schema = pg
    view = f"open_tickets_{uuid.uuid4().hex[:8]}"
    admin = psycopg2.connect(
        host=params["host"], port=params["port"], user=params["user"],
        password=params["password"], dbname=params["database"],
    )  # fmt: skip
    admin.autocommit = True
    fields = {name: TICKET_FIELDS[name] for name in ("id", "tenant_id", "status")}
    try:
        with admin.cursor() as cur:
            cur.execute(
                f'CREATE MATERIALIZED VIEW "{schema}"."{view}" AS '
                f'SELECT id, tenant_id, status FROM "{schema}".tickets WHERE status = %s',
                ["open"],
            )
        params_, _ = pg
        config = native_config(params_, schema, view, fields)
        with opened(twin, config) as db:
            assert ids(db.search()) == named(T1, T3)
    finally:
        with admin.cursor() as cur:
            cur.execute(f'DROP MATERIALIZED VIEW IF EXISTS "{schema}"."{view}"')
        admin.close()


def test_a_role_that_cannot_use_the_schema_is_told_so(
    pg: tuple[dict[str, Any], str], twin: str
) -> None:
    """Bug: ``to_regclass`` raises for a schema the role has no ``USAGE`` on, so
    ``connect()`` failed with the driver's bare ``permission denied for
    schema`` rather than the refusal naming what to grant.

    The table is there; what is missing is the grant, and the refusal says so.
    """
    psycopg2 = pytest.importorskip("psycopg2")
    params, schema = pg
    admin = psycopg2.connect(
        host=params["host"], port=params["port"], user=params["user"],
        password=params["password"], dbname=params["database"],
    )  # fmt: skip
    admin.autocommit = True
    with admin.cursor() as cur:
        cur.execute("SELECT rolcreaterole OR rolsuper FROM pg_roles WHERE rolname = current_user")
        if not cur.fetchone()[0]:
            admin.close()
            pytest.skip("the connecting user cannot create a role without USAGE")
    role, password = f"native_outsider_{uuid.uuid4().hex[:8]}", uuid.uuid4().hex
    try:
        with admin.cursor() as cur:
            cur.execute(f"CREATE ROLE {role} LOGIN PASSWORD %s", [password])
        with pytest.raises(RuntimeError) as caught:
            with opened(twin, tickets(pg, user=role, password=password)):
                pass
        message = str(caught.value)
        assert f"USAGE on schema {schema}" in message and "layout: native" in message
        assert "does not exist" not in message
    finally:
        with admin.cursor() as cur:
            cur.execute(f"DROP ROLE {role}")
        admin.close()


def json_layout(config: dict[str, Any], **keep: object) -> dict[str, Any]:
    """The same table and connection, read through the JSON layout."""
    plain = {k: v for k, v in config.items() if k not in ("layout", "id_column", "scope", "schema")}
    plain.update(keep)
    return plain
