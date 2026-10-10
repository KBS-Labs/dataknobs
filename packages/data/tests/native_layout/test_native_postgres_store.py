"""The Postgres backend reads a table it does not own, through its own columns.

Both twins, built from configuration alone (``layout: native``), over a
helpdesk table two tenants share. The store reads the columns it declares,
one tenant's rows (its ``scope``), and refuses every write.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import pytest
from _helpdesk import (
    ACME_ONLY,
    LAPTOPS,
    O1,
    O2,
    ORDER_FIELDS,
    T1,
    T2,
    T3,
    T4,
    T5,
    T6,
    TICKET_FIELDS,
    TWINS,
    built,
    helpdesk_schema,
    ids,
    named,
    native_config,
    opened,
)

from dataknobs_common.exceptions import OperationError, ValidationError
from dataknobs_common.testing import requires_postgres
from dataknobs_data.query import Filter, Operator, Query, SortOrder, SortSpec
from dataknobs_data.query_logic import (
    ComplexQuery,
    FilterCondition,
    LogicCondition,
    LogicOperator,
)
from dataknobs_data.records import Record
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


def orders(pg: tuple[dict[str, Any], str]) -> dict[str, Any]:
    params, schema = pg
    return native_config(params, schema, "orders", ORDER_FIELDS, id_column="order_id")


# --- reading native columns --------------------------------------------------------


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


def test_a_row_is_its_declared_columns(pg: tuple[dict[str, Any], str], twin: str) -> None:
    """Declared columns come back as fields; an undeclared one does not come back."""
    with opened(twin, tickets(pg)) as db:
        found = db.search(Query(filters=[Filter("number", Operator.EQ, 101)]))

    assert len(found) == 1
    row = found[0]
    assert row.storage_id == str(T1)
    assert row.get_value("subject") == "Laptop will not boot"
    assert row.get_value("category_id") == str(LAPTOPS), "a UUID column reads as its string"
    assert row.get_value("tags") == ["boot", "hardware"], "an array column reads as a list"
    assert row.get_value("priority") == 2
    assert row.get_value("opened_at") == datetime(2026, 1, 5, 9, 0, tzinfo=UTC)
    assert "internal_note" not in row.to_dict(), "an undeclared column stays unread"


def test_read_and_exists_go_through_the_id_column(
    pg: tuple[dict[str, Any], str], twin: str
) -> None:
    with opened(twin, tickets(pg)) as db:
        record = db.read(str(T3))
        assert record is not None and record.get_value("subject") == "Licence key rejected"
        assert db.exists(str(T2))
        assert db.read("00000000-0000-0000-0000-00000000ffff") is None


def test_a_projection_keeps_the_fields_asked_for(pg: tuple[dict[str, Any], str], twin: str) -> None:
    with opened(twin, tickets(pg)) as db:
        found = db.search(Query(filters=[Filter("number", Operator.EQ, 101)], fields=["subject"]))
    assert [list(record.fields) for record in found] == [["subject"]]
    assert found[0].get_value("subject") == "Laptop will not boot"
    assert found[0].storage_id == str(T1)


def test_the_operators_a_lookup_sends(pg: tuple[dict[str, Any], str], twin: str) -> None:
    """``EQ`` for one row, ``IN`` for several, ``NOT_EXISTS`` for a missing reference."""
    with opened(twin, tickets(pg)) as db:
        many = db.search(Query(filters=[Filter("id", Operator.IN, [str(T2), str(T3)])]))
        uncategorised = db.search(Query(filters=[Filter("category_id", Operator.NOT_EXISTS)]))
        filed = db.search(Query(filters=[Filter("category_id", Operator.EQ, str(LAPTOPS))]))

    assert ids(many) == named(T2, T3)
    assert ids(uncategorised) == named(T4)
    assert ids(filed) == named(T1)


def test_a_value_the_column_cannot_hold_matches_nothing(
    pg: tuple[dict[str, Any], str], twin: str
) -> None:
    """*Order 12* cannot be a UUID, so no UUID column equals it: no row, not an error."""
    with opened(twin, orders(pg)) as db:
        assert db.read("12") is None
        assert not db.exists("not-a-uuid")
        assert ids(
            db.search(Query(filters=[Filter("order_id", Operator.IN, ["12", str(O2)])]))
        ) == named(O2), "an IN keeps the members that can match and drops the ones that cannot"
        assert db.search(Query(filters=[Filter("order_id", Operator.IN, ["x", "y"])])) == []
        assert db.count(Query(filters=[Filter("order_id", Operator.NEQ, "12")])) == 2
        assert db.search(Query(filters=[Filter("quantity", Operator.EQ, "abc")])) == []
        assert ids(db.search(Query(filters=[Filter("quantity", Operator.GTE, 3.5)]))) == named(
            O2
        ), "a fractional bound is not truncated to 3"


def test_an_aware_bound_relates_to_a_zoned_column(
    pg: tuple[dict[str, Any], str], twin: str
) -> None:
    """An aware bound compares with zoned instants; a naive one relates to none of them."""
    after = Filter("placed_at", Operator.GT, datetime(2026, 2, 1, tzinfo=UTC))
    same_instant = Filter("placed_at", Operator.EQ, "2026-01-05T10:00:00+01:00")
    naive = Filter("placed_at", Operator.GT, datetime(2026, 2, 1))
    with opened(twin, orders(pg)) as db:
        assert ids(db.search(Query(filters=[after]))) == named(O2)
        assert ids(db.search(Query(filters=[same_instant]))) == named(O1)
        assert db.search(Query(filters=[naive])) == []


def test_sort_and_limit_are_pushed_down(pg: tuple[dict[str, Any], str], twin: str) -> None:
    with opened(twin, tickets(pg)) as db:
        found = db.search(Query(sort_specs=[SortSpec("priority", SortOrder.DESC)], limit_value=2))
    assert [record.get_value("number") for record in found] == [104, 103]


# --- the scope -----------------------------------------------------------------------


def test_the_scope_holds_on_every_read(pg: tuple[dict[str, Any], str], twin: str) -> None:
    """No path out of the scope: search, read, exists, count and stream all stay inside it."""
    with opened(twin, tickets(pg)) as db:
        assert ids(db.search()) == named(T1, T2, T3, T4)
        assert db.read(str(T5)) is None, "another tenant's ticket is not there to read"
        assert not db.exists(str(T6))
        assert db.count() == 4
        assert db.count(Query(filters=[Filter("category_id", Operator.NOT_EXISTS)])) == 1
        assert ids(db.stream(Query(), StreamConfig(batch_size=2))) == named(T1, T2, T3, T4)


def test_an_or_does_not_escape_the_scope(pg: tuple[dict[str, Any], str], twin: str) -> None:
    """``tenant AND (status OR priority)``, never ``tenant AND status OR priority``."""
    query = ComplexQuery(
        condition=LogicCondition(
            operator=LogicOperator.OR,
            conditions=[
                FilterCondition(Filter("status", Operator.EQ, "open")),
                FilterCondition(Filter("priority", Operator.EQ, 1)),
            ],
        )
    )
    with opened(twin, tickets(pg)) as db:
        assert ids(db.search(query)) == named(T1, T3)


def test_no_scope_reads_the_whole_table(pg: tuple[dict[str, Any], str], twin: str) -> None:
    with opened(twin, tickets(pg, scope=[])) as db:
        assert db.count() == 6


# --- streaming -----------------------------------------------------------------------


def test_stream_read_batches_inside_the_scope(pg: tuple[dict[str, Any], str], twin: str) -> None:
    """Smaller batches than rows, so a second fetch has to happen and stay scoped."""
    with opened(twin, tickets(pg)) as db:
        seen = db.stream(Query(sort_specs=[SortSpec("number")]), StreamConfig(batch_size=2))
    assert [int(record.get_value("number")) for record in seen] == [101, 102, 103, 104]


def test_stream_read_without_a_sort_streams_what_search_returns(
    pg: tuple[dict[str, Any], str], twin: str
) -> None:
    """With no sort a stream promises no order, as ``search`` does, and one
    cursor in one snapshot reads every row once whatever the batch size.
    """
    with opened(twin, tickets(pg)) as db:
        seen = db.stream(Query(), StreamConfig(batch_size=3))
    assert sorted(record.storage_id for record in seen) == sorted(str(t) for t in (T1, T2, T3, T4))


def test_stream_read_honours_a_limit(pg: tuple[dict[str, Any], str], twin: str) -> None:
    with opened(twin, tickets(pg)) as db:
        seen = db.stream(
            Query(sort_specs=[SortSpec("number")], limit_value=3), StreamConfig(batch_size=2)
        )
    assert [int(record.get_value("number")) for record in seen] == [101, 102, 103]


# --- refusals ------------------------------------------------------------------------


def test_an_undeclared_column_is_refused_not_interpolated(
    pg: tuple[dict[str, Any], str], twin: str
) -> None:
    """A column name reaches SQL only if the store declared it."""
    with opened(twin, tickets(pg)) as db:
        for field in ("internal_note", 'subject" OR 1=1 --'):
            with pytest.raises(ValidationError, match="no declared column"):
                db.search(Query(filters=[Filter(field, Operator.EQ, "x")]))
        with pytest.raises(ValidationError, match="no declared column"):
            db.search(Query(sort_specs=[SortSpec("internal_note")]))


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


def test_a_missing_table_is_named_as_one_somebody_else_owns(
    pg: tuple[dict[str, Any], str], twin: str
) -> None:
    """Bug: the refusal told the reader to run their migrations, for a table
    a native store reads and never creates.
    """
    params, schema = pg
    config = native_config(params, schema, "no_such_table", {"id": TICKET_FIELDS["id"]}, scope=[])
    with pytest.raises(RuntimeError) as caught:
        with opened(twin, config):
            pass
    message = str(caught.value)
    assert "no_such_table" in message and "layout: native" in message
    assert "migrations" not in message


def test_a_scope_on_an_undeclared_column_is_refused_at_construction(
    pg: tuple[dict[str, Any], str], twin: str
) -> None:
    with pytest.raises(ValidationError, match="no declared column"):
        built(twin, tickets(pg, scope=[{"field": "internal_note", "operator": "=", "value": "x"}]))


def test_an_id_column_must_be_declared(pg: tuple[dict[str, Any], str], twin: str) -> None:
    with pytest.raises(ValidationError, match="id_column"):
        built(twin, tickets(pg, id_column="internal_note"))


def test_an_unknown_sql_type_is_refused(pg: tuple[dict[str, Any], str], twin: str) -> None:
    """A misspelled type would otherwise switch the guard off without a word."""
    fields = {**TICKET_FIELDS, "id": {"type": "string", "metadata": {"sql_type": "uuidd"}}}
    with pytest.raises(ValidationError, match="uuidd"):
        built(twin, tickets(pg, schema={"fields": fields}))


def test_a_refusal_names_the_backend_and_table(pg: tuple[dict[str, Any], str], twin: str) -> None:
    with pytest.raises(ValidationError, match="'tickets'"):
        built(twin, tickets(pg, id_column="internal_note"))


def test_writes_are_refused(pg: tuple[dict[str, Any], str], twin: str) -> None:
    """Read-only is the contract, stated by raising rather than by a silent no-op."""
    with opened(twin, tickets(pg)) as db:
        with pytest.raises(OperationError, match="read-only"):
            db.call("create", Record({"subject": "Mouse is unresponsive"}))
        with pytest.raises(OperationError, match="read-only"):
            db.call("update", str(T2), Record({"status": "open"}))
        with pytest.raises(OperationError, match="read-only"):
            db.call("delete", str(T2))
        assert db.read(str(T2)) is not None


def json_layout(config: dict[str, Any], **keep: object) -> dict[str, Any]:
    """The same table and connection, read through the JSON layout."""
    plain = {k: v for k, v in config.items() if k not in ("layout", "id_column", "scope", "schema")}
    plain.update(keep)
    return plain


def test_a_native_store_claims_no_conditional_write(
    pg: tuple[dict[str, Any], str], twin: str
) -> None:
    from dataknobs_common.capabilities import Capability

    assert not built(twin, tickets(pg)).supports(Capability.CONDITIONAL_WRITE)
    assert built(twin, json_layout(tickets(pg))).supports(Capability.CONDITIONAL_WRITE)


def test_scope_keys_are_refused_under_the_json_layout(
    pg: tuple[dict[str, Any], str], twin: str
) -> None:
    with pytest.raises(ValidationError, match="scope"):
        built(twin, json_layout(tickets(pg), scope=ACME_ONLY))
