"""A backend reads a table it does not own, through its own columns, on every engine.

Both twins of the SQLite, DuckDB and Postgres backends, built from
configuration alone (``layout: native``), over a helpdesk table two tenants
share. The store reads the columns it declares, one tenant's rows (its
``scope``), and refuses every write. What only one engine has -- a column grant,
a materialized view, a file opened read-only -- is tested beside that engine.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import pytest
from _helpdesk import (
    ACME_ONLY,
    ENGINES,
    LAPTOPS,
    O1,
    O2,
    ORDER_FIELDS,
    T1,
    T2,
    T3,
    T4,
    TICKET_FIELDS,
    TWINS,
    Helpdesk,
    built,
    helpdesk_on,
    ids,
    named,
    opened,
)

from dataknobs_common.capabilities import Capability
from dataknobs_common.exceptions import OperationError, ValidationError
from dataknobs_data.query import Filter, Operator, Query, SortOrder, SortSpec
from dataknobs_data.query_logic import (
    ComplexQuery,
    FilterCondition,
    LogicCondition,
    LogicOperator,
)
from dataknobs_data.records import Record
from dataknobs_data.streaming import StreamConfig


@pytest.fixture(scope="module", params=ENGINES)
def helpdesk(request: pytest.FixtureRequest) -> Iterator[Helpdesk]:
    with helpdesk_on(request, str(request.param)) as desk:
        yield desk


@pytest.fixture(params=TWINS)
def twin(request: pytest.FixtureRequest) -> str:
    return str(request.param)


def tickets(helpdesk: Helpdesk, **overrides: object) -> dict[str, Any]:
    """One tenant's view of the shared ``tickets`` table."""
    return helpdesk.config("tickets", TICKET_FIELDS, **overrides)


def orders(helpdesk: Helpdesk) -> dict[str, Any]:
    return helpdesk.config("orders", ORDER_FIELDS, id_column="order_id")


def json_layout(config: dict[str, Any], **keep: object) -> dict[str, Any]:
    """The same table and connection, read through the JSON layout."""
    plain = {k: v for k, v in config.items() if k not in ("layout", "id_column", "scope", "schema")}
    plain.update(keep)
    return plain


# --- reading native columns --------------------------------------------------------


def test_a_row_is_its_declared_columns(helpdesk: Helpdesk, twin: str) -> None:
    """Declared columns come back as fields; an undeclared one does not come back."""
    with opened(twin, tickets(helpdesk)) as db:
        found = db.search(Query(filters=[Filter("number", Operator.EQ, 101)]))

    assert len(found) == 1
    row = found[0]
    assert row.storage_id == str(T1)
    assert row.get_value("subject") == "Laptop will not boot"
    assert row.get_value("category_id") == str(LAPTOPS), "a UUID column reads as its string"
    # A json column is read as the driver returns it: an array as a list, and
    # on SQLite, which has no array, the JSON text the column holds.
    tags = ["boot", "hardware"] if helpdesk.engine != "sqlite" else '["boot", "hardware"]'
    assert row.get_value("tags") == tags
    assert row.get_value("priority") == 2
    assert row.get_value("opened_at") == datetime(2026, 1, 5, 9, 0, tzinfo=UTC)
    assert "internal_note" not in row.to_dict(), "an undeclared column stays unread"


def test_read_and_exists_go_through_the_id_column(helpdesk: Helpdesk, twin: str) -> None:
    with opened(twin, tickets(helpdesk)) as db:
        record = db.read(str(T3))
        assert record is not None and record.get_value("subject") == "Licence key rejected"
        assert record.storage_id == str(T3)
        assert db.exists(str(T2))
        assert db.read("00000000-0000-0000-0000-00000000ffff") is None


def test_a_projection_keeps_the_fields_asked_for(helpdesk: Helpdesk, twin: str) -> None:
    with opened(twin, tickets(helpdesk)) as db:
        found = db.search(Query(filters=[Filter("number", Operator.EQ, 101)], fields=["subject"]))
    assert [list(record.fields) for record in found] == [["subject"]]
    assert found[0].get_value("subject") == "Laptop will not boot"
    assert found[0].storage_id == str(T1)


def test_the_operators_a_lookup_sends(helpdesk: Helpdesk, twin: str) -> None:
    """``EQ`` for one row, ``IN`` for several, ``NOT_EXISTS`` for a missing reference."""
    with opened(twin, tickets(helpdesk)) as db:
        many = db.search(Query(filters=[Filter("id", Operator.IN, [str(T2), str(T3)])]))
        uncategorised = db.search(Query(filters=[Filter("category_id", Operator.NOT_EXISTS)]))
        filed = db.search(Query(filters=[Filter("category_id", Operator.EQ, str(LAPTOPS))]))

    assert ids(many) == named(T2, T3)
    assert ids(uncategorised) == named(T4)
    assert ids(filed) == named(T1)


def test_a_value_the_column_cannot_hold_matches_nothing(helpdesk: Helpdesk, twin: str) -> None:
    """*Order 12* cannot be a UUID, so no UUID column equals it: no row, not an error."""
    with opened(twin, orders(helpdesk)) as db:
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


def test_an_aware_bound_relates_to_a_zoned_column(helpdesk: Helpdesk, twin: str) -> None:
    """An aware bound compares with zoned instants; a naive one relates to none of them."""
    after = Filter("placed_at", Operator.GT, datetime(2026, 2, 1, tzinfo=UTC))
    same_instant = Filter("placed_at", Operator.EQ, "2026-01-05T10:00:00+01:00")
    naive = Filter("placed_at", Operator.GT, datetime(2026, 2, 1))
    with opened(twin, orders(helpdesk)) as db:
        assert ids(db.search(Query(filters=[after]))) == named(O2)
        assert ids(db.search(Query(filters=[same_instant]))) == named(O1)
        assert db.search(Query(filters=[naive])) == []


def test_sort_and_limit_are_pushed_down(helpdesk: Helpdesk, twin: str) -> None:
    with opened(twin, tickets(helpdesk)) as db:
        found = db.search(Query(sort_specs=[SortSpec("priority", SortOrder.DESC)], limit_value=2))
    assert [record.get_value("number") for record in found] == [104, 103]


# --- the scope -----------------------------------------------------------------------


def test_the_scope_holds_on_every_read(helpdesk: Helpdesk, twin: str) -> None:
    """No path out of the scope: search, read, exists, count and stream all stay inside it.

    ``count()`` with no query is counted through ``_count_all``, which on SQLite
    and DuckDB was a hand-written ``COUNT(*)`` the scope never reached.
    """
    from _helpdesk import T5, T6

    with opened(twin, tickets(helpdesk)) as db:
        assert ids(db.search()) == named(T1, T2, T3, T4)
        assert db.read(str(T5)) is None, "another tenant's ticket is not there to read"
        assert not db.exists(str(T6))
        assert db.count() == 4
        assert db.call("_count_all") == 4
        assert db.count(Query(filters=[Filter("category_id", Operator.NOT_EXISTS)])) == 1
        assert ids(db.stream(Query(), StreamConfig(batch_size=2))) == named(T1, T2, T3, T4)


def test_an_or_does_not_escape_the_scope(helpdesk: Helpdesk, twin: str) -> None:
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
    with opened(twin, tickets(helpdesk)) as db:
        assert ids(db.search(query)) == named(T1, T3)


def test_no_scope_reads_the_whole_table(helpdesk: Helpdesk, twin: str) -> None:
    with opened(twin, tickets(helpdesk, scope=[])) as db:
        assert db.count() == 6
        assert db.call("_count_all") == 6


def test_two_disjoint_scopes_over_one_table_read_disjoint_rows(
    helpdesk: Helpdesk, twin: str
) -> None:
    zenith = [{"field": "tenant_id", "operator": "=", "value": "zenith"}]
    with (
        opened(twin, tickets(helpdesk)) as acme,
        opened(twin, tickets(helpdesk, scope=zenith)) as z,
    ):
        assert ids(acme.search()) & ids(z.search()) == set()
        assert acme.count() + z.count() == 6


# --- streaming -----------------------------------------------------------------------


def test_stream_read_batches_inside_the_scope(helpdesk: Helpdesk, twin: str) -> None:
    """Smaller batches than rows, so a second fetch has to happen and stay scoped."""
    with opened(twin, tickets(helpdesk)) as db:
        seen = db.stream(Query(sort_specs=[SortSpec("number")]), StreamConfig(batch_size=2))
    assert [int(record.get_value("number")) for record in seen] == [101, 102, 103, 104]


def test_stream_read_without_a_sort_reads_every_row_once(helpdesk: Helpdesk, twin: str) -> None:
    """With no sort a stream promises no order, as ``search`` does, and reads
    every row once whatever the batch size.
    """
    with opened(twin, tickets(helpdesk)) as db:
        seen = db.stream(Query(), StreamConfig(batch_size=3))
    assert sorted(record.storage_id for record in seen) == sorted(str(t) for t in (T1, T2, T3, T4))


def test_stream_read_with_a_sort_that_ties_reads_every_row_once(
    helpdesk: Helpdesk, twin: str
) -> None:
    """Two tickets share a status, so the sort alone does not order the pages."""
    with opened(twin, tickets(helpdesk)) as db:
        seen = db.stream(Query(sort_specs=[SortSpec("status")]), StreamConfig(batch_size=1))
    assert sorted(record.storage_id for record in seen) == sorted(str(t) for t in (T1, T2, T3, T4))
    statuses = [record.get_value("status") for record in seen]
    assert statuses == sorted(statuses)


def test_stream_read_honours_a_limit(helpdesk: Helpdesk, twin: str) -> None:
    with opened(twin, tickets(helpdesk)) as db:
        seen = db.stream(
            Query(sort_specs=[SortSpec("number")], limit_value=3), StreamConfig(batch_size=2)
        )
    assert [int(record.get_value("number")) for record in seen] == [101, 102, 103]


def test_stream_read_honours_an_offset(helpdesk: Helpdesk, twin: str) -> None:
    with opened(twin, tickets(helpdesk)) as db:
        seen = db.stream(
            Query(sort_specs=[SortSpec("number")], offset_value=1, limit_value=2),
            StreamConfig(batch_size=1),
        )
    assert [int(record.get_value("number")) for record in seen] == [102, 103]


def test_stream_read_projects_every_page(helpdesk: Helpdesk, twin: str) -> None:
    with opened(twin, tickets(helpdesk)) as db:
        seen = db.stream(
            Query(sort_specs=[SortSpec("number")], fields=["subject"]), StreamConfig(batch_size=3)
        )
    assert [list(record.fields) for record in seen] == [["subject"]] * 4


# --- refusals ------------------------------------------------------------------------


def test_an_undeclared_column_is_refused_not_interpolated(helpdesk: Helpdesk, twin: str) -> None:
    """A column name reaches SQL only if the store declared it."""
    with opened(twin, tickets(helpdesk)) as db:
        for field in ("internal_note", 'subject" OR 1=1 --'):
            with pytest.raises(ValidationError, match="no declared column"):
                db.search(Query(filters=[Filter(field, Operator.EQ, "x")]))
        with pytest.raises(ValidationError, match="no declared column"):
            db.search(Query(sort_specs=[SortSpec("internal_note")]))


def test_a_missing_table_is_named_as_one_somebody_else_owns(helpdesk: Helpdesk, twin: str) -> None:
    """Bug: the refusal told the reader to run their migrations, for a table
    a native store reads and never creates.
    """
    config = helpdesk.config("no_such_table", {"id": TICKET_FIELDS["id"]}, scope=[])
    with pytest.raises(RuntimeError) as caught:
        with opened(twin, config):
            pass
    message = str(caught.value)
    assert "no_such_table" in message and "layout: native" in message
    assert "migrations" not in message


def test_a_scope_on_an_undeclared_column_is_refused_at_construction(
    helpdesk: Helpdesk, twin: str
) -> None:
    scope = [{"field": "internal_note", "operator": "=", "value": "x"}]
    with pytest.raises(ValidationError, match="no declared column"):
        built(twin, tickets(helpdesk, scope=scope))


def test_an_id_column_must_be_declared(helpdesk: Helpdesk, twin: str) -> None:
    with pytest.raises(ValidationError, match="id_column"):
        built(twin, tickets(helpdesk, id_column="internal_note"))


def test_an_unknown_sql_type_is_refused(helpdesk: Helpdesk, twin: str) -> None:
    """A misspelled type would otherwise switch the guard off without a word."""
    fields = {**TICKET_FIELDS, "id": {"type": "string", "metadata": {"sql_type": "uuidd"}}}
    with pytest.raises(ValidationError, match="uuidd"):
        built(twin, tickets(helpdesk, schema={"fields": fields}))


def test_a_refusal_names_the_backend_and_table(helpdesk: Helpdesk, twin: str) -> None:
    with pytest.raises(ValidationError, match="'tickets'"):
        built(twin, tickets(helpdesk, id_column="internal_note"))


def test_writes_are_refused(helpdesk: Helpdesk, twin: str) -> None:
    """Read-only is the contract, stated by raising rather than by a silent no-op."""
    with opened(twin, tickets(helpdesk)) as db:
        with pytest.raises(OperationError, match="read-only"):
            db.call("create", Record({"subject": "Mouse is unresponsive"}))
        with pytest.raises(OperationError, match="read-only"):
            db.call("update", str(T2), Record({"status": "open"}))
        with pytest.raises(OperationError, match="read-only"):
            db.call("delete", str(T2))
        assert db.read(str(T2)) is not None


def test_a_native_store_claims_no_conditional_write(helpdesk: Helpdesk, twin: str) -> None:
    assert not built(twin, tickets(helpdesk)).supports(Capability.CONDITIONAL_WRITE)
    assert built(twin, json_layout(tickets(helpdesk))).supports(Capability.CONDITIONAL_WRITE)


def test_scope_keys_are_refused_under_the_json_layout(helpdesk: Helpdesk, twin: str) -> None:
    with pytest.raises(ValidationError, match="scope"):
        built(twin, json_layout(tickets(helpdesk), scope=ACME_ONLY))
