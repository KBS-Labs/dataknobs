"""A ``datetime`` and a ``date`` are ordered against each other, not refused.

``Filter.matches`` is the in-memory oracle for every ordering operator, and
the memory, file and S3 backends sort in Python. Both used to hand a
``datetime`` and a ``date`` straight to ``<``/``>``, which Python refuses with
``TypeError`` even though ``datetime`` subclasses ``date``. Measured before the
fix:

======================================  =================================
Case                                    Answer
======================================  =================================
``datetime(2024-03-05 12:00) > date``   ``False`` (swallowed ``TypeError``)
``... NOT_BETWEEN [date, date]``        ``True``, although inside the range
``"2024-03-05T12:00:00" > date``        ``False`` (the string parses to a
                                        ``datetime``, then the same raise)
``date > "2024-02-01"``                 ``False``, the same way round
``search(sort by t)`` over a mix        ``TypeError`` out of ``search()``
======================================  =================================

PostgreSQL and DuckDB answer these over a native ``timestamp`` column by
promoting the ``date`` to midnight. That is the rule taken here: a plain
``date`` compared with a ``datetime`` is that day's midnight --- in the
``datetime``'s own zone when it has one, since a ``date`` carries none.
sqlite has no timestamp type and compares the text, so it is not the model.

Only ordering is in scope. Whether ``EQ``/``IN`` should also call a midnight
``datetime`` equal to its ``date`` is a separate question: Python answers
"unequal" without raising, so nothing is refused there.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from dataknobs_data import Filter, Operator, Query, Record, SortOrder, SortSpec
from dataknobs_data.backends.memory import AsyncMemoryDatabase, SyncMemoryDatabase
from dataknobs_data.query import sort_key_for
from dataknobs_data.query_logic import ComplexQuery

LATER = datetime(2024, 3, 5, 12, 0)
FIRST_OF_MARCH = date(2024, 3, 1)
FIRST_OF_APRIL = date(2024, 4, 1)


class TestAnOrderingOperatorPromotesTheDate:
    @pytest.mark.parametrize(
        ("operator", "value", "expected"),
        [
            (Operator.GT, FIRST_OF_MARCH, True),
            (Operator.GTE, FIRST_OF_MARCH, True),
            (Operator.LT, FIRST_OF_MARCH, False),
            (Operator.LTE, FIRST_OF_MARCH, False),
            (Operator.GT, FIRST_OF_APRIL, False),
            (Operator.LT, FIRST_OF_APRIL, True),
            (Operator.BETWEEN, [FIRST_OF_MARCH, FIRST_OF_APRIL], True),
            (Operator.NOT_BETWEEN, [FIRST_OF_MARCH, FIRST_OF_APRIL], False),
        ],
    )
    def test_a_datetime_record_against_a_date(
        self, operator: Operator, value: object, expected: bool
    ) -> None:
        assert Filter("t", operator, value).matches(LATER) is expected

    def test_a_date_record_against_a_datetime(self) -> None:
        assert Filter("t", Operator.GT, datetime(2024, 2, 1)).matches(FIRST_OF_MARCH)
        assert not Filter("t", Operator.LT, datetime(2024, 2, 1)).matches(FIRST_OF_MARCH)

    def test_the_date_is_midnight(self) -> None:
        midnight = datetime(2024, 3, 1, 0, 0)
        assert Filter("t", Operator.GTE, FIRST_OF_MARCH).matches(midnight)
        assert Filter("t", Operator.LTE, FIRST_OF_MARCH).matches(midnight)
        assert not Filter("t", Operator.GT, FIRST_OF_MARCH).matches(midnight)
        assert Filter("t", Operator.GT, FIRST_OF_MARCH).matches(
            midnight + timedelta(microseconds=1)
        )

    def test_an_iso_string_record_against_a_date(self) -> None:
        assert Filter("t", Operator.GT, FIRST_OF_MARCH).matches("2024-03-05T12:00:00")
        assert not Filter("t", Operator.GT, FIRST_OF_APRIL).matches("2024-03-05T12:00:00")

    def test_a_date_record_against_an_iso_string(self) -> None:
        assert Filter("t", Operator.GT, "2024-02-01").matches(FIRST_OF_MARCH)
        assert not Filter("t", Operator.GT, "2024-04-01T00:00:00").matches(FIRST_OF_MARCH)

    def test_an_aware_datetime_takes_the_date_into_its_own_zone(self) -> None:
        plus_five = timezone(timedelta(hours=5))
        # 00:30 on 1 March at +05:00: after that zone's midnight.
        just_after = datetime(2024, 3, 1, 0, 30, tzinfo=plus_five)
        assert Filter("t", Operator.GT, FIRST_OF_MARCH).matches(just_after)
        assert Filter("t", Operator.LT, FIRST_OF_MARCH).matches(
            datetime(2024, 2, 29, 23, 30, tzinfo=plus_five)
        )

    def test_two_dates_and_two_datetimes_are_unchanged(self) -> None:
        assert Filter("t", Operator.GT, FIRST_OF_MARCH).matches(FIRST_OF_APRIL)
        assert Filter("t", Operator.GT, datetime(2024, 3, 1)).matches(LATER)


def _mixed_records() -> list[Record]:
    """A datetime, a date and a datetime, out of order."""
    return [
        Record({"t": LATER}),
        Record({"t": FIRST_OF_MARCH}),
        Record({"t": datetime(2024, 2, 1, 9, 0)}),
    ]


ASCENDING = [datetime(2024, 2, 1, 9, 0), FIRST_OF_MARCH, LATER]


def _complex_sorted(order: SortOrder) -> ComplexQuery:
    """An ``OR`` no single ``Query`` expresses, so the in-memory path sorts it."""
    query = ComplexQuery.OR(
        [
            Query().filter("t", ">", date(2000, 1, 1)),
            Query().filter("t", "<", date(1990, 1, 1)),
        ]
    )
    query.sort_specs = [SortSpec("t", order)]
    return query


class TestASortOverAMixedFieldOrdersIt:
    @pytest.mark.parametrize("order", [SortOrder.ASC, SortOrder.DESC])
    def test_sync_query(self, order: SortOrder) -> None:
        db = SyncMemoryDatabase()
        for record in _mixed_records():
            db.create(record)
        found = db.search(Query(sort_specs=[SortSpec("t", order)]))
        expected = ASCENDING if order == SortOrder.ASC else ASCENDING[::-1]
        assert [r.get_value("t") for r in found] == expected

    @pytest.mark.parametrize("order", [SortOrder.ASC, SortOrder.DESC])
    async def test_async_query(self, order: SortOrder) -> None:
        db = AsyncMemoryDatabase()
        for record in _mixed_records():
            await db.create(record)
        found = await db.search(Query(sort_specs=[SortSpec("t", order)]))
        expected = ASCENDING if order == SortOrder.ASC else ASCENDING[::-1]
        assert [r.get_value("t") for r in found] == expected

    @pytest.mark.parametrize("order", [SortOrder.ASC, SortOrder.DESC])
    def test_sync_complex_query(self, order: SortOrder) -> None:
        db = SyncMemoryDatabase()
        for record in _mixed_records():
            db.create(record)
        found = db.search(_complex_sorted(order))
        expected = ASCENDING if order == SortOrder.ASC else ASCENDING[::-1]
        assert [r.get_value("t") for r in found] == expected

    @pytest.mark.parametrize("order", [SortOrder.ASC, SortOrder.DESC])
    async def test_async_complex_query(self, order: SortOrder) -> None:
        db = AsyncMemoryDatabase()
        for record in _mixed_records():
            await db.create(record)
        found = await db.search(_complex_sorted(order))
        expected = ASCENDING if order == SortOrder.ASC else ASCENDING[::-1]
        assert [r.get_value("t") for r in found] == expected


class TestOnlyAMixedFieldPaysForTheAlignment:
    """The aligned comparison is a Python callback per comparison, so a field
    that does not mix a ``date`` with a ``datetime`` sorts by its raw values.
    """

    @pytest.mark.parametrize(
        "values",
        [
            ["b", "a", "c"],
            [3, 1, 2.5],
            [LATER, datetime(2024, 1, 1)],
            [FIRST_OF_APRIL, FIRST_OF_MARCH],
            [],
        ],
    )
    def test_a_homogeneous_field_sorts_by_its_raw_values(self, values: list[object]) -> None:
        key = sort_key_for(values)
        assert all(key(value) is value for value in values)

    def test_a_mixed_field_sorts_by_the_aligned_key(self) -> None:
        values = [LATER, FIRST_OF_MARCH]
        key = sort_key_for(values)
        assert sorted(values, key=key) == [FIRST_OF_MARCH, LATER]
        assert key(FIRST_OF_MARCH) is not FIRST_OF_MARCH
