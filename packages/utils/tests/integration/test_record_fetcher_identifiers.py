"""``PostgresRecordFetcher.get_records`` must quote the identifiers it inlines.

The method built its SQL with three unquoted f-string slots — the field list,
the table name and the ID field name — while every other SQL site in the module
had already adopted ``quote_ident``. One method kept the old idiom.

``fields_to_retrieve`` is the one that matters, because it is a **per-call
argument** rather than constructor configuration. Verified against a live
server before the fix: a fetcher configured for one table returned a value from
another one, plus ``current_user``, through nothing but that parameter.

The ``ids`` argument is *not* a vector and the fix does not treat it as one:
values go through ``str(value + offset)``, which raises ``TypeError`` on
anything non-numeric. That is asserted below so the reasoning stays checkable
rather than remembered.

What the family says the parameter is settles the contract question. The
sibling fetchers do ``df[fields_to_retrieve]`` — pandas column selection by
bare label — so quoting each name makes the Postgres fetcher agree with them
rather than narrowing it. An expression was never in the contract; it was only
ever reachable because the slot was unquoted.

Requires a reachable PostgreSQL instance (``bin/dk up``).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import psycopg2
import pytest
from dataknobs_common.testing import requires_postgres

from dataknobs_utils.sql_utils import PostgresDB, PostgresRecordFetcher

pytestmark = [requires_postgres, pytest.mark.postgres, pytest.mark.integration]

#: What must not come back through a fetcher configured for another table.
_SECRET = "sk-SUPER-SECRET"


@pytest.fixture
def fetcher_db(make_postgres_test_db: Any) -> Iterator[dict[str, Any]]:
    yield from make_postgres_test_db("test_rf_idents_")


@pytest.fixture
def db(fetcher_db: dict[str, Any]) -> Iterator[PostgresDB]:
    database = PostgresDB(
        host=fetcher_db["host"],
        db=fetcher_db["database"],
        user=fetcher_db["user"],
        pwd=fetcher_db["password"],
        port=fetcher_db["port"],
    )
    yield database
    database.close()


@pytest.fixture
def tables(db: PostgresDB, fetcher_db: dict[str, Any]) -> Iterator[dict[str, str]]:
    """A records table with an awkward-but-legal column, and a secrets table.

    The second table exists only so exfiltration has somewhere to come *from*:
    an injection that stays inside the configured table is hard to tell from
    the feature.
    """
    main = fetcher_db["table"]
    other = f"{main}_secrets"
    db.execute(f'DROP TABLE IF EXISTS "{main}"')
    db.execute(f'CREATE TABLE "{main}" (id integer, "Mixed Case" text, note text)')
    db.execute(
        f"""INSERT INTO "{main}" VALUES (1, 'mc-one', 'row-one'), (2, 'mc-two', 'row-two')"""
    )
    db.execute(f'CREATE TABLE "{other}" (id integer, api_key text)')
    db.execute(f"""INSERT INTO "{other}" VALUES (1, '{_SECRET}')""")
    yield {"main": main, "other": other}
    db.execute(f'DROP TABLE IF EXISTS "{other}"')


class TestFieldsToRetrieveIsNotAnInjectionVector:
    """The per-call argument, which is what makes this reachable."""

    def test_cross_table_exfiltration_is_refused(
        self, db: PostgresDB, tables: dict[str, str]
    ) -> None:
        """Bug: this returned ``{'exfiltrated': 'sk-SUPER-SECRET'}`` from a
        fetcher configured for a different table.
        """
        fetcher = PostgresRecordFetcher(db, tables["main"])
        payload = (
            f'id, (SELECT api_key FROM "{tables["other"]}" LIMIT 1) AS exfiltrated, '
            "current_user AS whoami"
        )

        with pytest.raises(psycopg2.Error) as caught:
            fetcher.get_records([1], fields_to_retrieve=[payload])

        assert _SECRET not in str(caught.value), "the secret reached the error text"

    def test_a_quoted_payload_is_treated_as_one_column_name(
        self, db: PostgresDB, tables: dict[str, str]
    ) -> None:
        """Failing closed means the payload becomes a (missing) identifier, not
        a fragment of the statement — so the error names a column, not syntax.
        """
        fetcher = PostgresRecordFetcher(db, tables["main"])

        with pytest.raises(psycopg2.errors.UndefinedColumn):
            fetcher.get_records([1], fields_to_retrieve=["note FROM pg_class --"])


class TestLegalIdentifiersSurvive:
    """Quoting is also what makes ordinary names work; they did not before."""

    def test_mixed_case_field(self, db: PostgresDB, tables: dict[str, str]) -> None:
        """Bug: ``UndefinedColumn: column "mixed" does not exist`` — the
        unquoted name was folded to lowercase and split at the space.
        """
        fetcher = PostgresRecordFetcher(db, tables["main"])

        got = fetcher.get_records([1], fields_to_retrieve=["id", "Mixed Case"])

        assert list(got.columns) == ["id", "Mixed Case"]
        assert got["Mixed Case"].iloc[0] == "mc-one"

    def test_mixed_case_id_field(self, db: PostgresDB, tables: dict[str, str]) -> None:
        """Bug: ``SyntaxError: syntax error at or near "Case"``."""
        db.execute(f'ALTER TABLE "{tables["main"]}" ADD COLUMN "Row Id" integer')
        db.execute(f'UPDATE "{tables["main"]}" SET "Row Id" = id')
        fetcher = PostgresRecordFetcher(db, tables["main"], id_field_name="Row Id")

        got = fetcher.get_records([2])

        assert len(got) == 1
        assert got["note"].iloc[0] == "row-two"

    def test_constructor_fields_are_quoted_too(
        self, db: PostgresDB, tables: dict[str, str]
    ) -> None:
        """The instance default takes the same path as the per-call argument."""
        fetcher = PostgresRecordFetcher(db, tables["main"], fields_to_retrieve=["id", "Mixed Case"])

        assert list(fetcher.get_records([1]).columns) == ["id", "Mixed Case"]


class TestIdsAreBoundNotInlined:
    """The ``ids`` clause stops relying on arithmetic to keep it safe.

    Leaving ``ids`` inlined was defensible — ``str(value + offset)`` raises
    ``TypeError`` on anything non-numeric, so caller text could not reach the
    SQL. But safety by side effect covers only what the side effect happens to
    cover, and two things fell outside it: an empty list produced ``IN ()``,
    which is a syntax error, and a ``nan``/``inf`` passed the arithmetic to
    become the literal ``IN (nan)``, which the server rejects.

    Binding the values removes the question rather than answering it, and the
    two gaps close with it.
    """

    def test_no_ids_returns_nothing_instead_of_failing(
        self, db: PostgresDB, tables: dict[str, str]
    ) -> None:
        """Bug: ``IN ()`` is not valid SQL, so asking for no records raised
        ``SyntaxError`` from the server rather than returning none.

        The sibling fetchers answer an empty request with an empty frame, which
        is also the only reading that makes sense.
        """
        got = PostgresRecordFetcher(db, tables["main"]).get_records([])

        assert len(got) == 0
        assert set(got.columns) >= {"id", "Mixed Case", "note"}, (
            "an empty result must carry the same columns a non-empty one does"
        )

    def test_the_empty_frame_has_the_same_columns_as_a_populated_one(
        self, db: PostgresDB, tables: dict[str, str]
    ) -> None:
        """Bug: the empty case returned a *zero-column* frame, so ``got["id"]``
        raised ``KeyError`` on a result that merely had no rows — and
        ``pd.concat`` over batched calls produced a frame that was not the
        union of the non-empty batches' columns.

        Both sibling fetchers return every column for an empty request, so this
        was also the odd one out of three.
        """
        fetcher = PostgresRecordFetcher(db, tables["main"])

        populated = fetcher.get_records([1])
        empty = fetcher.get_records([])

        assert list(empty.columns) == list(populated.columns)
        assert list(empty["id"]) == []

    def test_the_empty_frame_honours_fields_to_retrieve(
        self, db: PostgresDB, tables: dict[str, str]
    ) -> None:
        """The projection has to apply to the empty case the same way."""
        fetcher = PostgresRecordFetcher(db, tables["main"])

        empty = fetcher.get_records([], fields_to_retrieve=["id", "note"])

        assert list(empty.columns) == ["id", "note"]

    @pytest.mark.parametrize("value", [float("nan"), float("inf")])
    def test_a_non_finite_id_is_refused_rather_than_sent(
        self, db: PostgresDB, tables: dict[str, str], value: float
    ) -> None:
        """Bug: these survive ``value + offset``, so they reached the SQL as the
        bare literals ``nan`` / ``inf`` and failed inside the server.
        """
        fetcher = PostgresRecordFetcher(db, tables["main"])

        with pytest.raises(TypeError):
            fetcher.get_records([value])  # type: ignore[list-item]

    @pytest.mark.parametrize("value", ["5", 1.9, 2.0])
    def test_a_non_integer_id_is_refused_rather_than_coerced(
        self, db: PostgresDB, tables: dict[str, str], value: object
    ) -> None:
        """``ids`` is declared ``List[int]``, so it should mean it.

        ``int()`` accepted ``"5"`` and silently truncated ``1.9`` to ``1`` —
        returning a *different, wrong row* rather than the empty result the
        caller used to get. ``operator.index`` is the "must be an integer"
        test: it rejects strings and floats, including whole-valued ones,
        while accepting ``int`` and numpy integers.
        """
        fetcher = PostgresRecordFetcher(db, tables["main"])

        with pytest.raises(TypeError):
            fetcher.get_records([value])  # type: ignore[list-item]

    def test_numpy_integers_are_still_accepted(
        self, db: PostgresDB, tables: dict[str, str]
    ) -> None:
        """The guard against over-tightening: ids commonly arrive from a
        DataFrame column, whose entries are ``np.int64`` rather than ``int``.
        """
        import numpy as np

        got = PostgresRecordFetcher(db, tables["main"]).get_records([np.int64(1)])

        assert list(got["id"]) == [1]

    def test_binding_does_not_disturb_a_normal_fetch(
        self, db: PostgresDB, tables: dict[str, str]
    ) -> None:
        got = PostgresRecordFetcher(db, tables["main"]).get_records([1, 2])

        assert sorted(got["id"]) == [1, 2]


class TestUnchangedBehaviour:
    """What the fix must not move."""

    def test_no_fields_still_selects_everything(
        self, db: PostgresDB, tables: dict[str, str]
    ) -> None:
        got = PostgresRecordFetcher(db, tables["main"]).get_records([1, 2])

        assert set(got.columns) >= {"id", "Mixed Case", "note"}
        assert len(got) == 2

    def test_ids_remain_non_injectable(self, db: PostgresDB, tables: dict[str, str]) -> None:
        """Not a vector before the identifier fix, and not one now.

        The mechanism changed and the property did not. It used to rest on
        ``str(value + offset)`` raising ``TypeError`` on a non-number; the
        values are bound now, and ``int()`` refuses the same input a step
        earlier — as ``ValueError``, since that is what ``int`` raises on a
        string it cannot parse. Asserting the refusal rather than its type is
        the point: the guarantee is that caller text cannot reach the SQL, not
        that a particular exception carries the news.
        """
        fetcher = PostgresRecordFetcher(db, tables["main"])

        with pytest.raises((TypeError, ValueError)):
            fetcher.get_records(["1 OR 1=1"])  # type: ignore[list-item]

    def test_one_based_offset_still_applies(self, db: PostgresDB, tables: dict[str, str]) -> None:
        """Quoting the identifiers must not disturb the id arithmetic."""
        fetcher = PostgresRecordFetcher(db, tables["main"], one_based_ids=True)

        got = fetcher.get_records([1], one_based=False)

        assert got["note"].iloc[0] == "row-two", "expected the 0-based 1 to mean id 2"
