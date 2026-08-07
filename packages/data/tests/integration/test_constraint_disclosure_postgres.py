"""A non-duplicate constraint violation on Postgres (real service).

The sibling of ``tests/test_constraint_disclosure.py``, which covers SQLite
behaviourally and the rest of the backends structurally. Postgres was outside
both: it catches ``UniqueViolation`` *by type* and nothing else, so a
``NOT NULL`` or ``CHECK`` failure was never mapped at all — the raw driver
exception propagated, and the structural guard (which checks that no backend
*constructs* ``RecordValidationError`` itself) reported green because postgres
did not construct one.

Two things follow, and both are worse than the wrong-status framing suggests.
A caller's invalid write comes back as a 500 rather than the 422 it is, since
a raw ``psycopg2`` error is not a ``DataknobsError`` and reaches the generic
handler. And psycopg2's text for a ``CHECK`` violation carries a ``DETAIL:``
line quoting **the failing row** — so the value that was rejected travels in
the exception a library caller is likely to log.

Requires a running Postgres; the module skips when unavailable.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Generator

import pytest
from dataknobs_common.testing import requires_postgres

from dataknobs_data import DuplicateRecordError, Record
from dataknobs_data.backends.postgres import AsyncPostgresDatabase, SyncPostgresDatabase
from dataknobs_data.exceptions import RecordValidationError

pytestmark = requires_postgres

#: The value the constraint rejects. Postgres echoes the failing row in its
#: ``DETAIL:`` line, so this is what must not survive into the message.
_SECRET = "hunter2"

#: A constraint a deployment could plausibly add to the JSON payload column,
#: chosen because it is *not* a primary-key collision — the case the existing
#: ``UniqueViolation`` handler already covers.
_CHECK = (
    "ADD CONSTRAINT no_tenant_secret "
    "CHECK ((data->>'tenant_secret') IS NULL)"
)


def _assert_bounded(exc: RecordValidationError) -> None:
    """Message says what happened; the driver's text is on ``__cause__``."""
    assert _SECRET not in str(exc), "the rejected value reached the message"
    assert "constraint" in str(exc)
    assert exc.__cause__ is not None, "the driver's exception must stay reachable"
    assert _SECRET in str(exc.__cause__), (
        "the diagnostic must survive on __cause__, not merely be deleted"
    )


@pytest.fixture
def sync_pg(make_postgres_test_db) -> Generator[SyncPostgresDatabase, None, None]:
    for pg in make_postgres_test_db("test_constraint_disclosure_"):
        db = SyncPostgresDatabase(pg)
        db.connect()
        db.db.execute(f'ALTER TABLE "{pg["table"]}" {_CHECK}')
        try:
            yield db
        finally:
            db.close()


@pytest.fixture
async def async_pg(make_postgres_test_db) -> AsyncGenerator[AsyncPostgresDatabase, None]:
    for pg in make_postgres_test_db("test_constraint_disclosure_async_"):
        setup = SyncPostgresDatabase(pg)
        setup.connect()
        setup.db.execute(f'ALTER TABLE "{pg["table"]}" {_CHECK}')
        setup.close()

        db = AsyncPostgresDatabase(pg)
        await db.connect()
        try:
            yield db
        finally:
            await db.close()


class TestSyncPostgres:
    def test_create_maps_the_violation(self, sync_pg: SyncPostgresDatabase) -> None:
        with pytest.raises(RecordValidationError) as excinfo:
            sync_pg.create(Record({"tenant_secret": _SECRET}))

        _assert_bounded(excinfo.value)

    def test_create_batch_maps_the_violation(
        self, sync_pg: SyncPostgresDatabase
    ) -> None:
        with pytest.raises(RecordValidationError) as excinfo:
            sync_pg.create_batch(
                [Record({"v": 1}), Record({"tenant_secret": _SECRET})]
            )

        _assert_bounded(excinfo.value)

    def test_a_duplicate_id_is_still_told_apart(
        self, sync_pg: SyncPostgresDatabase
    ) -> None:
        """Widening the catch must not swallow the case it already handled.

        ``DuplicateRecordError`` and ``RecordValidationError`` are different
        answers — one means "pick another id", the other "fix the record" —
        and postgres distinguishes them by exception type, which is more
        precise than the text matching the other backends need.
        """
        sync_pg.create(Record({"v": "first"}, id="taken"))

        with pytest.raises(DuplicateRecordError):
            sync_pg.create(Record({"v": "second"}, id="taken"))


class TestAsyncPostgres:
    async def test_create_maps_the_violation(
        self, async_pg: AsyncPostgresDatabase
    ) -> None:
        with pytest.raises(RecordValidationError) as excinfo:
            await async_pg.create(Record({"tenant_secret": _SECRET}))

        _assert_bounded(excinfo.value)

    async def test_a_duplicate_id_is_still_told_apart(
        self, async_pg: AsyncPostgresDatabase
    ) -> None:
        await async_pg.create(Record({"v": "first"}, id="taken"))

        with pytest.raises(DuplicateRecordError):
            await async_pg.create(Record({"v": "second"}, id="taken"))
