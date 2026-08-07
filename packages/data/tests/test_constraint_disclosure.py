"""A rejected write must not publish the schema that rejected it.

``RecordValidationError`` is a ``dataknobs_common.exceptions.ValidationError``,
which the ``dataknobs-bots`` API layer renders as a 422 **with its message
returned to the caller**. Every SQL backend used to build that message from
``str(driver_exception)``, and a driver names the physical table and column it
enforced: ``NOT NULL constraint failed: records.tenant_secret``. So a write
rejected through an HTTP route answered with a piece of the storage schema.

The driver's text is not lost — every site raises ``from`` it, so it stays on
``__cause__``, in the traceback a library caller sees and in the line the API
handler logs. These tests pin that both halves of that are true.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from dataknobs_data import Record
from dataknobs_data.exceptions import RecordValidationError

#: Names a column that must not reach a caller, so a relay is unmistakable in
#: the assertion rather than a judgement call about what counts as schema.
_GUARDED_COLUMN = "records.tenant_secret"

_ABORT_TRIGGER = (
    "CREATE TRIGGER reject_marked BEFORE INSERT ON records "
    "WHEN NEW.id LIKE 'reject-%' "
    f"BEGIN SELECT RAISE(ABORT, 'NOT NULL constraint failed: {_GUARDED_COLUMN}'); END"
)


def _assert_bounded(exc: RecordValidationError) -> None:
    """The message says what happened; the driver's text is on ``__cause__``."""
    assert _GUARDED_COLUMN not in str(exc), "driver text reached the message"
    assert "constraint" in str(exc)
    assert exc.__cause__ is not None, "the driver's exception must stay reachable"
    assert _GUARDED_COLUMN in str(exc.__cause__), (
        "the diagnostic must survive on __cause__, not merely be deleted"
    )


class TestSyncSqlite:
    """``RAISE(ABORT)`` surfaces as a real ``sqlite3.IntegrityError``.

    Which is what makes this a test of the backend's own ``except`` block
    rather than of a substitute for it: the same branch runs, with the same
    driver exception type, for a genuine ``NOT NULL`` violation.
    """

    @pytest.fixture
    def db(self):
        from dataknobs_data.backends.sqlite import SyncSQLiteDatabase

        db = SyncSQLiteDatabase({"database": ":memory:", "table": "records"})
        db.connect()
        db.create(Record({"a": 1}, id="seed"))
        db.conn.execute(_ABORT_TRIGGER)
        yield db
        db.close()

    def test_create_does_not_relay_the_driver_text(self, db):
        with pytest.raises(RecordValidationError) as excinfo:
            db.create(Record({"a": 2}, id="reject-1"))

        _assert_bounded(excinfo.value)
        assert "reject-1" in str(excinfo.value), "the caller's own id is useful"

    def test_create_batch_does_not_relay_the_driver_text(self, db):
        with pytest.raises(RecordValidationError) as excinfo:
            db.create_batch([Record({"a": 2}, id="reject-2")])

        _assert_bounded(excinfo.value)


class TestAsyncSqlite:
    """The async twin has the same four sites and had the same defect."""

    @pytest.fixture
    async def db(self):
        from dataknobs_data.backends.sqlite_async import AsyncSQLiteDatabase

        db = AsyncSQLiteDatabase({"database": ":memory:", "table": "records"})
        await db.connect()
        await db.create(Record({"a": 1}, id="seed"))
        await db.db.execute(_ABORT_TRIGGER)
        yield db
        await db.close()

    async def test_create_does_not_relay_the_driver_text(self, db):
        with pytest.raises(RecordValidationError) as excinfo:
            await db.create(Record({"a": 2}, id="reject-1"))

        _assert_bounded(excinfo.value)
        assert "reject-1" in str(excinfo.value)

    async def test_create_batch_does_not_relay_the_driver_text(self, db):
        with pytest.raises(RecordValidationError) as excinfo:
            await db.create_batch([Record({"a": 2}, id="reject-2")])

        _assert_bounded(excinfo.value)


class TestNoBackendBuildsTheErrorItself:
    """The structural half, covering the backends a trigger cannot reach.

    DuckDB has no triggers and cannot add a ``CHECK`` constraint to an existing
    table, so its four sites have no behavioural equivalent of the tests above.
    They are covered instead by the invariant that makes the defect
    unreproducible: a backend does not construct this error, it asks
    ``constraint_violation_error`` for one. A ninth site pasted from the eight
    fails here.
    """

    #: Where the factory lives, and so the one module that must build one.
    _FACTORY_MODULE = "sql_base.py"

    def test_no_direct_construction_in_a_backend(self):
        backends = Path(__file__).resolve().parents[1] / "src" / "dataknobs_data" / "backends"
        modules = [p for p in sorted(backends.glob("*.py")) if p.name != self._FACTORY_MODULE]
        assert modules, f"no backend modules found under {backends}"
        assert (backends / self._FACTORY_MODULE).is_file(), (
            f"{self._FACTORY_MODULE} moved; this guard is now excluding nothing"
        )

        offenders = []
        for module in modules:
            tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "RecordValidationError"
                ):
                    offenders.append(f"{module.name}:{node.lineno}")

        assert not offenders, (
            f"{', '.join(offenders)} builds RecordValidationError directly. "
            f"Raise `constraint_violation_error(record_id) from exc` instead — "
            f"this error's message is returned to an HTTP caller, so the "
            f"driver's text belongs on __cause__."
        )
