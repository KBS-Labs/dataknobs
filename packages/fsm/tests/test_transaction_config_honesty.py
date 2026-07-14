"""ExecutionContext logical-transaction bookkeeping honesty.

``ExecutionContext.{start,commit,rollback}_transaction`` maintain in-memory
*logical* transaction bookkeeping only. They must never touch the attached
database — real database atomicity is driven through the async
``AsyncDatabase.transaction`` primitive (and the ``DatabaseTransaction`` FSM
function), which a synchronous context cannot drive. These tests pin that the
bookkeeping tracks state in memory and leaves the database untouched.
"""

from __future__ import annotations

import pytest

from dataknobs_data import async_database_factory
from dataknobs_fsm.core.modes import TransactionMode
from dataknobs_fsm.execution.context import ExecutionContext


@pytest.mark.asyncio
async def test_execution_context_transaction_does_not_drive_async_database():
    """With an ``AsyncDatabase`` attached and a non-NONE mode, the sync
    transaction methods must track in-memory state and leave the database
    untouched — never invoke the async ``begin_transaction`` un-awaited.
    """
    db = async_database_factory.create(backend="memory")
    await db.connect()
    try:
        ctx = ExecutionContext(
            transaction_mode=TransactionMode.PER_RECORD, database=db
        )
        assert ctx.start_transaction() is True
        assert ctx.current_transaction is not None
        # The database must be untouched by the in-memory bookkeeping.
        assert await db.count() == 0
        assert ctx.commit_transaction() is True
        assert ctx.current_transaction is None
        # A fresh transaction can be rolled back cleanly.
        assert ctx.start_transaction() is True
        assert ctx.rollback_transaction() is True
        assert await db.count() == 0
    finally:
        await db.close()


def test_execution_context_transaction_noop_when_mode_none():
    """With the default ``TransactionMode.NONE``, start is a no-op (returns
    False) — no logical transaction is opened and the database is untouched.
    """
    ctx = ExecutionContext()  # transaction_mode defaults to NONE
    assert ctx.start_transaction() is False
