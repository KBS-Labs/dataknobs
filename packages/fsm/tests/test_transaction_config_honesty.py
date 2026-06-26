"""W4 — neighboring no-op reconciliation + transaction-config honesty (170-FU2).

These cover the reconciliation of the formerly-silent transaction sites once a
real :meth:`AsyncDatabase.transaction` primitive exists:

* ``ExecutionContext.{start,commit,rollback}_transaction`` no longer call a
  ``hasattr``-guarded ``self.database.<method>()`` — which silently no-op'd on
  backends without the method and, once ``AsyncDatabase.begin_transaction``
  became an async coroutine, would have invoked it un-awaited (a silent miss).
  They keep their in-memory logical bookkeeping and DEBUG-log the decoupling.
* The builder warns loudly when a non-default ``transaction.strategy`` is
  configured, because the in-memory ``TransactionManager`` it builds is not
  consulted by the execution engines to drive database commit/rollback — the
  config knob would otherwise silently fail to deliver database atomicity.
"""

from __future__ import annotations

import logging

import pytest

from dataknobs_data import async_database_factory
from dataknobs_fsm.config.builder import FSMBuilder
from dataknobs_fsm.config.schema import TransactionConfig
from dataknobs_fsm.core.modes import TransactionMode
from dataknobs_fsm.core.transactions import TransactionStrategy
from dataknobs_fsm.execution.context import ExecutionContext


# --------------------------------------------------------------------------- #
# ExecutionContext: the reconciled (no broken DB call) transaction bookkeeping
# --------------------------------------------------------------------------- #
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
    False) — unchanged behavior, just no longer touching the database.
    """
    ctx = ExecutionContext()  # transaction_mode defaults to NONE
    assert ctx.start_transaction() is False


# --------------------------------------------------------------------------- #
# Builder: loud warning on a non-default (inert) transaction strategy
# --------------------------------------------------------------------------- #
def test_builder_warns_on_non_default_transaction_strategy(caplog):
    builder = FSMBuilder()
    with caplog.at_level(logging.WARNING, logger="dataknobs_fsm.config.builder"):
        builder._init_transaction_manager(
            TransactionConfig(strategy=TransactionStrategy.BATCH)
        )
    assert any(
        "do not use" in rec.message and "database commit/rollback" in rec.message
        for rec in caplog.records
    ), f"expected an honesty warning, got: {[r.message for r in caplog.records]}"


def test_builder_quiet_on_default_single_strategy(caplog):
    builder = FSMBuilder()
    with caplog.at_level(logging.WARNING, logger="dataknobs_fsm.config.builder"):
        builder._init_transaction_manager(
            TransactionConfig(strategy=TransactionStrategy.SINGLE)
        )
    # SINGLE is the default present on every config; it must not warn (noise).
    assert not caplog.records
