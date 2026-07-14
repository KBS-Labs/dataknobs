"""Retirement of the strategy-based FSM transaction coordinator.

The strategy-based coordinator configured an in-memory object the execution
engines never consulted, so it delivered no database atomicity. It has been
removed. Database atomicity is provided by the dataknobs-data transaction
primitives — ``AsyncDatabase.transaction()``, the ``DatabaseTransaction``
function, and ``BatchCommit(atomicity="require")``.

These tests assert the coordinator is gone and that the sanctioned replacement
(``AsyncDatabase.transaction()``) delivers real atomic rollback.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from dataknobs_data import Record, async_database_factory
from dataknobs_fsm.api.advanced import AdvancedFSM, ExecutionHook
from dataknobs_fsm.config.schema import FSMConfig, validate_config


def _minimal_config_dict() -> dict:
    """A minimal valid FSM configuration dictionary."""
    return {
        "name": "retirement_fsm",
        "networks": [
            {
                "name": "main",
                "states": [
                    {"name": "start", "is_start": True},
                    {"name": "end", "is_end": True},
                ],
            }
        ],
        "main_network": "main",
    }


# --------------------------------------------------------------------------- #
# The strategy-based coordinator is gone
# --------------------------------------------------------------------------- #
def test_core_transactions_module_removed():
    """The ``dataknobs_fsm.core.transactions`` module no longer exists."""
    with pytest.raises(ImportError):
        import dataknobs_fsm.core.transactions  # noqa: F401


def test_transaction_manager_symbols_removed():
    """The coordinator classes/factory are no longer importable."""
    for symbol in (
        "TransactionManager",
        "TransactionStrategy",
        "SingleTransactionManager",
        "BatchTransactionManager",
        "ManualTransactionManager",
        "create_transaction_manager",
    ):
        with pytest.raises(ImportError):
            __import__(
                "dataknobs_fsm.core.transactions", fromlist=[symbol]
            )


def test_transaction_config_removed_from_schema():
    """``TransactionConfig`` is no longer a config-schema type."""
    from dataknobs_fsm.config import schema

    assert not hasattr(schema, "TransactionConfig")


def test_fsmconfig_has_no_transaction_field():
    """``FSMConfig`` no longer declares a ``transaction`` field."""
    assert "transaction" not in FSMConfig.model_fields


def test_advanced_fsm_has_no_configure_transactions():
    """``AdvancedFSM`` no longer exposes ``configure_transactions``."""
    assert not hasattr(AdvancedFSM, "configure_transactions")


def test_execution_hook_has_no_transaction_callbacks():
    """The dead transaction-lifecycle hooks are gone from ``ExecutionHook``."""
    field_names = set(ExecutionHook.__dataclass_fields__)
    assert not field_names & {
        "on_transaction_begin",
        "on_transaction_commit",
        "on_transaction_rollback",
    }


# --------------------------------------------------------------------------- #
# A leftover ``transaction:`` config key loads (ignored) with a warning
# --------------------------------------------------------------------------- #
def test_legacy_transaction_key_loads_with_warning(caplog):
    """A leftover ``transaction`` block does not hard-fail; it warns once."""
    config = _minimal_config_dict()
    config["transaction"] = {"strategy": "batch", "batch_size": 1000}

    with caplog.at_level(logging.WARNING, logger="dataknobs_fsm.config.schema"):
        result = validate_config(config)

    # Loads successfully, key dropped (no longer a field).
    assert isinstance(result, FSMConfig)
    assert not hasattr(result, "transaction")
    # And a single discoverability warning was emitted.
    assert any(
        "'transaction' configuration is no longer supported" in rec.message
        for rec in caplog.records
    ), f"expected a removal warning, got: {[r.message for r in caplog.records]}"


def test_config_without_transaction_key_is_quiet(caplog):
    """A config with no ``transaction`` block emits no removal warning."""
    with caplog.at_level(logging.WARNING, logger="dataknobs_fsm.config.schema"):
        validate_config(_minimal_config_dict())

    assert not any(
        "'transaction' configuration is no longer supported" in rec.message
        for rec in caplog.records
    )


# --------------------------------------------------------------------------- #
# The sanctioned replacement delivers real atomic rollback
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_db_transaction_primitive_rolls_back_atomically(tmp_path: Path):
    """``AsyncDatabase.transaction()`` — the real primitive — rolls back a
    multi-write sequence when the block raises, leaving nothing persisted.

    This is the honest atomicity the retired strategy knob never provided.
    Uses a transaction-capable backend (sqlite) with the fail-closed
    ``policy="strict"``.
    """
    db = async_database_factory.create(
        name="retirement", backend="sqlite", path=str(tmp_path / "retirement.db")
    )
    await db.connect()
    try:
        with pytest.raises(RuntimeError, match="boom"):
            async with db.transaction(policy="strict") as tx:
                await tx.create(Record({"id": "a", "value": 1}))
                await tx.create(Record({"id": "b", "value": 2}))
                raise RuntimeError("boom")

        # Neither staged write was applied — atomic rollback.
        assert await db.count() == 0

        # A clean transaction commits both writes together.
        async with db.transaction(policy="strict") as tx:
            await tx.create(Record({"id": "a", "value": 1}))
            await tx.create(Record({"id": "b", "value": 2}))
        assert await db.count() == 2
    finally:
        await db.close()
