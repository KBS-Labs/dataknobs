"""Tests for AdvancedFSM.save_history() and load_history() methods.

Bug: These methods called non-existent methods on IHistoryStorage:
- save_history() called self._storage.save() instead of self._storage.save_history()
- load_history() called self._storage.load() instead of self._storage.load_history()
"""

import pytest

from dataknobs_fsm.api.advanced import AdvancedFSM
from dataknobs_fsm.config.builder import FSMBuilder
from dataknobs_fsm.config.loader import ConfigLoader
from dataknobs_fsm.storage.base import StorageBackend, StorageConfig, StorageFactory


def _build_simple_fsm():
    """Build a minimal 2-state FSM for testing."""
    config = {
        "name": "history_test_fsm",
        "main_network": "main",
        "networks": [
            {
                "name": "main",
                "states": [
                    {"name": "start", "is_start": True},
                    {"name": "end", "is_end": True},
                ],
                "arcs": [{"from": "start", "to": "end", "name": "finish"}],
            }
        ],
    }
    loader = ConfigLoader()
    fsm_config = loader.load_from_dict(config)
    builder = FSMBuilder()
    return builder.build(fsm_config)


@pytest.mark.asyncio
async def test_save_history_calls_storage():
    """save_history() should call storage.save_history() and return True."""
    config = StorageConfig(backend=StorageBackend.MEMORY)
    storage = StorageFactory.create(config)
    await storage.initialize()

    fsm = _build_simple_fsm()
    advanced = AdvancedFSM(fsm)
    advanced.enable_history(storage=storage)

    # History is populated by enable_history() — add a step to make it realistic
    history = advanced.get_history()
    assert history is not None
    history.add_step("start", "main")
    history.finalize()

    # Save — should succeed
    result = await advanced.save_history()
    assert result is True

    # Verify it was actually persisted
    loaded = await storage.load_history(history.execution_id)
    assert loaded is not None
    assert loaded.execution_id == history.execution_id


@pytest.mark.asyncio
async def test_load_history_calls_storage():
    """load_history() should call storage.load_history() and populate _history."""
    config = StorageConfig(backend=StorageBackend.MEMORY)
    storage = StorageFactory.create(config)
    await storage.initialize()

    fsm = _build_simple_fsm()
    advanced = AdvancedFSM(fsm)
    advanced.enable_history(storage=storage)

    # Save a history
    history = advanced.get_history()
    assert history is not None
    history.add_step("start", "main")
    history.finalize()
    await advanced.save_history()
    execution_id = history.execution_id

    # Create a fresh AdvancedFSM and load
    advanced2 = AdvancedFSM(fsm)
    advanced2.enable_history(storage=storage)
    result = await advanced2.load_history(execution_id)
    assert result is True
    assert advanced2.get_history() is not None
    assert advanced2.get_history().execution_id == execution_id


@pytest.mark.asyncio
async def test_save_history_no_storage():
    """save_history() returns False when no storage is configured."""
    fsm = _build_simple_fsm()
    advanced = AdvancedFSM(fsm)
    advanced.enable_history()  # No storage

    result = await advanced.save_history()
    assert result is False


@pytest.mark.asyncio
async def test_load_history_no_storage():
    """load_history() returns False when no storage is configured."""
    fsm = _build_simple_fsm()
    advanced = AdvancedFSM(fsm)

    result = await advanced.load_history("nonexistent")
    assert result is False


@pytest.mark.asyncio
async def test_load_history_nonexistent_id():
    """load_history() returns False when ID doesn't exist in storage."""
    config = StorageConfig(backend=StorageBackend.MEMORY)
    storage = StorageFactory.create(config)
    await storage.initialize()

    fsm = _build_simple_fsm()
    advanced = AdvancedFSM(fsm)
    advanced.enable_history(storage=storage)

    result = await advanced.load_history("does_not_exist")
    assert result is False
