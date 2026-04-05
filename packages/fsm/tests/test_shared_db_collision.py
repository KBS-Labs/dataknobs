"""Tests for shared-DB collision fix (Item 67, Bug B14).

When a single AsyncDatabase instance is shared between history and step
storage, queries must filter by record_type to avoid returning the wrong
record type.  These tests verify that all read methods on
UnifiedDatabaseStorage work correctly when the DB is shared.

Uses real AsyncMemoryDatabase instances — no mocks.
"""

import time

import pytest

from dataknobs_data.backends.memory import AsyncMemoryDatabase

from dataknobs_fsm.core.data_modes import DataHandlingMode
from dataknobs_fsm.execution.history import (
    ExecutionHistory,
    ExecutionStatus,
    ExecutionStep,
)
from dataknobs_fsm.storage.base import StorageBackend, StorageConfig
from dataknobs_fsm.storage.database import UnifiedDatabaseStorage
from dataknobs_fsm.storage.memory import InMemoryStorage


def _make_config() -> StorageConfig:
    return StorageConfig(backend=StorageBackend.MEMORY)


def _make_history(execution_id: str, fsm_name: str = "test_fsm") -> ExecutionHistory:
    history = ExecutionHistory(
        fsm_name=fsm_name,
        execution_id=execution_id,
        data_mode=DataHandlingMode.DIRECT,
    )
    history.start_time = 1000.0
    history.add_step(
        state_name="state_a",
        network_name="main",
        data=None,
    )
    history.end_time = 1001.0
    return history


def _make_step(
    step_id: str = "step-1",
    state_name: str = "state_b",
) -> ExecutionStep:
    return ExecutionStep(
        step_id=step_id,
        state_name=state_name,
        network_name="main",
        timestamp=time.time(),
        data_mode=DataHandlingMode.DIRECT,
        status=ExecutionStatus.COMPLETED,
    )


class TestSharedDatabaseCollision:
    """Verify all read methods work when history and steps share one DB."""

    @pytest.fixture
    async def shared_storage(self) -> UnifiedDatabaseStorage:
        """Create storage with a single shared database."""
        db = AsyncMemoryDatabase()
        config = _make_config()
        storage = UnifiedDatabaseStorage(config, database=db)
        await storage.initialize()
        return storage

    @pytest.mark.asyncio
    async def test_load_steps_does_not_crash(
        self, shared_storage: UnifiedDatabaseStorage
    ) -> None:
        """Bug B14: load_steps() KeyError when history records in same DB."""
        exec_id = "exec-1"
        await shared_storage.save_history(_make_history(exec_id))
        await shared_storage.save_step(exec_id, _make_step("step-1"))

        steps = await shared_storage.load_steps(exec_id)

        assert len(steps) == 1
        assert steps[0].step_id == "step-1"

    @pytest.mark.asyncio
    async def test_load_history_does_not_return_steps(
        self, shared_storage: UnifiedDatabaseStorage
    ) -> None:
        """Symmetric: load_history() must not find step records."""
        exec_id = "exec-1"
        await shared_storage.save_history(_make_history(exec_id))
        await shared_storage.save_step(exec_id, _make_step("step-1"))

        history = await shared_storage.load_history(exec_id)

        assert history is not None
        assert history.execution_id == exec_id
        assert history.fsm_name == "test_fsm"

    @pytest.mark.asyncio
    async def test_query_histories_excludes_steps(
        self, shared_storage: UnifiedDatabaseStorage
    ) -> None:
        """query_histories() must return only history records."""
        exec_id = "exec-1"
        await shared_storage.save_history(_make_history(exec_id))
        await shared_storage.save_step(exec_id, _make_step("step-1"))

        results = await shared_storage.query_histories({})

        assert len(results) == 1
        assert results[0]["id"] == exec_id

    @pytest.mark.asyncio
    async def test_get_statistics_counts_only_histories(
        self, shared_storage: UnifiedDatabaseStorage
    ) -> None:
        """get_statistics() must count history records, not steps."""
        exec_id = "exec-1"
        await shared_storage.save_history(_make_history(exec_id))
        await shared_storage.save_step(exec_id, _make_step("step-1"))
        await shared_storage.save_step(exec_id, _make_step("step-2"))

        # Overall stats
        stats = await shared_storage.get_statistics()
        assert stats["total_histories"] == 1

        # Per-execution stats
        exec_stats = await shared_storage.get_statistics(execution_id=exec_id)
        assert exec_stats["execution_id"] == exec_id
        assert exec_stats["fsm_name"] == "test_fsm"

    @pytest.mark.asyncio
    async def test_delete_history_scoped_correctly(
        self, shared_storage: UnifiedDatabaseStorage
    ) -> None:
        """delete_history() must delete both history and step records."""
        exec_id = "exec-1"
        await shared_storage.save_history(_make_history(exec_id))
        await shared_storage.save_step(exec_id, _make_step("step-1"))

        deleted = await shared_storage.delete_history(exec_id)

        assert deleted is True
        assert await shared_storage.load_history(exec_id) is None
        assert await shared_storage.load_steps(exec_id) == []

    @pytest.mark.asyncio
    async def test_cleanup_only_removes_histories(
        self, shared_storage: UnifiedDatabaseStorage
    ) -> None:
        """cleanup() must only target history records for deletion."""
        exec_id = "exec-1"
        await shared_storage.save_history(_make_history(exec_id))
        await shared_storage.save_step(exec_id, _make_step("step-1"))

        # cleanup with a future timestamp to match everything
        deleted = await shared_storage.cleanup(
            before_timestamp=time.time() + 9999,
            keep_failed=False,
        )

        assert deleted == 1
        # Verify both history and associated steps were removed
        assert await shared_storage.load_history(exec_id) is None
        assert await shared_storage.load_steps(exec_id) == []

    @pytest.mark.asyncio
    async def test_multiple_executions_isolated(
        self, shared_storage: UnifiedDatabaseStorage
    ) -> None:
        """Records from different executions don't leak across queries."""
        await shared_storage.save_history(_make_history("exec-1"))
        await shared_storage.save_step("exec-1", _make_step("step-1a"))

        await shared_storage.save_history(_make_history("exec-2", fsm_name="other"))
        await shared_storage.save_step("exec-2", _make_step("step-2a"))

        steps_1 = await shared_storage.load_steps("exec-1")
        steps_2 = await shared_storage.load_steps("exec-2")
        assert len(steps_1) == 1
        assert steps_1[0].step_id == "step-1a"
        assert len(steps_2) == 1
        assert steps_2[0].step_id == "step-2a"

        history_1 = await shared_storage.load_history("exec-1")
        history_2 = await shared_storage.load_history("exec-2")
        assert history_1 is not None
        assert history_1.fsm_name == "test_fsm"
        assert history_2 is not None
        assert history_2.fsm_name == "other"


class TestBackwardCompat:
    """Verify backward compatibility and non-shared-DB setups."""

    @pytest.mark.asyncio
    async def test_separate_databases_no_regression(self) -> None:
        """When two DBs are provided, each stores its own record type."""
        history_db = AsyncMemoryDatabase()
        steps_db = AsyncMemoryDatabase()
        config = _make_config()
        storage = UnifiedDatabaseStorage(
            config, database=history_db, steps_database=steps_db
        )
        await storage.initialize()

        exec_id = "exec-1"
        await storage.save_history(_make_history(exec_id))
        await storage.save_step(exec_id, _make_step("step-1"))

        # Both work independently — no cross-contamination possible
        history = await storage.load_history(exec_id)
        assert history is not None

        steps = await storage.load_steps(exec_id)
        assert len(steps) == 1

    @pytest.mark.asyncio
    async def test_legacy_records_without_record_type(self) -> None:
        """Legacy records lacking record_type are still returned correctly.

        The EXISTS filter checks for history_data / step_data, not
        record_type, so pre-fix records are naturally included.
        """
        from dataknobs_data.records import Record

        db = AsyncMemoryDatabase()
        config = _make_config()
        storage = UnifiedDatabaseStorage(config, database=db)
        await storage.initialize()

        # First, save records normally to get correctly serialized data
        history = _make_history("exec-legacy")
        await storage.save_history(history)
        step = _make_step("step-legacy")
        await storage.save_step("exec-legacy", step)

        # Now simulate legacy records by re-inserting without record_type.
        # Read back the real records, strip record_type, and re-insert.
        from dataknobs_data.query import Query

        all_records = await db.search(Query())
        await db.clear()

        for record in all_records:
            data = dict(record.data)
            data.pop('record_type', None)  # Strip discriminator
            legacy = Record(data)
            await db.upsert(legacy)

        # Verify record_type is gone
        reloaded = await db.search(Query())
        for r in reloaded:
            assert r.get_value('record_type') is None

        # Both legacy records are retrievable via the EXISTS filter
        loaded_history = await storage.load_history('exec-legacy')
        assert loaded_history is not None
        assert loaded_history.fsm_name == 'test_fsm'

        steps = await storage.load_steps('exec-legacy')
        assert len(steps) == 1
        assert steps[0].step_id == 'step-legacy'

        # query_histories also finds legacy history
        results = await storage.query_histories({})
        assert len(results) == 1
        assert results[0]['id'] == 'exec-legacy'

    @pytest.mark.asyncio
    async def test_in_memory_storage_still_works(self) -> None:
        """InMemoryStorage continues to work correctly."""
        config = _make_config()
        storage = InMemoryStorage(config)
        await storage.initialize()

        exec_id = "exec-1"
        await storage.save_history(_make_history(exec_id))
        await storage.save_step(exec_id, _make_step("step-1"))

        history = await storage.load_history(exec_id)
        assert history is not None

        steps = await storage.load_steps(exec_id)
        assert len(steps) == 1
        assert steps[0].step_id == "step-1"

    @pytest.mark.asyncio
    async def test_factory_path_shared_db(self) -> None:
        """Factory-created storage (no injected DB) handles mixed records."""
        config = _make_config()
        storage = UnifiedDatabaseStorage(config)
        await storage.initialize()

        exec_id = "exec-1"
        await storage.save_history(_make_history(exec_id))
        await storage.save_step(exec_id, _make_step("step-1"))

        history = await storage.load_history(exec_id)
        assert history is not None

        steps = await storage.load_steps(exec_id)
        assert len(steps) == 1
        assert steps[0].step_id == "step-1"
