"""Tests for FSM storage database instance injection (Item 6).

Verifies that UnifiedDatabaseStorage (and subclasses) accept pre-built
AsyncDatabase instances via keyword-only ``database`` and ``steps_database``
parameters, enabling connection pool sharing and clean testability.

Uses real AsyncMemoryDatabase instances — no mocks.
"""

import time

import pytest

from dataknobs_data.backends.memory import AsyncMemoryDatabase

from dataknobs_fsm.core.data_modes import DataHandlingMode
from dataknobs_fsm.execution.history import ExecutionHistory, ExecutionStep
from dataknobs_fsm.storage.base import StorageBackend, StorageConfig
from dataknobs_fsm.storage.database import UnifiedDatabaseStorage
from dataknobs_fsm.storage.memory import InMemoryStorage


class TestDatabaseInjection:
    """Tests for injecting pre-built AsyncDatabase instances."""

    @pytest.fixture
    def config(self) -> StorageConfig:
        return StorageConfig(backend=StorageBackend.MEMORY)

    @pytest.fixture
    def sample_history(self) -> ExecutionHistory:
        history = ExecutionHistory(
            fsm_name="test_fsm",
            execution_id="exec_1",
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

    @pytest.mark.asyncio
    async def test_injected_database_used_directly(self, config: StorageConfig) -> None:
        """Injected AsyncDatabase is stored and used; factory is not called."""
        db = AsyncMemoryDatabase()
        storage = UnifiedDatabaseStorage(config, database=db)

        assert storage._db is db
        assert storage._steps_db is db

    @pytest.mark.asyncio
    async def test_injected_database_skips_setup_backend(
        self, config: StorageConfig
    ) -> None:
        """_setup_backend() returns early when database is injected."""
        db = AsyncMemoryDatabase()
        storage = UnifiedDatabaseStorage(config, database=db)

        # initialize() calls _setup_backend(), which should be a no-op
        await storage.initialize()

        # The same db instance should still be in place (not replaced by factory)
        assert storage._db is db
        assert storage._steps_db is db
        assert storage._initialized is True

    @pytest.mark.asyncio
    async def test_save_and_load_with_injected_database(
        self, config: StorageConfig, sample_history: ExecutionHistory
    ) -> None:
        """Full round-trip: save history, load it back, verify data."""
        db = AsyncMemoryDatabase()
        storage = UnifiedDatabaseStorage(config, database=db)
        await storage.initialize()

        history_id = await storage.save_history(sample_history)
        loaded = await storage.load_history(history_id)

        assert loaded is not None
        assert loaded.fsm_name == "test_fsm"
        assert loaded.data_mode == DataHandlingMode.DIRECT
        assert loaded.total_steps == 1

    @pytest.mark.asyncio
    async def test_steps_database_defaults_to_main(
        self, config: StorageConfig
    ) -> None:
        """When only database is given, steps_database mirrors it."""
        db = AsyncMemoryDatabase()
        storage = UnifiedDatabaseStorage(config, database=db)

        assert storage._steps_db is db

    @pytest.mark.asyncio
    async def test_separate_steps_database(self, config: StorageConfig) -> None:
        """When both are given, each is used independently."""
        main_db = AsyncMemoryDatabase()
        steps_db = AsyncMemoryDatabase()
        storage = UnifiedDatabaseStorage(
            config, database=main_db, steps_database=steps_db
        )

        assert storage._db is main_db
        assert storage._steps_db is steps_db
        assert storage._db is not storage._steps_db

    @pytest.mark.asyncio
    async def test_no_injection_backward_compat(self, config: StorageConfig) -> None:
        """Omitting injection params gives identical behavior to pre-change code."""
        storage = UnifiedDatabaseStorage(config)

        # Before initialize, both are None (factory hasn't run)
        assert storage._db is None
        assert storage._steps_db is None

        # After initialize, factory creates the database
        await storage.initialize()
        assert storage._db is not None
        assert storage._steps_db is not None

    @pytest.mark.asyncio
    async def test_injection_through_subclass(self, config: StorageConfig) -> None:
        """InMemoryStorage(config, database=db) works correctly."""
        db = AsyncMemoryDatabase()
        storage = InMemoryStorage(config, database=db)

        assert storage._db is db
        assert storage._steps_db is db

        await storage.initialize()
        assert storage._db is db

    @pytest.mark.asyncio
    async def test_initialize_idempotent_with_injection(
        self, config: StorageConfig
    ) -> None:
        """Calling initialize() twice doesn't replace the injected db."""
        db = AsyncMemoryDatabase()
        storage = UnifiedDatabaseStorage(config, database=db)

        await storage.initialize()
        await storage.initialize()  # second call

        assert storage._db is db
        assert storage._steps_db is db

    @pytest.mark.asyncio
    async def test_save_step_with_injected_steps_database(
        self, config: StorageConfig, sample_history: ExecutionHistory
    ) -> None:
        """Steps are saved to the injected steps_database."""
        main_db = AsyncMemoryDatabase()
        steps_db = AsyncMemoryDatabase()
        storage = UnifiedDatabaseStorage(
            config, database=main_db, steps_database=steps_db
        )
        await storage.initialize()

        step = ExecutionStep(
            step_id="s1",
            state_name="state_a",
            network_name="main",
            timestamp=time.time(),
        )
        step.complete("to_next")

        step_id = await storage.save_step("exec_1", step)
        assert step_id == "s1"

        # Verify step is in steps_db
        loaded_steps = await storage.load_steps("exec_1")
        assert len(loaded_steps) == 1
        assert loaded_steps[0].step_id == "s1"
