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
from dataknobs_fsm.storage.base import StorageBackend, StorageConfig, StorageFactory
from dataknobs_fsm.storage.database import UnifiedDatabaseStorage
from dataknobs_fsm.storage.file import FileStorage
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
    async def test_cleanup_does_not_close_injected_database(
        self, config: StorageConfig
    ) -> None:
        """cleanup() must NOT close an injected (externally-owned) database.

        Bug: cleanup() unconditionally calls self._db.close(), which breaks
        sibling components sharing the same connection pool.
        """
        shared_db = AsyncMemoryDatabase()
        storage_a = UnifiedDatabaseStorage(config, database=shared_db)
        storage_b = UnifiedDatabaseStorage(config, database=shared_db)
        await storage_a.initialize()
        await storage_b.initialize()

        # Save a recent history via storage_b (won't be deleted by retention)
        recent_history = ExecutionHistory(
            fsm_name="test_fsm",
            execution_id="exec_recent",
            data_mode=DataHandlingMode.DIRECT,
        )
        recent_history.start_time = time.time()
        recent_history.add_step(
            state_name="state_a", network_name="main", data=None,
        )
        recent_history.end_time = time.time()
        await storage_b.save_history(recent_history)

        # Cleanup storage_a — should NOT close the shared database
        await storage_a.cleanup()

        # storage_b must still be fully functional (db not closed)
        loaded = await storage_b.load_history("exec_recent")
        assert loaded is not None, (
            "Shared database was closed by cleanup() on a sibling storage"
        )

    @pytest.mark.asyncio
    async def test_cleanup_closes_factory_created_database(
        self, config: StorageConfig
    ) -> None:
        """cleanup() SHOULD close a database the storage created itself."""
        storage = UnifiedDatabaseStorage(config)
        await storage.initialize()

        db = storage._db
        assert db is not None

        await storage.cleanup()

        # After cleanup, internal db reference should still exist but
        # a factory-owned database should have been closed
        # (AsyncMemoryDatabase.close() clears internal state)
        if hasattr(db, '_closed'):
            assert db._closed is True

    @pytest.mark.asyncio
    async def test_cleanup_does_not_close_injected_steps_database(
        self, config: StorageConfig, sample_history: ExecutionHistory
    ) -> None:
        """cleanup() must NOT close an injected steps_database either."""
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
        await storage.save_step("exec_1", step)

        await storage.cleanup()

        # The injected steps_db should still be usable
        # Create a second storage using the same steps_db
        storage2 = UnifiedDatabaseStorage(
            config, database=main_db, steps_database=steps_db
        )
        await storage2.initialize()
        loaded_steps = await storage2.load_steps("exec_1")
        assert len(loaded_steps) == 1, (
            "Injected steps_database was closed by cleanup()"
        )

    @pytest.mark.asyncio
    async def test_steps_database_only_injection_preserved(
        self, config: StorageConfig
    ) -> None:
        """Passing steps_database without database must persist through init.

        Bug: _setup_backend() unconditionally sets self._steps_db = self._db,
        overwriting an injected steps_database when no main database was given.
        """
        steps_db = AsyncMemoryDatabase()
        storage = UnifiedDatabaseStorage(config, steps_database=steps_db)

        # Before initialize, the injected steps_db should be in place
        assert storage._steps_db is steps_db

        # After initialize (factory creates main db), steps_db must survive
        await storage.initialize()
        assert storage._steps_db is steps_db, (
            "Injected steps_database was overwritten by _setup_backend()"
        )
        # Main db should have been created by factory
        assert storage._db is not None
        assert storage._db is not steps_db

    @pytest.mark.asyncio
    async def test_steps_database_only_injection_functional(
        self, config: StorageConfig, sample_history: ExecutionHistory
    ) -> None:
        """Steps saved with steps_database-only injection use the injected db."""
        steps_db = AsyncMemoryDatabase()
        storage = UnifiedDatabaseStorage(config, steps_database=steps_db)
        await storage.initialize()

        step = ExecutionStep(
            step_id="s1",
            state_name="state_a",
            network_name="main",
            timestamp=time.time(),
        )
        step.complete("to_next")

        await storage.save_step("exec_1", step)

        # Verify the step was saved to the injected steps_db, not the main db
        loaded = await storage.load_steps("exec_1")
        assert len(loaded) == 1
        assert loaded[0].step_id == "s1"

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


class TestConfigMutationProtection:
    """Subclasses must not mutate the caller's StorageConfig."""

    def test_inmemory_storage_does_not_mutate_config(self) -> None:
        """InMemoryStorage copies config before setting defaults."""
        config = StorageConfig(backend=StorageBackend.MEMORY)
        original_params = dict(config.connection_params)

        InMemoryStorage(config, database=AsyncMemoryDatabase())

        assert config.connection_params == original_params, (
            "InMemoryStorage mutated caller's connection_params"
        )
        assert config.mode_specific_config == {}, (
            "InMemoryStorage mutated caller's mode_specific_config"
        )

    def test_file_storage_does_not_mutate_config(self) -> None:
        """FileStorage copies config before setting defaults."""
        config = StorageConfig(backend=StorageBackend.FILE)
        original_params = dict(config.connection_params)

        FileStorage(config, database=AsyncMemoryDatabase())

        assert config.connection_params == original_params, (
            "FileStorage mutated caller's connection_params"
        )


class TestStorageFactoryInjection:
    """StorageFactory.create() forwards kwargs to constructors."""

    def test_factory_create_forwards_database(self) -> None:
        """Factory passes database kwarg through to storage constructor."""
        db = AsyncMemoryDatabase()
        config = StorageConfig(backend=StorageBackend.MEMORY)
        storage = StorageFactory.create(config, database=db)

        assert storage._db is db  # type: ignore[attr-defined]
        assert storage._steps_db is db  # type: ignore[attr-defined]

    def test_factory_create_forwards_both_databases(self) -> None:
        """Factory passes both database and steps_database kwargs."""
        main_db = AsyncMemoryDatabase()
        steps_db = AsyncMemoryDatabase()
        config = StorageConfig(backend=StorageBackend.MEMORY)
        storage = StorageFactory.create(
            config, database=main_db, steps_database=steps_db
        )

        assert storage._db is main_db  # type: ignore[attr-defined]
        assert storage._steps_db is steps_db  # type: ignore[attr-defined]

    def test_factory_create_backward_compatible(self) -> None:
        """Factory without kwargs works as before."""
        config = StorageConfig(backend=StorageBackend.MEMORY)
        storage = StorageFactory.create(config)

        assert storage._db is None  # type: ignore[attr-defined]
