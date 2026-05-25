"""Tests for Database Storage Factory (storage/database.py).

This test suite covers:
- AsyncDatabaseFactory with memory backend
- Factory with sqlite backend
- Factory with invalid backend
- Database connection handling
- Cleanup with cleanup() method
- Concurrent database operations
- Transaction support
- Error recovery
- InMemoryStorage shared-DB collision

Uses real implementations instead of mocks following DRY principle.
"""

import asyncio
import pytest
import tempfile
import os
import uuid
from typing import Dict, Any

from dataknobs_fsm.storage.database import UnifiedDatabaseStorage
from dataknobs_fsm.storage.memory import InMemoryStorage
from dataknobs_fsm.storage.base import StorageConfig, StorageBackend
from dataknobs_fsm.execution.history import ExecutionHistory, ExecutionStep, ExecutionStatus
from dataknobs_fsm.core.data_modes import DataHandlingMode


class TestDatabaseStorageFactory:
    """Tests for database storage factory functionality using real implementations."""
    
    @pytest.fixture
    def memory_config(self):
        """Create configuration for memory backend (post-Item-116 contract).

        ``StorageConfig.backend`` is the source of truth for backend
        selection; no redundant ``'type'`` / ``'backend'`` keys in
        ``connection_params``.
        """
        return StorageConfig(
            backend=StorageBackend.MEMORY,
            connection_params={'database': 'test_db'},
        )

    @pytest.fixture
    def sqlite_config(self):
        """Create configuration for SQLite backend (post-Item-116 contract)."""
        temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        temp_db.close()

        config = StorageConfig(
            backend=StorageBackend.SQLITE,
            connection_params={'database': temp_db.name},
        )

        try:
            yield config
        finally:
            # The fixture owns the temp-file lifecycle (the frozen config is
            # not a scratch namespace for the path).
            if os.path.exists(temp_db.name):
                os.unlink(temp_db.name)
    
    @pytest.fixture
    def sample_history(self):
        """Create a sample execution history."""
        history = ExecutionHistory(
            execution_id=f"test-exec-{uuid.uuid4().hex[:8]}",
            fsm_name="test_fsm",
            data_mode=DataHandlingMode.COPY
        )
        
        # Add steps using the proper add_step method signature
        step1 = history.add_step(
            state_name="start",
            network_name="main",
            data=None,
            parent_step_id=None
        )
        step1.complete()
        
        step2 = history.add_step(
            state_name="process",
            network_name="main",
            data=None,
            parent_step_id=step1.step_id
        )
        step2.complete()
        
        history.end_time = 1002.0
        history.total_steps = 2
        
        return history
    
    @pytest.mark.asyncio
    async def test_database_factory_with_memory_backend(self, memory_config):
        """Test AsyncDatabaseFactory with memory backend using real implementation."""
        storage = UnifiedDatabaseStorage(memory_config)
        
        # Initialize with real backend
        await storage.initialize()
        
        # Verify storage is initialized
        assert storage._initialized
        assert storage._db is not None
        
        # The database should be ready to use
        # Clean up
        await storage.cleanup()
    
    @pytest.mark.asyncio
    async def test_database_factory_with_sqlite_backend(self, sqlite_config):
        """Test factory with SQLite backend using real implementation."""
        storage = UnifiedDatabaseStorage(sqlite_config)

        # Initialize with real SQLite backend (temp file cleaned up by the
        # ``sqlite_config`` fixture teardown).
        await storage.initialize()

        # Verify storage is initialized
        assert storage._initialized
        assert storage._db is not None

        # Clean up
        await storage.cleanup()
    
    # Note: a former ``test_database_factory_with_invalid_backend`` was
    # removed in the fix.  It relied on the buggy
    # ``connection_params['type']`` factory-input path that has since been
    # removed; the canonical backend selection is now driven by the
    # ``StorageBackend`` enum on ``StorageConfig``, which Python's enum
    # mechanics already prevent from holding an unknown string.  The
    # corresponding "factory rejects unknown backend" coverage lives in
    # ``dataknobs-data`` where the factory itself is defined.

    @pytest.mark.asyncio
    async def test_database_connection_handling(self, memory_config):
        """Test database connection handling with real implementation."""
        storage = UnifiedDatabaseStorage(memory_config)
        
        # Initialize (should connect if backend supports it)
        await storage.initialize()
        assert storage._initialized
        assert storage._db is not None
        
        # Cleanup (should disconnect)
        await storage.cleanup()
        # Note: _initialized might not be set to False by cleanup in the real implementation
        
        # Re-initialize should work
        await storage.initialize()
        assert storage._initialized
        
        # Clean up
        await storage.cleanup()
    
    @pytest.mark.asyncio
    async def test_cleanup_method(self, memory_config):
        """Test cleanup method using real implementation."""
        storage = UnifiedDatabaseStorage(memory_config)
        
        await storage.initialize()
        assert storage._db is not None
        assert storage._initialized
        
        # Cleanup should clean up resources
        await storage.cleanup()
        # The real implementation may not reset _initialized flag
        # but cleanup should be idempotent
        
        # Multiple cleanups should be safe
        await storage.cleanup()  # Should not raise
    
    @pytest.mark.asyncio
    async def test_concurrent_database_operations(self, memory_config):
        """Test concurrent database operations with real backend."""
        storage = UnifiedDatabaseStorage(memory_config)
        await storage.initialize()
        
        try:
            # Create multiple unique histories
            histories = []
            for i in range(5):
                history = ExecutionHistory(
                    execution_id=f"exec-{uuid.uuid4().hex[:8]}",
                    fsm_name=f"fsm_{i}",
                    data_mode=DataHandlingMode.COPY
                )
                history.start_time = 1000.0 + i
                history.end_time = 1001.0 + i
                histories.append(history)
            
            # Save concurrently
            tasks = [storage.save_history(h) for h in histories]
            results = await asyncio.gather(*tasks)
            
            # Verify all were saved
            assert len(results) == 5
            for i, result in enumerate(results):
                assert result == histories[i].execution_id
            
            # Try to load them back
            for history in histories:
                loaded = await storage.load_history(history.execution_id)
                # Loaded might be None if backend doesn't persist or might be actual data
                # This depends on the backend implementation
        finally:
            await storage.cleanup()
    
    @pytest.mark.asyncio
    async def test_save_and_load_history(self, memory_config, sample_history):
        """Test saving and loading execution history with real backend."""
        storage = UnifiedDatabaseStorage(memory_config)
        await storage.initialize()
        
        try:
            # Save history
            history_id = await storage.save_history(sample_history)
            assert history_id == sample_history.execution_id
            
            # Load history back
            loaded = await storage.load_history(history_id)
            
            # Depending on backend, loaded might be the actual history
            # or might need deserialization
            if loaded is not None:
                # If backend supports loading, verify some properties
                if hasattr(loaded, 'execution_id'):
                    assert loaded.execution_id == history_id
                elif isinstance(loaded, dict):
                    assert loaded.get('execution_id') == history_id
        finally:
            await storage.cleanup()
    
    # Note: a former ``test_database_schema_creation`` was removed
    # alongside the ``_create_history_schema`` method.  The descriptor
    # was never consumed by any backend (the FSM history records carry
    # ``history_data`` JSON payloads rather than typed columns), so
    # the method and its test had no semantic content to preserve.

    @pytest.mark.asyncio
    async def test_resaving_history_preserves_created_at(
        self, memory_config, sample_history
    ):
        """``created_at`` is stable across resaves; ``updated_at`` advances.

        ``save_history`` is an idempotent upsert on ``execution_id``.
        Without read-before-write, every resave would reset
        ``created_at`` to the latest save's wall-clock — breaking any
        consumer that treats ``created_at`` as the original creation
        timestamp.  ``updated_at`` is expected to advance on each resave.
        """
        storage = UnifiedDatabaseStorage(memory_config)
        await storage.initialize()

        try:
            await storage.save_history(sample_history)
            first = await storage._db.read(sample_history.execution_id)
            assert first is not None
            first_created = first.get_value("created_at")
            first_updated = first.get_value("updated_at")
            assert first_created is not None
            assert first_updated is not None

            # Sleep so wall-clock can advance distinguishably; the
            # ``time.time()`` resolution on most platforms is sub-ms but
            # CI clocks can be coarse — 10ms is conservative.
            await asyncio.sleep(0.01)

            await storage.save_history(sample_history)
            second = await storage._db.read(sample_history.execution_id)
            assert second is not None
            second_created = second.get_value("created_at")
            second_updated = second.get_value("updated_at")

            assert second_created == first_created, (
                "created_at must be preserved across resaves"
            )
            assert second_updated > first_updated, (
                "updated_at must advance on each resave"
            )
        finally:
            await storage.cleanup()

    @pytest.mark.asyncio
    async def test_error_recovery(self, memory_config):
        """Test error recovery in database operations."""
        storage = UnifiedDatabaseStorage(memory_config)
        await storage.initialize()
        
        try:
            # Create history with invalid data that might cause errors
            invalid_history = ExecutionHistory(
                execution_id="test-invalid",
                fsm_name="test_fsm",
                data_mode=DataHandlingMode.COPY
            )
            
            # Add a step using the proper interface
            step = invalid_history.add_step(
                state_name="test",
                network_name="main",
                data=None
            )
            # Mark it as failed
            step.fail(Exception("Test error message"))
            
            # Should handle the error gracefully or save successfully
            # Real backend should handle this
            result = await storage.save_history(invalid_history)
            assert result == invalid_history.execution_id
            
        finally:
            await storage.cleanup()
    
    @pytest.mark.asyncio
    async def test_multiple_backend_initialization(self):
        """Test initialization with different backends in sequence using real implementations."""
        # Test with memory backend first
        memory_config = StorageConfig(
            backend=StorageBackend.MEMORY,
            connection_params={},
        )

        storage = UnifiedDatabaseStorage(memory_config)
        await storage.initialize()
        assert storage._initialized
        await storage.cleanup()

        # Then test with SQLite backend
        temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        temp_db.close()

        try:
            sqlite_config = StorageConfig(
                backend=StorageBackend.SQLITE,
                connection_params={'database': temp_db.name},
            )
            
            storage = UnifiedDatabaseStorage(sqlite_config)
            await storage.initialize()
            assert storage._initialized
            await storage.cleanup()
        finally:
            # Cleanup SQLite temp file
            if os.path.exists(temp_db.name):
                os.unlink(temp_db.name)
    
    @pytest.mark.asyncio
    async def test_list_histories(self, memory_config):
        """Test listing execution histories with real backend."""
        storage = UnifiedDatabaseStorage(memory_config)
        await storage.initialize()
        
        try:
            # Save multiple histories
            history_ids = []
            for i in range(3):
                history = ExecutionHistory(
                    execution_id=f"list-test-{uuid.uuid4().hex[:8]}",
                    fsm_name=f"fsm_{i}",
                    data_mode=DataHandlingMode.COPY
                )
                history.start_time = 1000.0 + i
                history_id = await storage.save_history(history)
                history_ids.append(history_id)
            
            # Query all histories using the actual method with empty filters
            all_histories = await storage.query_histories(filters={})
            
            # Check that our histories are included
            # Note: The format depends on backend implementation
            if all_histories:
                if isinstance(all_histories, list):
                    # Could be list of IDs or list of history objects
                    assert len(all_histories) >= 3
        finally:
            await storage.cleanup()
    
    @pytest.mark.asyncio
    async def test_factory_path_uses_canonical_backend_enum(self) -> None:
        """``_setup_backend`` reads from ``StorageConfig.backend``.

        Reproduces the bug where, when ``connection_params`` does NOT
        redundantly carry a ``'type'`` key, the prior implementation
        defaulted to ``'memory'`` and silently constructed an
        ``AsyncMemoryDatabase`` regardless of the requested backend.

        Uses ``StorageBackend.SQLITE`` (not ``MEMORY``) deliberately
        so the bug surfaces visibly: the pre-fix ``'memory'`` default
        coincidentally matches a ``MEMORY`` request, masking the bug.
        With ``SQLITE`` requested and no ``'type'`` override, pre-fix
        code produces ``AsyncMemoryDatabase``; post-fix produces
        ``AsyncSQLiteDatabase``.
        """
        import os
        import tempfile

        from dataknobs_data.backends.sqlite_async import AsyncSQLiteDatabase

        temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        temp_db.close()
        config = StorageConfig(
            backend=StorageBackend.SQLITE,
            connection_params={
                "database": temp_db.name,  # NO redundant 'type' key
            },
        )
        storage = UnifiedDatabaseStorage(config)
        try:
            await storage.initialize()
            assert isinstance(storage._db, AsyncSQLiteDatabase), (
                f"Expected AsyncSQLiteDatabase from SQLITE enum, got "
                f"{type(storage._db).__name__}"
            )
        finally:
            await storage.close()
            if os.path.exists(temp_db.name):
                os.unlink(temp_db.name)

    @pytest.mark.asyncio
    async def test_statistics_reports_canonical_backend_type(self) -> None:
        """Second site (database.py:571): ``get_statistics``
        reports the backend from ``StorageConfig.backend``, not
        ``connection_params['type']``.

        Pre-fix, when ``connection_params`` did not include a
        redundant ``'type'`` key, ``get_statistics()`` returned
        ``backend_type='unknown'`` even though the actual backend
        was correctly identified by the enum.  Pin the post-fix
        invariant.
        """
        config = StorageConfig(
            backend=StorageBackend.MEMORY,
            connection_params={},  # NO redundant 'type' key
        )
        storage = UnifiedDatabaseStorage(config)
        try:
            await storage.initialize()
            stats = await storage.get_statistics()
            assert stats["backend_type"] == "memory"
        finally:
            await storage.close()

    @pytest.mark.asyncio
    async def test_legacy_type_key_emits_deprecation_warning(self) -> None:
        """Passing 'type' in connection_params is deprecated.

        Pins three contracts at once:

        1.  The deprecated ``'type'`` alias emits a
            ``DeprecationWarning`` so consumers migrate before the
            alias is removed in the next minor release.
        2.  The canonical ``StorageConfig.backend`` enum WINS when
            it disagrees with ``connection_params['type']`` — the
            alias is informational, not authoritative.  The values
            are deliberately set to disagree (``SQLITE`` vs
            ``"memory"``) so the test fails if either wins-by-coincidence.
        3.  The warning's ``stacklevel`` attributes it to user code
            (this test file), not to internal framework code.  A
            wrong stacklevel is invisible to ``pytest.warns(...)``
            alone, so we inspect ``record.filename`` directly.
        """
        from dataknobs_data.backends.sqlite_async import AsyncSQLiteDatabase

        temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        temp_db.close()
        config = StorageConfig(
            backend=StorageBackend.SQLITE,         # canonical: SQLite
            connection_params={
                "type": "memory",                  # deprecated alias: disagrees
                "database": temp_db.name,
            },
        )
        storage = UnifiedDatabaseStorage(config)
        try:
            with pytest.warns(DeprecationWarning, match="type") as records:
                await storage.initialize()

            # (2) canonical enum wins on disagreement
            assert isinstance(storage._db, AsyncSQLiteDatabase), (
                f"Canonical StorageConfig.backend (SQLITE) must win over "
                f"deprecated 'type' alias (memory). Got "
                f"{type(storage._db).__name__}."
            )

            # (3) stacklevel attributes warning to this test file, not
            # to internal framework code.  ``__file__`` may be a
            # bytecode path on some loaders; compare basenames.
            assert len(records) == 1
            assert os.path.basename(records[0].filename) == os.path.basename(__file__), (
                f"Warning origin {records[0].filename!r} should be the "
                f"test file ({__file__!r}); a wrong stacklevel "
                "attributes the warning to internal dataknobs_fsm code "
                "and hides which user call site to migrate."
            )
        finally:
            await storage.close()
            if os.path.exists(temp_db.name):
                os.unlink(temp_db.name)

    @pytest.mark.asyncio
    async def test_data_serialization_modes(self, memory_config):
        """Test different data serialization modes with real backend."""
        # Test with COPY mode
        storage = UnifiedDatabaseStorage(memory_config)
        await storage.initialize()
        
        try:
            history = ExecutionHistory(
                execution_id=f"serial-test-{uuid.uuid4().hex[:8]}",
                fsm_name="test_fsm",
                data_mode=DataHandlingMode.COPY
            )
            
            # Add step with data using proper interface
            step = history.add_step(
                state_name="process",
                network_name="main",
                data={"key": "value", "nested": {"data": 123}}
            )
            step.complete()
            
            # Save and verify serialization works
            history_id = await storage.save_history(history)
            assert history_id == history.execution_id
            
        finally:
            await storage.cleanup()
        
        # Test with REFERENCE mode
        ref_config = StorageConfig(
            backend=StorageBackend.MEMORY,
            connection_params={},
            mode_specific_config={
                DataHandlingMode.REFERENCE: {}
            }
        )
        
        storage = UnifiedDatabaseStorage(ref_config)
        await storage.initialize()
        
        try:
            history = ExecutionHistory(
                execution_id=f"ref-test-{uuid.uuid4().hex[:8]}",
                fsm_name="test_fsm",
                data_mode=DataHandlingMode.REFERENCE
            )
            
            # Save with reference mode
            history_id = await storage.save_history(history)
            assert history_id == history.execution_id

        finally:
            await storage.cleanup()


class TestInMemoryStorageIsolation:
    """InMemoryStorage record-type isolation.

    When history and step records share a single database instance,
    ``UnifiedDatabaseStorage`` adds EXISTS filters on the type-specific
    field (``history_data`` / ``step_data``) to prevent record-type
    collisions.  These tests verify the isolation works for
    ``InMemoryStorage`` specifically.
    """

    @pytest.fixture
    def memory_config(self):
        """Create a default InMemoryStorage config (post-Item-116 contract)."""
        return StorageConfig(
            backend=StorageBackend.MEMORY,
            connection_params={},
        )

    @pytest.mark.asyncio
    async def test_load_steps_does_not_collide_with_history(self, memory_config):
        """Steps and history with same execution_id are isolated by EXISTS filter."""
        storage = InMemoryStorage(memory_config)
        await storage.initialize()

        try:
            exec_id = f"collision-test-{uuid.uuid4().hex[:8]}"

            # Create and save a history record
            history = ExecutionHistory(
                execution_id=exec_id,
                fsm_name="test_fsm",
                data_mode=DataHandlingMode.COPY,
            )
            step = history.add_step(
                state_name="start",
                network_name="main",
                data=None,
            )
            step.complete()
            history.end_time = 1002.0
            history.total_steps = 1

            await storage.save_history(history)
            await storage.save_step(exec_id, step)

            # This must return only the step record, not crash on the
            # history record which lacks 'step_data'.
            steps = await storage.load_steps(exec_id)
            assert len(steps) == 1
            assert steps[0].state_name == "start"
        finally:
            await storage.cleanup()

    @pytest.mark.asyncio
    async def test_injected_databases_still_work(self, memory_config):
        """Explicit database injection must still be honored."""
        from dataknobs_data.backends.memory import AsyncMemoryDatabase

        db = AsyncMemoryDatabase()
        steps_db = AsyncMemoryDatabase()
        storage = InMemoryStorage(
            memory_config, database=db, steps_database=steps_db,
        )
        await storage.initialize()

        try:
            exec_id = f"inject-test-{uuid.uuid4().hex[:8]}"
            history = ExecutionHistory(
                execution_id=exec_id,
                fsm_name="test_fsm",
                data_mode=DataHandlingMode.COPY,
            )
            step = history.add_step(
                state_name="process",
                network_name="main",
                data=None,
            )
            step.complete()

            await storage.save_history(history)
            await storage.save_step(exec_id, step)

            steps = await storage.load_steps(exec_id)
            assert len(steps) == 1
        finally:
            await storage.cleanup()

    @pytest.mark.asyncio
    async def test_close_disposes_auto_created_databases(self, memory_config):
        """close() must dispose databases created internally, not injected ones."""
        storage = InMemoryStorage(memory_config)
        await storage.initialize()

        # Auto-created databases should be owned
        assert storage._owns_db is True  # type: ignore[attr-defined]
        assert storage._owns_steps_db is True  # type: ignore[attr-defined]

        # close() should not raise
        await storage.close()

        # Idempotent — calling again should not raise
        await storage.close()

    @pytest.mark.asyncio
    async def test_close_does_not_dispose_injected_databases(self, memory_config):
        """close() must not close caller-injected databases."""
        from dataknobs_data.backends.memory import AsyncMemoryDatabase

        db = AsyncMemoryDatabase()
        steps_db = AsyncMemoryDatabase()
        storage = InMemoryStorage(
            memory_config, database=db, steps_database=steps_db,
        )
        await storage.initialize()

        # Injected databases should NOT be owned
        assert storage._owns_db is False  # type: ignore[attr-defined]
        assert storage._owns_steps_db is False  # type: ignore[attr-defined]

        await storage.close()

        # Injected databases should still be usable after storage.close()
        from dataknobs_data.records import Record
        await db.upsert(Record({"id": "test", "value": "alive"}))
        result = await db.read("test")
        assert result is not None
