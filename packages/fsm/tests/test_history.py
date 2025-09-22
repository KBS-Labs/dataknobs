"""Tests for execution history and storage."""

import asyncio
import tempfile
import time
from typing import Any, Dict, List

import pytest

from dataknobs_fsm.core.data_modes import DataHandlingMode
from dataknobs_fsm.execution.history import (
    ExecutionHistory,
    ExecutionStep,
    ExecutionStatus
)
from dataknobs_fsm.storage import (
    StorageBackend,
    StorageConfig,
    StorageFactory,
    InMemoryStorage,
    FileStorage,
    UnifiedDatabaseStorage
)


class TestExecutionHistory:
    """Test execution history tracking."""
    
    def test_create_history(self):
        """Test creating execution history."""
        history = ExecutionHistory(
            fsm_name="test_fsm",
            execution_id="exec_123",
            data_mode=DataHandlingMode.COPY
        )
        
        assert history.fsm_name == "test_fsm"
        assert history.execution_id == "exec_123"
        assert history.data_mode == DataHandlingMode.COPY
        assert history.total_steps == 0
        assert history.failed_steps == 0
        assert history.skipped_steps == 0
    
    def test_add_step(self):
        """Test adding execution steps."""
        history = ExecutionHistory(
            fsm_name="test_fsm",
            execution_id="exec_123"
        )
        
        # Add first step
        step1 = history.add_step(
            state_name="start",
            network_name="main",
            data={"value": 1}
        )
        
        assert step1.state_name == "start"
        assert step1.network_name == "main"
        assert step1.status == ExecutionStatus.PENDING
        assert history.total_steps == 1
        
        # Add second step
        step2 = history.add_step(
            state_name="process",
            network_name="main",
            data={"value": 2}
        )
        
        assert history.total_steps == 2
        assert len(history.get_path_to_current()) == 2
    
    def test_update_step(self):
        """Test updating execution steps."""
        history = ExecutionHistory(
            fsm_name="test_fsm",
            execution_id="exec_123"
        )
        
        step = history.add_step("start", "main")
        
        # Start the step
        step.start()
        assert step.status == ExecutionStatus.IN_PROGRESS
        assert step.start_time is not None
        
        # Complete the step
        step.complete(arc_taken="to_process")
        assert step.status == ExecutionStatus.COMPLETED
        assert step.arc_taken == "to_process"
        assert step.end_time is not None
        assert step.duration is not None
        
        # Update via history
        success = history.update_step(
            step.step_id,
            metrics={"processed": 10}
        )
        assert success
        assert step.metrics["processed"] == 10
    
    def test_step_failure(self):
        """Test handling step failures."""
        history = ExecutionHistory(
            fsm_name="test_fsm",
            execution_id="exec_123"
        )
        
        step = history.add_step("process", "main")
        error = ValueError("Processing failed")
        
        step.fail(error)
        assert step.status == ExecutionStatus.FAILED
        assert step.error == error
        
        history.update_step(
            step.step_id,
            status=ExecutionStatus.FAILED,
            error=error
        )
        assert history.failed_steps == 1
    
    def test_branching_paths(self):
        """Test branching execution paths."""
        history = ExecutionHistory(
            fsm_name="test_fsm",
            execution_id="exec_123"
        )
        
        # Main path
        step1 = history.add_step("start", "main")
        step2 = history.add_step("process", "main")
        
        # Branch from step1
        step3 = history.add_step(
            "alternate",
            "main",
            parent_step_id=step1.step_id
        )
        
        paths = history.get_all_paths()
        assert len(paths) == 2  # Two paths: start->process and start->alternate
    
    def test_resource_tracking(self):
        """Test resource usage tracking."""
        history = ExecutionHistory(
            fsm_name="test_fsm",
            execution_id="exec_123"
        )
        
        step = history.add_step("process", "main")
        
        # Add resource usage
        step.add_resource_usage("database", {
            "queries": 5,
            "duration": 1.2
        })
        step.add_resource_usage("llm", {
            "tokens": 500,
            "duration": 3.5
        })
        
        usage = history.get_resource_usage()
        assert "database" in usage
        assert "llm" in usage
        assert usage["database"]["total_calls"] == 1
        assert usage["llm"]["total_calls"] == 1
    
    def test_stream_tracking(self):
        """Test stream progress tracking."""
        history = ExecutionHistory(
            fsm_name="test_fsm",
            execution_id="exec_123"
        )
        
        step = history.add_step("stream_process", "main")
        
        # Update stream progress
        step.update_stream_progress(
            chunks=10,
            records=1000,
            current_position=5000
        )
        
        assert step.chunks_processed == 10
        assert step.records_processed == 1000
        assert step.stream_progress["position"] == 5000
        
        progress = history.get_stream_progress()
        assert progress["total_chunks"] == 10
        assert progress["total_records"] == 1000
    
    def test_mode_specific_storage(self):
        """Test mode-specific storage strategies."""
        # COPY mode - full storage
        copy_history = ExecutionHistory(
            fsm_name="test",
            execution_id="copy_123",
            data_mode=DataHandlingMode.COPY,
            enable_data_snapshots=True
        )
        step = copy_history.add_step("state", "net", data={"key": "value"})
        assert step.data_snapshot is not None
        
        # REFERENCE mode - reference only
        ref_history = ExecutionHistory(
            fsm_name="test",
            execution_id="ref_123",
            data_mode=DataHandlingMode.REFERENCE,
            enable_data_snapshots=True
        )
        step = ref_history.add_step("state", "net", data={"key": "value"})
        if step.data_snapshot:
            assert "type" in step.data_snapshot
            assert "id" in step.data_snapshot
        
        # DIRECT mode - minimal
        direct_history = ExecutionHistory(
            fsm_name="test",
            execution_id="direct_123",
            data_mode=DataHandlingMode.DIRECT,
            enable_data_snapshots=True
        )
        step = direct_history.add_step("state", "net", data={"key": "value"})
        if step.data_snapshot:
            assert "type" in step.data_snapshot
    
    def test_pruning(self):
        """Test history pruning based on max_depth."""
        history = ExecutionHistory(
            fsm_name="test",
            execution_id="exec_123",
            data_mode=DataHandlingMode.DIRECT,
            max_depth=3
        )
        
        # Add more steps than max_depth
        for i in range(5):
            history.add_step(f"state_{i}", "main")
        
        # For DIRECT mode, should prune to keep only recent steps
        path = history.get_path_to_current()
        assert len(path) <= 3
    
    def test_summary(self):
        """Test execution summary generation."""
        history = ExecutionHistory(
            fsm_name="test_fsm",
            execution_id="exec_123"
        )
        
        # Add some steps
        step1 = history.add_step("start", "main")
        step1.complete()
        
        step2 = history.add_step("process", "main")
        step2.fail(Exception("Error"))
        history.update_step(step2.step_id, status=ExecutionStatus.FAILED, error=Exception("Error"))
        
        step3 = history.add_step("skip", "main")
        step3.skip("Not needed")
        history.update_step(step3.step_id, status=ExecutionStatus.SKIPPED)
        
        history.finalize()
        
        summary = history.get_summary()
        assert summary["fsm_name"] == "test_fsm"
        assert summary["execution_id"] == "exec_123"
        assert summary["total_steps"] == 3
        assert summary["failed_steps"] == 1
        assert summary["skipped_steps"] == 1
        assert summary["completed_steps"] == 1
        assert summary["duration"] is not None


@pytest.mark.asyncio
class TestStorageBackends:
    """Test storage backend implementations."""
    
    async def test_memory_storage_basic(self):
        """Test basic memory storage operations."""
        config = StorageConfig(
            backend=StorageBackend.MEMORY,
            connection_params={"max_size": 100}
        )
        
        storage = StorageFactory.create(config)
        await storage.initialize()
        
        # Create and save history
        history = ExecutionHistory(
            fsm_name="test",
            execution_id="mem_123"
        )
        history.add_step("start", "main")
        history.finalize()
        
        # Save
        history_id = await storage.save_history(history, {"test": "metadata"})
        assert history_id == "mem_123"
        
        # Load
        loaded = await storage.load_history(history_id)
        assert loaded is not None
        assert loaded.fsm_name == "test"
        assert loaded.execution_id == "mem_123"
        
        # Query
        results = await storage.query_histories({"fsm_name": "test"})
        assert len(results) > 0
        assert results[0]["id"] == "mem_123"
        
        # Delete
        deleted = await storage.delete_history(history_id)
        assert deleted
        
        # Verify deleted
        loaded = await storage.load_history(history_id)
        assert loaded is None
    
    async def test_file_storage(self):
        """Test file storage operations."""
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a file path within the directory, not the directory itself
            file_path = os.path.join(tmpdir, "fsm_history.json")
            config = StorageConfig(
                backend=StorageBackend.FILE,
                connection_params={
                    "path": file_path,
                    "format": "json"
                }
            )
            
            storage = StorageFactory.create(config)
            await storage.initialize()
            
            # Save history
            history = ExecutionHistory(
                fsm_name="test",
                execution_id="file_123"
            )
            history.add_step("start", "main")
            
            history_id = await storage.save_history(history)
            
            # Verify file exists
            # Note: Actual file location depends on dataknobs_data implementation
            assert history_id == "file_123"
            
            # Load back
            loaded = await storage.load_history(history_id)
            assert loaded is not None
            assert loaded.execution_id == "file_123"
    
    async def test_step_storage(self):
        """Test storing individual steps."""
        config = StorageConfig(backend=StorageBackend.MEMORY)
        storage = StorageFactory.create(config)
        await storage.initialize()
        
        # Create step
        step = ExecutionStep(
            step_id="step_1",
            state_name="process",
            network_name="main",
            timestamp=time.time()
        )
        step.complete("to_next")
        
        # Save step
        step_id = await storage.save_step("exec_123", step)
        assert step_id == "step_1"
        
        # Load steps
        steps = await storage.load_steps("exec_123")
        assert len(steps) == 1
        assert steps[0].step_id == "step_1"
        assert steps[0].arc_taken == "to_next"
        
        # Filter steps
        steps = await storage.load_steps(
            "exec_123",
            filters={"state_name": "process"}
        )
        assert len(steps) == 1
        
        steps = await storage.load_steps(
            "exec_123",
            filters={"state_name": "other"}
        )
        assert len(steps) == 0
    
    async def test_statistics(self):
        """Test storage statistics."""
        config = StorageConfig(backend=StorageBackend.MEMORY)
        storage = StorageFactory.create(config)
        await storage.initialize()
        
        # Save multiple histories
        for i in range(3):
            history = ExecutionHistory(
                fsm_name=f"fsm_{i % 2}",
                execution_id=f"exec_{i}",
                data_mode=DataHandlingMode.COPY if i == 0 else DataHandlingMode.REFERENCE
            )
            history.add_step("start", "main")
            if i == 1:
                step = history.add_step("fail", "main")
                step.fail(Exception("Error"))
                history.update_step(step.step_id, status=ExecutionStatus.FAILED, error=Exception("Error"))
            history.finalize()
            await storage.save_history(history)
        
        # Get overall stats
        stats = await storage.get_statistics()
        assert stats["total_histories"] == 3
        assert "mode_distribution" in stats
        assert stats["backend_type"] == "memory"
        
        # Get specific stats
        stats = await storage.get_statistics("exec_1")
        assert stats["execution_id"] == "exec_1"
        assert stats["failed_steps"] == 1
    
    async def test_cleanup(self):
        """Test cleanup of old histories."""
        config = StorageConfig(backend=StorageBackend.MEMORY)
        storage = StorageFactory.create(config)
        await storage.initialize()
        
        # Save old history
        old_history = ExecutionHistory(
            fsm_name="old",
            execution_id="old_123"
        )
        old_history.start_time = time.time() - (10 * 86400)  # 10 days ago
        old_history.finalize()
        await storage.save_history(old_history)
        
        # Save recent history
        new_history = ExecutionHistory(
            fsm_name="new",
            execution_id="new_123"
        )
        new_history.finalize()
        await storage.save_history(new_history)
        
        # Save failed history (old)
        failed_history = ExecutionHistory(
            fsm_name="failed",
            execution_id="failed_123"
        )
        failed_history.start_time = time.time() - (10 * 86400)
        step = failed_history.add_step("fail", "main")
        step.fail(Exception("Error"))
        failed_history.update_step(step.step_id, status=ExecutionStatus.FAILED, error=Exception("Error"))
        failed_history.finalize()
        await storage.save_history(failed_history)
        
        # Cleanup old, keep failed
        before = time.time() - (5 * 86400)  # 5 days ago
        deleted = await storage.cleanup(before_timestamp=before, keep_failed=True)
        
        assert deleted == 1  # Only old_123 deleted
        
        # Verify
        assert await storage.load_history("old_123") is None
        assert await storage.load_history("new_123") is not None
        assert await storage.load_history("failed_123") is not None
    
    async def test_mode_specific_config(self):
        """Test mode-specific storage configuration."""
        config = StorageConfig(
            backend=StorageBackend.MEMORY,
            mode_specific_config={
                DataHandlingMode.COPY: {"store_snapshots": True},
                DataHandlingMode.REFERENCE: {"compress": True},
                DataHandlingMode.DIRECT: {"max_history": 5}
            }
        )
        
        storage = StorageFactory.create(config)
        await storage.initialize()
        
        # Test each mode
        for mode in [DataHandlingMode.COPY, DataHandlingMode.REFERENCE, DataHandlingMode.DIRECT]:
            history = ExecutionHistory(
                fsm_name="test",
                execution_id=f"{mode.value}_123",
                data_mode=mode
            )
            history.add_step("start", "main")
            history.finalize()
            
            await storage.save_history(history)
            loaded = await storage.load_history(f"{mode.value}_123")
            assert loaded is not None
            assert loaded.data_mode == mode
    
    async def test_concurrent_access(self):
        """Test concurrent access to storage."""
        config = StorageConfig(backend=StorageBackend.MEMORY)
        storage = StorageFactory.create(config)
        await storage.initialize()
        
        # Create multiple histories concurrently
        async def save_history(i: int):
            history = ExecutionHistory(
                fsm_name="concurrent",
                execution_id=f"concurrent_{i}"
            )
            history.add_step(f"step_{i}", "main")
            history.finalize()
            return await storage.save_history(history)
        
        # Save 10 histories concurrently
        tasks = [save_history(i) for i in range(10)]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 10
        assert all(f"concurrent_{i}" in results for i in range(10))
        
        # Query all
        histories = await storage.query_histories({"fsm_name": "concurrent"})
        assert len(histories) == 10


class TestStorageFactory:
    """Test storage factory."""
    
    def test_factory_registration(self):
        """Test backend registration."""
        backends = StorageFactory.get_available_backends()
        
        assert StorageBackend.MEMORY in backends
        assert StorageBackend.FILE in backends
        assert StorageBackend.SQLITE in backends
        assert StorageBackend.POSTGRES in backends
    
    def test_factory_create(self):
        """Test creating storage instances."""
        # Memory backend
        config = StorageConfig(backend=StorageBackend.MEMORY)
        storage = StorageFactory.create(config)
        assert isinstance(storage, InMemoryStorage)
        
        # File backend with temporary file
        with tempfile.NamedTemporaryFile(suffix='.json', delete=True) as tmp:
            config = StorageConfig(
                backend=StorageBackend.FILE,
                connection_params={"path": tmp.name}
            )
            storage = StorageFactory.create(config)
            assert isinstance(storage, FileStorage)
        
        # Database backends all use UnifiedDatabaseStorage
        for backend in [StorageBackend.SQLITE, StorageBackend.POSTGRES]:
            config = StorageConfig(backend=backend)
            storage = StorageFactory.create(config)
            assert isinstance(storage, UnifiedDatabaseStorage)
    
    def test_invalid_backend(self):
        """Test creating storage with invalid backend."""
        # Create a backend that doesn't exist
        config = StorageConfig(backend=None)  # type: ignore
        config.backend = "invalid"  # type: ignore
        
        with pytest.raises(ValueError, match="Unknown storage backend"):
            StorageFactory.create(config)
