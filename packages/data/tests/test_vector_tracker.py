"""Tests for vector change tracking functionality."""

import asyncio
import os
from datetime import datetime

import pytest

# Skip all tests if PostgreSQL is not available
pytestmark = pytest.mark.skipif(
    not os.environ.get("TEST_POSTGRES", "").lower() == "true",
    reason="Vector tracker tests require TEST_POSTGRES=true and a running PostgreSQL instance with pgvector"
)

from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_data.fields import FieldType
from dataknobs_data.records import Record
from dataknobs_data.schema import DatabaseSchema, FieldSchema
from dataknobs_data.vector.tracker import (
    ChangeEvent,
    ChangeTracker,
    UpdateTask,
)


@pytest.fixture
async def memory_database():
    """Create an in-memory database for testing."""
    # Create schema with vector fields
    schema = DatabaseSchema()
    schema.add_text_field("content")
    schema.add_vector_field("embedding", dimensions=384, source_field="content")
    schema.add_text_field("title")
    schema.add_vector_field("title_embedding", dimensions=384, source_field="title")
    schema.add_text_field("description")
    schema.add_field(FieldSchema(name="metadata", type=FieldType.JSON))
    
    # Pass schema in config
    db = AsyncMemoryDatabase(config={"schema": schema})
    await db.connect()
    yield db
    await db.close()


class TestChangeEvent:
    """Test ChangeEvent data class."""
    
    def test_change_event_creation(self):
        """Test creating a change event."""
        event = ChangeEvent(
            record_id="123",
            field_name="content",
            old_value="old",
            new_value="new",
            event_type="update"
        )
        
        assert event.record_id == "123"
        assert event.field_name == "content"
        assert event.old_value == "old"
        assert event.new_value == "new"
        assert event.event_type == "update"
        assert isinstance(event.timestamp, datetime)
    
    def test_change_event_repr(self):
        """Test string representation of change event."""
        event = ChangeEvent(
            record_id="123",
            field_name="content",
            old_value="old",
            new_value="new",
        )
        
        repr_str = repr(event)
        assert "record=123" in repr_str
        assert "field=content" in repr_str
        assert "type=update" in repr_str


class TestUpdateTask:
    """Test UpdateTask data class."""
    
    def test_update_task_creation(self):
        """Test creating an update task."""
        task = UpdateTask(
            record_id="123",
            vector_fields={"embedding", "title_embedding"},
            source_fields={"content": "new content", "title": "new title"},
            priority=5
        )
        
        assert task.record_id == "123"
        assert task.vector_fields == {"embedding", "title_embedding"}
        assert task.source_fields["content"] == "new content"
        assert task.priority == 5
        assert task.attempts == 0
        assert task.last_error is None
    
    def test_update_task_comparison(self):
        """Test priority-based comparison of tasks."""
        task1 = UpdateTask(
            record_id="1",
            vector_fields=set(),
            source_fields={},
            priority=1
        )
        
        task2 = UpdateTask(
            record_id="2",
            vector_fields=set(),
            source_fields={},
            priority=5
        )
        
        # Higher priority should come first (less than)
        assert task2 < task1
        
        # Same priority - newer should come first
        task3 = UpdateTask(
            record_id="3",
            vector_fields=set(),
            source_fields={},
            priority=5
        )
        
        assert task3 < task2  # task3 is newer


class TestChangeTracker:
    """Test ChangeTracker functionality with real database."""
    
    def test_initialization(self, memory_database):
        """Test tracker initialization."""
        tracker = ChangeTracker(
            database=memory_database,
            max_queue_size=100,
            batch_size=10,
        )
        
        assert tracker.database == memory_database
        assert tracker.max_queue_size == 100
        assert tracker.batch_size == 10
        assert len(tracker._dependencies) == 2  # content -> embedding, title -> title_embedding
        assert len(tracker._vector_fields) == 2
    
    def test_track_change(self, memory_database):
        """Test tracking field changes."""
        tracker = ChangeTracker(database=memory_database)
        
        # Track change to source field
        result = tracker.track_change(
            record_id="123",
            field_name="content",
            old_value="old content",
            new_value="new content",
        )
        
        assert result is True
        assert len(tracker._pending_updates) == 1
        assert len(tracker._update_queue) == 1
        
        task = tracker._pending_updates["123"]
        assert "embedding" in task.vector_fields
        assert task.source_fields["content"] == "new content"
        
        # Track change to non-vector field
        result = tracker.track_change(
            record_id="456",
            field_name="metadata",
            old_value={"key": "old"},
            new_value={"key": "new"},
        )
        
        assert result is False  # No vectors affected
        assert "456" not in tracker._pending_updates
    
    def test_track_multiple_changes(self, memory_database):
        """Test tracking multiple changes to same record."""
        tracker = ChangeTracker(database=memory_database)
        
        # First change
        tracker.track_change(
            record_id="123",
            field_name="content",
            old_value="old content",
            new_value="new content",
        )
        
        # Second change to same record
        tracker.track_change(
            record_id="123",
            field_name="title",
            old_value="old title",
            new_value="new title",
        )
        
        # Should have single task with both fields
        assert len(tracker._pending_updates) == 1
        assert len(tracker._update_queue) == 1
        
        task = tracker._pending_updates["123"]
        assert "embedding" in task.vector_fields
        assert "title_embedding" in task.vector_fields
        assert task.source_fields["content"] == "new content"
        assert task.source_fields["title"] == "new title"
    
    def test_queue_overflow(self, memory_database):
        """Test behavior when queue is full."""
        tracker = ChangeTracker(
            database=memory_database,
            max_queue_size=3,
        )
        
        # Fill the queue
        for i in range(3):
            result = tracker.track_change(
                record_id=str(i),
                field_name="content",
                old_value="old",
                new_value=f"new {i}",
            )
            assert result is True
        
        assert len(tracker._update_queue) == 3
        
        # Try to add one more
        result = tracker.track_change(
            record_id="overflow",
            field_name="content",
            old_value="old",
            new_value="new",
        )
        
        assert result is False  # Should be rejected
        assert len(tracker._update_queue) == 3
        assert "overflow" not in tracker._pending_updates
    
    @pytest.mark.asyncio
    async def test_on_create(self, memory_database):
        """Test tracking record creation."""
        tracker = ChangeTracker(database=memory_database)
        
        record = Record(
            id="123",
            data={
                "content": "new content",
                "title": "new title",
                "metadata": {"key": "value"},
            }
        )
        
        await tracker.on_create(record)
        
        # Should track vector-related fields
        assert len(tracker._pending_updates) == 1
        task = tracker._pending_updates["123"]
        assert "embedding" in task.vector_fields
        assert "title_embedding" in task.vector_fields
    
    @pytest.mark.asyncio
    async def test_on_update(self, memory_database):
        """Test tracking record updates."""
        tracker = ChangeTracker(database=memory_database)
        
        old_data = {
            "content": "old content",
            "title": "same title",
            "metadata": {"key": "old"},
        }
        
        new_data = {
            "content": "new content",
            "title": "same title",
            "metadata": {"key": "new"},
        }
        
        await tracker.on_update("123", old_data, new_data)
        
        # Should track only changed vector source fields
        assert len(tracker._pending_updates) == 1
        task = tracker._pending_updates["123"]
        assert "embedding" in task.vector_fields
        assert "title_embedding" not in task.vector_fields  # Title didn't change
    
    @pytest.mark.asyncio
    async def test_on_delete(self, memory_database):
        """Test handling record deletion."""
        tracker = ChangeTracker(database=memory_database)
        
        # Add a pending update
        tracker.track_change(
            record_id="123",
            field_name="content",
            old_value="old",
            new_value="new",
        )
        
        assert "123" in tracker._pending_updates
        assert len(tracker._update_queue) == 1
        
        # Delete the record
        await tracker.on_delete("123")
        
        assert "123" not in tracker._pending_updates
        assert len(tracker._update_queue) == 0
    
    def test_get_pending_updates(self, memory_database):
        """Test retrieving pending updates."""
        tracker = ChangeTracker(database=memory_database)
        
        # Add some updates
        for i in range(3):
            tracker.track_change(
                record_id=str(i),
                field_name="content",
                old_value="old",
                new_value=f"new {i}",
            )
        
        pending = tracker.get_pending_updates()
        assert len(pending) == 3
        assert all(isinstance(task, UpdateTask) for task in pending)
    
    def test_get_change_history(self, memory_database):
        """Test retrieving change history."""
        tracker = ChangeTracker(database=memory_database)
        
        # Track various changes
        tracker.track_change("1", "content", "old1", "new1")
        tracker.track_change("2", "title", "old2", "new2")
        tracker.track_change("1", "title", "old3", "new3")
        tracker.track_change("3", "content", "old4", "new4")
        
        # Get all history
        history = tracker.get_change_history()
        assert len(history) == 4
        
        # Filter by record ID
        history = tracker.get_change_history(record_id="1")
        assert len(history) == 2
        assert all(e.record_id == "1" for e in history)
        
        # Filter by field name
        history = tracker.get_change_history(field_name="content")
        assert len(history) == 2
        assert all(e.field_name == "content" for e in history)
        
        # Limit results
        history = tracker.get_change_history(limit=2)
        assert len(history) == 2
    
    @pytest.mark.asyncio
    async def test_process_batch(self, memory_database):
        """Test batch processing of updates."""
        tracker = ChangeTracker(
            database=memory_database,
            batch_size=2,
        )
        
        # Track callback invocations
        processed_tasks = []
        async def test_callback(task):
            processed_tasks.append(task.record_id)
            await asyncio.sleep(0.001)  # Simulate work
        
        tracker.add_update_callback(test_callback)
        
        # Add some updates
        for i in range(5):
            tracker.track_change(
                record_id=str(i),
                field_name="content",
                old_value="old",
                new_value=f"new {i}",
            )
        
        # Process first batch
        count = await tracker.process_batch()
        assert count == 2
        assert len(processed_tasks) == 2
        assert len(tracker._update_queue) == 3
        
        # Process next batch
        count = await tracker.process_batch()
        assert count == 2
        assert len(processed_tasks) == 4
        assert len(tracker._update_queue) == 1
    
    @pytest.mark.asyncio
    async def test_error_handling_and_retry(self, memory_database):
        """Test error handling and retry logic."""
        tracker = ChangeTracker(database=memory_database)
        
        # Callback that fails first time
        call_counts = {}
        async def failing_callback(task):
            if task.record_id not in call_counts:
                call_counts[task.record_id] = 0
            call_counts[task.record_id] += 1
            
            if call_counts[task.record_id] == 1:
                raise Exception("First attempt fails")
        
        tracker.add_update_callback(failing_callback)
        
        # Add update
        tracker.track_change("123", "content", "old", "new")
        
        # First process - should fail and re-queue
        count = await tracker.process_batch()
        assert count == 0  # Failed
        assert len(tracker._update_queue) == 1  # Re-queued
        
        task = tracker._pending_updates["123"]
        assert task.attempts == 1
        assert task.last_error == "First attempt fails"
        assert task.priority == 1  # Increased priority
        
        # Second process - should succeed
        count = await tracker.process_batch()
        assert count == 1
        assert len(tracker._update_queue) == 0
    
    @pytest.mark.asyncio
    async def test_processing_loop(self, memory_database):
        """Test background processing loop."""
        tracker = ChangeTracker(
            database=memory_database,
            process_interval=0.05,  # 50ms for faster testing
        )
        
        processed_records = []
        async def track_callback(task):
            processed_records.append(task.record_id)
        
        tracker.add_update_callback(track_callback)
        
        # Start processing
        await tracker.start_processing()
        
        # Add updates while processing
        for i in range(3):
            tracker.track_change(str(i), "content", "old", f"new {i}")
            await asyncio.sleep(0.02)
        
        # Wait for processing
        await asyncio.sleep(0.2)
        
        # Stop processing
        await tracker.stop_processing()
        
        # Should have processed all updates
        assert len(processed_records) == 3
        assert tracker._processing_task is None
    
    @pytest.mark.asyncio
    async def test_flush(self, memory_database):
        """Test flushing all pending updates."""
        tracker = ChangeTracker(
            database=memory_database,
            batch_size=2,
        )
        
        processed = []
        async def callback(task):
            processed.append(task.record_id)
        
        tracker.add_update_callback(callback)
        
        # Add many updates
        for i in range(10):
            tracker.track_change(str(i), "content", "old", f"new {i}")
        
        # Flush all
        total = await tracker.flush()
        assert total == 10
        assert len(processed) == 10
        assert len(tracker._update_queue) == 0
        assert len(tracker._pending_updates) == 0
    
    def test_get_stats(self, memory_database):
        """Test statistics gathering."""
        tracker = ChangeTracker(database=memory_database)
        
        # Add some updates
        for i in range(5):
            tracker.track_change(str(i), "content", "old", f"new {i}")
        
        stats = tracker.get_stats()
        
        assert stats["pending_updates"] == 5
        assert stats["queue_size"] == 5
        assert stats["max_queue_size"] == 10000
        assert stats["history_size"] == 5
        assert stats["dependencies"]["content"] == 1  # content -> embedding
        assert stats["dependencies"]["title"] == 1  # title -> title_embedding
        assert stats["is_processing"] is False
    
    @pytest.mark.asyncio
    async def test_concurrent_callbacks(self, memory_database):
        """Test multiple concurrent callbacks."""
        tracker = ChangeTracker(database=memory_database)
        
        callback1_runs = []
        callback2_runs = []
        
        async def callback1(task):
            await asyncio.sleep(0.01)
            callback1_runs.append(task.record_id)
        
        async def callback2(task):
            await asyncio.sleep(0.005)
            callback2_runs.append(task.record_id)
        
        tracker.add_update_callback(callback1)
        tracker.add_update_callback(callback2)
        
        # Add updates
        for i in range(3):
            tracker.track_change(str(i), "content", "old", f"new {i}")
        
        # Process
        await tracker.process_batch()
        
        # Both callbacks should have been called
        assert len(callback1_runs) == 3
        assert len(callback2_runs) == 3
        assert set(callback1_runs) == set(callback2_runs)