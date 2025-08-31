"""Change tracking for automatic vector updates."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from ..database import Database
    from ..records import Record

logger = logging.getLogger(__name__)


@dataclass
class ChangeEvent:
    """Represents a change event for a record field."""

    record_id: str
    field_name: str
    old_value: Any
    new_value: Any
    timestamp: datetime = field(default_factory=datetime.utcnow)
    event_type: str = "update"  # create, update, delete

    def __repr__(self) -> str:
        """String representation of the event."""
        return (
            f"ChangeEvent(record={self.record_id}, field={self.field_name}, "
            f"type={self.event_type}, time={self.timestamp.isoformat()})"
        )


@dataclass
class UpdateTask:
    """Represents a pending vector update task."""

    record_id: str
    vector_fields: set[str]
    source_fields: dict[str, Any]  # source field -> new value
    priority: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    attempts: int = 0
    last_error: str | None = None

    def __lt__(self, other: UpdateTask) -> bool:
        """Enable priority queue sorting (higher priority first)."""
        if self.priority != other.priority:
            return self.priority > other.priority  # Higher priority comes first
        return self.created_at > other.created_at  # Newer tasks come first for same priority


class ChangeTracker:
    """Tracks field changes and manages automatic vector updates."""

    def __init__(
        self,
        database: Database,
        tracked_fields: list[str] | None = None,
        vector_field: str = "embedding",
        max_queue_size: int = 10000,
        batch_size: int = 100,
        process_interval: float = 5.0,
    ):
        """Initialize the change tracker with simplified API.
        
        Args:
            database: The database to track changes for
            tracked_fields: Fields to track for changes (if None, tracks all)
            vector_field: Vector field that depends on tracked fields
            max_queue_size: Maximum number of pending updates
            batch_size: Number of updates to process in a batch
            process_interval: Seconds between batch processing
        """
        self.database = database
        self.tracked_fields = tracked_fields or []
        self.vector_field = vector_field
        self.max_queue_size = max_queue_size
        self.batch_size = batch_size
        self.process_interval = process_interval

        # Field dependency mapping: source_field -> [vector_fields]
        self._dependencies: dict[str, list[str]] = defaultdict(list)
        self._vector_fields: dict[str, dict[str, Any]] = {}

        # Set up dependencies for tracked fields
        for field_name in self.tracked_fields:
            self._dependencies[field_name].append(self.vector_field)
        self._vector_fields[self.vector_field] = {"source_fields": self.tracked_fields}

        # Update queue and history
        self._update_queue: deque[UpdateTask] = deque(maxlen=max_queue_size)
        self._pending_updates: dict[str, UpdateTask] = {}  # record_id -> task
        self._change_history: deque[ChangeEvent] = deque(maxlen=1000)

        # Processing state
        self._processing_task: asyncio.Task | None = None
        self._shutdown_event = asyncio.Event()
        self._update_callbacks: list[Callable] = []

        self._initialize_dependencies()

    def _initialize_dependencies(self) -> None:
        """Initialize field dependency mappings."""
        # Use schema if available
        if hasattr(self.database, 'schema') and self.database.schema:
            for field_name, field_schema in self.database.schema.fields.items():
                if field_schema.is_vector_field():
                    self._vector_fields[field_name] = field_schema.metadata
                    source_field = field_schema.get_source_field()
                    if source_field:
                        self._dependencies[source_field].append(field_name)

    def add_update_callback(
        self,
        callback: Callable[[UpdateTask], None] | Callable[[UpdateTask], Coroutine[Any, Any, None]]
    ) -> None:
        """Add a callback to be called when updates are processed.
        
        Args:
            callback: Function to call with update tasks
        """
        self._update_callbacks.append(callback)

    def track_change(
        self,
        record_id: str,
        field_name: str,
        old_value: Any,
        new_value: Any,
        event_type: str = "update",
    ) -> bool:
        """Track a field change event.
        
        Args:
            record_id: ID of the changed record
            field_name: Name of the changed field
            old_value: Previous value
            new_value: New value
            event_type: Type of event (create, update, delete)
            
        Returns:
            True if change affects vectors and was queued
        """
        # Record the change event
        event = ChangeEvent(
            record_id=record_id,
            field_name=field_name,
            old_value=old_value,
            new_value=new_value,
            event_type=event_type,
        )
        self._change_history.append(event)

        # Check if this field affects any vectors
        affected_vectors = self._dependencies.get(field_name, [])
        if not affected_vectors:
            return False

        # Create or update task
        if record_id in self._pending_updates:
            task = self._pending_updates[record_id]
            task.vector_fields.update(affected_vectors)
            task.source_fields[field_name] = new_value
        else:
            task = UpdateTask(
                record_id=record_id,
                vector_fields=set(affected_vectors),
                source_fields={field_name: new_value},
            )
            self._pending_updates[record_id] = task

            # Add to queue if not full
            if len(self._update_queue) < self.max_queue_size:
                self._update_queue.append(task)
            else:
                logger.warning(f"Update queue full, dropping task for record {record_id}")
                del self._pending_updates[record_id]
                return False

        return True

    async def on_create(self, record: Record) -> None:
        """Handle record creation.
        
        Args:
            record: The created record
        """
        # Skip if record has no ID
        if record.id is None:
            return
            
        for field_name in record.fields.keys():
            value = record.get_value(field_name)
            if field_name in self._dependencies:
                self.track_change(
                    record_id=record.id,
                    field_name=field_name,
                    old_value=None,
                    new_value=value,
                    event_type="create",
                )

    async def on_update(
        self,
        record_id: str,
        old_data: dict[str, Any],
        new_data: dict[str, Any],
    ) -> None:
        """Handle record update.
        
        Args:
            record_id: ID of the updated record
            old_data: Previous data
            new_data: New data
        """
        for field_name in self._dependencies:
            old_value = old_data.get(field_name)
            new_value = new_data.get(field_name)

            if old_value != new_value:
                self.track_change(
                    record_id=record_id,
                    field_name=field_name,
                    old_value=old_value,
                    new_value=new_value,
                    event_type="update",
                )

    async def on_delete(self, record_id: str) -> None:
        """Handle record deletion.
        
        Args:
            record_id: ID of the deleted record
        """
        # Remove from pending updates
        if record_id in self._pending_updates:
            task = self._pending_updates[record_id]
            if task in self._update_queue:
                self._update_queue.remove(task)
            del self._pending_updates[record_id]

    def get_pending_updates(self) -> list[UpdateTask]:
        """Get list of pending update tasks.
        
        Returns:
            List of pending tasks
        """
        return list(self._update_queue)

    async def start_processing(self) -> None:
        """Start background processing of changes."""
        if self._processing_task and not self._processing_task.done():
            return  # Already running

        # Initialize content hashes for existing vector fields if we have tracked fields
        if self.tracked_fields:
            await self._initialize_content_hashes()

        self._shutdown_event.clear()
        self._processing_task = asyncio.create_task(self._process_loop())

    async def start_tracking(self, tracked_fields: list[str] | None = None, vector_field: str | None = None) -> None:
        """Legacy method for compatibility - redirects to start_processing."""
        if tracked_fields:
            self.tracked_fields = tracked_fields
        if vector_field:
            self.vector_field = vector_field

        # Update dependencies
        self._dependencies.clear()
        for field_name in self.tracked_fields:
            self._dependencies[field_name].append(self.vector_field)

        # Initialize content hashes for existing vector fields that don't have them
        await self._initialize_content_hashes()

        await self.start_processing()

    async def get_outdated_records(self) -> list[Record]:
        """Get records with outdated vector fields.
        
        Returns:
            List of records that need vector updates
        """
        import hashlib

        from ..query import Query

        # Get all records
        all_records = await self.database.search(Query())
        outdated = []

        for record in all_records:
            # Check if vector field exists
            if self.vector_field not in record.fields:
                outdated.append(record)
                continue

            # Check if any tracked field is newer than vector
            # by comparing content hashes
            vector_field = record.fields.get(self.vector_field)
            if vector_field and hasattr(vector_field, 'metadata'):
                stored_hash = vector_field.metadata.get('content_hash')

                # If no content hash is stored, auto-generate it and consider record up-to-date
                if stored_hash is None:
                    # Calculate and store content hash
                    content_parts = []
                    for field_name in self.tracked_fields:
                        field_value = record.get_value(field_name)
                        if field_value:
                            content_parts.append(str(field_value))

                    if content_parts:
                        current_content = " ".join(content_parts)
                        content_hash = hashlib.md5(current_content.encode()).hexdigest()

                        # Update the vector field metadata
                        vector_field.metadata['content_hash'] = content_hash

                        # Update the record in the database
                        try:
                            await self.database.update(record.id, record)
                            logger.debug(f"Auto-initialized content hash for record {record.id}")
                        except Exception as e:
                            logger.warning(f"Failed to auto-initialize content hash for record {record.id}: {e}")
                            # If we can't update, consider it outdated for safety
                            outdated.append(record)
                    continue

                # Compute current content hash from tracked fields
                content_parts = []
                for field_name in self.tracked_fields:
                    field_value = record.get_value(field_name)
                    if field_value:
                        content_parts.append(str(field_value))

                if content_parts:
                    current_content = " ".join(content_parts)
                    current_hash = hashlib.md5(current_content.encode()).hexdigest()

                    # If hashes don't match, the record is outdated
                    if stored_hash != current_hash:
                        outdated.append(record)
                        continue

        return outdated

    async def mark_updated(self, record_id: str) -> None:
        """Mark a record as having updated vectors.
        
        Args:
            record_id: ID of the updated record
        """
        # Remove from pending updates if present
        if record_id in self._pending_updates:
            task = self._pending_updates[record_id]
            if task in self._update_queue:
                self._update_queue.remove(task)
            del self._pending_updates[record_id]

    def get_change_history(
        self,
        record_id: str | None = None,
        field_name: str | None = None,
        limit: int = 100,
    ) -> list[ChangeEvent]:
        """Get change history with optional filters.
        
        Args:
            record_id: Filter by record ID
            field_name: Filter by field name
            limit: Maximum events to return
            
        Returns:
            List of change events
        """
        events = list(self._change_history)

        if record_id:
            events = [e for e in events if e.record_id == record_id]

        if field_name:
            events = [e for e in events if e.field_name == field_name]

        return events[-limit:]

    async def process_batch(self) -> int:
        """Process a batch of pending updates.
        
        Returns:
            Number of tasks processed
        """
        processed = 0
        batch = []

        # Get batch of tasks
        while self._update_queue and len(batch) < self.batch_size:
            task = self._update_queue.popleft()
            if task.record_id in self._pending_updates:
                batch.append(task)
                del self._pending_updates[task.record_id]

        if not batch:
            return 0

        # Process tasks
        for task in batch:
            try:
                # Call update callbacks
                for callback in self._update_callbacks:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(task)
                    else:
                        await asyncio.to_thread(callback, task)

                processed += 1

            except Exception as e:
                logger.error(f"Failed to process update for record {task.record_id}: {e}")
                task.attempts += 1
                task.last_error = str(e)

                # Retry if under max attempts (3)
                if task.attempts < 3:
                    task.priority += 1  # Increase priority for retries
                    if len(self._update_queue) < self.max_queue_size:
                        self._update_queue.append(task)
                        self._pending_updates[task.record_id] = task

        logger.info(f"Processed {processed} vector update tasks")
        return processed

    async def _process_loop(self) -> None:
        """Background processing loop for updates."""
        logger.info("Started change tracker processing loop")

        while not self._shutdown_event.is_set():
            try:
                # Process batch
                await self.process_batch()

                # Wait for interval or shutdown
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=self.process_interval
                    )
                except asyncio.TimeoutError:
                    continue

            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                await asyncio.sleep(1)

        logger.info("Stopped change tracker processing loop")

    async def stop_processing(self, timeout: float = 10.0) -> None:
        """Stop background processing.
        
        Args:
            timeout: Maximum time to wait for graceful shutdown
        """
        if not self._processing_task:
            return

        self._shutdown_event.set()

        try:
            await asyncio.wait_for(self._processing_task, timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("Processing task did not stop gracefully, cancelling")
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass

        self._processing_task = None

    def get_stats(self) -> dict[str, Any]:
        """Get tracker statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            "pending_updates": len(self._pending_updates),
            "queue_size": len(self._update_queue),
            "max_queue_size": self.max_queue_size,
            "history_size": len(self._change_history),
            "dependencies": {
                field: len(vectors)
                for field, vectors in self._dependencies.items()
            },
            "is_processing": bool(
                self._processing_task and not self._processing_task.done()
            ),
        }

    async def flush(self) -> int:
        """Process all pending updates immediately.
        
        Returns:
            Number of tasks processed
        """
        total_processed = 0

        while self._update_queue:
            processed = await self.process_batch()
            total_processed += processed

            if processed == 0:
                break

        return total_processed

    async def _initialize_content_hashes(self) -> None:
        """Initialize content hashes for existing vector fields that don't have them."""
        if not self.tracked_fields:
            return

        import hashlib

        from ..query import Query

        # Get all records
        all_records = await self.database.search(Query())

        for record in all_records:
            # Check if record has vector field but no content hash
            vector_field = record.fields.get(self.vector_field)
            if vector_field and hasattr(vector_field, 'metadata'):
                stored_hash = vector_field.metadata.get('content_hash')

                if stored_hash is None:
                    # Calculate and store content hash
                    content_parts = []
                    for field_name in self.tracked_fields:
                        field_value = record.get_value(field_name)
                        if field_value:
                            content_parts.append(str(field_value))

                    if content_parts:
                        current_content = " ".join(content_parts)
                        content_hash = hashlib.md5(current_content.encode()).hexdigest()

                        # Update the vector field metadata
                        vector_field.metadata['content_hash'] = content_hash

                        # Update the record in the database
                        try:
                            await self.database.update(record.id, record)
                            logger.debug(f"Initialized content hash for record {record.id}")
                        except Exception as e:
                            logger.warning(f"Failed to initialize content hash for record {record.id}: {e}")
