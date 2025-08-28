"""Migration tools for adding vector support to existing data."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Coroutine

import numpy as np

from ..fields import FieldType
from ..query import Query
from ..records import Record
from ..schema import FieldSchema
from .sync import SyncConfig, VectorTextSynchronizer
from .types import VectorMetadata

if TYPE_CHECKING:
    from ..database import Database

logger = logging.getLogger(__name__)


@dataclass
class MigrationConfig:
    """Configuration for vector migration."""
    
    batch_size: int = 100
    max_workers: int = 4
    checkpoint_interval: int = 1000
    enable_rollback: bool = True
    verify_migration: bool = True
    retry_failed: bool = True
    max_retries: int = 3
    max_consecutive_failures: int = 5  # Fail fast after this many consecutive failures
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {self.batch_size}")
        if self.max_workers <= 0:
            raise ValueError(f"Max workers must be positive, got {self.max_workers}")


@dataclass
class MigrationStatus:
    """Status of a migration operation."""
    
    total_records: int = 0
    migrated_records: int = 0
    verified_records: int = 0
    failed_records: int = 0
    rollback_records: int = 0
    errors: list[dict[str, Any]] = field(default_factory=list)
    checkpoints: list[dict[str, Any]] = field(default_factory=list)
    start_time: datetime | None = None
    end_time: datetime | None = None
    
    @property
    def success_rate(self) -> float:
        """Calculate the success rate of the migration."""
        if self.total_records == 0:
            return 0.0
        return self.migrated_records / self.total_records
    
    @property
    def duration(self) -> float | None:
        """Calculate the duration of the migration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    @property
    def records_per_second(self) -> float:
        """Calculate the migration speed."""
        duration = self.duration
        if duration and duration > 0:
            return self.migrated_records / duration
        return 0.0
    
    def add_checkpoint(self, name: str, record_id: str | None = None) -> None:
        """Add a checkpoint to the migration."""
        self.checkpoints.append({
            "name": name,
            "record_id": record_id,
            "timestamp": datetime.utcnow().isoformat(),
            "migrated": self.migrated_records,
            "failed": self.failed_records,
        })
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "total_records": self.total_records,
            "migrated_records": self.migrated_records,
            "verified_records": self.verified_records,
            "failed_records": self.failed_records,
            "rollback_records": self.rollback_records,
            "success_rate": self.success_rate,
            "duration": self.duration,
            "records_per_second": self.records_per_second,
            "errors": self.errors,
            "checkpoints": self.checkpoints,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
        }


class VectorMigration:
    """Manages migration of existing data to include vector embeddings."""
    
    def __init__(
        self,
        source_db: Database,
        target_db: Database | None = None,
        embedding_fn: Callable[[str], np.ndarray] | Callable[[str], Coroutine[Any, Any, np.ndarray]] = None,
        config: MigrationConfig | None = None,
        model_name: str | None = None,
        model_version: str | None = None,
    ):
        """Initialize the migration manager.
        
        Args:
            source_db: Source database to migrate from
            target_db: Target database (None to migrate in-place)
            embedding_fn: Function to generate embeddings
            config: Migration configuration
            model_name: Name of the embedding model
            model_version: Version of the embedding model
        """
        self.source_db = source_db
        self.target_db = target_db or source_db
        self.embedding_fn = embedding_fn
        self.config = config or MigrationConfig()
        self.config.validate()
        self.model_name = model_name
        self.model_version = model_version
        
        # Track rollback data if enabled
        self._rollback_data: dict[str, dict[str, Any]] = {}
    
    async def add_vectors_to_existing(
        self,
        vector_fields: dict[str, str],  # vector_field -> source_field mapping
        filter_query: dict[str, Any] | None = None,
        progress_callback: Callable[[MigrationStatus], None] | None = None,
    ) -> MigrationStatus:
        """Add vector fields to existing records.
        
        Args:
            vector_fields: Mapping of vector field names to source text fields
            filter_query: Optional filter to select records to migrate
            progress_callback: Callback for progress updates
            
        Returns:
            Migration status
        """
        if not self.embedding_fn:
            raise ValueError("Embedding function required for adding vectors")
        
        status = MigrationStatus(start_time=datetime.utcnow())
        
        try:
            # Get records to migrate
            if filter_query:
                # Convert filter_query dict to Query object
                query = Query()
                for field, value in filter_query.items():
                    query = query.filter(field, "==", value)
                records = await self.source_db.search(query)
            else:
                records = await self.source_db.all()
            
            status.total_records = len(records)
            logger.info(f"Starting migration of {status.total_records} records")
            
            # Create synchronizer with wrapped embedding function
            sync_config = SyncConfig(
                batch_size=self.config.batch_size,
                max_retries=self.config.max_retries,
            )
            
            # Track the last embedding exception
            last_embedding_exception = None
            
            # Create wrapper that captures exceptions
            async def embedding_wrapper(text: str) -> np.ndarray:
                nonlocal last_embedding_exception
                try:
                    if asyncio.iscoroutinefunction(self.embedding_fn):
                        result = await self.embedding_fn(text)
                    else:
                        result = await asyncio.to_thread(self.embedding_fn, text)
                    return result
                except Exception as e:
                    last_embedding_exception = e
                    raise
            
            synchronizer = VectorTextSynchronizer(
                database=self.target_db,
                embedding_fn=embedding_wrapper,
                config=sync_config,
                model_name=self.model_name,
                model_version=self.model_version,
            )
            
            # Process in batches
            consecutive_batch_failures = 0
            for i in range(0, len(records), self.config.batch_size):
                batch = records[i:i + self.config.batch_size]
                
                # Process batch with workers
                tasks = []
                for record in batch:
                    # Store original data for rollback
                    if self.config.enable_rollback:
                        # Store original field values for rollback
                        self._rollback_data[record.id] = {
                            field_name: record.get_value(field_name)
                            for field_name in record.fields.keys()
                        }
                    
                    # Add vector fields to record
                    for vector_field, source_field in vector_fields.items():
                        if record.get_value(source_field) is None:
                            continue
                        
                        # Add vector field schema if needed
                        if vector_field not in self.target_db.schema.fields:
                            source_text = record.get_value(source_field)
                            if source_text:
                                # Get dimensions from first embedding
                                sample_embedding = await self._get_embedding(str(source_text))
                                if sample_embedding is not None:
                                    dimensions = len(sample_embedding)
                                    # Add schema for vector field
                                    field_schema = FieldSchema(
                                        name=vector_field,
                                        type=FieldType.VECTOR,
                                        metadata={
                                            "dimensions": dimensions,
                                            "source_field": source_field,
                                        }
                                    )
                                    self.target_db.add_field_schema(field_schema)
                    
                    # Create migration task
                    task = self._migrate_record(
                        synchronizer,
                        record,
                        vector_fields,
                        status,
                    )
                    tasks.append(task)
                
                # Wait for batch to complete  
                results = await asyncio.gather(*tasks, return_exceptions=False)
                
                # Check for batch failures and fail fast if needed
                batch_failed_count = sum(1 for r in results if r is False)
                if batch_failed_count == len(results) and len(results) > 0:
                    consecutive_batch_failures += 1
                    # If multiple consecutive batches completely fail, re-raise the last exception
                    if consecutive_batch_failures >= 2 and self.config.enable_rollback:
                        if last_embedding_exception:
                            raise last_embedding_exception
                        else:
                            raise Exception("Migration failed: consecutive batch failures")
                else:
                    consecutive_batch_failures = 0
                
                # Checkpoint if needed
                if status.migrated_records % self.config.checkpoint_interval == 0:
                    status.add_checkpoint(
                        f"Batch {i // self.config.batch_size + 1}",
                        batch[-1].id if batch else None,
                    )
                    if progress_callback:
                        progress_callback(status)
            
            # Verify migration if enabled
            if self.config.verify_migration:
                await self._verify_migration(vector_fields, records, status)
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            if self.config.enable_rollback:
                await self._rollback(status)
            raise
            
        finally:
            status.end_time = datetime.utcnow()
        
        logger.info(
            f"Migration completed: {status.migrated_records}/{status.total_records} "
            f"migrated, {status.failed_records} failed"
        )
        
        return status
    
    async def _get_embedding(self, text: str) -> np.ndarray | None:
        """Get embedding for text."""
        try:
            if asyncio.iscoroutinefunction(self.embedding_fn):
                result = await self.embedding_fn(text)
            else:
                result = await asyncio.to_thread(self.embedding_fn, text)
            
            if isinstance(result, np.ndarray):
                return result
            elif isinstance(result, list):
                return np.array(result)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            return None
    
    async def _migrate_record(
        self,
        synchronizer: VectorTextSynchronizer,
        record: Record,
        vector_fields: dict[str, str],
        status: MigrationStatus,
    ) -> bool:
        """Migrate a single record.
        
        Returns:
            True if migration succeeded, False otherwise
        """
        try:
            # Sync vectors
            success, updated_fields = await synchronizer.sync_record(record, force=True)
            
            if success and updated_fields:
                # Update record in target database
                await self.target_db.update(record.id, record)
                status.migrated_records += 1
                return True
            else:
                status.failed_records += 1
                status.errors.append({
                    "record_id": record.id,
                    "error": "Failed to generate vectors",
                })
                return False
                
        except Exception as e:
            status.failed_records += 1
            status.errors.append({
                "record_id": record.id,
                "error": str(e),
            })
            logger.error(f"Failed to migrate record {record.id}: {e}")
            return False
    
    async def _verify_migration(
        self,
        vector_fields: dict[str, str],
        records: list[Record],
        status: MigrationStatus,
    ) -> None:
        """Verify that migration was successful."""
        logger.info("Verifying migration...")
        
        for record in records:
            try:
                # Get updated record
                migrated = await self.target_db.read(record.id)
                
                # Check vector fields
                all_present = True
                for vector_field, source_field in vector_fields.items():
                    source_value = record.get_value(source_field)
                    if source_value:
                        if vector_field not in migrated.data:
                            all_present = False
                            break
                        
                        vector_data = migrated.data[vector_field]
                        if not vector_data or not isinstance(vector_data, (list, np.ndarray)):
                            all_present = False
                            break
                
                if all_present:
                    status.verified_records += 1
                    
            except Exception as e:
                logger.error(f"Failed to verify record {record.id}: {e}")
    
    async def _rollback(self, status: MigrationStatus) -> None:
        """Rollback migration on failure."""
        if not self._rollback_data:
            return
        
        logger.info(f"Rolling back {len(self._rollback_data)} records...")
        
        for record_id, original_data in self._rollback_data.items():
            try:
                # Restore original record
                original_record = Record(id=record_id, data=original_data)
                await self.target_db.update(record_id, original_record)
                status.rollback_records += 1
            except Exception as e:
                logger.error(f"Failed to rollback record {record_id}: {e}")
    
    async def migrate_between_backends(
        self,
        field_mapping: dict[str, str] | None = None,
        transform_fn: Callable[[Record], Record] | None = None,
        progress_callback: Callable[[MigrationStatus], None] | None = None,
    ) -> MigrationStatus:
        """Migrate vector data between different backends.
        
        Args:
            field_mapping: Optional field name mapping
            transform_fn: Optional record transformation function
            progress_callback: Callback for progress updates
            
        Returns:
            Migration status
        """
        status = MigrationStatus(start_time=datetime.utcnow())
        
        try:
            # Get all records with vectors
            records = await self.source_db.all()
            status.total_records = len(records)
            
            logger.info(
                f"Migrating {status.total_records} records from "
                f"{self.source_db.__class__.__name__} to "
                f"{self.target_db.__class__.__name__}"
            )
            
            # Process in batches
            for i in range(0, len(records), self.config.batch_size):
                batch = records[i:i + self.config.batch_size]
                
                for record in batch:
                    try:
                        # Apply field mapping
                        if field_mapping:
                            new_data = {}
                            for old_field, new_field in field_mapping.items():
                                old_value = record.get_value(old_field)
                                if old_value is not None:
                                    new_data[new_field] = old_value
                            # Update record with new field mapping
                            for field_name, value in new_data.items():
                                record.set_value(field_name, value)
                        
                        # Apply transformation
                        if transform_fn:
                            record = transform_fn(record)
                        
                        # Create in target database
                        await self.target_db.create(record)
                        status.migrated_records += 1
                        
                    except Exception as e:
                        status.failed_records += 1
                        status.errors.append({
                            "record_id": record.id,
                            "error": str(e),
                        })
                        logger.error(f"Failed to migrate record {record.id}: {e}")
                
                # Progress update
                if progress_callback:
                    progress_callback(status)
            
        finally:
            status.end_time = datetime.utcnow()
        
        return status


class IncrementalVectorizer:
    """Manages incremental vectorization of large datasets."""
    
    def __init__(
        self,
        database: Database,
        embedding_fn: Callable[[str], np.ndarray] | Callable[[str], Coroutine[Any, Any, np.ndarray]],
        vector_field: str,
        source_field: str,
        batch_size: int = 100,
        max_workers: int = 4,
        model_name: str | None = None,
        model_version: str | None = None,
    ):
        """Initialize the incremental vectorizer.
        
        Args:
            database: Database to vectorize
            embedding_fn: Function to generate embeddings
            vector_field: Name of the vector field
            source_field: Name of the source text field
            batch_size: Size of processing batches
            max_workers: Maximum concurrent workers
            model_name: Name of the embedding model
            model_version: Version of the embedding model
        """
        self.database = database
        self.embedding_fn = embedding_fn
        self.vector_field = vector_field
        self.source_field = source_field
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.model_name = model_name
        self.model_version = model_version
        
        # Processing state
        self._queue: asyncio.Queue[Record] = asyncio.Queue()
        self._processing_task: asyncio.Task | None = None
        self._workers: list[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()
        self._stats = {
            "processed": 0,
            "failed": 0,
            "queued": 0,
        }
    
    async def _worker(self, worker_id: int) -> None:
        """Worker task for processing records."""
        logger.info(f"Worker {worker_id} started")
        
        while not self._shutdown_event.is_set():
            try:
                # Get record from queue with timeout
                try:
                    record = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Process record
                await self._process_record(record)
                self._stats["processed"] += 1
                
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                self._stats["failed"] += 1
        
        logger.info(f"Worker {worker_id} stopped")
    
    async def _process_record(self, record: Record) -> None:
        """Process a single record to add vectors."""
        try:
            # Get source text
            source_text = record.get_value(self.source_field)
            if not source_text:
                return
            
            # Check if vector already exists
            vector_data = record.get_value(self.vector_field)
            if vector_data is not None:
                if vector_data and isinstance(vector_data, (list, np.ndarray)):
                    return
            
            # Generate embedding
            if asyncio.iscoroutinefunction(self.embedding_fn):
                embedding = await self.embedding_fn(str(source_text))
            else:
                embedding = await asyncio.to_thread(self.embedding_fn, str(source_text))
            
            if embedding is None:
                return
            
            # Update record
            update_data = {
                self.vector_field: embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
            }
            
            # Add metadata
            if self.model_name:
                metadata = VectorMetadata(
                    dimensions=len(embedding),
                    source_field=self.source_field,
                    model_name=self.model_name,
                    model_version=self.model_version,
                    updated_at=datetime.utcnow().isoformat(),
                )
                update_data[f"{self.vector_field}_metadata"] = metadata.to_dict()
            
            # Update the record with the new vector data
            for key, value in update_data.items():
                record.set_value(key, value)
            await self.database.update(record.id, record)
            
        except Exception as e:
            logger.error(f"Failed to process record {record.id}: {e}")
            raise
    
    async def start(self) -> None:
        """Start incremental vectorization."""
        if self._processing_task and not self._processing_task.done():
            logger.warning("Vectorization already running")
            return
        
        self._shutdown_event.clear()
        
        # Start workers
        self._workers = [
            asyncio.create_task(self._worker(i))
            for i in range(self.max_workers)
        ]
        
        # Start queue loader
        self._processing_task = asyncio.create_task(self._load_queue())
        
        logger.info(f"Started incremental vectorization with {self.max_workers} workers")
    
    async def _load_queue(self) -> None:
        """Load records into processing queue."""
        while not self._shutdown_event.is_set():
            try:
                # Get records without vectors
                filter_query = {
                    self.vector_field: {"$exists": False},
                    self.source_field: {"$exists": True, "$ne": ""},
                }
                
                records = await self.database.filter(filter_query, limit=self.batch_size)
                
                if not records:
                    # No more records to process
                    await asyncio.sleep(60)  # Check again in a minute
                    continue
                
                # Add to queue
                for record in records:
                    await self._queue.put(record)
                    self._stats["queued"] += 1
                
            except Exception as e:
                logger.error(f"Failed to load queue: {e}")
                await asyncio.sleep(10)
    
    async def stop(self, timeout: float = 30.0) -> None:
        """Stop incremental vectorization.
        
        Args:
            timeout: Maximum time to wait for graceful shutdown
        """
        if not self._processing_task:
            return
        
        logger.info("Stopping incremental vectorization...")
        self._shutdown_event.set()
        
        # Cancel queue loader
        self._processing_task.cancel()
        try:
            await self._processing_task
        except asyncio.CancelledError:
            pass
        
        # Wait for workers to finish
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._workers),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.warning("Workers did not stop gracefully, cancelling")
            for worker in self._workers:
                worker.cancel()
            
            await asyncio.gather(*self._workers, return_exceptions=True)
        
        self._workers.clear()
        self._processing_task = None
        
        logger.info(f"Vectorization stopped. Stats: {self._stats}")
    
    def get_stats(self) -> dict[str, Any]:
        """Get vectorization statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            **self._stats,
            "queue_size": self._queue.qsize(),
            "workers": len(self._workers),
            "is_running": bool(
                self._processing_task and not self._processing_task.done()
            ),
        }
    
    async def wait_for_completion(self, check_interval: float = 5.0) -> None:
        """Wait for all queued records to be processed.
        
        Args:
            check_interval: Seconds between queue checks
        """
        while self._queue.qsize() > 0:
            await asyncio.sleep(check_interval)
        
        logger.info("All queued records processed")