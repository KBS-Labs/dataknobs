"""Migration tools for adding vector support to existing data."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np

from dataknobs_common.callbacks import is_async_callable

from ..fields import FieldType
from ..query import Query
from ..records import Record
from ..schema import FieldSchema
from .sync import SyncConfig, VectorTextSynchronizer
from .content import assemble_source_text
from .embedding_fn import call_embedding_fn
from .types import VectorMetadata

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Coroutine

    from ..database import AsyncDatabase

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
    def total_processed(self) -> int:
        """Total number of processed records (migrated + failed)."""
        return self.migrated_records + self.failed_records

    @property
    def failed_count(self) -> int:
        """Alias for failed_records for compatibility."""
        return self.failed_records

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
        self.checkpoints.append(
            {
                "name": name,
                "record_id": record_id,
                "timestamp": datetime.now(UTC).isoformat(),
                "migrated": self.migrated_records,
                "failed": self.failed_records,
            }
        )

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
        source_db: AsyncDatabase,
        target_db: AsyncDatabase | None = None,
        embedding_fn: Callable[[str], np.ndarray]
        | Callable[[str], Coroutine[Any, Any, np.ndarray]]
        | None = None,
        text_fields: list[str] | None = None,
        vector_field: str = "embedding",
        field_separator: str = " ",
        batch_size: int = 100,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        model_name: str | None = None,
        model_version: str | None = None,
        config: MigrationConfig | None = None,
    ):
        """Initialize the migration manager with simplified API.

        Args:
            source_db: Source database to migrate from
            target_db: Target database (None to migrate in-place)
            embedding_fn: Function to generate embeddings
            text_fields: Fields to concatenate for embedding
            vector_field: Name of the vector field to create
            field_separator: Separator for concatenating text fields
            batch_size: Batch size for processing
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries
            model_name: Name of the embedding model
            model_version: Version of the embedding model
            config: Advanced configuration (overrides other params)
        """
        self.source_db = source_db
        self.target_db = target_db or source_db
        self.embedding_fn = embedding_fn
        self.embedding_function = embedding_fn  # Alias for compatibility
        self.text_fields = text_fields or []
        self.vector_field = vector_field
        self.field_separator = field_separator
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.model_name = model_name
        self.model_version = model_version

        # Use config if provided, otherwise create from params
        if config:
            self.config = config
        else:
            self.config = MigrationConfig(
                batch_size=batch_size,
                max_retries=max_retries,
            )
        self.config.validate()

        # Migration status
        self.status = MigrationStatus()

        # Track rollback data if enabled
        self._rollback_data: dict[str, dict[str, Any]] = {}

    async def run(
        self, progress_callback: Callable[[MigrationStatus], None] | None = None
    ) -> MigrationStatus:
        """Run the complete migration.

        Args:
            progress_callback: Optional callback for progress updates

        Returns:
            Migration status
        """
        self.status = MigrationStatus(start_time=datetime.now(UTC))

        try:
            # Get all records from source
            all_records = await self.source_db.search(Query())
            self.status.total_records = len(all_records)

            # Process in batches
            for i in range(0, len(all_records), self.batch_size):
                batch = all_records[i : i + self.batch_size]

                for record in batch:
                    try:
                        text = assemble_source_text(record, self.text_fields, self.field_separator)

                        if text and self.embedding_fn is not None:
                            embedding_fn = self.embedding_fn

                            # Generate embedding
                            embedding = await call_embedding_fn(embedding_fn, text)

                            # Create VectorField
                            from ..fields import VectorField

                            vector_field_obj = VectorField(
                                value=embedding,
                                name=self.vector_field,
                                source_field=self.text_fields[0]
                                if len(self.text_fields) == 1
                                else None,
                                model_name=self.model_name,
                                model_version=self.model_version,
                            )

                            # Add to record
                            record.fields[self.vector_field] = vector_field_obj

                        # Create in target database
                        await self.target_db.create(record)
                        self.status.migrated_records += 1

                    except Exception as e:
                        logger.error(f"Failed to migrate record {record.id}: {e}")
                        self.status.failed_records += 1
                        self.status.errors.append({"record_id": record.id, "error": str(e)})

                if progress_callback:
                    progress_callback(self.status)

            self.status.end_time = datetime.now(UTC)
            return self.status

        except Exception as e:
            logger.error(f"Migration failed: {e}")
            self.status.failed_records = self.status.total_records - self.status.migrated_records
            self.status.end_time = datetime.now(UTC)
            return self.status

    async def start(self) -> None:
        """Start migration (for compatibility)."""
        # Migration runs synchronously in run() method
        pass

    async def wait_for_completion(
        self, progress_callback: Callable[[MigrationStatus], None] | None = None
    ) -> MigrationStatus:
        """Wait for migration completion (for compatibility)."""
        # Since run() is synchronous, just return current status
        return self.status

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
        if self.embedding_fn is None:
            raise ValueError("Embedding function required for adding vectors")
        embedding_fn = self.embedding_fn

        status = MigrationStatus(start_time=datetime.now(UTC))

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
                    result = await call_embedding_fn(embedding_fn, text)
                    return np.asarray(result)
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
                batch = records[i : i + self.config.batch_size]

                # Process batch with workers
                tasks = []
                for record in batch:
                    # Store original data for rollback
                    if self.config.enable_rollback and record.id is not None:
                        # Store original field values for rollback. A record
                        # with no id cannot be written back, so recording one
                        # would only produce a rollback that silently does
                        # nothing.
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
                                        },
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
            status.end_time = datetime.now(UTC)

        logger.info(
            f"Migration completed: {status.migrated_records}/{status.total_records} "
            f"migrated, {status.failed_records} failed"
        )

        return status

    async def _get_embedding(self, text: str) -> np.ndarray | None:
        """Get embedding for text."""
        if self.embedding_fn is None:
            logger.error("No embedding function configured")
            return None
        embedding_fn = self.embedding_fn
        try:
            result = await call_embedding_fn(embedding_fn, text)

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

            if success and updated_fields and record.id is not None:
                # Update record in target database
                if not await self.target_db.update(record.id, record):
                    status.failed_records += 1
                    status.errors.append(
                        {
                            "record_id": record.id,
                            "error": "no record stored under that id in the target database",
                        }
                    )
                    return False
                status.migrated_records += 1
                return True
            else:
                status.failed_records += 1
                status.errors.append(
                    {
                        "record_id": record.id,
                        "error": "Failed to generate vectors",
                    }
                )
                return False

        except Exception as e:
            status.failed_records += 1
            status.errors.append(
                {
                    "record_id": record.id,
                    "error": str(e),
                }
            )
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
                if record.id is None:
                    continue
                # Get updated record
                migrated = await self.target_db.read(record.id)
                if migrated is None:
                    # Nothing arrived in the target, which is a verification
                    # failure rather than something to read fields off of.
                    status.errors.append(
                        {
                            "record_id": record.id,
                            "error": "record is absent from the target database",
                        }
                    )
                    continue

                # Check vector fields
                all_present = True
                for vector_field, source_field in vector_fields.items():
                    source_value = record.get_value(source_field)
                    if source_value:
                        # Check if vector field exists (could be in fields or data)
                        vector_data = migrated.get_value(vector_field)
                        if vector_data is None:
                            all_present = False
                            break

                        # For VectorField objects, check the value
                        from ..fields import VectorField

                        if isinstance(migrated.fields.get(vector_field), VectorField):
                            vector_data = migrated.fields[vector_field].value

                        if not isinstance(vector_data, (list, np.ndarray)):
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
        status = MigrationStatus(start_time=datetime.now(UTC))

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
                batch = records[i : i + self.config.batch_size]

                for original_record in batch:
                    try:
                        record = original_record
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
                        status.errors.append(
                            {
                                "record_id": record.id,
                                "error": str(e),
                            }
                        )
                        logger.error(f"Failed to migrate record {record.id}: {e}")

                # Progress update
                if progress_callback:
                    progress_callback(status)

        finally:
            status.end_time = datetime.now(UTC)

        return status

    @classmethod
    def from_config(
        cls,
        source_db: AsyncDatabase,
        target_db: AsyncDatabase | None,
        embedding_fn: Callable[[str], np.ndarray]
        | Callable[[str], Coroutine[Any, Any, np.ndarray]],
        config: MigrationConfig,
        text_fields: list[str] | None = None,
        vector_field: str = "embedding",
        model_name: str | None = None,
        model_version: str | None = None,
    ) -> VectorMigration:
        """Create migration from a config object for advanced use cases.

        Args:
            source_db: Source database
            target_db: Target database (None for in-place)
            embedding_fn: Function to generate embeddings
            config: Migration configuration
            text_fields: Text field names (optional)
            vector_field: Name of the vector field
            model_name: Name of the embedding model
            model_version: Version of the embedding model

        Returns:
            Configured VectorMigration instance
        """
        return cls(
            source_db=source_db,
            target_db=target_db,
            embedding_fn=embedding_fn,
            text_fields=text_fields,
            vector_field=vector_field,
            batch_size=config.batch_size,
            model_name=model_name,
            model_version=model_version,
            config=config,
        )


class IncrementalVectorizer:
    """Manages incremental vectorization of large datasets.

    Examples:
        import numpy as np
        from dataknobs_data import database_factory

        # Create database and embedding function
        db = database_factory.create(backend="memory")

        def embedding_fn(text):
            # In practice, use a real model like sentence-transformers.
            # `default_rng` builds its own generator; `np.random.rand` would
            # read — and `np.random.seed` would mutate — state shared by the
            # whole process.
            return np.random.default_rng().random(384, dtype=np.float32)

        # Simple usage with single field
        vectorizer = IncrementalVectorizer(
            db,
            embedding_fn=embedding_fn,
            text_fields="content"  # Can be string or list
        )
        result = await vectorizer.run()

        # Resume from checkpoint
        result = await vectorizer.run(resume_from=last_checkpoint)

        # Process limited batch
        result = await vectorizer.run_batch(limit=1000)
    """

    def __init__(
        self,
        database: AsyncDatabase,
        embedding_fn: Callable[[str], np.ndarray]
        | Callable[[str], Coroutine[Any, Any, np.ndarray]],
        text_fields: list[str] | str | None = None,  # Support multiple fields
        vector_field: str = "embedding",  # Sensible default
        field_separator: str = " ",
        batch_size: int = 100,
        checkpoint_interval: int = 1000,
        max_workers: int = 4,
        model_name: str | None = None,
        model_version: str | None = None,
        idle_interval: float = 60.0,
        error_retry_interval: float = 10.0,
    ):
        """Initialize the incremental vectorizer with simplified parameters.

        Args:
            database: The database to vectorize
            embedding_fn: Function to generate embeddings
            text_fields: Text field names to concatenate for embeddings
            vector_field: Name of the vector field to create
            field_separator: Separator for concatenating multiple text fields
            batch_size: Size of processing batches
            checkpoint_interval: Records between checkpoints
            max_workers: Maximum concurrent workers
            model_name: Name of the embedding model
            model_version: Version of the embedding model
            idle_interval: Seconds to wait before re-querying once the source
                has nothing left to vectorize. Shutdown interrupts it, so a
                long interval costs nothing at stop time.
            error_retry_interval: Seconds to wait after a failed load before
                retrying. Also interrupted by shutdown.
        """
        self.database = database
        self.embedding_fn = embedding_fn
        self.embedding_function = embedding_fn  # Alias for compatibility

        # Handle text fields
        if isinstance(text_fields, str):
            text_fields = [text_fields]
        elif text_fields is None:
            # Try to auto-detect from database schema
            text_fields = self._detect_text_fields()
        self.text_fields = text_fields

        self.vector_field = vector_field
        self.field_separator = field_separator
        self.batch_size = batch_size
        self.checkpoint_interval = checkpoint_interval
        self.max_workers = max_workers
        self.model_name = model_name
        self.model_version = model_version
        self.idle_interval = idle_interval
        self.error_retry_interval = error_retry_interval

        # Processing state
        self._queue: asyncio.Queue[Record] = asyncio.Queue()
        self._processing_task: asyncio.Task | None = None
        self._workers: list[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()
        # Set by the loader when a query finds nothing left to vectorize, and
        # cleared when one finds something. The queue cannot answer this: an
        # empty queue means the loader has not enqueued *yet* just as readily
        # as it means there is nothing left to enqueue, and those are the two
        # states `wait_for_completion` has to tell apart.
        self._source_drained = asyncio.Event()
        self._stats = {
            "processed": 0,
            "failed": 0,
            "queued": 0,
        }
        self._last_checkpoint: str | None = None
        self._progress: VectorizationProgress | None = None

    def _detect_text_fields(self) -> list[str]:
        """Auto-detect text fields from database schema."""
        text_fields = []
        if hasattr(self.database, "schema") and self.database.schema:
            for field_name, field_schema in self.database.schema.fields.items():
                if field_schema.type in (FieldType.STRING, FieldType.TEXT):
                    text_fields.append(field_name)

        # Default to common field names if no schema
        if not text_fields:
            text_fields = ["content", "text", "description"]

        return text_fields

    async def _worker(self, worker_id: int) -> None:
        """Worker task for processing records."""
        logger.info(f"Worker {worker_id} started")

        while not self._shutdown_event.is_set():
            # Get record from queue with timeout
            try:
                record = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except TimeoutError:
                continue

            try:
                await self._process_record(record)
                self._stats["processed"] += 1
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                self._stats["failed"] += 1
            finally:
                # Exactly one `task_done` per `get`, whatever the outcome. This
                # is what makes the record's whole lifetime -- queued, taken,
                # embedded, written -- visible to a waiter, where `qsize()`
                # went back to zero the moment it was taken.
                self._queue.task_done()

        logger.info(f"Worker {worker_id} stopped")

    async def _process_record(self, record: Record) -> None:
        """Process a single record to add vectors."""
        try:
            # The same assembly the synchronizer and the tracker use. This was
            # a third independent copy of the loop, identical in every respect
            # including the rule that drops falsy values -- which is precisely
            # the kind of agreement that holds until it quietly does not.
            source_text = assemble_source_text(record, self.text_fields, self.field_separator)

            if not source_text:
                return

            # Check if vector already exists
            vector_data = record.get_value(self.vector_field)
            if vector_data is not None:
                if vector_data and isinstance(vector_data, (list, np.ndarray)):
                    return

            # Generate embedding
            embedding = await call_embedding_fn(self.embedding_fn, str(source_text))

            if embedding is None:
                return

            # Update record
            update_data = {
                self.vector_field: embedding.tolist()
                if isinstance(embedding, np.ndarray)
                else embedding,
            }

            # Add metadata
            if self.model_name:
                metadata = VectorMetadata(
                    dimensions=len(embedding),
                    # A list of field names, comma-joined -- which is how the
                    # only reader of this key parses it. Joining them on
                    # `field_separator` mixed a content separator into a field
                    # list, so on any non-default separator the names came back
                    # as one unsplittable string.
                    source_field=",".join(self.text_fields),
                    model_name=self.model_name,
                    model_version=self.model_version,
                    updated_at=datetime.now(UTC).isoformat(),
                )
                update_data[f"{self.vector_field}_metadata"] = metadata.to_dict()

            # Update the record with the new vector data
            for key, value in update_data.items():
                record.set_value(key, value)
            if record.id is None:
                logger.warning("Vectorized a record with no id; nothing was persisted")
                return
            if not await self.database.update(record.id, record):
                logger.warning(
                    "Vectorized record %s but no record is stored under that id", record.id
                )

        except Exception as e:
            logger.error(f"Failed to process record {record.id}: {e}")
            raise

    async def start(self) -> None:
        """Start incremental vectorization."""
        if self._processing_task and not self._processing_task.done():
            logger.warning("Vectorization already running")
            return

        self._shutdown_event.clear()
        # A restart begins with the source unproven, exactly as a first start
        # does; leaving a previous run's verdict standing would let the first
        # `wait_for_completion` of the new run return on it.
        self._source_drained.clear()

        # Start workers
        self._workers = [asyncio.create_task(self._worker(i)) for i in range(self.max_workers)]

        # Start queue loader
        self._processing_task = asyncio.create_task(self._load_queue())

        logger.info(f"Started incremental vectorization with {self.max_workers} workers")

    async def _load_pending_records(self) -> list[Record]:
        """Fetch a batch of records that still need a vector.

        `AsyncDatabase` has no `filter`, and neither does any backend --- this
        was a mongo-shaped dict passed to a method that does not exist, so the
        call raised `AttributeError` into the caller's `except Exception` and
        the queue was never loaded. Nothing said so, because the class
        annotated its database with a class name that does not exist either.

        The "at least one non-empty text field" half of the original filter is
        not expressed here: it is not a conjunction, and `_process_record`
        already returns without embedding when the assembled text is empty. So
        the query fetches a few records that are then skipped, rather than
        excluding them up front.
        """
        from ..query import Filter, Operator, Query

        return await self.database.search(
            Query(
                filters=[Filter(self.vector_field, Operator.NOT_EXISTS)],
                limit_value=self.batch_size,
            )
        )

    async def _load_queue(self) -> None:
        """Load records into processing queue.

        The loop waits for the batch it queued to be *finished* before fetching
        again. Without that it re-queries the instant it has enqueued, gets
        back the records the workers have not written yet, and enqueues them a
        second time --- a busy loop that grows the queue faster than the
        workers can empty it and never terminates.

        Waiting for the queue to merely *empty* is not enough either: a record
        taken but not yet written is gone from the queue and still absent from
        the database, so the next query returns it and it is embedded twice.
        `Queue.join()` is the distinction, and it is why the workers call
        `task_done()`.

        None of this was reachable while the fetch itself was broken: the loop
        raised into its own `except` on every pass and enqueued nothing.
        """
        while not self._shutdown_event.is_set():
            try:
                if not await self._until_shutdown(self._queue.join()):
                    break

                records = await self._load_pending_records()

                if not records:
                    # Nothing left to vectorize. Say so -- it is the half of
                    # "done" the queue cannot express -- then idle until there
                    # might be, or until shutdown, whichever comes first.
                    self._source_drained.set()
                    await self._until_shutdown(asyncio.sleep(self.idle_interval))
                    continue

                self._source_drained.clear()
                for record in records:
                    await self._queue.put(record)
                    self._stats["queued"] += 1

            except Exception as e:
                logger.error(f"Failed to load queue: {e}")
                await self._until_shutdown(asyncio.sleep(self.error_retry_interval))

    async def _until_shutdown(self, awaitable: Any) -> bool:
        """Await ``awaitable``, abandoning it if shutdown is requested first.

        Every wait in this class is a race between the thing being waited for
        and the instruction to stop. Polling a flag instead makes the shutdown
        latency the poll interval and the code an `ASYNC110` busy-wait; racing
        the two makes the latency zero and lets each caller see which won.

        Args:
            awaitable: The work, delay or join to wait for. It is cancelled if
                shutdown wins, so it must be safe to abandon.

        Returns:
            ``True`` if the awaitable finished first, ``False`` on shutdown.
        """
        work = asyncio.ensure_future(awaitable)
        shutdown = asyncio.ensure_future(self._shutdown_event.wait())
        try:
            done, _ = await asyncio.wait({work, shutdown}, return_when=asyncio.FIRST_COMPLETED)
            return work in done
        finally:
            for task in (work, shutdown):
                task.cancel()
            await asyncio.gather(work, shutdown, return_exceptions=True)

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
            await asyncio.wait_for(asyncio.gather(*self._workers), timeout=timeout)
        except TimeoutError:
            logger.warning("Workers did not stop gracefully, cancelling")
            for worker in self._workers:
                worker.cancel()

            await asyncio.gather(*self._workers, return_exceptions=True)

        self._workers.clear()
        self._processing_task = None

    async def run(
        self,
        # The body has always awaited an async callback; the annotation said it
        # would not, which is the kind of disagreement only a type checker with
        # something to narrow ever reports.
        progress_callback: Callable[[int, int, list], Awaitable[None] | None] | None = None,
        max_workers: int | None = None,
    ) -> dict[str, Any]:
        """Run the complete vectorization.

        Args:
            progress_callback: Optional callback (completed, total, current_batch)
            max_workers: Override default max_workers

        Returns:
            Results dictionary
        """
        if max_workers:
            self.max_workers = max_workers

        # Get all records that need vectors
        from ..query import Query

        all_records = await self.database.search(Query())

        to_process = []
        for record in all_records:
            # Check if needs vectorization
            if self.vector_field not in record.fields:
                # Check if has text to vectorize
                has_text = False
                for field in self.text_fields:
                    if record.get_value(field):
                        has_text = True
                        break
                if has_text:
                    to_process.append(record)

        total = len(to_process)
        processed = 0
        failed = 0

        # Process in batches
        for i in range(0, total, self.batch_size):
            batch = to_process[i : i + self.batch_size]

            for record in batch:
                try:
                    await self._process_record(record)
                    processed += 1
                except Exception as e:
                    logger.error(f"Failed to process record {record.id}: {e}")
                    failed += 1

                if progress_callback:
                    # Same classification question as the embedding function,
                    # so the same predicate answers it -- a callable object
                    # with an async ``__call__`` is the natural shape for a
                    # progress reporter that accumulates.
                    if is_async_callable(progress_callback):
                        await progress_callback(processed, total, batch)
                    else:
                        progress_callback(processed, total, batch)

        return {
            "processed": processed,
            "failed": failed,
            "total": total,
        }

    async def get_status(self) -> dict[str, Any]:
        """Get current vectorization status.

        Returns:
            Status dictionary
        """
        # Count records with and without vectors
        from ..query import Query

        all_records = await self.database.search(Query())

        total = 0
        completed = 0

        for record in all_records:
            # Check if has text fields
            has_text = False
            for field_name in self.text_fields:
                if record.get_value(field_name):
                    has_text = True
                    break

            if has_text:
                total += 1
                if self.vector_field in record.fields:
                    completed += 1

        return {
            "total": total,
            "completed": completed,
            "remaining": total - completed,
            "percentage": (completed / total * 100) if total > 0 else 0,
        }

    def get_stats(self) -> dict[str, Any]:
        """Get vectorization statistics.

        Returns:
            Dictionary of statistics
        """
        return {
            **self._stats,
            "queue_size": self._queue.qsize(),
            "workers": len(self._workers),
            "is_running": bool(self._processing_task and not self._processing_task.done()),
        }

    async def wait_for_completion(self, timeout: float | None = None) -> bool:
        """Block until there is nothing left to vectorize.

        "Nothing left" is two conditions, and the queue can only express one
        of them:

        * **The queue is joined** --- every record put on it has been taken
          *and* finished. `qsize()` drops to zero the moment a worker takes a
          record, which is before it has embedded or written it.
        * **The source is drained** --- the loader's last query found no
          pending records. Without this, a call made immediately after
          `start()` returns at once, because the loader is a task that has not
          run yet and the queue it will fill is empty. Measured on a corpus of
          twelve pending records: the previous implementation returned with
          zero of them vectorized.

        Args:
            timeout: Seconds to wait, or ``None`` to wait indefinitely. The
                previous signature took a poll interval and had no timeout at
                all, so a stalled worker hung the caller forever.

        Returns:
            ``True`` if everything is vectorized; ``False`` on timeout or if
            :meth:`stop` was called while waiting. A waiter is never left
            behind by a shutdown.
        """

        async def _idle() -> None:
            while True:
                await self._queue.join()
                if self._source_drained.is_set():
                    return
                # The loader has more to enqueue. Wait for its verdict rather
                # than re-checking on a timer.
                await self._source_drained.wait()

        idle = asyncio.ensure_future(_idle())
        shutdown = asyncio.ensure_future(self._shutdown_event.wait())
        try:
            done, _ = await asyncio.wait(
                {idle, shutdown}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
            )
            if idle in done:
                logger.info("All queued records processed")
                return True
            return False
        finally:
            for task in (idle, shutdown):
                task.cancel()
            await asyncio.gather(idle, shutdown, return_exceptions=True)

    async def run_with_checkpoint(self, resume_from: str | None = None) -> VectorizationResult:
        """Run the complete vectorization with checkpoint support.

        Args:
            resume_from: Optional checkpoint ID to resume from

        Returns:
            Vectorization result with statistics
        """
        await self.start()
        await self.wait_for_completion()

        return VectorizationResult(
            processed=self._stats["processed"],
            failed=self._stats["failed"],
            checkpoint=self._last_checkpoint,
        )

    async def run_batch(self, limit: int | None = None) -> VectorizationResult:
        """Process a limited number of records.

        Args:
            limit: Maximum number of records to process

        Returns:
            Vectorization result with statistics
        """
        # Temporarily modify batch size if limit provided
        original_batch_size = self.batch_size
        if limit:
            self.batch_size = min(self.batch_size, limit)

        try:
            await self.start()

            # Wait for limited processing
            while self._stats["processed"] < (limit or float("inf")):
                if self._queue.empty() and (
                    self._processing_task is None or self._processing_task.done()
                ):
                    break
                await asyncio.sleep(0.1)

            await self.stop()

            return VectorizationResult(
                processed=self._stats["processed"],
                failed=self._stats["failed"],
                checkpoint=self._last_checkpoint,
            )
        finally:
            self.batch_size = original_batch_size

    @property
    def progress(self) -> VectorizationProgress:
        """Get current progress."""
        return VectorizationProgress(
            total_records=self._stats.get("total", 0),
            processed_records=self._stats["processed"],
            failed_records=self._stats["failed"],
            queued_records=self._queue.qsize(),
            checkpoint=self._last_checkpoint,
        )

    async def get_checkpoint(self) -> str:
        """Get checkpoint ID for resuming."""
        # Save current progress as checkpoint
        self._last_checkpoint = f"checkpoint_{self._stats['processed']}"
        return self._last_checkpoint


@dataclass
class VectorizationResult:
    """Result of a vectorization operation."""

    processed: int
    failed: int
    checkpoint: str | None = None


@dataclass
class VectorizationProgress:
    """Current progress of vectorization."""

    total_records: int
    processed_records: int
    failed_records: int
    queued_records: int
    checkpoint: str | None = None
