"""Migration tools for adding vector support to existing data."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np

from dataknobs_common.callbacks import is_async_callable, run_callback

from ..fields import FieldType
from ..query import Query
from ..records import Record
from ..schema import FieldSchema
from .sync import SyncConfig, VectorTextSynchronizer
from .bulk_embed_mixin import attach_vector_field
from .content import assemble_source_text, compute_content_hash, content_hash_metadata
from .embedding import (
    TextEmbedder,
    default_model_name,
    embed_text,
    require_embedding_source,
)
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
        *,
        embedder: TextEmbedder | None = None,
    ):
        """Initialize the migration manager with simplified API.

        Args:
            source_db: Source database to migrate from
            target_db: Target database (None to migrate in-place)
            embedding_fn: Function to generate embeddings. Optional, as
                *embedder* is --- a migration that only adds the schema field
                needs neither, so the demand for a source is made where a
                vector is actually produced.
            text_fields: Fields to concatenate for embedding
            vector_field: Name of the vector field to create
            field_separator: Separator for concatenating text fields
            batch_size: Batch size for processing
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries
            model_name: Name of the embedding model
            model_version: Version of the embedding model
            config: Advanced configuration (overrides other params)
            embedder: A :class:`~dataknobs_data.vector.TextEmbedder`, in place
                of *embedding_fn*

        Raises:
            ValueError: Both *embedding_fn* and *embedder* were given.
        """
        require_embedding_source(embedder, embedding_fn, allow_neither=True)

        self.source_db = source_db
        self.target_db = target_db or source_db
        self.embedder = embedder
        self.embedding_fn = embedding_fn
        self.embedding_function = embedding_fn  # Alias for compatibility
        self.text_fields = text_fields or []
        self.vector_field = vector_field
        self.field_separator = field_separator
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.model_name = default_model_name(model_name, embedder)
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
        self, progress_callback: Callable[[MigrationStatus], Awaitable[None] | None] | None = None
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

                        if text and self.has_embedding_source:
                            embedding = await self._embed_one(text)
                            # The field the whole package builds, rather than a
                            # fifth copy of building it. The copy here recorded
                            # no content digest, and a vector carrying none is
                            # one nothing can judge --- so a synchronizer
                            # sweeping the migrated corpus called every record
                            # current however far its text had since drifted.
                            attach_vector_field(
                                record,
                                self.vector_field,
                                embedding,
                                text,
                                self.text_fields,
                                self.field_separator,
                                self.model_name,
                                self.model_version,
                            )

                        # Create in target database
                        await self.target_db.create(record)
                        self.status.migrated_records += 1

                    except Exception as e:
                        logger.error(f"Failed to migrate record {record.id}: {e}")
                        self.status.failed_records += 1
                        self.status.errors.append({"record_id": record.id, "error": str(e)})

                if progress_callback:
                    await run_callback(progress_callback, self.status)

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

    @property
    def has_embedding_source(self) -> bool:
        """Whether this migration was given anything that can produce vectors.

        Construction permits neither source, because adding the schema field
        is a useful migration on its own. So the demand is made here, at the
        three places that actually embed --- each of which used to ask
        ``self.embedding_fn is not None`` and would otherwise now have to ask
        about two attributes instead of one.
        """
        return self.embedder is not None or self.embedding_fn is not None

    async def _embed_one(self, text: str) -> Any:
        """One vector, from whichever source was configured.

        Three sites in this class embed a single text, and each carried its
        own dispatch call. Routing them through one method is what keeps a
        later change --- a retry policy, a timeout, a batch --- from being
        applied to two of the three.
        """
        return await embed_text(text, embedder=self.embedder, embedding_fn=self.embedding_fn)

    async def add_vectors_to_existing(
        self,
        vector_fields: dict[str, str],  # vector_field -> source_field mapping
        filter_query: dict[str, Any] | None = None,
        progress_callback: Callable[[MigrationStatus], Awaitable[None] | None] | None = None,
    ) -> MigrationStatus:
        """Add vector fields to existing records.

        Args:
            vector_fields: Mapping of vector field names to source text fields
            filter_query: Optional filter to select records to migrate
            progress_callback: Callback for progress updates

        Returns:
            Migration status
        """
        if not self.has_embedding_source:
            raise ValueError(
                "an embedding source is required for adding vectors: pass "
                "`embedder=` or `embedding_fn=` when building the migration"
            )

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
                    result = await self._embed_one(text)
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
                        await run_callback(progress_callback, status)

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
        if not self.has_embedding_source:
            logger.error("No embedding source configured")
            return None
        try:
            result = await self._embed_one(text)

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
        transform_fn: Callable[[Record], Record | Awaitable[Record]] | None = None,
        progress_callback: Callable[[MigrationStatus], Awaitable[None] | None] | None = None,
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
                            record = await run_callback(transform_fn, record)

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
                    await run_callback(progress_callback, status)

        finally:
            status.end_time = datetime.now(UTC)

        return status

    @classmethod
    def from_config(
        cls,
        source_db: AsyncDatabase,
        target_db: AsyncDatabase | None,
        embedding_fn: Callable[[str], np.ndarray]
        | Callable[[str], Coroutine[Any, Any, np.ndarray]]
        | None = None,
        config: MigrationConfig | None = None,
        text_fields: list[str] | None = None,
        vector_field: str = "embedding",
        model_name: str | None = None,
        model_version: str | None = None,
        *,
        embedder: TextEmbedder | None = None,
    ) -> VectorMigration:
        """Create migration from a config object for advanced use cases.

        Args:
            source_db: Source database
            target_db: Target database (None for in-place)
            embedding_fn: Function to generate embeddings, in place of
                *embedder*
            config: Migration configuration. Optional so that ``embedder=``
                may be passed by keyword without also restating a config.
            text_fields: Text field names (optional)
            vector_field: Name of the vector field
            model_name: Name of the embedding model, defaulted from *embedder*
                when one is given
            model_version: Version of the embedding model
            embedder: A :class:`~dataknobs_data.vector.TextEmbedder`

        Returns:
            Configured VectorMigration instance

        Raises:
            ValueError: Both *embedding_fn* and *embedder* were given.
        """
        config = config or MigrationConfig()
        return cls(
            source_db=source_db,
            target_db=target_db,
            embedding_fn=embedding_fn,
            embedder=embedder,
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
        | Callable[[str], Coroutine[Any, Any, np.ndarray]]
        | None = None,
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
        *,
        embedder: TextEmbedder | None = None,
    ):
        """Initialize the incremental vectorizer with simplified parameters.

        Args:
            database: The database to vectorize
            embedding_fn: Function to generate embeddings. Optional since
                *embedder* arrived; exactly one of the two is required.
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
            embedder: A :class:`~dataknobs_data.vector.TextEmbedder`, in place
                of *embedding_fn*

        Raises:
            ValueError: Neither *embedding_fn* nor *embedder* was given, or
                both were. Unlike :class:`VectorMigration`, this class exists
                only to embed, so having no source is a construction error
                rather than a supported mode.
        """
        require_embedding_source(embedder, embedding_fn)

        self.database = database
        self.embedder = embedder
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
        self.model_name = default_model_name(model_name, embedder)
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
        # Pulsed by a worker each time a record leaves its hands, so a waiter
        # counting records can wait for the next one instead of polling for
        # it. Cleared and re-awaited by the waiter, which is why it is an
        # Event rather than a counter.
        self._record_finished = asyncio.Event()
        # Ids a worker completed without writing a vector. Such a record still
        # matches the loader's `NOT_EXISTS(vector_field)` query, so without
        # this "the query returned nothing" and "there is nothing left to do"
        # are different questions --- and only the second one ends a drain.
        self._declined: set[str] = set()
        self._stats = {
            "processed": 0,
            "failed": 0,
            "skipped": 0,
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
                if await self._process_record(record):
                    self._stats["processed"] += 1
                else:
                    # Completed without writing a vector. Remembered by id so
                    # the loader stops re-fetching it: the record still matches
                    # its `NOT_EXISTS(vector_field)` query and will until
                    # something changes about the record itself.
                    self._stats["skipped"] += 1
                    if record.id is not None:
                        self._declined.add(record.id)
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                self._stats["failed"] += 1
            finally:
                # Exactly one `task_done` per `get`, whatever the outcome. This
                # is what makes the record's whole lifetime -- queued, taken,
                # embedded, written -- visible to a waiter, where `qsize()`
                # went back to zero the moment it was taken.
                self._queue.task_done()
                # ...and the pulse a *counting* waiter needs, which `join()`
                # cannot give it: `join()` answers "all of it", not "one more".
                self._record_finished.set()

        logger.info(f"Worker {worker_id} stopped")

    async def _process_record(self, record: Record) -> bool:
        """Vectorize one record, reporting whether a vector was written.

        ``False`` is not a failure --- it is the pipeline declining a record
        it completed. There are four ways that happens: the assembled text is
        empty, the record already carries a vector, the embedding function
        returned ``None``, or nothing is stored under the record's id (or it
        has none). What they share is that the record still matches the
        loader's ``NOT_EXISTS(vector_field)`` query afterwards, so a caller
        that cannot tell this outcome from a write re-fetches it forever.
        """
        try:
            # The same assembly the synchronizer and the tracker use. This was
            # a third independent copy of the loop, identical in every respect
            # including the rule that drops falsy values -- which is precisely
            # the kind of agreement that holds until it quietly does not.
            source_text = assemble_source_text(record, self.text_fields, self.field_separator)

            if not source_text:
                return False

            # Check if vector already exists
            vector_data = record.get_value(self.vector_field)
            if vector_data is not None:
                if vector_data and isinstance(vector_data, (list, np.ndarray)):
                    return False

            # Generate embedding
            embedding = await embed_text(
                str(source_text), embedder=self.embedder, embedding_fn=self.embedding_fn
            )

            if embedding is None:
                return False

            # Update record
            update_data = {
                self.vector_field: embedding.tolist()
                if isinstance(embedding, np.ndarray)
                else embedding,
            }

            # Describe the vector well enough to be judged. This class stores a
            # plain list rather than a `VectorField`, so its description lives
            # in a sidecar record field -- a different place, not a different
            # contract. It carried no digest, which made everything this class
            # wrote permanently exempt from staleness: a synchronizer sweeping
            # the same corpus found nothing to compare and called every record
            # current, however far its source text had drifted.
            #
            # Written whether or not a model was named, because the digest is
            # the half that does not depend on one.
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
            update_data[f"{self.vector_field}_metadata"] = {
                **metadata.to_dict(),
                **content_hash_metadata(
                    self.text_fields,
                    self.field_separator,
                    compute_content_hash(source_text),
                ),
            }

            # Update the record with the new vector data
            for key, value in update_data.items():
                record.set_value(key, value)
            if record.id is None:
                logger.warning("Vectorized a record with no id; nothing was persisted")
                return False
            if not await self.database.update(record.id, record):
                logger.warning(
                    "Vectorized record %s but no record is stored under that id", record.id
                )
                return False
            return True

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
                fresh = self._forget_what_the_pipeline_declined(records)

                if not fresh:
                    # Nothing left to vectorize. Say so -- it is the half of
                    # "done" the queue cannot express -- then idle until there
                    # might be, or until shutdown, whichever comes first.
                    self._source_drained.set()
                    await self._until_shutdown(asyncio.sleep(self.idle_interval))
                    continue

                self._source_drained.clear()
                for record in fresh:
                    await self._queue.put(record)
                    self._stats["queued"] += 1

            except Exception as e:
                logger.error(f"Failed to load queue: {e}")
                await self._until_shutdown(asyncio.sleep(self.error_retry_interval))

    def _forget_what_the_pipeline_declined(self, records: list[Record]) -> list[Record]:
        """The records of this page a worker has not already declined.

        `_load_pending_records` deliberately over-fetches --- its docstring
        says so --- and the pipeline completes some of what it fetches
        *without writing a vector*. Such a record still matches
        `NOT_EXISTS(vector_field)`, so the loader fetched it again on the next
        pass and every pass after that: `_source_drained` was never set, and
        every caller racing it --- `run_batch()` and `wait_for_completion()`
        on their `timeout=None` defaults --- waited forever on a corpus
        containing one record with no text.

        The workers report the outcome directly rather than having the loader
        re-derive it from the query, because the worker is the only thing that
        knows *why*. A record with no id cannot be remembered by one, so it is
        dropped here instead; `_process_record` refuses to persist it anyway.

        Bounded by the number of records the pipeline declines rather than by
        the size of the corpus: a record that is written stops matching.
        """
        fresh = []
        for record in records:
            if record.id is None:
                logger.warning("Skipping a record with no id; it cannot be persisted")
                continue
            if record.id not in self._declined:
                fresh.append(record)
        return fresh

    async def _until_shutdown(self, *awaitables: Any, timeout: float | None = None) -> bool:
        """Await the first of ``awaitables``, abandoning them all on shutdown.

        Every wait in this class is a race between the things being waited for
        and the instruction to stop. Polling a flag instead makes the shutdown
        latency the poll interval and the code an `ASYNC110` busy-wait; racing
        makes the latency zero and lets each caller see which won.

        It takes a *set* of awaitables because "done" has more than one form:
        a batch is finished when the source drains, and equally when the
        caller's record budget is spent. Each caller names the conditions that
        end its own wait; the shutdown and the timeout are added to every race
        here, so no caller can forget either.

        Args:
            awaitables: The work, delay, join or condition to wait for. Every
                one is cancelled as soon as the race is decided, so each must
                be safe to abandon.
            timeout: Seconds to wait before giving up on all of them, or
                ``None`` to wait as long as it takes.

        Returns:
            ``True`` if one of ``awaitables`` finished first, ``False`` on
            shutdown or on the timeout.

        Raises:
            Whatever an awaitable raised. A raise lands in ``done`` exactly as
            a completion does, so counting it as "the work finished" would
            report success for a wait that failed --- and the cleanup below
            gathers with ``return_exceptions=True``, which would retrieve and
            discard the exception without even a "never retrieved" warning.
            None of the conditions this class currently races can raise; the
            method is a general racer, and this is the reading of ``done``
            that stays right when one can.
        """
        work = [asyncio.ensure_future(awaitable) for awaitable in awaitables]
        shutdown = asyncio.ensure_future(self._shutdown_event.wait())
        try:
            done, _ = await asyncio.wait(
                {*work, shutdown}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
            )
            finished = [task for task in work if task in done]
            for task in finished:
                error = None if task.cancelled() else task.exception()
                if error is not None:
                    raise error
            return bool(finished)
        finally:
            for task in (*work, shutdown):
                task.cancel()
            await asyncio.gather(*work, shutdown, return_exceptions=True)

    async def _until_idle(self) -> None:
        """Return once there is nothing queued and nothing left to queue.

        Those are the two halves of "done", and the queue can express only
        one: ``join()`` says every record put on it has been taken *and*
        finished, and ``_source_drained`` says the loader's last query found
        nothing. Without the second, a wait entered immediately after
        ``start()`` returns at once --- the loader is a task that has not run
        yet, and the queue it is about to fill is empty.
        """
        while True:
            await self._queue.join()
            if self._source_drained.is_set():
                return
            # The loader has more to enqueue. Wait for its verdict rather than
            # re-checking on a timer.
            await self._source_drained.wait()

    def _attempted(self) -> int:
        """Records this vectorizer has finished with, successfully or not.

        Attempted, not vectorized: a worker bumps ``failed`` rather than
        ``processed`` when a record raises, so a budget counting only
        successes is one a corpus that cannot be embedded never meets.
        """
        return self._stats["processed"] + self._stats["failed"] + self._stats["skipped"]

    async def _until_counted(self, target: int) -> None:
        """Return once the attempted count reaches ``target``.

        ``target`` is an absolute count rather than a quantity, because
        ``_stats`` is the vectorizer's lifetime tally and is never reset ---
        so a caller wanting "``n`` more from here" passes
        ``self._attempted() + n`` and gets a budget measured from its own
        starting point rather than from the instance's.
        """
        while self._attempted() < target:
            # Cleared before the re-check, so a worker finishing in between
            # leaves the event set and the wait below returns at once rather
            # than missing the pulse.
            self._record_finished.clear()
            if self._attempted() >= target:
                return
            await self._record_finished.wait()

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

        "Nothing left" is the two conditions :meth:`_until_idle` waits for,
        and the queue can express only one of them --- `qsize()` drops to zero
        the moment a worker *takes* a record, which is before it has embedded
        or written it, and an empty queue means "not started yet" as readily
        as it means "finished". Measured on a corpus of twelve pending
        records: the implementation that read `qsize()` returned with zero of
        them vectorized.

        Args:
            timeout: Seconds to wait, or ``None`` to wait indefinitely. The
                previous signature took a poll interval and had no timeout at
                all, so a stalled worker hung the caller forever.

        Returns:
            ``True`` if everything is vectorized; ``False`` on timeout or if
            :meth:`stop` was called while waiting. A waiter is never left
            behind by a shutdown.
        """
        completed = await self._until_shutdown(self._until_idle(), timeout=timeout)
        if completed:
            logger.info("All queued records processed")
        return completed

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
            skipped=self._stats["skipped"],
            checkpoint=self._last_checkpoint,
        )

    async def run_batch(
        self, limit: int | None = None, timeout: float | None = None
    ) -> VectorizationResult:
        """Process the pending records, or at most ``limit`` of them.

        The wait ends on whichever comes first: the source draining, the
        record budget being spent, a :meth:`stop` from elsewhere, or
        ``timeout``. The vectorizer is left stopped on every one of those
        paths, including an exception.

        This used to poll ``self._queue.empty()`` for a break that could never
        be taken --- the other half of the condition was
        ``self._processing_task.done()``, and that task is the loader, whose
        loop idles rather than returning when the source drains and exits only
        on the shutdown this method sets *after* the loop. So ``run_batch()``
        with no argument never returned (``None or float("inf")``), nor did
        ``run_batch(0)`` (the same, through ``or``), nor ``run_batch(n)``
        whenever fewer than ``n`` records succeeded.

        Args:
            limit: Maximum number of records to *attempt* on this call, or
                ``None`` for all of them. A record that raises counts against
                the budget; a budget of successes could not be met by a corpus
                that fails. The budget is measured from this call rather than
                from the vectorizer's lifetime totals, so successive batches
                over one corpus each do a batch's worth of work.
            timeout: Seconds to wait before giving up, or ``None`` to wait as
                long as it takes. It bounds the shutdown too --- returning
                after the full graceful-stop budget would break the promise
                the timeout made.

        Returns:
            The work *this call* did, not the vectorizer's running totals.
        """
        original_batch_size = self.batch_size
        already_attempted = self._attempted()
        processed_before = self._stats["processed"]
        failed_before = self._stats["failed"]
        skipped_before = self._stats["skipped"]
        # `is not None`, not truthiness: `limit=0` is a request for no work,
        # and `if limit` read it as "no limit given".
        if limit is not None and limit > 0:
            self.batch_size = min(self.batch_size, limit)

        try:
            await self.start()

            done_when = [self._until_idle()]
            if limit is not None:
                done_when.append(self._until_counted(already_attempted + limit))
            await self._until_shutdown(*done_when, timeout=timeout)
        finally:
            # Restored before the stop rather than after it: `stop()` gathers
            # the workers without `return_exceptions`, so it can raise --- and
            # a raise here would both replace the body's exception and leave
            # the instance clamped to this call's `limit` for good.
            self.batch_size = original_batch_size
            if timeout is None:
                await self.stop()
            else:
                await self.stop(timeout=timeout)

        # Read after the stop, so a record a worker finished during the
        # graceful shutdown is counted rather than dropped from the total.
        return VectorizationResult(
            processed=self._stats["processed"] - processed_before,
            failed=self._stats["failed"] - failed_before,
            skipped=self._stats["skipped"] - skipped_before,
            checkpoint=self._last_checkpoint,
        )

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
    #: Records the pipeline completed without writing a vector --- an empty
    #: assembled text, an embedding function that returned ``None``, a record
    #: already carrying a vector, or nothing stored under its id. Distinct
    #: from ``failed``, which counts records that raised. Defaulted, so a
    #: caller constructing one positionally is unaffected.
    skipped: int = 0
    checkpoint: str | None = None


@dataclass
class VectorizationProgress:
    """Current progress of vectorization."""

    total_records: int
    processed_records: int
    failed_records: int
    queued_records: int
    checkpoint: str | None = None
