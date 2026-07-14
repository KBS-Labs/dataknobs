"""SQLite backend implementation with sync and async support."""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
from dataknobs_common.structured_config import StructuredConfigConsumer

from ..database import SyncDatabase, enforce_content_version
from ..exceptions import DuplicateRecordError, RecordValidationError
from ..query import Query
from ..query_logic import ComplexQuery
from ..records import Record
from ..vector.bulk_embed_mixin import BulkEmbedMixin
from ..vector.mixins import VectorOperationsMixin
from ..vector.python_vector_search import PythonVectorSearchMixin
from .config import SyncSQLiteDatabaseConfig
from .sql_base import (
    SQLQueryBuilder,
    SQLRecordSerializer,
    SQLTableManager,
    is_duplicate_key_error,
)
from .sqlite_mixins import SQLiteVectorSupport
from .vector_config_mixin import VectorConfigMixin

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import ClassVar
    from ..streaming import StreamConfig, StreamResult
    from ..vector.types import DistanceMetric, VectorSearchResult


logger = logging.getLogger(__name__)


class SyncSQLiteDatabase(  # type: ignore[misc]
    StructuredConfigConsumer[SyncSQLiteDatabaseConfig],
    SyncDatabase,
    VectorConfigMixin,
    PythonVectorSearchMixin,  # Provides python_vector_search_sync
    BulkEmbedMixin,  # Must come before VectorOperationsMixin to override bulk_embed_and_store
    VectorOperationsMixin,
    SQLiteVectorSupport,
    SQLRecordSerializer,  # Use the standard SQL serializer
):
    """Synchronous SQLite database backend.

    Constructed through :class:`SyncSQLiteDatabaseConfig` — every
    documented config key is a typed field on that dataclass, so
    ``self.config`` is the typed config (not a dict) and the
    ``from_config`` / factory paths share one construction route.
    """

    CONFIG_CLS: ClassVar[type[SyncSQLiteDatabaseConfig]] = SyncSQLiteDatabaseConfig

    def _setup(self) -> None:
        """Derive backend attributes from the typed config.

        Runs after the cooperative base chain has set ``self.schema`` and
        run ``_initialize`` (a no-op for SQLite — connection setup is
        deferred to :meth:`connect`).
        """
        cfg = self.config
        self._apply_vector_config(cfg.vector_enabled, cfg.vector_metric)
        self._init_vector_state()

        self.db_path = cfg.path
        self.table_name = cfg.table
        self.timeout = cfg.timeout
        self.check_same_thread = cfg.check_same_thread
        self.journal_mode = cfg.journal_mode
        self.synchronous = cfg.synchronous
        self.auto_create_table = cfg.auto_create_table

        self.query_builder = SQLQueryBuilder(self.table_name, dialect="sqlite", param_style="qmark")
        self.table_manager = SQLTableManager(self.table_name, dialect="sqlite")

        self.conn: sqlite3.Connection | None = None
        self._connected = False

    def connect(self) -> None:
        """Connect to the SQLite database."""
        if self._connected:
            return

        # Create directory if needed for file-based database
        if self.db_path != ":memory:":
            db_file = Path(self.db_path)
            db_file.parent.mkdir(parents=True, exist_ok=True)

        # Connect to database
        self.conn = sqlite3.connect(
            self.db_path,
            timeout=self.timeout,
            check_same_thread=self.check_same_thread
        )

        # Enable row factory for dict-like access
        self.conn.row_factory = sqlite3.Row

        # Configure SQLite for better performance
        self._configure_sqlite()

        # Create table if it doesn't exist
        self._ensure_table()

        self._connected = True
        logger.info(f"Connected to SQLite database: {self.db_path}")

    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            self._connected = False
            logger.info(f"Disconnected from SQLite database: {self.db_path}")

    def _configure_sqlite(self) -> None:
        """Configure SQLite settings for performance."""
        if not self.conn:
            return

        cursor = self.conn.cursor()

        # Set journal mode if specified
        if self.journal_mode:
            cursor.execute(f"PRAGMA journal_mode = {self.journal_mode}")
            logger.debug(f"Set journal_mode to {self.journal_mode}")

        # Set synchronous mode if specified
        if self.synchronous:
            cursor.execute(f"PRAGMA synchronous = {self.synchronous}")
            logger.debug(f"Set synchronous to {self.synchronous}")

        # Enable foreign keys
        cursor.execute("PRAGMA foreign_keys = ON")

        # Optimize for performance
        cursor.execute("PRAGMA temp_store = MEMORY")
        cursor.execute("PRAGMA mmap_size = 30000000000")

        cursor.close()

    def _ensure_table(self) -> None:
        """Ensure the table exists.

        When ``auto_create_table=True`` (default), runs ``CREATE TABLE IF NOT
        EXISTS …``. When ``auto_create_table=False``, verifies the table is
        present and raises ``RuntimeError`` if it isn't.
        """
        if not self.conn:
            raise RuntimeError("Database not connected. Call connect() first.")

        if not self.auto_create_table:
            exists_sql, params = self.table_manager.get_table_exists_sql()
            cursor = self.conn.cursor()
            try:
                cursor.execute(exists_sql, params)
                row = cursor.fetchone()
                exists = bool(row[0]) if row else False
            finally:
                cursor.close()
            if not exists:
                raise RuntimeError(
                    f"Table {self.table_name} does not exist and "
                    "auto_create_table is disabled. Run your migrations "
                    "before starting the application."
                )
            return

        cursor = self.conn.cursor()
        try:
            cursor.executescript(self.table_manager.get_create_table_sql())
            self.conn.commit()
        finally:
            cursor.close()

    def _check_connection(self) -> None:
        """Check if database is connected."""
        if not self._connected or not self.conn:
            raise RuntimeError("Database not connected. Call connect() first.")

    def create(self, record: Record) -> str:
        """Create a new record."""
        self._check_connection()

        # Update vector dimensions tracking if needed
        if self._has_vector_fields(record):
            self._update_vector_dimensions(record)

        # Use centralized method to prepare record
        record, storage_id = self._prepare_record_for_storage(record)

        # Use the standard SQL serializer
        data_json = self.record_to_json(record)
        metadata_json = json.dumps(record.metadata) if record.metadata else None

        # Build insert query for SQLite's standard table structure
        query = f"INSERT INTO {self.table_manager.qualified_table} (id, data, metadata) VALUES (?, ?, ?)"
        params = [storage_id, data_json, metadata_json]

        cursor = self.conn.cursor()

        try:
            cursor.execute(query, params)
            self.conn.commit()
            return storage_id
        except sqlite3.IntegrityError as e:
            self.conn.rollback()
            if is_duplicate_key_error(e):
                raise DuplicateRecordError(storage_id) from e
            # NOT NULL / CHECK / other column constraint — surface truthfully
            # instead of mislabeling it as a duplicate id.
            raise RecordValidationError(str(e)) from e
        finally:
            cursor.close()

    def read(self, id: str) -> Record | None:
        """Read a record by ID."""
        self._check_connection()

        query, params = self.query_builder.build_read_query(id)
        cursor = self.conn.cursor()

        try:
            cursor.execute(query, params)
            row = cursor.fetchone()

            if row:
                # Use the standard SQL serializer
                record = self.row_to_record(dict(row))
                # Use centralized method to prepare record
                return self._prepare_record_from_storage(record, id)
            return None
        finally:
            cursor.close()

    def update(
        self, id: str, record: Record, *, expected_version: str | None = None
    ) -> bool:
        """Update an existing record.

        Args:
            id: The record ID to update
            record: The record data to update with
            expected_version: Optional optimistic-concurrency token from
                ``get_version(id)`` (a content hash for SQLite). When provided,
                the read-compare-write runs inside one transaction so the
                compare-and-set is atomic within the connection; a stale token
                raises ``ConcurrencyError`` instead of overwriting. When
                ``None`` the update is unconditional, byte-identical to prior
                behavior.

        Returns:
            True if the record was updated, False if no record with the given ID exists

        Raises:
            ConcurrencyError: If ``expected_version`` does not match the
                record's current version token.
        """
        self._check_connection()

        # Update vector dimensions tracking if needed
        if self._has_vector_fields(record):
            self._update_vector_dimensions(record)

        # Use the standard SQL serializer
        data_json = self.record_to_json(record)
        metadata_json = json.dumps(record.metadata) if record.metadata else None

        # Build update query
        query = f"UPDATE {self.table_manager.qualified_table} SET data = ?, metadata = ? WHERE id = ?"
        params = [data_json, metadata_json, id]

        # Conditional write: compare the current content-hash token before
        # issuing the UPDATE. Reusing read() guarantees the token compared
        # here is byte-identical to the one get_version() returned. On a
        # single connection the read and write are effectively serialized;
        # cross-connection atomicity is out of scope (see the module docs on
        # the in-process content-hash backends).
        if expected_version is not None:
            current = self.read(id)
            if current is None:
                return False
            enforce_content_version(id, expected_version, current)

        cursor = self.conn.cursor()

        try:
            cursor.execute(query, params)
            self.conn.commit()
            rows_affected = cursor.rowcount

            if rows_affected == 0:
                logger.warning(f"Update affected 0 rows for id={id}. Record may not exist.")

            return rows_affected > 0
        finally:
            cursor.close()

    def delete(self, id: str, *, expected_version: str | None = None) -> bool:
        """Delete a record by ID.

        When ``expected_version`` is provided the current content-hash token is
        compared before the ``DELETE``; a stale token raises
        ``ConcurrencyError`` and a missing record returns ``False``. Reusing
        ``read()`` keeps the compared token byte-identical to ``get_version()``.
        On a single connection the read and delete are serialized;
        cross-connection atomicity is out of scope (see the module docs on the
        in-process content-hash backends). When ``None`` the delete is
        unconditional, byte-identical to prior behavior.
        """
        self._check_connection()

        if expected_version is not None:
            current = self.read(id)
            if current is None:
                return False
            enforce_content_version(id, expected_version, current)

        query, params = self.query_builder.build_delete_query(id)
        cursor = self.conn.cursor()

        try:
            cursor.execute(query, params)
            self.conn.commit()
            return cursor.rowcount > 0
        finally:
            cursor.close()

    def exists(self, id: str) -> bool:
        """Check if a record exists."""
        self._check_connection()

        query, params = self.query_builder.build_exists_query(id)
        cursor = self.conn.cursor()

        try:
            cursor.execute(query, params)
            result = cursor.fetchone()
            return result is not None
        finally:
            cursor.close()

    def clear(self) -> int:
        """Clear all records from the database."""
        self._check_connection()
        
        cursor = self.conn.cursor()
        try:
            # Get count before clearing
            cursor.execute(f"SELECT COUNT(*) FROM {self.table_manager.qualified_table}")
            count = cursor.fetchone()[0]

            # Clear the table
            cursor.execute(f"DELETE FROM {self.table_manager.qualified_table}")
            self.conn.commit()
            
            return count
        finally:
            cursor.close()
    
    def search(self, query: Query | ComplexQuery) -> list[Record]:
        """Search for records matching a query."""
        self._check_connection()

        # Handle ComplexQuery with native SQL support
        if isinstance(query, ComplexQuery):
            sql_query, params = self.query_builder.build_complex_search_query(query)
        else:
            sql_query, params = self.query_builder.build_search_query(query)

        cursor = self.conn.cursor()

        try:
            cursor.execute(sql_query, params)
            rows = cursor.fetchall()

            records = []
            for row in rows:
                row_dict = dict(row)
                record = self.row_to_record(row_dict)

                # Populate storage_id from database ID
                record.storage_id = str(row_dict['id'])

                records.append(record)

            # Apply field projection if specified
            if query.fields:
                records = [r.project(query.fields) for r in records]

            return records
        finally:
            cursor.close()

    def count(self, query: Query | None = None) -> int:
        """Count records matching a query."""
        self._check_connection()

        sql_query, params = self.query_builder.build_count_query(query)
        cursor = self.conn.cursor()

        try:
            cursor.execute(sql_query, params)
            result = cursor.fetchone()
            return result[0] if result else 0
        finally:
            cursor.close()

    def create_batch(self, records: list[Record]) -> list[str]:
        """Create multiple records efficiently using a single query.
        
        Uses multi-value INSERT for better performance.
        """
        if not records:
            return []

        self._check_connection()

        # Use the shared batch create query builder
        query, params, ids = self.query_builder.build_batch_create_query(records)

        cursor = self.conn.cursor()
        try:
            # Execute the batch insert in a transaction
            cursor.execute("BEGIN TRANSACTION")
            cursor.execute(query, params)
            self.conn.commit()

            # Return the generated IDs
            return ids
        except Exception:
            self.conn.rollback()
            raise
        finally:
            cursor.close()

    def update_batch(self, updates: list[tuple[str, Record]]) -> list[bool]:
        """Update multiple records efficiently using a single query.
        
        Uses CASE expressions for batch updates, similar to PostgreSQL.
        """
        if not updates:
            return []

        self._check_connection()

        # Use the shared batch update query builder
        query, params = self.query_builder.build_batch_update_query(updates)

        cursor = self.conn.cursor()
        try:
            # Execute the batch update in a transaction
            cursor.execute("BEGIN TRANSACTION")
            cursor.execute(query, params)
            self.conn.commit()

            # Check which records were actually updated
            # SQLite doesn't have RETURNING, so we need to verify each ID
            update_ids = [record_id for record_id, _ in updates]
            placeholders = ", ".join(["?" for _ in update_ids])
            check_query = f"SELECT id FROM {self.table_manager.qualified_table} WHERE id IN ({placeholders})"
            cursor.execute(check_query, update_ids)
            existing_ids = {row[0] for row in cursor.fetchall()}

            # Return results for each update
            results = []
            for record_id, _ in updates:
                results.append(record_id in existing_ids)

            return results
        except Exception:
            self.conn.rollback()
            raise
        finally:
            cursor.close()

    def delete_batch(self, ids: list[str]) -> list[bool]:
        """Delete multiple records efficiently using a single query.
        
        Uses single DELETE with IN clause for better performance.
        """
        if not ids:
            return []

        self._check_connection()

        # Check which IDs exist before deletion
        placeholders = ", ".join(["?" for _ in ids])
        check_query = f"SELECT id FROM {self.table_manager.qualified_table} WHERE id IN ({placeholders})"

        cursor = self.conn.cursor()
        try:
            cursor.execute(check_query, ids)
            existing_ids = {row[0] for row in cursor.fetchall()}

            # Use the shared batch delete query builder
            query, params = self.query_builder.build_batch_delete_query(ids)

            # Execute the batch delete in a transaction
            cursor.execute("BEGIN TRANSACTION")
            cursor.execute(query, params)
            self.conn.commit()

            # Return results based on which IDs existed
            results = []
            for id in ids:
                results.append(id in existing_ids)

            return results
        except Exception:
            self.conn.rollback()
            raise
        finally:
            cursor.close()

    def _initialize(self) -> None:
        """Initialize method - connection setup handled in connect()."""
        pass

    def _count_all(self) -> int:
        """Count all records in the database."""
        self._check_connection()
        cursor = self.conn.cursor()
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {self.table_manager.qualified_table}")
            result = cursor.fetchone()
            return result[0] if result else 0
        finally:
            cursor.close()

    def stream_read(
        self,
        query: Query | None = None,
        config: StreamConfig | None = None
    ) -> Iterator[Record]:
        """Stream records from database."""
        from ..streaming import StreamConfig

        config = config or StreamConfig()
        query = query or Query()

        # Use the existing stream method's logic but yield individual records
        offset = 0
        while True:
            # Fetch a batch
            query_copy = query.copy()
            query_copy.offset(offset).limit(config.batch_size)
            batch = self.search(query_copy)

            if not batch:
                break

            for record in batch:
                yield record

            offset += len(batch)

            # If we got less than batch_size, we're done
            if len(batch) < config.batch_size:
                break

    def stream_write(
        self,
        records: Iterator[Record],
        config: StreamConfig | None = None
    ) -> StreamResult:
        """Stream records into database.

        Honors ``config.on_conflict``: INSERT uses the ``create_batch`` bulk
        fast-path; UPSERT/SKIP write per-record via ``upsert``/``create``.
        """
        from ..streaming import (
            ConflictPolicy,
            StreamConfig,
            StreamResult,
            resolve_conflict_write,
            run_stream_write,
        )

        config = config or StreamConfig()

        if config.on_conflict != ConflictPolicy.INSERT:
            # UPSERT/SKIP have no conflict-aware bulk verb: write per record.
            batch_write_func, single_write_func, skip_on_duplicate = resolve_conflict_write(
                config.on_conflict,
                insert_batch_func=None,
                single_create_func=self.create,
                upsert_func=self.upsert,
            )
            return run_stream_write(
                records,
                batch_write_func=batch_write_func,
                single_write_func=single_write_func,
                skip_on_duplicate=skip_on_duplicate,
                config=config,
            )

        batch = []
        total_written = 0
        start_time = time.time()

        for record in records:
            batch.append(record)

            if len(batch) >= config.batch_size:
                # Write the batch
                self.create_batch(batch)
                total_written += len(batch)
                batch = []

        # Write any remaining records
        if batch:
            self.create_batch(batch)
            total_written += len(batch)

        elapsed = time.time() - start_time

        return StreamResult(
            total_processed=total_written,
            successful=total_written,
            failed=0,
            duration=elapsed,
            total_batches=(total_written + config.batch_size - 1) // config.batch_size
        )

    # Vector support methods
    def has_vector_support(self) -> bool:
        """Check if this backend has vector support.
        
        Returns:
            False - SQLite has no native vector support, uses Python-based similarity
        """
        return False  # No native vector support

    def enable_vector_support(self) -> bool:
        """Enable vector support for this backend.
        
        Returns:
            True - Vector support is always available (Python-based)
        """
        # SQLite doesn't need any special setup for vector support
        # We handle vectors as JSON strings
        self.vector_enabled = True
        return True

    def vector_search(
        self,
        query_vector: np.ndarray,
        field_name: str = "embedding",
        k: int = 10,
        filter: Query | None = None,
        metric: DistanceMetric | None = None,
        **kwargs
    ) -> list[VectorSearchResult]:
        """Perform vector similarity search using Python-based calculations.
        
        Delegates to PythonVectorSearchMixin for the implementation.
        
        Args:
            query_vector: Query vector
            field_name: Name of the vector field to search
            k: Number of results to return
            filter: Optional filter conditions
            metric: Distance metric (uses instance default if not specified)
            **kwargs: Additional arguments for compatibility
            
        Returns:
            List of search results with scores
        """
        self._check_connection()

        # Delegate to the mixin's implementation
        return self.python_vector_search_sync(
            query_vector=query_vector,
            vector_field=field_name,
            k=k,
            filter=filter,
            metric=metric,
            **kwargs
        )

    def add_vectors(
        self,
        vectors: list[np.ndarray],
        ids: list[str] | None = None,
        metadata: list[dict[str, Any]] | None = None,
        field_name: str = "embedding",
    ) -> list[str]:
        """Add vectors to the database.
        
        Args:
            vectors: List of vectors to add
            ids: Optional list of IDs
            metadata: Optional list of metadata dicts
            field_name: Name of the vector field
            
        Returns:
            List of created record IDs
        """
        from collections import OrderedDict

        from ..fields import VectorField

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in vectors]

        # Create records with vector fields
        records = []
        for i, vector in enumerate(vectors):
            # Create vector field
            vector_field = VectorField(
                name=field_name,
                value=vector,
                dimensions=len(vector) if isinstance(vector, (list, np.ndarray)) else None
            )

            # Create record
            record_metadata = metadata[i] if metadata and i < len(metadata) else {}
            record = Record(
                data=OrderedDict({field_name: vector_field}),
                metadata=record_metadata,
                storage_id=ids[i]
            )
            records.append(record)

        # Use batch create for efficiency
        return self.create_batch(records)
