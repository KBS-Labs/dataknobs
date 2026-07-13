"""DuckDB backend implementation for analytical workloads.

DuckDB is an embedded columnar database optimized for analytics,
providing 10-100x performance improvement over SQLite for
aggregations, joins, and analytical queries.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb
from dataknobs_common.structured_config import StructuredConfigConsumer

from ..database import AsyncDatabase, SyncDatabase, enforce_content_version
from ..exceptions import DuplicateRecordError, RecordValidationError
from ..query import Query
from ..query_logic import ComplexQuery
from .config import AsyncDuckDBDatabaseConfig, SyncDuckDBDatabaseConfig
from .sql_base import (
    SQLQueryBuilder,
    SQLRecordSerializer,
    SQLTableManager,
    is_duplicate_key_error,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator
    from typing import ClassVar

    from ..records import Record
    from ..streaming import StreamConfig, StreamResult


logger = logging.getLogger(__name__)


class AsyncDuckDBDatabase(  # type: ignore[misc]
    StructuredConfigConsumer[AsyncDuckDBDatabaseConfig],
    AsyncDatabase,
):
    """Asynchronous DuckDB database backend for analytical workloads.

    DuckDB is an embedded columnar database optimized for analytics.
    Provides 10-100x performance improvement over SQLite for
    aggregations, joins, and analytical queries.

    Features:
    - Columnar storage for fast analytical queries
    - Parallel execution for multi-threaded query processing
    - Native Parquet integration for efficient data import/export
    - Advanced analytics support (window functions, CTEs, complex aggregations)

    Usage:
        ```python
        from dataknobs_data import async_database_factory

        # File-based database
        db = async_database_factory("duckdb:///path/to/data.duckdb")

        # In-memory database
        db = async_database_factory("duckdb:///:memory:")

        async with db:
            # Perform CRUD operations
            await db.create(record)
            results = await db.search(query)
        ```
    """

    CONFIG_CLS: ClassVar[type[AsyncDuckDBDatabaseConfig]] = (
        AsyncDuckDBDatabaseConfig
    )

    def _setup(self) -> None:
        """Derive backend attributes from the typed config.

        Runs after the cooperative base chain has set ``self.schema`` and
        run ``_initialize`` (a no-op — connection setup is deferred to
        :meth:`connect`).
        """
        cfg = self.config
        self.db_path = cfg.path
        self.table_name = cfg.table
        self.timeout = cfg.timeout
        self.max_workers = cfg.max_workers
        self.read_only = cfg.read_only
        self.auto_create_table = cfg.auto_create_table

        # Thread pool for async operations (DuckDB has no native async support)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

        # Reuse SQL infrastructure
        self.query_builder = SQLQueryBuilder(
            self.table_name,
            dialect="duckdb",
            param_style="qmark"  # DuckDB uses ? placeholders
        )
        self.serializer = SQLRecordSerializer()
        self.table_manager = SQLTableManager(
            self.table_name,
            dialect="duckdb"
        )

        self.conn: duckdb.DuckDBPyConnection | None = None
        self._connected = False
        self._lock = threading.Lock()  # Thread safety lock for DuckDB connection

    async def connect(self) -> None:
        """Connect to the DuckDB database."""
        if self._connected:
            return

        # Create directory if needed for file-based database (off the loop).
        if self.db_path != ":memory:":
            db_file = Path(self.db_path)
            await asyncio.to_thread(
                db_file.parent.mkdir, parents=True, exist_ok=True
            )

        # Connect to database (in thread pool since DuckDB is sync)
        loop = asyncio.get_event_loop()
        self.conn = await loop.run_in_executor(
            self.executor,
            self._connect_sync
        )

        # Create table if it doesn't exist
        await self._ensure_table()

        self._connected = True
        logger.info(f"Connected to async DuckDB database: {self.db_path}")

    def _connect_sync(self) -> duckdb.DuckDBPyConnection:
        """Synchronous connection helper."""
        return duckdb.connect(
            self.db_path,
            read_only=self.read_only
        )

    async def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor,
                self.conn.close
            )
            self.conn = None
            self._connected = False
            logger.info(f"Disconnected from async DuckDB database: {self.db_path}")

        # Shutdown executor
        self.executor.shutdown(wait=True)

    async def _ensure_table(self) -> None:
        """Ensure the table exists."""
        if not self.conn:
            raise RuntimeError("Database not connected. Call connect() first.")

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self.executor,
            self._ensure_table_sync
        )

    def _ensure_table_sync(self) -> None:
        """Synchronous table creation/verification.

        When ``auto_create_table=True`` (default), creates the table. When
        ``False``, verifies it exists and raises ``RuntimeError`` if missing.
        ``read_only=True`` skips this entirely — no DDL is meaningful in
        read-only mode, and the fail-fast existence check is also skipped.
        If you rely on ``auto_create_table=False`` to detect a missing table at
        startup, do not combine it with ``read_only=True``; the check will not
        run and no error will be raised.
        """
        if self.read_only:
            if not self.auto_create_table:
                logger.warning(
                    "auto_create_table=False has no effect when read_only=True — "
                    "the table existence check is skipped in read-only mode."
                )
            return

        with self._lock:
            if not self.auto_create_table:
                exists_sql, params = self.table_manager.get_table_exists_sql()
                row = self.conn.execute(exists_sql, list(params)).fetchone()
                exists = bool(row[0]) if row else False
                if not exists:
                    raise RuntimeError(
                        f"Table {self.table_name} does not exist and "
                        "auto_create_table is disabled. Run your migrations "
                        "before starting the application."
                    )
                return

            create_sql = self.table_manager.get_create_table_sql()
            self.conn.execute(create_sql)

    def _check_connection(self) -> None:
        """Check if database is connected."""
        if not self._connected or not self.conn:
            raise RuntimeError("Database not connected. Call connect() first.")

    async def create(self, record: Record) -> str:
        """Create a new record.

        Args:
            record: The record to create

        Returns:
            The record ID
        """
        self._check_connection()

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._create_sync,
            record
        )

    def _create_sync(self, record: Record) -> str:
        """Synchronous create implementation."""
        query, params = self.query_builder.build_create_query(record)

        try:
            with self._lock:
                self.conn.execute(query, params)
            # DuckDB doesn't support RETURNING, so we use the ID we generated
            record_id = params[0]  # ID is the first parameter
            return record_id
        except duckdb.ConstraintException as e:
            if is_duplicate_key_error(e):
                raise DuplicateRecordError(params[0]) from e
            # NOT NULL / CHECK / other column constraint — surface truthfully
            # instead of mislabeling it as a duplicate id.
            raise RecordValidationError(str(e)) from e

    async def read(self, id: str) -> Record | None:
        """Read a record by ID.

        Args:
            id: The record ID

        Returns:
            The record if found, None otherwise
        """
        self._check_connection()

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._read_sync,
            id
        )

    def _read_sync(self, id: str) -> Record | None:
        """Synchronous read implementation."""
        query, params = self.query_builder.build_read_query(id)

        with self._lock:
            result = self.conn.execute(query, params).fetchone()

            if result:
                # Convert tuple result to dict
                columns = self.conn.description
                row_dict = {columns[i][0]: result[i] for i in range(len(columns))}
                return SQLQueryBuilder.row_to_record(row_dict)
        return None

    async def update(
        self, id: str, record: Record, *, expected_version: str | None = None
    ) -> bool:
        """Update an existing record.

        Args:
            id: The record ID to update
            record: The record data to update with
            expected_version: Optional optimistic-concurrency token from
                ``get_version(id)`` (a content hash for DuckDB). When provided,
                the read-compare-write runs inside the connection lock so the
                compare-and-set is atomic within the connection; a stale token
                raises ``ConcurrencyError`` instead of overwriting. When
                ``None`` the update is unconditional, byte-identical to prior
                behavior.

        Returns:
            True if the record was updated, False if no record exists

        Raises:
            ConcurrencyError: If ``expected_version`` does not match the
                record's current version token.
        """
        self._check_connection()

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._update_sync,
            id,
            record,
            expected_version,
        )

    def _update_sync(
        self, id: str, record: Record, expected_version: str | None = None
    ) -> bool:
        """Synchronous update implementation."""
        query, params = self.query_builder.build_update_query(id, record)

        with self._lock:
            if expected_version is not None:
                # Conditional write: read the current row and apply the update
                # under the connection lock so the compare-and-set is atomic.
                read_query, read_params = self.query_builder.build_read_query(id)
                result = self.conn.execute(read_query, read_params).fetchone()
                if result is None:
                    # Conditional update of an absent record is a documented
                    # False return, not a warning-worthy event (matches the
                    # SQLite backend's quiet miss).
                    return False
                columns = self.conn.description
                row_dict = {columns[i][0]: result[i] for i in range(len(columns))}
                current = SQLQueryBuilder.row_to_record(row_dict)
                enforce_content_version(id, expected_version, current)
                self.conn.execute(query, params)
                return True

            # Check if record exists
            exists_query, exists_params = self.query_builder.build_exists_query(id)
            exists = self.conn.execute(exists_query, exists_params).fetchone() is not None

            if exists:
                self.conn.execute(query, params)
                return True

            logger.warning(f"Update affected 0 rows for id={id}. Record may not exist.")
            return False

    async def delete(
        self, id: str, *, expected_version: str | None = None
    ) -> bool:
        """Delete a record by ID.

        Args:
            id: The record ID
            expected_version: Optional content-hash token from
                ``get_version(id)``. When provided, the read-compare-delete
                runs under the connection lock so the compare-and-set is atomic
                within the connection; a stale token raises ``ConcurrencyError``
                and a missing record returns ``False``. When ``None`` the
                delete is unconditional, byte-identical to prior behavior.

        Returns:
            True if deleted, False if not found

        Raises:
            ConcurrencyError: If ``expected_version`` does not match the
                record's current version token.
        """
        self._check_connection()

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._delete_sync,
            id,
            expected_version,
        )

    def _delete_sync(self, id: str, expected_version: str | None = None) -> bool:
        """Synchronous delete implementation."""
        query, params = self.query_builder.build_delete_query(id)

        with self._lock:
            if expected_version is not None:
                # Conditional delete: read the current row and apply the delete
                # under the connection lock so the compare-and-set is atomic.
                read_query, read_params = self.query_builder.build_read_query(id)
                result = self.conn.execute(read_query, read_params).fetchone()
                if result is None:
                    return False
                columns = self.conn.description
                row_dict = {columns[i][0]: result[i] for i in range(len(columns))}
                current = SQLQueryBuilder.row_to_record(row_dict)
                enforce_content_version(id, expected_version, current)
                self.conn.execute(query, params)
                return True

            # First check if the record exists
            exists_query, exists_params = self.query_builder.build_exists_query(id)
            exists = self.conn.execute(exists_query, exists_params).fetchone() is not None

            if exists:
                self.conn.execute(query, params)
                return True
        return False

    async def exists(self, id: str) -> bool:
        """Check if a record exists.

        Args:
            id: The record ID

        Returns:
            True if exists, False otherwise
        """
        self._check_connection()

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._exists_sync,
            id
        )

    def _exists_sync(self, id: str) -> bool:
        """Synchronous exists implementation."""
        query, params = self.query_builder.build_exists_query(id)

        with self._lock:
            result = self.conn.execute(query, params).fetchone()
        return result is not None

    async def search(self, query: Query | ComplexQuery) -> list[Record]:
        """Search for records matching a query.

        Args:
            query: The query specification

        Returns:
            List of matching records
        """
        self._check_connection()

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._search_sync,
            query
        )

    def _search_sync(self, query: Query | ComplexQuery) -> list[Record]:
        """Synchronous search implementation."""
        # Handle ComplexQuery with native SQL support
        if isinstance(query, ComplexQuery):
            sql_query, params = self.query_builder.build_complex_search_query(query)
        else:
            sql_query, params = self.query_builder.build_search_query(query)

        with self._lock:
            results = self.conn.execute(sql_query, params).fetchall()
            columns = self.conn.description

        records = []
        for result in results:
            # Convert tuple result to dict
            row_dict = {columns[i][0]: result[i] for i in range(len(columns))}
            record = SQLQueryBuilder.row_to_record(row_dict)
            records.append(record)

        # Apply field projection if specified
        if hasattr(query, 'fields') and query.fields:
            records = [r.project(query.fields) for r in records]

        return records

    async def count(self, query: Query | None = None) -> int:
        """Count records matching a query.

        Args:
            query: Optional query specification

        Returns:
            Count of matching records
        """
        self._check_connection()

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._count_sync,
            query
        )

    def _count_sync(self, query: Query | None = None) -> int:
        """Synchronous count implementation."""
        sql_query, params = self.query_builder.build_count_query(query)

        with self._lock:
            result = self.conn.execute(sql_query, params).fetchone()
        return result[0] if result else 0

    def supports_transactions(self) -> bool:
        """DuckDB batch ops run inside an explicit ``begin``/``commit``."""
        return True

    async def create_batch(self, records: list[Record]) -> list[str]:
        """Create multiple records efficiently.

        Args:
            records: List of records to create

        Returns:
            List of record IDs
        """
        if not records:
            return []

        self._check_connection()

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._create_batch_sync,
            records
        )

    def _create_batch_sync(self, records: list[Record]) -> list[str]:
        """Synchronous batch create implementation."""
        # Use the shared batch create query builder
        query, params, ids = self.query_builder.build_batch_create_query(records)

        # Execute the batch insert in a transaction
        with self._lock:
            try:
                self.conn.begin()
                self.conn.execute(query, params)
                self.conn.commit()
                return ids
            except Exception:
                self.conn.rollback()
                raise

    async def update_batch(self, updates: list[tuple[str, Record]]) -> list[bool]:
        """Update multiple records efficiently.

        Args:
            updates: List of (record_id, record) tuples

        Returns:
            List of success indicators
        """
        if not updates:
            return []

        self._check_connection()

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._update_batch_sync,
            updates
        )

    def _update_batch_sync(self, updates: list[tuple[str, Record]]) -> list[bool]:
        """Synchronous batch update implementation."""
        # Use the shared batch update query builder
        query, params = self.query_builder.build_batch_update_query(updates)

        # Execute the batch update in a transaction
        with self._lock:
            try:
                self.conn.begin()
                self.conn.execute(query, params)
                self.conn.commit()

                # Check which records were actually updated
                update_ids = [record_id for record_id, _ in updates]
                placeholders = ", ".join(["?" for _ in update_ids])
                check_query = f"SELECT id FROM {self.table_manager.qualified_table} WHERE id IN ({placeholders})"

                rows = self.conn.execute(check_query, update_ids).fetchall()
                existing_ids = {row[0] for row in rows}

                # Return results for each update
                results = []
                for record_id, _ in updates:
                    results.append(record_id in existing_ids)

                return results
            except Exception:
                self.conn.rollback()
                raise

    async def delete_batch(self, ids: list[str]) -> list[bool]:
        """Delete multiple records efficiently.

        Args:
            ids: List of record IDs to delete

        Returns:
            List of success indicators
        """
        if not ids:
            return []

        self._check_connection()

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._delete_batch_sync,
            ids
        )

    def _delete_batch_sync(self, ids: list[str]) -> list[bool]:
        """Synchronous batch delete implementation."""
        with self._lock:
            # Check which IDs exist before deletion
            placeholders = ", ".join(["?" for _ in ids])
            check_query = f"SELECT id FROM {self.table_manager.qualified_table} WHERE id IN ({placeholders})"

            rows = self.conn.execute(check_query, ids).fetchall()
            existing_ids = {row[0] for row in rows}

            # Use the shared batch delete query builder
            query, params = self.query_builder.build_batch_delete_query(ids)

            # Execute the batch delete in a transaction
            try:
                self.conn.begin()
                self.conn.execute(query, params)
                self.conn.commit()

                # Return results based on which IDs existed
                results = []
                for id in ids:
                    results.append(id in existing_ids)

                return results
            except Exception:
                self.conn.rollback()
                raise

    def _initialize(self) -> None:
        """Initialize method - connection setup handled in connect()."""
        pass

    async def _count_all(self) -> int:
        """Count all records in the database."""
        self._check_connection()

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._count_all_sync
        )

    def _count_all_sync(self) -> int:
        """Synchronous count all implementation."""
        with self._lock:
            result = self.conn.execute(f"SELECT COUNT(*) FROM {self.table_manager.qualified_table}").fetchone()
        return result[0] if result else 0

    async def stream_read(
        self,
        query: Query | None = None,
        config: StreamConfig | None = None
    ) -> AsyncIterator[Record]:
        """Stream records from database.

        Args:
            query: Optional query specification
            config: Stream configuration

        Yields:
            Records one at a time
        """
        from ..streaming import StreamConfig

        config = config or StreamConfig()
        query = query or Query()

        # Use the existing stream method's logic but yield individual records
        offset = 0
        while True:
            # Fetch a batch
            query_copy = query.copy()
            query_copy.offset(offset).limit(config.batch_size)
            batch = await self.search(query_copy)

            if not batch:
                break

            for record in batch:
                yield record

            offset += len(batch)

            # If we got less than batch_size, we're done
            if len(batch) < config.batch_size:
                break

    async def stream_write(
        self,
        records: AsyncIterator[Record],
        config: StreamConfig | None = None
    ) -> StreamResult:
        """Stream records into database.

        Args:
            records: Async iterator of records
            config: Stream configuration

        Returns:
            Stream result with statistics

        Honors ``config.on_conflict``: INSERT uses the ``create_batch`` bulk
        fast-path; UPSERT/SKIP write per-record via ``upsert``/``create``.
        """
        import time

        from ..streaming import (
            ConflictPolicy,
            StreamConfig,
            StreamResult,
            async_run_stream_write,
            resolve_conflict_write,
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
            return await async_run_stream_write(
                records,
                batch_write_func=batch_write_func,
                single_write_func=single_write_func,
                skip_on_duplicate=skip_on_duplicate,
                config=config,
            )

        batch = []
        total_written = 0
        start_time = time.time()

        async for record in records:
            batch.append(record)

            if len(batch) >= config.batch_size:
                # Write the batch
                await self.create_batch(batch)
                total_written += len(batch)
                batch = []

        # Write any remaining records
        if batch:
            await self.create_batch(batch)
            total_written += len(batch)

        elapsed = time.time() - start_time

        return StreamResult(
            total_processed=total_written,
            successful=total_written,
            failed=0,
            duration=elapsed,
            total_batches=(total_written + config.batch_size - 1) // config.batch_size
        )


class SyncDuckDBDatabase(  # type: ignore[misc]
    StructuredConfigConsumer[SyncDuckDBDatabaseConfig],
    SyncDatabase,
):
    """Synchronous DuckDB database backend for analytical workloads.

    DuckDB is an embedded columnar database optimized for analytics.
    Provides 10-100x performance improvement over SQLite for
    aggregations, joins, and analytical queries.

    Features:
    - Columnar storage for fast analytical queries
    - Native Parquet integration for efficient data import/export
    - Advanced analytics support (window functions, CTEs, complex aggregations)

    Usage:
        ```python
        from dataknobs_data.backends.duckdb import SyncDuckDBDatabase

        # File-based database
        db = SyncDuckDBDatabase({"path": "/path/to/data.duckdb"})

        # In-memory database
        db = SyncDuckDBDatabase({"path": ":memory:"})

        with db:
            # Perform CRUD operations
            db.create(record)
            results = db.search(query)
        ```
    """

    CONFIG_CLS: ClassVar[type[SyncDuckDBDatabaseConfig]] = (
        SyncDuckDBDatabaseConfig
    )

    def _setup(self) -> None:
        """Derive backend attributes from the typed config.

        Runs after the cooperative base chain has set ``self.schema`` and
        run ``_initialize`` (a no-op — connection setup is deferred to
        :meth:`connect`).
        """
        cfg = self.config
        self.db_path = cfg.path
        self.table_name = cfg.table
        self.timeout = cfg.timeout
        self.read_only = cfg.read_only
        self.auto_create_table = cfg.auto_create_table

        # Reuse SQL infrastructure
        self.query_builder = SQLQueryBuilder(
            self.table_name,
            dialect="duckdb",
            param_style="qmark"
        )
        self.serializer = SQLRecordSerializer()
        self.table_manager = SQLTableManager(
            self.table_name,
            dialect="duckdb"
        )

        self.conn: duckdb.DuckDBPyConnection | None = None
        self._connected = False

    def connect(self) -> None:
        """Connect to the DuckDB database."""
        if self._connected:
            return

        # Create directory if needed for file-based database
        if self.db_path != ":memory:":
            db_file = Path(self.db_path)
            db_file.parent.mkdir(parents=True, exist_ok=True)

        # Connect to database
        self.conn = duckdb.connect(
            self.db_path,
            read_only=self.read_only
        )

        # Create table if it doesn't exist
        self._ensure_table()

        self._connected = True
        logger.info(f"Connected to sync DuckDB database: {self.db_path}")

    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            self._connected = False
            logger.info(f"Disconnected from sync DuckDB database: {self.db_path}")

    def _ensure_table(self) -> None:
        """Ensure the table exists.

        When ``auto_create_table=True`` (default), creates the table. When
        ``False``, verifies it exists and raises ``RuntimeError`` if missing.
        ``read_only=True`` skips this entirely — no DDL is meaningful in
        read-only mode.
        """
        if not self.conn:
            raise RuntimeError("Database not connected. Call connect() first.")

        if self.read_only:
            if not self.auto_create_table:
                logger.warning(
                    "auto_create_table=False has no effect when read_only=True — "
                    "the table existence check is skipped in read-only mode."
                )
            return

        if not self.auto_create_table:
            exists_sql, params = self.table_manager.get_table_exists_sql()
            row = self.conn.execute(exists_sql, list(params)).fetchone()
            exists = bool(row[0]) if row else False
            if not exists:
                raise RuntimeError(
                    f"Table {self.table_name} does not exist and "
                    "auto_create_table is disabled. Run your migrations "
                    "before starting the application."
                )
            return

        create_sql = self.table_manager.get_create_table_sql()
        self.conn.execute(create_sql)

    def _check_connection(self) -> None:
        """Check if database is connected."""
        if not self._connected or not self.conn:
            raise RuntimeError("Database not connected. Call connect() first.")

    def create(self, record: Record) -> str:
        """Create a new record.

        Args:
            record: The record to create

        Returns:
            The record ID
        """
        self._check_connection()
        query, params = self.query_builder.build_create_query(record)

        try:
            self.conn.execute(query, params)
            record_id = params[0]  # ID is the first parameter
            return record_id
        except duckdb.ConstraintException as e:
            if is_duplicate_key_error(e):
                raise DuplicateRecordError(params[0]) from e
            # NOT NULL / CHECK / other column constraint — surface truthfully
            # instead of mislabeling it as a duplicate id.
            raise RecordValidationError(str(e)) from e

    def read(self, id: str) -> Record | None:
        """Read a record by ID.

        Args:
            id: The record ID

        Returns:
            The record if found, None otherwise
        """
        self._check_connection()
        query, params = self.query_builder.build_read_query(id)

        result = self.conn.execute(query, params).fetchone()

        if result:
            columns = self.conn.description
            row_dict = {columns[i][0]: result[i] for i in range(len(columns))}
            return SQLQueryBuilder.row_to_record(row_dict)
        return None

    def update(
        self, id: str, record: Record, *, expected_version: str | None = None
    ) -> bool:
        """Update an existing record.

        Args:
            id: The record ID to update
            record: The record data to update with
            expected_version: Optional optimistic-concurrency token from
                ``get_version(id)`` (a content hash for DuckDB). When provided,
                a stale token raises ``ConcurrencyError`` instead of
                overwriting. When ``None`` the update is unconditional,
                byte-identical to prior behavior.

        Returns:
            True if the record was updated, False if no record exists

        Raises:
            ConcurrencyError: If ``expected_version`` does not match the
                record's current version token.
        """
        self._check_connection()
        query, params = self.query_builder.build_update_query(id, record)

        # Conditional write: compare the current content-hash token before
        # issuing the UPDATE. Reusing read() guarantees the token compared
        # here matches the one get_version() returned. On a single connection
        # the read and write are serialized; cross-connection atomicity is out
        # of scope (see the module docs on the in-process content-hash
        # backends).
        if expected_version is not None:
            current = self.read(id)
            if current is None:
                # Conditional update of an absent record is a documented False
                # return, not a warning-worthy event (matches SQLite).
                return False
            enforce_content_version(id, expected_version, current)

        # Check if record exists
        exists_query, exists_params = self.query_builder.build_exists_query(id)
        exists = self.conn.execute(exists_query, exists_params).fetchone() is not None

        if exists:
            self.conn.execute(query, params)
            return True

        logger.warning(f"Update affected 0 rows for id={id}. Record may not exist.")
        return False

    def delete(self, id: str, *, expected_version: str | None = None) -> bool:
        """Delete a record by ID.

        Args:
            id: The record ID
            expected_version: Optional content-hash token from
                ``get_version(id)``. When provided, a stale token raises
                ``ConcurrencyError`` and a missing record returns ``False``.
                When ``None`` the delete is unconditional, byte-identical to
                prior behavior.

        Returns:
            True if deleted, False if not found

        Raises:
            ConcurrencyError: If ``expected_version`` does not match the
                record's current version token.
        """
        self._check_connection()
        query, params = self.query_builder.build_delete_query(id)

        if expected_version is not None:
            current = self.read(id)
            if current is None:
                return False
            enforce_content_version(id, expected_version, current)
            self.conn.execute(query, params)
            return True

        # First check if the record exists
        exists_query, exists_params = self.query_builder.build_exists_query(id)
        exists = self.conn.execute(exists_query, exists_params).fetchone() is not None

        if exists:
            self.conn.execute(query, params)
            return True
        return False

    def exists(self, id: str) -> bool:
        """Check if a record exists.

        Args:
            id: The record ID

        Returns:
            True if exists, False otherwise
        """
        self._check_connection()
        query, params = self.query_builder.build_exists_query(id)

        result = self.conn.execute(query, params).fetchone()
        return result is not None

    def search(self, query: Query | ComplexQuery) -> list[Record]:
        """Search for records matching a query.

        Args:
            query: The query specification

        Returns:
            List of matching records
        """
        self._check_connection()

        # Handle ComplexQuery with native SQL support
        if isinstance(query, ComplexQuery):
            sql_query, params = self.query_builder.build_complex_search_query(query)
        else:
            sql_query, params = self.query_builder.build_search_query(query)

        results = self.conn.execute(sql_query, params).fetchall()
        columns = self.conn.description

        records = []
        for result in results:
            row_dict = {columns[i][0]: result[i] for i in range(len(columns))}
            record = SQLQueryBuilder.row_to_record(row_dict)
            records.append(record)

        # Apply field projection if specified
        if hasattr(query, 'fields') and query.fields:
            records = [r.project(query.fields) for r in records]

        return records

    def count(self, query: Query | None = None) -> int:
        """Count records matching a query.

        Args:
            query: Optional query specification

        Returns:
            Count of matching records
        """
        self._check_connection()
        sql_query, params = self.query_builder.build_count_query(query)

        result = self.conn.execute(sql_query, params).fetchone()
        return result[0] if result else 0

    def create_batch(self, records: list[Record]) -> list[str]:
        """Create multiple records efficiently.

        Args:
            records: List of records to create

        Returns:
            List of record IDs
        """
        if not records:
            return []

        self._check_connection()
        query, params, ids = self.query_builder.build_batch_create_query(records)

        try:
            self.conn.begin()
            self.conn.execute(query, params)
            self.conn.commit()
            return ids
        except Exception:
            self.conn.rollback()
            raise

    def update_batch(self, updates: list[tuple[str, Record]]) -> list[bool]:
        """Update multiple records efficiently.

        Args:
            updates: List of (record_id, record) tuples

        Returns:
            List of success indicators
        """
        if not updates:
            return []

        self._check_connection()
        query, params = self.query_builder.build_batch_update_query(updates)

        try:
            self.conn.begin()
            self.conn.execute(query, params)
            self.conn.commit()

            # Check which records were actually updated
            update_ids = [record_id for record_id, _ in updates]
            placeholders = ", ".join(["?" for _ in update_ids])
            check_query = f"SELECT id FROM {self.table_manager.qualified_table} WHERE id IN ({placeholders})"

            rows = self.conn.execute(check_query, update_ids).fetchall()
            existing_ids = {row[0] for row in rows}

            results = []
            for record_id, _ in updates:
                results.append(record_id in existing_ids)

            return results
        except Exception:
            self.conn.rollback()
            raise

    def delete_batch(self, ids: list[str]) -> list[bool]:
        """Delete multiple records efficiently.

        Args:
            ids: List of record IDs to delete

        Returns:
            List of success indicators
        """
        if not ids:
            return []

        self._check_connection()

        # Check which IDs exist before deletion
        placeholders = ", ".join(["?" for _ in ids])
        check_query = f"SELECT id FROM {self.table_manager.qualified_table} WHERE id IN ({placeholders})"

        rows = self.conn.execute(check_query, ids).fetchall()
        existing_ids = {row[0] for row in rows}

        query, params = self.query_builder.build_batch_delete_query(ids)

        try:
            self.conn.begin()
            self.conn.execute(query, params)
            self.conn.commit()

            results = []
            for id in ids:
                results.append(id in existing_ids)

            return results
        except Exception:
            self.conn.rollback()
            raise

    def _initialize(self) -> None:
        """Initialize method - connection setup handled in connect()."""
        pass

    def _count_all(self) -> int:
        """Count all records in the database."""
        self._check_connection()

        result = self.conn.execute(f"SELECT COUNT(*) FROM {self.table_manager.qualified_table}").fetchone()
        return result[0] if result else 0

    def stream_read(
        self,
        query: Query | None = None,
        config: StreamConfig | None = None
    ) -> Iterator[Record]:
        """Stream records from database.

        Args:
            query: Optional query specification
            config: Stream configuration

        Yields:
            Records one at a time
        """
        from ..streaming import StreamConfig

        config = config or StreamConfig()
        query = query or Query()

        offset = 0
        while True:
            query_copy = query.copy()
            query_copy.offset(offset).limit(config.batch_size)
            batch = self.search(query_copy)

            if not batch:
                break

            for record in batch:
                yield record

            offset += len(batch)

            if len(batch) < config.batch_size:
                break

    def stream_write(
        self,
        records: Iterator[Record],
        config: StreamConfig | None = None
    ) -> StreamResult:
        """Stream records into database.

        Args:
            records: Iterator of records
            config: Stream configuration

        Returns:
            Stream result with statistics

        Honors ``config.on_conflict``: INSERT uses the ``create_batch`` bulk
        fast-path; UPSERT/SKIP write per-record via ``upsert``/``create``.
        """
        import time

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
