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
from typing import Any, TYPE_CHECKING

import duckdb
from dataknobs_config import ConfigurableBase

from ..database import AsyncDatabase, SyncDatabase
from ..query import Query
from ..query_logic import ComplexQuery
from .sql_base import SQLQueryBuilder, SQLRecordSerializer, SQLTableManager

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator
    from ..records import Record
    from ..streaming import StreamConfig, StreamResult


logger = logging.getLogger(__name__)


class AsyncDuckDBDatabase(AsyncDatabase, ConfigurableBase):  # type: ignore[misc]
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

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize async DuckDB database.

        Args:
            config: Configuration with the following optional keys:
                - path: Database file path (default: ":memory:")
                - table: Table name (default: "records")
                - timeout: Connection timeout in seconds (default: 5.0)
                - max_workers: Number of threads in pool (default: 4)
                - read_only: Open database in read-only mode (default: False)
        """
        super().__init__(config)
        config = config or {}
        self.db_path = config.get("path", ":memory:")
        self.table_name = config.get("table", "records")
        self.timeout = config.get("timeout", 5.0)
        self.max_workers = config.get("max_workers", 4)
        self.read_only = config.get("read_only", False)

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

    @classmethod
    def from_config(cls, config: dict) -> AsyncDuckDBDatabase:
        """Create from config dictionary."""
        return cls(config)

    async def connect(self) -> None:
        """Connect to the DuckDB database."""
        if self._connected:
            return

        # Create directory if needed for file-based database
        if self.db_path != ":memory:":
            db_file = Path(self.db_path)
            db_file.parent.mkdir(parents=True, exist_ok=True)

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
        """Synchronous table creation."""
        # Skip table creation in read-only mode
        if self.read_only:
            return

        with self._lock:
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
            raise ValueError(f"Record with ID {params[0]} already exists") from e

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

    async def update(self, id: str, record: Record) -> bool:
        """Update an existing record.

        Args:
            id: The record ID to update
            record: The record data to update with

        Returns:
            True if the record was updated, False if no record exists
        """
        self._check_connection()

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._update_sync,
            id,
            record
        )

    def _update_sync(self, id: str, record: Record) -> bool:
        """Synchronous update implementation."""
        query, params = self.query_builder.build_update_query(id, record)

        with self._lock:
            # Check if record exists
            exists_query, exists_params = self.query_builder.build_exists_query(id)
            exists = self.conn.execute(exists_query, exists_params).fetchone() is not None

            if exists:
                self.conn.execute(query, params)
                return True

            logger.warning(f"Update affected 0 rows for id={id}. Record may not exist.")
            return False

    async def delete(self, id: str) -> bool:
        """Delete a record by ID.

        Args:
            id: The record ID

        Returns:
            True if deleted, False if not found
        """
        self._check_connection()

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._delete_sync,
            id
        )

    def _delete_sync(self, id: str) -> bool:
        """Synchronous delete implementation."""
        query, params = self.query_builder.build_delete_query(id)

        with self._lock:
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

            # Populate storage_id from database ID
            record.storage_id = str(row_dict['id'])

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
                check_query = f"SELECT id FROM {self.table_name} WHERE id IN ({placeholders})"

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
            check_query = f"SELECT id FROM {self.table_name} WHERE id IN ({placeholders})"

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
            result = self.conn.execute(f"SELECT COUNT(*) FROM {self.table_name}").fetchone()
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
        """
        import time

        from ..streaming import StreamConfig, StreamResult

        config = config or StreamConfig()
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


class SyncDuckDBDatabase(SyncDatabase, ConfigurableBase):  # type: ignore[misc]
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

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize sync DuckDB database.

        Args:
            config: Configuration with the following optional keys:
                - path: Database file path (default: ":memory:")
                - table: Table name (default: "records")
                - timeout: Connection timeout in seconds (default: 5.0)
                - read_only: Open database in read-only mode (default: False)
        """
        super().__init__(config)
        config = config or {}
        self.db_path = config.get("path", ":memory:")
        self.table_name = config.get("table", "records")
        self.timeout = config.get("timeout", 5.0)
        self.read_only = config.get("read_only", False)

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

    @classmethod
    def from_config(cls, config: dict) -> SyncDuckDBDatabase:
        """Create from config dictionary."""
        return cls(config)

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
        """Ensure the table exists."""
        if not self.conn:
            raise RuntimeError("Database not connected. Call connect() first.")

        # Skip table creation in read-only mode
        if self.read_only:
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
            raise ValueError(f"Record with ID {params[0]} already exists") from e

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

    def update(self, id: str, record: Record) -> bool:
        """Update an existing record.

        Args:
            id: The record ID to update
            record: The record data to update with

        Returns:
            True if the record was updated, False if no record exists
        """
        self._check_connection()
        query, params = self.query_builder.build_update_query(id, record)

        # Check if record exists
        exists_query, exists_params = self.query_builder.build_exists_query(id)
        exists = self.conn.execute(exists_query, exists_params).fetchone() is not None

        if exists:
            self.conn.execute(query, params)
            return True

        logger.warning(f"Update affected 0 rows for id={id}. Record may not exist.")
        return False

    def delete(self, id: str) -> bool:
        """Delete a record by ID.

        Args:
            id: The record ID

        Returns:
            True if deleted, False if not found
        """
        self._check_connection()
        query, params = self.query_builder.build_delete_query(id)

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
            record.storage_id = str(row_dict['id'])
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
            check_query = f"SELECT id FROM {self.table_name} WHERE id IN ({placeholders})"

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
        check_query = f"SELECT id FROM {self.table_name} WHERE id IN ({placeholders})"

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

        result = self.conn.execute(f"SELECT COUNT(*) FROM {self.table_name}").fetchone()
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
        """
        import time

        from ..streaming import StreamConfig, StreamResult

        config = config or StreamConfig()
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
