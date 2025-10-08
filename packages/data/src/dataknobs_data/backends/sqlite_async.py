"""Async SQLite backend implementation using aiosqlite."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, TYPE_CHECKING

import aiosqlite
from dataknobs_config import ConfigurableBase

from ..database import AsyncDatabase
from ..pooling import ConnectionPoolManager
from ..query import Query
from ..query_logic import ComplexQuery
from ..vector import VectorOperationsMixin
from ..vector.bulk_embed_mixin import BulkEmbedMixin
from ..vector.python_vector_search import PythonVectorSearchMixin
from .sql_base import SQLQueryBuilder, SQLTableManager
from .sqlite_mixins import SQLiteVectorSupport
from .vector_config_mixin import VectorConfigMixin

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from ..records import Record
    from ..streaming import StreamConfig, StreamResult


logger = logging.getLogger(__name__)

# Global pool manager for SQLite connections
_pool_manager = ConnectionPoolManager()


class AsyncSQLiteDatabase(  # type: ignore[misc]
    AsyncDatabase,
    ConfigurableBase,
    VectorConfigMixin,
    SQLiteVectorSupport,
    PythonVectorSearchMixin,  # Provides python_vector_search_async
    BulkEmbedMixin,  # Must come before VectorOperationsMixin to override bulk_embed_and_store
    VectorOperationsMixin
):
    """Asynchronous SQLite database backend using aiosqlite."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize async SQLite database.
        
        Args:
            config: Configuration with the following optional keys:
                - path: Database file path (default: ":memory:")
                - table: Table name (default: "records")
                - timeout: Connection timeout in seconds (default: 5.0)
                - journal_mode: Journal mode (WAL, DELETE, etc.) (default: WAL for file-based)
                - synchronous: Synchronous mode (NORMAL, FULL, OFF) (default: NORMAL)
                - pool_size: Number of connections in pool (default: 5)
        """
        super().__init__(config)
        config = config or {}
        self.db_path = config.get("path", ":memory:")
        self.table_name = config.get("table", "records")
        self.timeout = config.get("timeout", 5.0)
        self.journal_mode = config.get("journal_mode", "WAL" if self.db_path != ":memory:" else None)
        self.synchronous = config.get("synchronous", "NORMAL")
        self.pool_size = config.get("pool_size", 5)

        # Start with standard query builder, will customize after mixins are initialized
        self.query_builder = SQLQueryBuilder(self.table_name, dialect="sqlite", param_style="qmark")
        self.table_manager = SQLTableManager(self.table_name, dialect="sqlite")

        self.db: aiosqlite.Connection | None = None
        self._connected = False

        # Initialize vector support
        self._parse_vector_config(config)
        self._init_vector_state()

    @classmethod
    def from_config(cls, config: dict) -> AsyncSQLiteDatabase:
        """Create from config dictionary."""
        return cls(config)

    async def connect(self) -> None:
        """Connect to the SQLite database."""
        if self._connected:
            return

        # Create directory if needed for file-based database
        if self.db_path != ":memory:":
            db_file = Path(self.db_path)
            db_file.parent.mkdir(parents=True, exist_ok=True)

        # Connect to database
        self.db = await aiosqlite.connect(
            self.db_path,
            timeout=self.timeout
        )

        # Enable row factory for dict-like access
        self.db.row_factory = aiosqlite.Row

        # Configure SQLite for better performance
        await self._configure_sqlite()

        # Create table if it doesn't exist
        await self._ensure_table()

        self._connected = True
        logger.info(f"Connected to async SQLite database: {self.db_path}")

    async def close(self) -> None:
        """Close the database connection."""
        if self.db:
            await self.db.close()
            self.db = None
            self._connected = False
            logger.info(f"Disconnected from async SQLite database: {self.db_path}")

    async def _configure_sqlite(self) -> None:
        """Configure SQLite settings for performance."""
        if not self.db:
            return

        # Set journal mode if specified
        if self.journal_mode:
            await self.db.execute(f"PRAGMA journal_mode = {self.journal_mode}")
            logger.debug(f"Set journal_mode to {self.journal_mode}")

        # Set synchronous mode
        await self.db.execute(f"PRAGMA synchronous = {self.synchronous}")
        logger.debug(f"Set synchronous to {self.synchronous}")

        # Enable foreign keys
        await self.db.execute("PRAGMA foreign_keys = ON")

        # Optimize for performance
        await self.db.execute("PRAGMA temp_store = MEMORY")
        await self.db.execute("PRAGMA mmap_size = 30000000000")

        await self.db.commit()

    async def _ensure_table(self) -> None:
        """Ensure the table exists."""
        if not self.db:
            raise RuntimeError("Database not connected. Call connect() first.")

        await self.db.executescript(self.table_manager.get_create_table_sql())
        await self.db.commit()

    def _check_connection(self) -> None:
        """Check if database is connected."""
        if not self._connected or not self.db:
            raise RuntimeError("Database not connected. Call connect() first.")

    async def create(self, record: Record) -> str:
        """Create a new record."""
        self._check_connection()

        query, params = self.query_builder.build_create_query(record)

        try:
            await self.db.execute(query, params)
            await self.db.commit()

            # SQLite doesn't support RETURNING, so we use the ID we generated
            record_id = params[0]  # ID is the first parameter
            return record_id
        except aiosqlite.IntegrityError as e:
            await self.db.rollback()
            raise ValueError(f"Record with ID {params[0]} already exists") from e

    async def read(self, id: str) -> Record | None:
        """Read a record by ID."""
        self._check_connection()

        query, params = self.query_builder.build_read_query(id)

        async with self.db.execute(query, params) as cursor:
            row = await cursor.fetchone()

            if row:
                return SQLQueryBuilder.row_to_record(dict(row))
            return None

    async def update(self, id: str, record: Record) -> bool:
        """Update an existing record.

        Args:
            id: The record ID to update
            record: The record data to update with

        Returns:
            True if the record was updated, False if no record with the given ID exists
        """
        self._check_connection()

        query, params = self.query_builder.build_update_query(id, record)

        cursor = await self.db.execute(query, params)
        await self.db.commit()
        rows_affected = cursor.rowcount

        if rows_affected == 0:
            logger.warning(f"Update affected 0 rows for id={id}. Record may not exist.")

        return rows_affected > 0

    async def delete(self, id: str) -> bool:
        """Delete a record by ID."""
        self._check_connection()

        query, params = self.query_builder.build_delete_query(id)

        cursor = await self.db.execute(query, params)
        await self.db.commit()
        return cursor.rowcount > 0

    async def exists(self, id: str) -> bool:
        """Check if a record exists."""
        self._check_connection()

        query, params = self.query_builder.build_exists_query(id)

        async with self.db.execute(query, params) as cursor:
            result = await cursor.fetchone()
            return result is not None

    async def search(self, query: Query | ComplexQuery) -> list[Record]:
        """Search for records matching a query."""
        self._check_connection()

        # Handle ComplexQuery with native SQL support
        if isinstance(query, ComplexQuery):
            sql_query, params = self.query_builder.build_complex_search_query(query)
        else:
            sql_query, params = self.query_builder.build_search_query(query)

        async with self.db.execute(sql_query, params) as cursor:
            rows = await cursor.fetchall()

            records = []
            for row in rows:
                row_dict = dict(row)
                record = SQLQueryBuilder.row_to_record(row_dict)

                # Populate storage_id from database ID
                record.storage_id = str(row_dict['id'])

                records.append(record)

            # Apply field projection if specified
            if query.fields:
                records = [r.project(query.fields) for r in records]

            return records

    async def count(self, query: Query | None = None) -> int:
        """Count records matching a query."""
        self._check_connection()

        sql_query, params = self.query_builder.build_count_query(query)

        async with self.db.execute(sql_query, params) as cursor:
            result = await cursor.fetchone()
            return result[0] if result else 0

    async def create_batch(self, records: list[Record]) -> list[str]:
        """Create multiple records efficiently using a single query.
        
        Uses multi-value INSERT for better performance.
        """
        if not records:
            return []

        self._check_connection()

        # Use the shared batch create query builder
        query, params, ids = self.query_builder.build_batch_create_query(records)

        # Execute the batch insert in a transaction
        await self.db.execute("BEGIN TRANSACTION")

        try:
            await self.db.execute(query, params)
            await self.db.commit()

            # Return the generated IDs
            return ids
        except Exception:
            await self.db.rollback()
            raise

    async def update_batch(self, updates: list[tuple[str, Record]]) -> list[bool]:
        """Update multiple records efficiently using a single query.
        
        Uses CASE expressions for batch updates, similar to PostgreSQL.
        """
        if not updates:
            return []

        self._check_connection()

        # Use the shared batch update query builder
        query, params = self.query_builder.build_batch_update_query(updates)

        # Execute the batch update in a transaction
        await self.db.execute("BEGIN TRANSACTION")

        try:
            await self.db.execute(query, params)
            await self.db.commit()

            # Check which records were actually updated
            # SQLite doesn't have RETURNING, so we need to verify each ID
            update_ids = [record_id for record_id, _ in updates]
            placeholders = ", ".join(["?" for _ in update_ids])
            check_query = f"SELECT id FROM {self.table_name} WHERE id IN ({placeholders})"

            async with self.db.execute(check_query, update_ids) as check_cursor:
                rows = await check_cursor.fetchall()
                existing_ids = {row[0] for row in rows}

            # Return results for each update
            results = []
            for record_id, _ in updates:
                results.append(record_id in existing_ids)

            return results
        except Exception:
            await self.db.rollback()
            raise

    async def delete_batch(self, ids: list[str]) -> list[bool]:
        """Delete multiple records efficiently using a single query.
        
        Uses single DELETE with IN clause for better performance.
        """
        if not ids:
            return []

        self._check_connection()

        # Check which IDs exist before deletion
        placeholders = ", ".join(["?" for _ in ids])
        check_query = f"SELECT id FROM {self.table_name} WHERE id IN ({placeholders})"

        async with self.db.execute(check_query, ids) as cursor:
            rows = await cursor.fetchall()
            existing_ids = {row[0] for row in rows}

        # Use the shared batch delete query builder
        query, params = self.query_builder.build_batch_delete_query(ids)

        # Execute the batch delete in a transaction
        await self.db.execute("BEGIN TRANSACTION")

        try:
            await self.db.execute(query, params)
            await self.db.commit()

            # Return results based on which IDs existed
            results = []
            for id in ids:
                results.append(id in existing_ids)

            return results
        except Exception:
            await self.db.rollback()
            raise

    def _initialize(self) -> None:
        """Initialize method - connection setup handled in connect()."""
        pass

    async def _count_all(self) -> int:
        """Count all records in the database."""
        self._check_connection()

        async with self.db.execute(f"SELECT COUNT(*) FROM {self.table_name}") as cursor:
            result = await cursor.fetchone()
            return result[0] if result else 0

    async def stream_read(
        self,
        query: Query | None = None,
        config: StreamConfig | None = None
    ) -> AsyncIterator[Record]:
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
        """Stream records into database."""
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

    async def vector_search(
        self,
        query_vector,
        vector_field: str = "embedding",
        k: int = 10,
        filter=None,
        metric=None,
        **kwargs
    ):
        """Perform async vector similarity search using Python-based calculations.
        
        Delegates to PythonVectorSearchMixin for the implementation.
        
        Args:
            query_vector: Query vector
            vector_field: Name of the vector field to search
            k: Number of results to return  
            filter: Optional filter conditions
            metric: Distance metric (uses instance default if not specified)
            **kwargs: Additional arguments for compatibility
            
        Returns:
            List of VectorSearchResult objects with scores
        """
        self._check_connection()

        # Delegate to the mixin's implementation
        return await self.python_vector_search_async(
            query_vector=query_vector,
            vector_field=vector_field,
            k=k,
            filter=filter,
            metric=metric,
            **kwargs
        )
