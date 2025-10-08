"""PostgreSQL backend implementation with proper connection management and vector support."""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any, cast

import asyncpg
from dataknobs_config import ConfigurableBase

from dataknobs_utils.sql_utils import DotenvPostgresConnector, PostgresDB

from ..database import AsyncDatabase, SyncDatabase
from ..pooling import ConnectionPoolManager
from ..pooling.postgres import PostgresPoolConfig, create_asyncpg_pool, validate_asyncpg_pool
from ..query import Operator, Query
from ..query_logic import ComplexQuery
from ..streaming import (
    StreamConfig,
    StreamResult,
    async_process_batch_with_fallback,
    process_batch_with_fallback,
)
from ..vector.mixins import VectorOperationsMixin
from .postgres_mixins import (
    PostgresBaseConfig,
    PostgresConnectionValidator,
    PostgresErrorHandler,
    PostgresTableManager,
    PostgresVectorSupport,
)
from .sql_base import SQLQueryBuilder, SQLRecordSerializer

if TYPE_CHECKING:
    import numpy as np

    from collections.abc import AsyncIterator, Iterator, Callable, Awaitable
    from ..fields import VectorField
    from ..records import Record
    from ..vector.types import DistanceMetric, VectorSearchResult

logger = logging.getLogger(__name__)


class SyncPostgresDatabase(
    SyncDatabase,
    ConfigurableBase,
    VectorOperationsMixin,
    SQLRecordSerializer,
    PostgresBaseConfig,
    PostgresTableManager,
    PostgresVectorSupport,
    PostgresConnectionValidator,
    PostgresErrorHandler,
):
    """Synchronous PostgreSQL database backend with proper connection management."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize PostgreSQL database configuration.

        Args:
            config: Configuration with the following optional keys:
                - host: PostgreSQL host (default: from env/localhost)
                - port: PostgreSQL port (default: 5432)
                - database: Database name (default: from env/postgres)
                - user: Username (default: from env/postgres)
                - password: Password (default: from env)
                - table: Table name (default: "records")
                - schema: Schema name (default: "public")
                - enable_vector: Enable vector support (default: False)
        """
        super().__init__(config)

        # Parse configuration using mixin
        table_name, schema_name, conn_config = self._parse_postgres_config(config or {})
        self._init_postgres_attributes(table_name, schema_name)

        # Store connection config for later use
        self._conn_config = conn_config
        self.db = None  # Will be initialized in connect()
        self.query_builder = None  # Will be initialized in connect()

    @classmethod
    def from_config(cls, config: dict) -> SyncPostgresDatabase:
        """Create from config dictionary."""
        return cls(config)

    def connect(self) -> None:
        """Connect to the PostgreSQL database."""
        if self._connected:
            return  # Already connected

        # Initialize query builder with pyformat style for psycopg2
        self.query_builder = SQLQueryBuilder(self.table_name, self.schema_name, dialect="postgres", param_style="pyformat")

        # Create connection using existing utilities
        if not any(key in self._conn_config for key in ["host", "database", "user"]):
            # Use dotenv connector for environment-based config
            connector = DotenvPostgresConnector()
            self.db = PostgresDB(connector)
        else:
            # Direct configuration - map 'database' to 'db' for PostgresDB
            self.db = PostgresDB(
                host=self._conn_config.get("host", "localhost"),
                db=self._conn_config.get("database", "postgres"),  # Note: PostgresDB expects 'db' not 'database'
                user=self._conn_config.get("user", "postgres"),
                pwd=self._conn_config.get("password"),  # Note: PostgresDB expects 'pwd' not 'password'
                port=self._conn_config.get("port", 5432),
            )

        # Create table if it doesn't exist
        self._ensure_table()

        # Detect and enable vector support if requested
        if self.vector_enabled:
            self._detect_vector_support()

        self._connected = True
        self.log_operation("connect", f"Connected to table: {self.schema_name}.{self.table_name}")

    def close(self) -> None:
        """Close the database connection."""
        if self.db:
            # PostgresDB manages its own connections via context managers
            # but we can mark as disconnected
            self._connected = False  # type: ignore[unreachable]

    def _initialize(self) -> None:
        """Initialize method - connection setup moved to connect()."""
        # Configuration parsing stays here, actual connection in connect()
        pass

    def _detect_vector_support(self) -> None:
        """Detect and enable vector support if pgvector is available."""
        from .postgres_vector import check_pgvector_extension_sync, install_pgvector_extension_sync

        try:
            # Check if pgvector is installed
            if check_pgvector_extension_sync(self.db):
                self._vector_enabled = True
                logger.info("pgvector extension detected and enabled")
            else:
                # Try to install it
                if install_pgvector_extension_sync(self.db):
                    self._vector_enabled = True
                    logger.info("pgvector extension installed and enabled")
                else:
                    logger.debug("pgvector extension not available")
        except Exception as e:
            logger.debug(f"Could not enable vector support: {e}")
            self._vector_enabled = False

    def _ensure_table(self) -> None:
        """Ensure the records table exists."""
        if not self.db:
            raise RuntimeError("Database not connected. Call connect() first.")

        create_table_sql = self.get_create_table_sql(self.schema_name, self.table_name)  # type: ignore[unreachable]
        self.db.execute(create_table_sql)


    def _record_to_row(self, record: Record, id: str | None = None) -> dict[str, Any]:
        """Convert a Record to a database row."""
        return {
            "id": id or str(uuid.uuid4()),
            "data": self.record_to_json(record),
            "metadata": json.dumps(record.metadata) if record.metadata else None,
        }

    def _row_to_record(self, row: dict[str, Any]) -> Record:
        """Convert a database row to a Record."""
        return self.row_to_record(row)

    def create(self, record: Record) -> str:
        """Create a new record."""
        self._check_connection()
        # Use record's ID if it has one, otherwise generate a new one
        id = record.id if record.id else str(uuid.uuid4())
        row = self._record_to_row(record, id)

        sql = f"""
        INSERT INTO {self.schema_name}.{self.table_name} (id, data, metadata)
        VALUES (%(id)s, %(data)s, %(metadata)s)
        """
        self.db.execute(sql, row)
        return id

    def read(self, id: str) -> Record | None:
        """Read a record by ID."""
        self._check_connection()
        sql = f"""
        SELECT id, data, metadata
        FROM {self.schema_name}.{self.table_name}
        WHERE id = %(id)s
        """
        df = self.db.query(sql, {"id": id})

        if df.empty:
            return None

        row = df.iloc[0].to_dict()
        return self._row_to_record(row)

    def update(self, id: str, record: Record) -> bool:
        """Update an existing record.

        Args:
            id: The record ID to update
            record: The record data to update with

        Returns:
            True if the record was updated, False if no record with the given ID exists
        """
        self._check_connection()
        row = self._record_to_row(record, id)

        sql = f"""
        UPDATE {self.schema_name}.{self.table_name}
        SET data = %(data)s, metadata = %(metadata)s, updated_at = CURRENT_TIMESTAMP
        WHERE id = %(id)s
        """
        result = self.db.execute(sql, row)

        # PostgresDB.execute returns number of affected rows
        rows_affected = result if isinstance(result, int) else 0

        if rows_affected == 0:
            logger.warning(f"Update affected 0 rows for id={id}. Record may not exist.")

        return rows_affected > 0

    def delete(self, id: str) -> bool:
        """Delete a record by ID."""
        self._check_connection()
        sql = f"""
        DELETE FROM {self.schema_name}.{self.table_name}
        WHERE id = %(id)s
        """
        result = self.db.execute(sql, {"id": id})
        return result > 0 if isinstance(result, int) else False

    def exists(self, id: str) -> bool:
        """Check if a record exists."""
        self._check_connection()
        sql = f"""
        SELECT 1 FROM {self.schema_name}.{self.table_name}
        WHERE id = %(id)s
        LIMIT 1
        """
        df = self.db.query(sql, {"id": id})
        return not df.empty

    def upsert(self, id_or_record: str | Record, record: Record | None = None) -> str:
        """Update or insert a record.
        
        Can be called as:
        - upsert(id, record) - explicit ID and record
        - upsert(record) - extract ID from record using Record's built-in logic
        """
        self._check_connection()
        
        # Determine ID and record based on arguments
        if isinstance(id_or_record, str):
            id = id_or_record
            if record is None:
                raise ValueError("Record required when ID is provided")
        else:
            record = id_or_record
            id = record.id
            if id is None:
                import uuid  # type: ignore[unreachable]
                id = str(uuid.uuid4())
                record.storage_id = id
        
        if self.exists(id):
            self.update(id, record)
        else:
            # Insert with specific ID
            row = self._record_to_row(record, id)
            sql = f"""
            INSERT INTO {self.schema_name}.{self.table_name} (id, data, metadata)
            VALUES (%(id)s, %(data)s, %(metadata)s)
            """
            self.db.execute(sql, row)
        return id

    def search(self, query: Query | ComplexQuery) -> list[Record]:
        """Search for records matching the query."""
        self._check_connection()

        # Handle ComplexQuery with native SQL support
        if isinstance(query, ComplexQuery):
            sql_query, params_list = self.query_builder.build_complex_search_query(query)
        else:
            sql_query, params_list = self.query_builder.build_search_query(query)

        # Build params dict for psycopg2
        # The query builder now generates %(p0)s style placeholders directly
        params_dict = {}
        if params_list:
            for i, param in enumerate(params_list):
                params_dict[f"p{i}"] = param

        # Execute query
        df = self.db.query(sql_query, params_dict)

        # Convert to records
        records = []
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            record = self._row_to_record(row_dict)

            # Populate storage_id from database ID
            record.storage_id = str(row_dict['id'])

            # Apply field projection if specified
            if query.fields:
                record = record.project(query.fields)

            records.append(record)

        return records

    def _count_all(self) -> int:
        """Count all records in the database."""
        self._check_connection()
        sql = f"SELECT COUNT(*) as count FROM {self.schema_name}.{self.table_name}"
        df = self.db.query(sql)
        return int(df.iloc[0]["count"]) if not df.empty else 0

    def clear(self) -> int:
        """Clear all records from the database."""
        self._check_connection()
        # Get count first
        count = self._count_all()

        # Delete all records
        sql = f"TRUNCATE TABLE {self.schema_name}.{self.table_name}"
        self.db.execute(sql)

        return count

    def create_batch(self, records: list[Record]) -> list[str]:
        """Create multiple records efficiently using a single query.
        
        Uses multi-value INSERT for better performance.
        
        Args:
            records: List of records to create
            
        Returns:
            List of created record IDs
        """
        if not records:
            return []

        self._check_connection()

        # Create a query builder for PostgreSQL with pyformat style
        from .sql_base import SQLQueryBuilder
        query_builder = SQLQueryBuilder(self.table_name, self.schema_name, dialect="postgres", param_style="pyformat")

        # Use the shared batch create query builder
        query, params_list, ids = query_builder.build_batch_create_query(records)

        # Build params dict for psycopg2
        params_dict = {}
        for i, param in enumerate(params_list):
            params_dict[f"p{i}"] = param

        # Execute the batch insert and get returned IDs
        result_df = self.db.query(query, params_dict)

        # PostgreSQL RETURNING clause gives us the actual inserted IDs
        if not result_df.empty:
            return result_df['id'].tolist()
        return ids

    def delete_batch(self, ids: list[str]) -> list[bool]:
        """Delete multiple records efficiently using a single query.
        
        Uses single DELETE with IN clause for better performance.
        
        Args:
            ids: List of record IDs to delete
            
        Returns:
            List of success flags for each deletion
        """
        if not ids:
            return []

        self._check_connection()

        # Create a query builder for PostgreSQL with pyformat style
        from .sql_base import SQLQueryBuilder
        query_builder = SQLQueryBuilder(self.table_name, self.schema_name, dialect="postgres", param_style="pyformat")

        # Use the shared batch delete query builder (includes RETURNING clause)
        query, params_list = query_builder.build_batch_delete_query(ids)

        # Build params dict for psycopg2
        params_dict = {}
        for i, param in enumerate(params_list):
            params_dict[f"p{i}"] = param

        # Execute the batch delete and get returned IDs
        result_df = self.db.query(query, params_dict)

        # Get list of deleted IDs from RETURNING clause
        deleted_ids = set(result_df['id'].tolist()) if not result_df.empty else set()

        # Return results based on which IDs were actually deleted
        results = []
        for id in ids:
            results.append(id in deleted_ids)

        return results

    def update_batch(self, updates: list[tuple[str, Record]]) -> list[bool]:
        """Update multiple records efficiently using a single query.
        
        Uses PostgreSQL's CASE expressions for batch updates via shared SQL builder.
        
        Args:
            updates: List of (id, record) tuples to update
            
        Returns:
            List of success flags for each update
        """
        if not updates:
            return []

        self._check_connection()

        # Create a query builder for PostgreSQL with pyformat style
        from .sql_base import SQLQueryBuilder
        query_builder = SQLQueryBuilder(self.table_name, self.schema_name, dialect="postgres", param_style="pyformat")

        # Use the shared batch update query builder
        query, params_list = query_builder.build_batch_update_query(updates)

        # Build params dict for psycopg2
        params_dict = {}
        for i, param in enumerate(params_list):
            params_dict[f"p{i}"] = param

        # Execute the batch update and get returned IDs (query now includes RETURNING clause)
        result_df = self.db.query(query, params_dict)

        # Get list of updated IDs from RETURNING clause
        updated_ids = set(result_df['id'].tolist()) if not result_df.empty else set()

        results = []
        for record_id, _ in updates:
            results.append(record_id in updated_ids)

        return results

    def stream_read(
        self,
        query: Query | None = None,
        config: StreamConfig | None = None
    ) -> Iterator[Record]:
        """Stream records from PostgreSQL."""
        self._check_connection()
        config = config or StreamConfig()

        # Build SQL query
        sql = f"SELECT id, data, metadata FROM {self.schema_name}.{self.table_name}"
        params = {}

        if query and query.filters:
            # Add WHERE clause (simplified for now)
            where_clauses = []
            for i, filter in enumerate(query.filters):
                field_path = f"data->>'{filter.field}'"
                param_name = f"param_{i}"

                if filter.operator == Operator.EQ:
                    where_clauses.append(f"{field_path} = %({param_name})s")
                    params[param_name] = str(filter.value)

            if where_clauses:
                sql += " WHERE " + " AND ".join(where_clauses)

        # Use cursor for streaming
        # Note: PostgresDB may need modification to support cursors
        # For now, we'll fetch in batches
        sql += f" LIMIT {config.batch_size} OFFSET %(offset)s"

        offset = 0
        while True:
            params["offset"] = offset
            df = self.db.query(sql, params)

            if df.empty:
                break

            for _, row in df.iterrows():
                record = self._row_to_record(row.to_dict())
                if query and query.fields:
                    record = record.project(query.fields)
                yield record

            offset += config.batch_size

            # If we got less than batch_size, we're done
            if len(df) < config.batch_size:
                break

    def stream_write(
        self,
        records: Iterator[Record],
        config: StreamConfig | None = None
    ) -> StreamResult:
        """Stream records into PostgreSQL."""
        self._check_connection()
        config = config or StreamConfig()
        result = StreamResult()
        start_time = time.time()
        quitting = False

        batch = []
        for record in records:
            batch.append(record)

            if len(batch) >= config.batch_size:
                # Write batch with graceful fallback
                # Use lambda wrapper for _write_batch
                continue_processing = process_batch_with_fallback(
                    batch,
                    lambda b: self._write_batch(b),
                    self.create,
                    result,
                    config
                )

                if not continue_processing:
                    quitting = True
                    break

                batch = []

        # Write remaining batch
        if batch and not quitting:
            process_batch_with_fallback(
                batch,
                lambda b: self._write_batch(b),
                self.create,
                result,
                config
            )

        result.duration = time.time() - start_time
        return result

    def _write_batch(self, records: list[Record]) -> list[str]:
        """Write a batch of records to the database.
        
        Returns:
            List of created record IDs
        """
        # Build batch insert SQL
        values = []
        params = {}
        ids = []

        for i, record in enumerate(records):
            id = str(uuid.uuid4())
            ids.append(id)
            row = self._record_to_row(record, id)
            values.append(f"(%(id_{i})s, %(data_{i})s, %(metadata_{i})s)")
            params[f"id_{i}"] = row["id"]
            params[f"data_{i}"] = row["data"]
            params[f"metadata_{i}"] = row["metadata"]

        sql = f"""
        INSERT INTO {self.schema_name}.{self.table_name} (id, data, metadata)
        VALUES {', '.join(values)}
        """
        self.db.execute(sql, params)
        return ids

    def vector_search(
        self,
        query_vector: np.ndarray | list[float] | VectorField,
        field_name: str,
        k: int = 10,
        filter: Query | None = None,
        metric: DistanceMetric | str = "cosine"
    ) -> list[VectorSearchResult]:
        """Search for similar vectors using PostgreSQL pgvector.
        
        Args:
            query_vector: Query vector (numpy array, list, or VectorField)
            field_name: Name of vector field to search (must be in data JSON)
            limit: Maximum number of results
            filters: Optional filters to apply
            metric: Distance metric to use (cosine, euclidean, l2, inner_product)
            
        Returns:
            List of VectorSearchResult objects ordered by similarity
        """
        if not self._vector_enabled:
            raise RuntimeError("Vector search not available - pgvector not installed")

        self._check_connection()

        from ..fields import VectorField
        from ..vector.types import DistanceMetric, VectorSearchResult
        from .postgres_vector import format_vector_for_postgres, get_vector_operator

        # Convert query vector to proper format
        if isinstance(query_vector, VectorField):
            vector_str = format_vector_for_postgres(query_vector.value)
        else:
            vector_str = format_vector_for_postgres(query_vector)

        # Get the appropriate operator
        if isinstance(metric, DistanceMetric):
            metric_str = metric.value
        else:
            metric_str = str(metric).lower()

        operator = get_vector_operator(metric_str)

        # Build the query - vectors are stored in JSON data field
        # Use centralized vector extraction logic
        vector_expr = self.get_vector_extraction_sql(field_name, dialect="postgres")

        # Build the base SQL with pyformat placeholders
        sql = f"""
        SELECT 
            id, 
            data,
            metadata,
            {vector_expr} {operator} %(p0)s::vector AS distance
        FROM {self.schema_name}.{self.table_name}
        WHERE data ? %(p1)s  -- Check field exists
        """

        params: list[Any] = [vector_str, field_name]

        # Add filters if provided using the query builder
        if filter:
            # Query builder will generate pyformat placeholders since we configured it that way
            where_clause, filter_params = self.query_builder.build_where_clause(filter, len(params) + 1)
            if where_clause:
                sql += where_clause
                params.extend(filter_params)

        # Order by distance and limit
        next_param = len(params)
        sql += f" ORDER BY distance LIMIT %(p{next_param})s"
        params.append(k)

        # Build param dict for psycopg2
        param_dict = {}
        for i, param in enumerate(params):
            param_dict[f"p{i}"] = param

        df = self.db.query(sql, param_dict)

        # Convert results
        results = []
        for _, row in df.iterrows():
            record = self._row_to_record(row)

            # Calculate similarity score from distance
            distance = row["distance"]
            if metric_str in ["cosine", "cosine_similarity"]:
                score = 1.0 - distance  # Cosine distance to similarity
            elif metric_str in ["euclidean", "l2"]:
                score = 1.0 / (1.0 + distance)  # Convert distance to similarity
            elif metric_str in ["inner_product", "dot_product"]:
                score = -distance  # Negative because pgvector uses negative for descending
            else:
                score = -distance  # Default: lower distance = better

            result = VectorSearchResult(
                record=record,
                score=float(score),
                vector_field=field_name
            )
            results.append(result)

        return results

    def has_vector_support(self) -> bool:
        """Check if this database has vector support enabled.
        
        Returns:
            True if vector operations are supported
        """
        return self._vector_enabled

    def enable_vector_support(self) -> bool:
        """Enable vector support for this database if possible.
        
        Returns:
            True if vector support is now enabled
        """
        if self._vector_enabled:
            return True

        self._detect_vector_support()
        return self._vector_enabled

    def bulk_embed_and_store(
        self,
        records: list[Record],
        text_field: str | list[str],
        vector_field: str = "embedding",
        embedding_fn: Any = None,
        batch_size: int = 100,
        model_name: str | None = None,
        model_version: str | None = None,
    ) -> list[str]:
        """Embed text fields and store vectors with records (stub for abstract requirement).
        
        This is a placeholder implementation to satisfy the abstract method requirement.
        Full implementation would require actual embedding function.
        """
        raise NotImplementedError("bulk_embed_and_store requires an embedding function")


# Global pool manager instance for async PostgreSQL connections
_pool_manager = ConnectionPoolManager[asyncpg.Pool]()


class AsyncPostgresDatabase(
    AsyncDatabase,
    VectorOperationsMixin,
    ConfigurableBase,
    PostgresBaseConfig,
    PostgresTableManager,
    PostgresVectorSupport,
    PostgresConnectionValidator,
    PostgresErrorHandler,
):
    """Native async PostgreSQL database backend with vector support and event loop-aware connection pooling."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize async PostgreSQL database."""
        super().__init__(config)

        # Parse configuration using mixin
        table_name, schema_name, conn_config = self._parse_postgres_config(config or {})
        self._init_postgres_attributes(table_name, schema_name)

        # Extract pool configuration
        self._pool_config = PostgresPoolConfig.from_dict(conn_config)
        self._pool: asyncpg.Pool | None = None

    @classmethod
    def from_config(cls, config: dict) -> AsyncPostgresDatabase:
        """Create from config dictionary."""
        return cls(config)

    async def connect(self) -> None:
        """Connect to the database."""
        if self._connected:
            return

        # Get or create pool for current event loop
        from ..pooling import BasePoolConfig
        self._pool = await _pool_manager.get_pool(
            self._pool_config,
            cast("Callable[[BasePoolConfig], Awaitable[Any]]", create_asyncpg_pool),
            validate_asyncpg_pool
        )

        # Initialize query builder
        self.query_builder = SQLQueryBuilder(self.table_name, self.schema_name, dialect="postgres")

        # Ensure table exists
        await self._ensure_table()

        # Check and enable vector support if requested
        if self.vector_enabled:
            await self._detect_vector_support()

        self._connected = True
        self.log_operation("connect", f"Connected to table: {self.schema_name}.{self.table_name}")

    async def close(self) -> None:
        """Close the database connection and properly close the pool."""
        if self._connected:
            # Properly close the pool if we have one
            if self._pool:
                try:
                    await self._pool.close()
                except Exception as e:
                    logger.warning(f"Error closing connection pool: {e}")
            self._pool = None
            self._connected = False

    def _initialize(self) -> None:
        """Initialize is handled in connect."""
        pass

    async def _ensure_table(self) -> None:
        """Ensure the records table exists."""
        if not self._pool:
            raise RuntimeError("Database not connected. Call connect() first.")

        create_table_sql = self.get_create_table_sql(self.schema_name, self.table_name)

        async with self._pool.acquire() as conn:
            await conn.execute(create_table_sql)

    async def _detect_vector_support(self) -> None:
        """Detect and enable vector support if pgvector is available."""
        from .postgres_vector import check_pgvector_extension, install_pgvector_extension

        async with self._pool.acquire() as conn:
            # Check if pgvector is available
            if await check_pgvector_extension(conn):
                self._vector_enabled = True
                logger.info("pgvector extension detected and enabled")
            else:
                # Try to install it
                if await install_pgvector_extension(conn):
                    self._vector_enabled = True
                    logger.info("pgvector extension installed and enabled")
                else:
                    logger.debug("pgvector extension not available")

    async def _ensure_vector_column(self, field_name: str, dimensions: int) -> None:
        """Ensure a vector column exists for the given field.
        
        Args:
            field_name: Name of the vector field
            dimensions: Number of dimensions
        """
        if not self._vector_enabled:
            return

        column_name = f"vector_{field_name}"

        # Check if column already exists
        check_sql = """
        SELECT column_name FROM information_schema.columns
        WHERE table_schema = $1 AND table_name = $2 AND column_name = $3
        """

        async with self._pool.acquire() as conn:
            existing = await conn.fetchval(check_sql, self.schema_name, self.table_name, column_name)

            if not existing:
                # Add vector column
                alter_sql = f"""
                ALTER TABLE {self.schema_name}.{self.table_name}
                ADD COLUMN IF NOT EXISTS {column_name} vector({dimensions})
                """
                try:
                    await conn.execute(alter_sql)
                    self._vector_dimensions[field_name] = dimensions
                    logger.info(f"Added vector column {column_name} with {dimensions} dimensions")

                    # Create index for the vector column
                    from .postgres_vector import build_vector_index_sql, get_optimal_index_type

                    # Get row count for optimal index selection
                    count_sql = f"SELECT COUNT(*) FROM {self.schema_name}.{self.table_name}"
                    count = await conn.fetchval(count_sql)

                    index_type, index_params = get_optimal_index_type(count)
                    index_sql = build_vector_index_sql(
                        self.table_name,
                        self.schema_name,
                        column_name,
                        dimensions,
                        metric="cosine",
                        index_type=index_type,
                        index_params=index_params
                    )

                    # Note: IVFFlat requires table to have data before creating index
                    if count > 0 or index_type != "ivfflat":
                        await conn.execute(index_sql)
                        logger.info(f"Created {index_type} index for {column_name}")

                except Exception as e:
                    logger.warning(f"Could not create vector column {column_name}: {e}")
            else:
                self._vector_dimensions[field_name] = dimensions

    def _check_connection(self) -> None:
        """Check if async database is connected."""
        self._check_async_connection()

    def _record_to_row(self, record: Record, id: str | None = None) -> dict[str, Any]:
        """Convert a Record to a database row using common serializer."""
        from .sql_base import SQLRecordSerializer

        return {
            "id": id or str(uuid.uuid4()),
            "data": SQLRecordSerializer.record_to_json(record),
            "metadata": json.dumps(record.metadata) if record.metadata else None,
        }

    def _row_to_record(self, row: asyncpg.Record) -> Record:
        """Convert a database row to a Record using the common serializer."""
        from .sql_base import SQLRecordSerializer

        # Convert asyncpg.Record to dict format expected by SQLRecordSerializer
        data_json = row.get("data", {})
        if not isinstance(data_json, str):
            data_json = json.dumps(data_json)

        metadata_json = row.get("metadata")
        if metadata_json and not isinstance(metadata_json, str):
            metadata_json = json.dumps(metadata_json)

        # Use the common serializer to reconstruct the record
        return SQLRecordSerializer.json_to_record(data_json, metadata_json)

    async def create(self, record: Record) -> str:
        """Create a new record with vector support."""
        self._check_connection()

        # Check for vector fields and ensure columns exist
        from ..fields import VectorField
        for field_name, field_obj in record.fields.items():
            if isinstance(field_obj, VectorField) and self._vector_enabled:
                await self._ensure_vector_column(field_name, field_obj.dimensions)

        # Use record's ID if it has one, otherwise generate a new one
        id = record.id if record.id else str(uuid.uuid4())
        row = self._record_to_row(record, id)

        # Build dynamic SQL based on vector columns present
        columns = ["id", "data", "metadata"]
        values = [row["id"], row["data"], row["metadata"]]
        placeholders = ["$1", "$2", "$3"]

        # Add vector columns
        param_num = 4
        for key, value in row.items():
            if key.startswith("vector_"):
                columns.append(key)
                values.append(value)
                placeholders.append(f"${param_num}")
                param_num += 1

        sql = f"""
        INSERT INTO {self.schema_name}.{self.table_name} ({', '.join(columns)})
        VALUES ({', '.join(placeholders)})
        """

        async with self._pool.acquire() as conn:
            await conn.execute(sql, *values)

        return id

    async def read(self, id: str) -> Record | None:
        """Read a record by ID."""
        self._check_connection()
        sql = f"""
        SELECT id, data, metadata
        FROM {self.schema_name}.{self.table_name}
        WHERE id = $1
        """

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(sql, id)

        if not row:
            return None

        return self._row_to_record(row)

    async def update(self, id: str, record: Record) -> bool:
        """Update an existing record.

        Args:
            id: The record ID to update
            record: The record data to update with

        Returns:
            True if the record was updated, False if no record with the given ID exists
        """
        self._check_connection()
        row = self._record_to_row(record, id)

        sql = f"""
        UPDATE {self.schema_name}.{self.table_name}
        SET data = $2, metadata = $3, updated_at = CURRENT_TIMESTAMP
        WHERE id = $1
        """

        async with self._pool.acquire() as conn:
            result = await conn.execute(sql, row["id"], row["data"], row["metadata"])

        # Returns UPDATE n where n is rows affected
        rows_affected = int(result.split()[-1])

        if rows_affected == 0:
            logger.warning(f"Update affected 0 rows for id={id}. Record may not exist.")

        return rows_affected > 0

    async def delete(self, id: str) -> bool:
        """Delete a record by ID."""
        self._check_connection()
        sql = f"""
        DELETE FROM {self.schema_name}.{self.table_name}
        WHERE id = $1
        """

        async with self._pool.acquire() as conn:
            result = await conn.execute(sql, id)

        # Returns DELETE n where n is rows affected
        return result.split()[-1] != "0"

    async def exists(self, id: str) -> bool:
        """Check if a record exists."""
        self._check_connection()
        sql = f"""
        SELECT 1 FROM {self.schema_name}.{self.table_name}
        WHERE id = $1
        LIMIT 1
        """

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(sql, id)

        return row is not None

    async def upsert(self, id_or_record: str | Record, record: Record | None = None) -> str:
        """Update or insert a record.
        
        Can be called as:
        - upsert(id, record) - explicit ID and record
        - upsert(record) - extract ID from record using Record's built-in logic
        """
        self._check_connection()
        
        # Determine ID and record based on arguments
        if isinstance(id_or_record, str):
            id = id_or_record
            if record is None:
                raise ValueError("Record required when ID is provided")
        else:
            record = id_or_record
            id = record.id
            if id is None:
                import uuid  # type: ignore[unreachable]
                id = str(uuid.uuid4())
                record.storage_id = id
        
        row = self._record_to_row(record, id)

        sql = f"""
        INSERT INTO {self.schema_name}.{self.table_name} (id, data, metadata)
        VALUES ($1, $2, $3)
        ON CONFLICT (id) DO UPDATE
        SET data = EXCLUDED.data, metadata = EXCLUDED.metadata, updated_at = CURRENT_TIMESTAMP
        """

        async with self._pool.acquire() as conn:
            await conn.execute(sql, row["id"], row["data"], row["metadata"])

        return id

    async def search(self, query: Query | ComplexQuery) -> list[Record]:
        """Search for records matching the query."""
        self._check_connection()

        # Initialize query builder if not already done
        if not hasattr(self, 'query_builder'):
            self.query_builder = SQLQueryBuilder(
                self.table_name, self.schema_name, dialect="postgres"
            )

        # Handle ComplexQuery with native SQL support
        if isinstance(query, ComplexQuery):
            sql, params = self.query_builder.build_complex_search_query(query)
        else:
            sql, params = self.query_builder.build_search_query(query)

        # Execute query with asyncpg (already uses positional parameters)
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)

        # Convert to records
        records = []
        for row in rows:
            record = self._row_to_record(row)

            # Populate storage_id from database ID
            record.storage_id = str(row['id'])

            # Apply field projection if specified
            if query.fields:
                record = record.project(query.fields)

            records.append(record)

        return records

    async def _count_all(self) -> int:
        """Count all records in the database."""
        self._check_connection()
        sql = f"SELECT COUNT(*) as count FROM {self.schema_name}.{self.table_name}"

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(sql)

        return row["count"] if row else 0

    async def clear(self) -> int:
        """Clear all records from the database."""
        self._check_connection()
        # Get count first
        count = await self._count_all()

        # Delete all records
        sql = f"TRUNCATE TABLE {self.schema_name}.{self.table_name}"

        async with self._pool.acquire() as conn:
            await conn.execute(sql)

        return count

    async def create_batch(self, records: list[Record]) -> list[str]:
        """Create multiple records efficiently using a single query.
        
        Uses multi-value INSERT with RETURNING for better performance.
        
        Args:
            records: List of records to create
            
        Returns:
            List of created record IDs
        """
        if not records:
            return []

        self._check_connection()

        # Create a query builder for PostgreSQL
        from .sql_base import SQLQueryBuilder
        query_builder = SQLQueryBuilder(self.table_name, self.schema_name, dialect="postgres")

        # Use the shared batch create query builder
        query, params, ids = query_builder.build_batch_create_query(records)

        # Execute the batch insert with RETURNING
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        # Return the actual inserted IDs from RETURNING clause
        if rows:
            return [row["id"] for row in rows]
        return ids  # Fallback to generated IDs

    async def delete_batch(self, ids: list[str]) -> list[bool]:
        """Delete multiple records efficiently using a single query.
        
        Uses single DELETE with IN clause and RETURNING for verification.
        
        Args:
            ids: List of record IDs to delete
            
        Returns:
            List of success flags for each deletion
        """
        if not ids:
            return []

        self._check_connection()

        # Create a query builder for PostgreSQL
        from .sql_base import SQLQueryBuilder
        query_builder = SQLQueryBuilder(self.table_name, self.schema_name, dialect="postgres")

        # Use the shared batch delete query builder
        query, params = query_builder.build_batch_delete_query(ids)

        # Execute the batch delete with RETURNING
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        # Convert returned rows to set of deleted IDs
        deleted_ids = {row["id"] for row in rows}

        # Return results for each deletion
        results = []
        for id in ids:
            results.append(id in deleted_ids)

        return results

    async def update_batch(self, updates: list[tuple[str, Record]]) -> list[bool]:
        """Update multiple records efficiently using a single query.
        
        Uses PostgreSQL's CASE expressions for batch updates with native asyncpg.
        
        Args:
            updates: List of (id, record) tuples to update
            
        Returns:
            List of success flags for each update
        """
        if not updates:
            return []

        self._check_connection()

        # Create a query builder for PostgreSQL
        from .sql_base import SQLQueryBuilder
        query_builder = SQLQueryBuilder(self.table_name, self.schema_name, dialect="postgres")

        # Use the shared batch update query builder
        # It already produces positional parameters ($1, $2) for PostgreSQL
        query, params = query_builder.build_batch_update_query(updates)

        # Add RETURNING clause for PostgreSQL to get updated IDs
        query = query.rstrip() + " RETURNING id"

        # Execute the batch update
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        # Convert returned rows to set of updated IDs
        updated_ids = {row["id"] for row in rows}

        # Return results for each update
        results = []
        for record_id, _ in updates:
            results.append(record_id in updated_ids)

        return results

    async def vector_search(
        self,
        query_vector: np.ndarray | list[float] | VectorField,
        field_name: str,
        k: int = 10,
        filter: Query | None = None,
        metric: DistanceMetric | str = "cosine"
    ) -> list[VectorSearchResult]:
        """Search for similar vectors using PostgreSQL pgvector.
        
        Args:
            query_vector: Query vector (numpy array, list, or VectorField)
            field_name: Name of vector field to search
            limit: Maximum number of results
            filters: Optional filters to apply
            metric: Distance metric to use
            
        Returns:
            List of VectorSearchResult objects
        """
        if not self._vector_enabled:
            raise RuntimeError("Vector search not available - pgvector not installed")

        self._check_connection()

        from ..fields import VectorField
        from ..vector.types import DistanceMetric, VectorSearchResult
        from .postgres_vector import format_vector_for_postgres, get_vector_operator

        # Convert query vector to proper format
        if isinstance(query_vector, VectorField):
            vector_str = format_vector_for_postgres(query_vector.value)
        else:
            vector_str = format_vector_for_postgres(query_vector)

        # Get the appropriate operator
        if isinstance(metric, DistanceMetric):
            metric_str = metric.value
        else:
            metric_str = str(metric).lower()
        operator = get_vector_operator(metric_str)

        vector_column = f"vector_{field_name}"

        # Build query
        sql = f"""
        SELECT id, data, metadata, {vector_column},
               {vector_column} {operator} $1::vector AS distance
        FROM {self.schema_name}.{self.table_name}
        WHERE {vector_column} IS NOT NULL
        """

        params = [vector_str]
        param_num = 2

        # Add filters if provided using the query builder
        if filter:
            # First get the where clause from query builder
            where_clause, filter_params = self.query_builder.build_where_clause(filter, param_num)
            if where_clause:
                # Convert %s placeholders to $N for asyncpg
                for param in filter_params:
                    where_clause = where_clause.replace("%s", f"${param_num}", 1)
                    params.append(param)
                    param_num += 1
                sql += where_clause

        # Order by distance and limit
        sql += f"""
        ORDER BY distance
        LIMIT {k}
        """

        # Execute query
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)

        # Convert to VectorSearchResult objects
        results = []
        for row in rows:
            record = self._row_to_record(row)

            # Convert distance to similarity score (1 - normalized_distance for cosine)
            distance = float(row['distance'])
            if metric_str == "cosine":
                score = 1.0 - min(distance, 2.0) / 2.0  # Normalize cosine distance [0,2] to similarity [0,1]
            elif metric_str in ["euclidean", "l2"]:
                score = 1.0 / (1.0 + distance)  # Convert distance to similarity
            else:
                score = 1.0 - distance  # Generic conversion

            result = VectorSearchResult(
                record=record,
                score=score,
                vector_field=field_name,
                metadata={"distance": distance, "metric": metric_str}
            )
            results.append(result)

        return results

    async def enable_vector_support(self) -> bool:
        """Enable vector support for this database.
        
        Returns:
            True if vector support is enabled
        """
        if self._vector_enabled:
            return True

        await self._detect_vector_support()
        return self._vector_enabled

    async def has_vector_support(self) -> bool:
        """Check if this database has vector support enabled.
        
        Returns:
            True if vector support is available
        """
        return self._vector_enabled

    async def bulk_embed_and_store(
        self,
        records: list[Record],
        text_field: str | list[str],
        vector_field: str,
        embedding_fn: Any | None = None,
        batch_size: int = 100,
        model_name: str | None = None,
        model_version: str | None = None,
    ) -> list[str]:
        """Embed text fields and store vectors with records.
        
        This is a placeholder implementation. In a real scenario, you would:
        1. Extract text from the specified fields
        2. Call the embedding function to generate vectors
        3. Store the vectors alongside the records
        
        Args:
            records: Records to process
            text_field: Field name(s) containing text to embed
            vector_field: Field name to store vectors in
            embedding_fn: Function to generate embeddings
            batch_size: Number of records to process at once
            model_name: Name of the embedding model
            model_version: Version of the embedding model
            
        Returns:
            List of record IDs that were processed
        """
        if not embedding_fn:
            raise ValueError("embedding_fn is required for bulk_embed_and_store")

        from ..fields import VectorField

        processed_ids = []

        # Process in batches
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]

            # Extract texts
            texts = []
            for record in batch:
                if isinstance(text_field, list):
                    text = " ".join(str(record.fields.get(f, {}).value) for f in text_field if f in record.fields)
                else:
                    text = str(record.fields.get(text_field, {}).value) if text_field in record.fields else ""
                texts.append(text)

            # Generate embeddings
            if texts:
                embeddings = await embedding_fn(texts)

                # Store vectors with records
                for j, record in enumerate(batch):
                    if j < len(embeddings):
                        vector = embeddings[j]

                        # Add vector field to record
                        record.fields[vector_field] = VectorField(
                            name=vector_field,
                            value=vector,
                            dimensions=len(vector) if hasattr(vector, '__len__') else None,
                            source_field=text_field if isinstance(text_field, str) else ",".join(text_field),
                            model_name=model_name,
                            model_version=model_version,
                        )

                        # Create or update record
                        if record.has_storage_id():
                            if record.storage_id is None:
                                raise ValueError("Record has_storage_id() returned True but storage_id is None")
                            await self.update(record.storage_id, record)
                        else:
                            record_id = await self.create(record)
                            record.storage_id = record_id

                        if record.storage_id is None:
                            raise ValueError("Record storage_id is None after create/update")
                        processed_ids.append(record.storage_id)

        return processed_ids

    async def create_vector_index(
        self,
        vector_field: str,
        dimensions: int,
        metric: DistanceMetric | str = "cosine",
        index_type: str = "ivfflat",
        lists: int | None = None,
    ) -> bool:
        """Create a vector index for efficient similarity search.
        
        Args:
            vector_field: Name of the vector field to index
            dimensions: Number of dimensions in the vectors
            metric: Distance metric for the index
            index_type: Type of index (ivfflat, hnsw)
            lists: Number of lists for IVFFlat index
            
        Returns:
            True if index was created successfully
        """
        from .postgres_vector import (
            build_vector_column_expression,
            build_vector_index_sql,
            get_optimal_index_type,
            get_vector_count_sql,
        )

        self._check_connection()

        if not self._vector_enabled:
            return False

        # Determine optimal parameters if not provided
        if not lists and index_type == "ivfflat":
            # Count vectors to determine optimal lists
            count_sql = get_vector_count_sql(self.schema_name, self.table_name, vector_field)
            async with self._pool.acquire() as conn:
                count = await conn.fetchval(count_sql) or 0
                _, params = get_optimal_index_type(count)
                lists = params.get("lists", 100)

        # Convert metric enum to string if needed
        if hasattr(metric, 'value'):
            metric_str = metric.value
        else:
            metric_str = str(metric).lower()

        # Build vector column expression for index
        column_expr = build_vector_column_expression(vector_field, dimensions, for_index=True)

        # Build index SQL - pass field_name for proper index naming
        index_sql = build_vector_index_sql(
            table_name=self.table_name,
            schema_name=self.schema_name,
            column_name=column_expr,
            dimensions=dimensions,
            metric=metric_str,
            index_type=index_type,
            index_params={"lists": lists} if lists else None,
            field_name=vector_field
        )

        # Create the index
        try:
            logger.debug(f"Creating vector index with SQL: {index_sql}")
            async with self._pool.acquire() as conn:
                await conn.execute(index_sql)
            return True
        except Exception as e:
            logger.warning(f"Failed to create vector index: {e}")
            logger.debug(f"Index SQL was: {index_sql}")
            return False

    async def drop_vector_index(self, vector_field: str, metric: str = "cosine") -> bool:
        """Drop a vector index.
        
        Args:
            vector_field: Name of the vector field
            metric: Distance metric used in the index
            
        Returns:
            True if index was dropped successfully
        """
        from .postgres_vector import get_vector_index_name

        self._check_connection()

        index_name = get_vector_index_name(self.table_name, vector_field, metric)

        try:
            async with self._pool.acquire() as conn:
                await conn.execute(f"DROP INDEX IF EXISTS {self.schema_name}.{index_name}")
            return True
        except Exception as e:
            logger.warning(f"Failed to drop vector index: {e}")
            return False

    async def get_vector_index_stats(self, vector_field: str) -> dict[str, Any]:
        """Get statistics about a vector field and its index.
        
        Args:
            vector_field: Name of the vector field
            
        Returns:
            Dictionary with index statistics
        """
        from .postgres_vector import get_index_check_sql, get_vector_count_sql

        self._check_connection()

        stats = {
            "field": vector_field,
            "indexed": False,
            "vector_count": 0,
        }

        try:
            async with self._pool.acquire() as conn:
                # Count vectors
                count_sql = get_vector_count_sql(self.schema_name, self.table_name, vector_field)
                stats["vector_count"] = await conn.fetchval(count_sql) or 0

                # Check for index
                index_sql, params = get_index_check_sql(self.schema_name, self.table_name, vector_field)
                stats["indexed"] = await conn.fetchval(index_sql, *params) or False
        except Exception as e:
            logger.warning(f"Failed to get vector index stats: {e}")

        return stats

    async def stream_read(
        self,
        query: Query | None = None,
        config: StreamConfig | None = None
    ) -> AsyncIterator[Record]:
        """Stream records from PostgreSQL using cursor."""
        self._check_connection()
        config = config or StreamConfig()

        # Build SQL query
        sql = f"SELECT id, data, metadata FROM {self.schema_name}.{self.table_name}"
        params = []

        if query and query.filters:
            where_clauses = []
            param_count = 0

            for filter in query.filters:
                param_count += 1
                field_path = f"data->>'{filter.field}'"

                if filter.operator == Operator.EQ:
                    where_clauses.append(f"{field_path} = ${param_count}")
                    params.append(str(filter.value))

            if where_clauses:
                sql += " WHERE " + " AND ".join(where_clauses)

        # Use cursor for efficient streaming
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                cursor = await conn.cursor(sql, *params)

                batch = []
                async for row in cursor:
                    record = self._row_to_record(row)
                    if query and query.fields:
                        record = record.project(query.fields)

                    batch.append(record)

                    if len(batch) >= config.batch_size:
                        for rec in batch:
                            yield rec
                        batch = []

                # Yield remaining records
                for rec in batch:
                    yield rec

    async def stream_write(
        self,
        records: AsyncIterator[Record],
        config: StreamConfig | None = None
    ) -> StreamResult:
        """Stream records into PostgreSQL using batch inserts."""
        self._check_connection()
        config = config or StreamConfig()
        result = StreamResult()
        start_time = time.time()
        quitting = False

        batch = []
        async for record in records:
            batch.append(record)

            if len(batch) >= config.batch_size:
                # Write batch with graceful fallback
                # Use lambda wrapper for _write_batch
                async def batch_func(b):
                    await self._write_batch(b)
                    return [r.id for r in b]

                continue_processing = await async_process_batch_with_fallback(
                    batch,
                    batch_func,
                    self.create,
                    result,
                    config
                )

                if not continue_processing:
                    quitting = True
                    break

                batch = []

        # Write remaining batch
        if batch and not quitting:
            async def batch_func(b):
                await self._write_batch(b)
                return [r.id for r in b]

            await async_process_batch_with_fallback(
                batch,
                batch_func,
                self.create,
                result,
                config
            )

        result.duration = time.time() - start_time
        return result

    async def _write_batch(self, records: list[Record]) -> list[str]:
        """Write a batch of records using COPY for performance.
        
        Returns:
            List of created record IDs
        """
        if not records:
            return []

        # Prepare data for COPY
        rows = []
        ids = []
        for record in records:
            row_data = self._record_to_row(record)
            ids.append(row_data["id"])
            rows.append((
                row_data["id"],
                row_data["data"],
                row_data["metadata"]
            ))

        # Use COPY for efficient bulk insert
        async with self._pool.acquire() as conn:
            await conn.copy_records_to_table(
                f"{self.schema_name}.{self.table_name}",
                records=rows,
                columns=["id", "data", "metadata"]
            )

        return ids
