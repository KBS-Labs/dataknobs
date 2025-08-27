"""PostgreSQL backend implementation with proper connection management and vector support."""

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from datetime import datetime
from typing import Any, TYPE_CHECKING

import asyncpg
from dataknobs_config import ConfigurableBase

from dataknobs_utils.sql_utils import DotenvPostgresConnector, PostgresDB

from ..database import AsyncDatabase, SyncDatabase
from ..pooling import ConnectionPoolManager
from ..pooling.postgres import PostgresPoolConfig, create_asyncpg_pool, validate_asyncpg_pool
from ..query import Operator, Query, SortOrder
from ..query_logic import ComplexQuery
from ..records import Record
from .sql_base import SQLQueryBuilder
from ..streaming import (
    StreamConfig,
    StreamResult,
    async_process_batch_with_fallback,
    process_batch_with_fallback,
)
from ..vector.mixins import VectorCapable, VectorOperationsMixin

if TYPE_CHECKING:
    import numpy as np
    from ..fields import VectorField
    from ..vector.types import DistanceMetric, VectorSearchResult

logger = logging.getLogger(__name__)


class SyncPostgresDatabase(SyncDatabase, ConfigurableBase):
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
        """
        super().__init__(config)
        self.db = None  # Will be initialized in connect()
        self._connected = False
        self.query_builder = None  # Will be initialized in connect()

    @classmethod
    def from_config(cls, config: dict) -> "SyncPostgresDatabase":
        """Create from config dictionary."""
        return cls(config)

    def connect(self) -> None:
        """Connect to the PostgreSQL database."""
        if self._connected:
            return  # Already connected

        config = self.config.copy()

        # Extract table configuration
        self.table_name = config.pop("table", "records")
        self.schema_name = config.pop("schema", "public")
        
        # Initialize query builder
        self.query_builder = SQLQueryBuilder(self.table_name, self.schema_name, dialect="postgres")

        # Create connection using existing utilities
        if not any(key in config for key in ["host", "database", "user"]):
            # Use dotenv connector for environment-based config
            connector = DotenvPostgresConnector()
            self.db = PostgresDB(connector)
        else:
            # Direct configuration - map 'database' to 'db' for PostgresDB
            self.db = PostgresDB(
                host=config.get("host", "localhost"),
                db=config.get("database", "postgres"),  # Note: PostgresDB expects 'db' not 'database'
                user=config.get("user", "postgres"),
                pwd=config.get("password"),  # Note: PostgresDB expects 'pwd' not 'password'
                port=config.get("port", 5432),
            )

        # Create table if it doesn't exist
        self._ensure_table()
        self._connected = True

    def close(self) -> None:
        """Close the database connection."""
        if self.db:
            # PostgresDB manages its own connections via context managers
            # but we can mark as disconnected
            self._connected = False

    def _initialize(self) -> None:
        """Initialize method - connection setup moved to connect()."""
        # Configuration parsing stays here, actual connection in connect()
        pass

    def _ensure_table(self) -> None:
        """Ensure the records table exists."""
        if not self.db:
            raise RuntimeError("Database not connected. Call connect() first.")

        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.schema_name}.{self.table_name} (
            id VARCHAR(255) PRIMARY KEY,
            data JSONB NOT NULL,
            metadata JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_{self.table_name}_data 
        ON {self.schema_name}.{self.table_name} USING GIN (data);
        
        CREATE INDEX IF NOT EXISTS idx_{self.table_name}_metadata
        ON {self.schema_name}.{self.table_name} USING GIN (metadata);
        """
        self.db.execute(create_table_sql)

    def _check_connection(self) -> None:
        """Check if database is connected."""
        if not self._connected or not self.db:
            raise RuntimeError("Database not connected. Call connect() first.")

    def _record_to_row(self, record: Record, id: str | None = None) -> dict[str, Any]:
        """Convert a Record to a database row."""
        data = {}
        for field_name, field_obj in record.fields.items():
            data[field_name] = field_obj.value

        return {
            "id": id or str(uuid.uuid4()),
            "data": json.dumps(data),
            "metadata": json.dumps(record.metadata) if record.metadata else None,
        }

    def _row_to_record(self, row: dict[str, Any]) -> Record:
        """Convert a database row to a Record."""
        data = row.get("data", {})
        if isinstance(data, str):
            data = json.loads(data)

        metadata = row.get("metadata", {})
        if isinstance(metadata, str) and metadata:
            metadata = json.loads(metadata)
        elif not metadata:
            metadata = {}

        return Record(data=data, metadata=metadata)

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
        """Update an existing record."""
        self._check_connection()
        row = self._record_to_row(record, id)

        sql = f"""
        UPDATE {self.schema_name}.{self.table_name}
        SET data = %(data)s, metadata = %(metadata)s, updated_at = CURRENT_TIMESTAMP
        WHERE id = %(id)s
        """
        result = self.db.execute(sql, row)
        # PostgresDB.execute returns number of affected rows
        return result > 0 if isinstance(result, int) else False

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

    def upsert(self, id: str, record: Record) -> str:
        """Update or insert a record with a specific ID."""
        self._check_connection()
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

        # Convert numbered parameters to named parameters for psycopg2
        params_dict = {}
        if params_list:
            # Replace placeholders in reverse order to avoid conflicts like $10 being replaced as $1 + "0"
            for i in range(len(params_list), 0, -1):
                param_name = f"p{i}"
                sql_query = sql_query.replace(f"${i}", f"%({param_name})s")
                params_dict[param_name] = params_list[i - 1]

        # Execute query
        df = self.db.query(sql_query, params_dict)

        # Convert to records
        records = []
        for _, row in df.iterrows():
            record = self._row_to_record(row.to_dict())

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
        
        # Create a query builder for PostgreSQL
        from .sql_base import SQLQueryBuilder
        query_builder = SQLQueryBuilder(self.table_name, self.schema_name, dialect="postgres")
        
        # Use the shared batch create query builder
        query, params_list, ids = query_builder.build_batch_create_query(records)
        
        # Convert positional parameters to named parameters for PostgresDB
        # Replace in reverse order to avoid $10 being replaced as $1 + 0
        params_dict = {}
        for i, param in enumerate(params_list, 1):
            param_name = f"p{i}"
            params_dict[param_name] = param
        
        # Replace placeholders in reverse order to avoid conflicts
        for i in range(len(params_list), 0, -1):
            param_name = f"p{i}"
            query = query.replace(f"${i}", f"%({param_name})s")
        
        # Execute the batch insert
        result = self.db.execute(query, params_dict)
        
        # PostgreSQL RETURNING clause gives us the actual inserted IDs
        # But our PostgresDB wrapper doesn't return them directly
        # So we'll use the generated IDs
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
        
        # Check which IDs exist before deletion
        check_sql = f"""
        SELECT id FROM {self.schema_name}.{self.table_name}
        WHERE id = ANY(%(ids)s)
        """
        existing_df = self.db.query(check_sql, {"ids": ids})
        existing_ids = set(existing_df["id"].tolist()) if not existing_df.empty else set()
        
        # Create a query builder for PostgreSQL
        from .sql_base import SQLQueryBuilder
        query_builder = SQLQueryBuilder(self.table_name, self.schema_name, dialect="postgres")
        
        # Use the shared batch delete query builder
        query, params_list = query_builder.build_batch_delete_query(ids)
        
        # Convert positional parameters to named parameters for PostgresDB
        # Replace in reverse order to avoid $10 being replaced as $1 + 0
        params_dict = {}
        for i, param in enumerate(params_list, 1):
            param_name = f"p{i}"
            params_dict[param_name] = param
        
        # Replace placeholders in reverse order to avoid conflicts
        for i in range(len(params_list), 0, -1):
            param_name = f"p{i}"
            query = query.replace(f"${i}", f"%({param_name})s")
        
        # Execute the batch delete
        result = self.db.execute(query, params_dict)
        
        # Return results based on which IDs existed
        results = []
        for id in ids:
            results.append(id in existing_ids)
        
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
        
        # Create a query builder for PostgreSQL
        from .sql_base import SQLQueryBuilder
        query_builder = SQLQueryBuilder(self.table_name, self.schema_name, dialect="postgres")
        
        # Note: The shared builder uses positional params ($1, $2) for postgres
        # but our PostgresDB uses named params. We need to convert.
        query, params_list = query_builder.build_batch_update_query(updates)
        
        # Convert positional parameters to named parameters for PostgresDB
        # Replace in reverse order to avoid $10 being replaced as $1 + 0
        params_dict = {}
        for i, param in enumerate(params_list, 1):
            param_name = f"p{i}"
            params_dict[param_name] = param
        
        # Replace placeholders in reverse order to avoid conflicts
        for i in range(len(params_list), 0, -1):
            param_name = f"p{i}"
            query = query.replace(f"${i}", f"%({param_name})s")
        
        # Execute the batch update
        result = self.db.execute(query, params_dict)
        
        # Check which records were actually updated
        update_ids = [record_id for record_id, _ in updates]
        check_sql = f"""
        SELECT id FROM {self.schema_name}.{self.table_name}
        WHERE id = ANY(%(ids)s)
        """
        existing_df = self.db.query(check_sql, {"ids": update_ids})
        existing_ids = set(existing_df["id"].tolist()) if not existing_df.empty else set()
        
        results = []
        for record_id, _ in updates:
            results.append(record_id in existing_ids)
        
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
                    lambda b: self._write_batch(b) or [r.id for r in b],  # _write_batch returns None, we need IDs
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
                lambda b: self._write_batch(b) or [r.id for r in b],
                self.create,
                result,
                config
            )

        result.duration = time.time() - start_time
        return result

    def _write_batch(self, records: list[Record]) -> None:
        """Write a batch of records to the database."""
        # Build batch insert SQL
        values = []
        params = {}

        for i, record in enumerate(records):
            id = str(uuid.uuid4())
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


# Global pool manager instance for async PostgreSQL connections
_pool_manager = ConnectionPoolManager[asyncpg.Pool]()


class AsyncPostgresDatabase(AsyncDatabase, VectorOperationsMixin, ConfigurableBase):
    """Native async PostgreSQL database backend with vector support and event loop-aware connection pooling."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize async PostgreSQL database."""
        super().__init__(config)
        config = config or {}
        self._pool_config = PostgresPoolConfig.from_dict(config)
        # Add table and schema to pool config from regular config
        self.table_name = config.get("table", "records")
        self.schema_name = config.get("schema", "public")
        self._pool: asyncpg.Pool | None = None
        self._connected = False
        self._vector_enabled = False
        self._vector_dimensions: dict[str, int] = {}  # Track dimensions per vector field

    @classmethod
    def from_config(cls, config: dict) -> "AsyncPostgresDatabase":
        """Create from config dictionary."""
        return cls(config)

    async def connect(self) -> None:
        """Connect to the database."""
        if self._connected:
            return

        # Get or create pool for current event loop
        self._pool = await _pool_manager.get_pool(
            self._pool_config,
            create_asyncpg_pool,
            validate_asyncpg_pool
        )

        # Initialize query builder
        self.query_builder = SQLQueryBuilder(self.table_name, self.schema_name, dialect="postgres")

        # Ensure table exists
        await self._ensure_table()
        
        # Check and enable vector support if available
        await self._detect_vector_support()
        
        self._connected = True

    async def close(self) -> None:
        """Close the database connection."""
        if self._connected:
            # Pool manager handles cleanup
            self._pool = None
            self._connected = False

    def _initialize(self) -> None:
        """Initialize is handled in connect."""
        pass

    async def _ensure_table(self) -> None:
        """Ensure the records table exists."""
        if not self._pool:
            raise RuntimeError("Database not connected. Call connect() first.")

        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.schema_name}.{self.table_name} (
            id VARCHAR(255) PRIMARY KEY,
            data JSONB NOT NULL,
            metadata JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_{self.table_name}_data 
        ON {self.schema_name}.{self.table_name} USING GIN (data);
        
        CREATE INDEX IF NOT EXISTS idx_{self.table_name}_metadata
        ON {self.schema_name}.{self.table_name} USING GIN (metadata);
        """

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
        check_sql = f"""
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
        """Check if database is connected."""
        if not self._connected or not self._pool:
            raise RuntimeError("Database not connected. Call connect() first.")

    def _record_to_row(self, record: Record, id: str | None = None) -> dict[str, Any]:
        """Convert a Record to a database row, extracting vector fields."""
        from ..fields import VectorField
        from .postgres_vector import format_vector_for_postgres
        
        data = {}
        vectors = {}
        
        for field_name, field_obj in record.fields.items():
            if isinstance(field_obj, VectorField) and self._vector_enabled:
                # Store vector in separate column
                if field_obj.value is not None:
                    vectors[f"vector_{field_name}"] = format_vector_for_postgres(field_obj.value)
                # Store vector metadata in JSON
                data[field_name] = {
                    "type": "vector",
                    "dimensions": field_obj.dimensions,
                    "source_field": field_obj.source_field,
                    "model_name": field_obj.model_name,
                    "model_version": field_obj.model_version,
                }
            else:
                data[field_name] = field_obj.value

        result = {
            "id": id or str(uuid.uuid4()),
            "data": json.dumps(data),
            "metadata": json.dumps(record.metadata) if record.metadata else None,
        }
        
        # Add vector columns
        result.update(vectors)
        
        return result

    def _row_to_record(self, row: asyncpg.Record) -> Record:
        """Convert a database row to a Record, reconstructing vector fields."""
        from ..fields import VectorField
        from .postgres_vector import parse_postgres_vector
        
        data = row.get("data", {})
        if isinstance(data, str):
            data = json.loads(data)

        metadata = row.get("metadata", {})
        if isinstance(metadata, str) and metadata:
            metadata = json.loads(metadata)
        elif not metadata:
            metadata = {}

        # Check for vector fields in data
        if self._vector_enabled:
            for field_name, field_value in list(data.items()):
                if isinstance(field_value, dict) and field_value.get("type") == "vector":
                    # Check if we have the vector data
                    vector_column = f"vector_{field_name}"
                    if vector_column in row:
                        vector_data = row[vector_column]
                        if vector_data:
                            # Parse PostgreSQL vector format
                            vector_list = parse_postgres_vector(str(vector_data))
                            # Reconstruct VectorField
                            data[field_name] = vector_list
                        else:
                            # No vector data stored
                            data[field_name] = None
                    else:
                        data[field_name] = None

        return Record(data=data, metadata=metadata)

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
        """Update an existing record."""
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
        return result.split()[-1] != "0"

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

    async def upsert(self, id: str, record: Record) -> str:
        """Update or insert a record with a specific ID."""
        self._check_connection()
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
        query_vector: "np.ndarray | list[float] | VectorField",
        field_name: str,
        limit: int = 10,
        filters: list[Any] | None = None,
        metric: "DistanceMetric | str" = "cosine"
    ) -> list["VectorSearchResult"]:
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
        
        # Add filters if provided
        if filters:
            for filter_obj in filters:
                if hasattr(filter_obj, 'field') and hasattr(filter_obj, 'operator') and hasattr(filter_obj, 'value'):
                    field_path = f"data->>'{filter_obj.field}'"
                    
                    if filter_obj.operator == Operator.EQ:
                        sql += f" AND {field_path} = ${param_num}"
                        params.append(str(filter_obj.value))
                        param_num += 1
                    elif filter_obj.operator == Operator.GT:
                        sql += f" AND ({field_path})::numeric > ${param_num}"
                        params.append(filter_obj.value)
                        param_num += 1
                    elif filter_obj.operator == Operator.LT:
                        sql += f" AND ({field_path})::numeric < ${param_num}"
                        params.append(filter_obj.value)
                        param_num += 1
        
        # Order by distance and limit
        sql += f"""
        ORDER BY distance
        LIMIT {limit}
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
                embeddings = embedding_fn(texts)
                
                # Store vectors with records
                for j, record in enumerate(batch):
                    if j < len(embeddings):
                        vector = embeddings[j]
                        
                        # Add vector field to record
                        record.fields[vector_field] = VectorField(
                            name=vector_field,
                            value=vector,
                            source_field=text_field if isinstance(text_field, str) else ",".join(text_field),
                            model_name=model_name,
                            model_version=model_version,
                        )
                        
                        # Create or update record
                        if record.id:
                            await self.update(record.id, record)
                        else:
                            record_id = await self.create(record)
                            record.id = record_id
                        
                        processed_ids.append(record.id)
        
        return processed_ids

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

    async def _write_batch(self, records: list[Record]) -> None:
        """Write a batch of records using COPY for performance."""
        if not records:
            return

        # Prepare data for COPY
        rows = []
        for record in records:
            row_data = self._record_to_row(record)
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
