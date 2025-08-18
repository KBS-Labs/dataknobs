"""PostgreSQL backend implementation with proper connection management."""

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from datetime import datetime
from typing import Any

import asyncpg
from dataknobs_config import ConfigurableBase

from dataknobs_utils.sql_utils import DotenvPostgresConnector, PostgresDB

from ..database import AsyncDatabase, SyncDatabase
from ..pooling import ConnectionPoolManager
from ..pooling.postgres import PostgresPoolConfig, create_asyncpg_pool, validate_asyncpg_pool
from ..query import Operator, Query, SortOrder
from ..records import Record
from ..streaming import (
    StreamConfig,
    StreamResult,
    async_process_batch_with_fallback,
    process_batch_with_fallback,
)

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

    def search(self, query: Query) -> list[Record]:
        """Search for records matching the query."""
        self._check_connection()
        # Build SQL query from Query object
        where_clauses = []
        params = {}

        # Build WHERE clauses for filters
        for i, filter in enumerate(query.filters):
            field_path = f"data->>'{filter.field}'"
            param_name = f"param_{i}"

            if filter.operator == Operator.EQ:
                # Handle different types appropriately
                if isinstance(filter.value, bool):
                    where_clauses.append(f"({field_path})::boolean = %({param_name})s")
                    params[param_name] = filter.value
                elif isinstance(filter.value, (int, float)):
                    where_clauses.append(f"({field_path})::numeric = %({param_name})s")
                    params[param_name] = filter.value
                else:
                    where_clauses.append(f"{field_path} = %({param_name})s")
                    params[param_name] = str(filter.value)
            elif filter.operator == Operator.NEQ:
                if isinstance(filter.value, bool):
                    where_clauses.append(f"({field_path})::boolean != %({param_name})s")
                    params[param_name] = filter.value
                elif isinstance(filter.value, (int, float)):
                    where_clauses.append(f"({field_path})::numeric != %({param_name})s")
                    params[param_name] = filter.value
                else:
                    where_clauses.append(f"{field_path} != %({param_name})s")
                    params[param_name] = str(filter.value)
            elif filter.operator == Operator.GT:
                where_clauses.append(f"({field_path})::numeric > %({param_name})s")
                params[param_name] = filter.value
            elif filter.operator == Operator.LT:
                where_clauses.append(f"({field_path})::numeric < %({param_name})s")
                params[param_name] = filter.value
            elif filter.operator == Operator.GTE:
                where_clauses.append(f"({field_path})::numeric >= %({param_name})s")
                params[param_name] = filter.value
            elif filter.operator == Operator.LTE:
                where_clauses.append(f"({field_path})::numeric <= %({param_name})s")
                params[param_name] = filter.value
            elif filter.operator == Operator.LIKE:
                where_clauses.append(f"{field_path} LIKE %({param_name})s")
                params[param_name] = f"%{filter.value}%"
            elif filter.operator == Operator.IN:
                # Convert values to strings for comparison with JSONB text fields
                values = [str(v) for v in filter.value]
                where_clauses.append(f"{field_path} = ANY(%({param_name})s)")
                params[param_name] = values
            elif filter.operator == Operator.NOT_IN:
                # Convert values to strings for comparison with JSONB text fields
                values = [str(v) for v in filter.value]
                where_clauses.append(f"{field_path} != ALL(%({param_name})s)")
                params[param_name] = values
            elif filter.operator == Operator.BETWEEN:
                # Optimize BETWEEN for different data types
                if isinstance(filter.value, (list, tuple)) and len(filter.value) == 2:
                    lower, upper = filter.value
                    param_lower = f"{param_name}_lower"
                    param_upper = f"{param_name}_upper"

                    # Try to determine the type for proper casting
                    if isinstance(lower, (int, float)) and isinstance(upper, (int, float)):
                        where_clauses.append(
                            f"({field_path})::numeric BETWEEN %({param_lower})s AND %({param_upper})s"
                        )
                    elif isinstance(lower, datetime) or isinstance(upper, datetime):
                        where_clauses.append(
                            f"({field_path})::timestamp BETWEEN %({param_lower})s AND %({param_upper})s"
                        )
                    else:
                        # String or unknown type
                        where_clauses.append(
                            f"{field_path} BETWEEN %({param_lower})s AND %({param_upper})s"
                        )

                    params[param_lower] = lower
                    params[param_upper] = upper
            elif filter.operator == Operator.NOT_BETWEEN:
                # Optimize NOT BETWEEN
                if isinstance(filter.value, (list, tuple)) and len(filter.value) == 2:
                    lower, upper = filter.value
                    param_lower = f"{param_name}_lower"
                    param_upper = f"{param_name}_upper"

                    # Try to determine the type for proper casting
                    if isinstance(lower, (int, float)) and isinstance(upper, (int, float)):
                        where_clauses.append(
                            f"({field_path})::numeric NOT BETWEEN %({param_lower})s AND %({param_upper})s"
                        )
                    elif isinstance(lower, datetime) or isinstance(upper, datetime):
                        where_clauses.append(
                            f"({field_path})::timestamp NOT BETWEEN %({param_lower})s AND %({param_upper})s"
                        )
                    else:
                        # String or unknown type
                        where_clauses.append(
                            f"{field_path} NOT BETWEEN %({param_lower})s AND %({param_upper})s"
                        )

                    params[param_lower] = lower
                    params[param_upper] = upper

        # Build SQL
        sql = f"SELECT id, data, metadata FROM {self.schema_name}.{self.table_name}"
        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)

        # Add ORDER BY
        if query.sort_specs:
            order_clauses = []
            for sort_spec in query.sort_specs:
                # Try to cast to numeric for proper sorting
                # This will sort numbers correctly while still working for strings
                field_path = f"data->>'{sort_spec.field}'"
                direction = "DESC" if sort_spec.order == SortOrder.DESC else "ASC"
                # Use a CASE statement to handle both numeric and string sorting
                order_clause = f"""
                    CASE 
                        WHEN {field_path} ~ '^[0-9]+(\\.[0-9]+)?$' 
                        THEN ({field_path})::numeric 
                        ELSE NULL 
                    END {direction} NULLS LAST,
                    {field_path} {direction}
                """
                order_clauses.append(order_clause)
            sql += " ORDER BY " + ", ".join(order_clauses)

        # Add LIMIT and OFFSET
        if query.limit_value:
            sql += f" LIMIT {query.limit_value}"
        if query.offset_value:
            sql += f" OFFSET {query.offset_value}"

        # Execute query
        df = self.db.query(sql, params)

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


class AsyncPostgresDatabase(AsyncDatabase, ConfigurableBase):
    """Native async PostgreSQL database backend with event loop-aware connection pooling."""

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

        # Ensure table exists
        await self._ensure_table()
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

    def _check_connection(self) -> None:
        """Check if database is connected."""
        if not self._connected or not self._pool:
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

    def _row_to_record(self, row: asyncpg.Record) -> Record:
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

    async def create(self, record: Record) -> str:
        """Create a new record."""
        self._check_connection()
        # Use record's ID if it has one, otherwise generate a new one
        id = record.id if record.id else str(uuid.uuid4())
        row = self._record_to_row(record, id)

        sql = f"""
        INSERT INTO {self.schema_name}.{self.table_name} (id, data, metadata)
        VALUES ($1, $2, $3)
        """

        async with self._pool.acquire() as conn:
            await conn.execute(sql, row["id"], row["data"], row["metadata"])

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

    async def search(self, query: Query) -> list[Record]:
        """Search for records matching the query."""
        self._check_connection()

        # Build SQL query from Query object
        where_clauses = []
        params = []
        param_count = 0

        # Build WHERE clauses for filters
        for filter in query.filters:
            param_count += 1
            field_path = f"data->>'{filter.field}'"

            if filter.operator == Operator.EQ:
                if isinstance(filter.value, bool):
                    where_clauses.append(f"({field_path})::boolean = ${param_count}")
                    params.append(filter.value)
                elif isinstance(filter.value, (int, float)):
                    where_clauses.append(f"({field_path})::numeric = ${param_count}")
                    params.append(filter.value)
                else:
                    where_clauses.append(f"{field_path} = ${param_count}")
                    params.append(str(filter.value))
            elif filter.operator == Operator.NEQ:
                if isinstance(filter.value, bool):
                    where_clauses.append(f"({field_path})::boolean != ${param_count}")
                    params.append(filter.value)
                elif isinstance(filter.value, (int, float)):
                    where_clauses.append(f"({field_path})::numeric != ${param_count}")
                    params.append(filter.value)
                else:
                    where_clauses.append(f"{field_path} != ${param_count}")
                    params.append(str(filter.value))
            elif filter.operator == Operator.GT:
                where_clauses.append(f"({field_path})::numeric > ${param_count}")
                params.append(filter.value)
            elif filter.operator == Operator.LT:
                where_clauses.append(f"({field_path})::numeric < ${param_count}")
                params.append(filter.value)
            elif filter.operator == Operator.GTE:
                where_clauses.append(f"({field_path})::numeric >= ${param_count}")
                params.append(filter.value)
            elif filter.operator == Operator.LTE:
                where_clauses.append(f"({field_path})::numeric <= ${param_count}")
                params.append(filter.value)
            elif filter.operator == Operator.LIKE:
                where_clauses.append(f"{field_path} LIKE ${param_count}")
                params.append(f"%{filter.value}%")
            elif filter.operator == Operator.IN:
                values = [str(v) for v in filter.value]
                where_clauses.append(f"{field_path} = ANY(${param_count})")
                params.append(values)
            elif filter.operator == Operator.NOT_IN:
                values = [str(v) for v in filter.value]
                where_clauses.append(f"{field_path} != ALL(${param_count})")
                params.append(values)
            elif filter.operator == Operator.BETWEEN:
                # Optimize BETWEEN for different data types
                if isinstance(filter.value, (list, tuple)) and len(filter.value) == 2:
                    lower, upper = filter.value

                    # Try to determine the type for proper casting
                    if isinstance(lower, (int, float)) and isinstance(upper, (int, float)):
                        where_clauses.append(
                            f"({field_path})::numeric BETWEEN ${param_count} AND ${param_count + 1}"
                        )
                    elif isinstance(lower, datetime) or isinstance(upper, datetime):
                        where_clauses.append(
                            f"({field_path})::timestamp BETWEEN ${param_count} AND ${param_count + 1}"
                        )
                    else:
                        # String or unknown type
                        where_clauses.append(
                            f"{field_path} BETWEEN ${param_count} AND ${param_count + 1}"
                        )

                    params.append(lower)
                    params.append(upper)
                    param_count += 1  # We used two parameters
            elif filter.operator == Operator.NOT_BETWEEN:
                # Optimize NOT BETWEEN
                if isinstance(filter.value, (list, tuple)) and len(filter.value) == 2:
                    lower, upper = filter.value

                    # Try to determine the type for proper casting
                    if isinstance(lower, (int, float)) and isinstance(upper, (int, float)):
                        where_clauses.append(
                            f"({field_path})::numeric NOT BETWEEN ${param_count} AND ${param_count + 1}"
                        )
                    elif isinstance(lower, datetime) or isinstance(upper, datetime):
                        where_clauses.append(
                            f"({field_path})::timestamp NOT BETWEEN ${param_count} AND ${param_count + 1}"
                        )
                    else:
                        # String or unknown type
                        where_clauses.append(
                            f"{field_path} NOT BETWEEN ${param_count} AND ${param_count + 1}"
                        )

                    params.append(lower)
                    params.append(upper)
                    param_count += 1  # We used two parameters

        # Build SQL
        sql = f"SELECT id, data, metadata FROM {self.schema_name}.{self.table_name}"
        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)

        # Add ORDER BY
        if query.sort_specs:
            order_clauses = []
            for sort_spec in query.sort_specs:
                field_path = f"data->>'{sort_spec.field}'"
                direction = "DESC" if sort_spec.order == SortOrder.DESC else "ASC"
                # Handle numeric sorting
                order_clause = f"""
                    CASE 
                        WHEN {field_path} ~ '^[0-9]+(\\.[0-9]+)?$' 
                        THEN ({field_path})::numeric 
                        ELSE NULL 
                    END {direction} NULLS LAST,
                    {field_path} {direction}
                """
                order_clauses.append(order_clause)
            sql += " ORDER BY " + ", ".join(order_clauses)

        # Add LIMIT and OFFSET
        if query.limit_value:
            sql += f" LIMIT {query.limit_value}"
        if query.offset_value:
            sql += f" OFFSET {query.offset_value}"

        # Execute query
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
