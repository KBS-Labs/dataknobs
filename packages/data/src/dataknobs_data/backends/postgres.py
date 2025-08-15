"""PostgreSQL backend implementation with proper connection management."""

import asyncio
import json
import time
import uuid
from typing import Any, AsyncIterator, Iterator, Optional

from dataknobs_config import ConfigurableBase
from dataknobs_utils.sql_utils import DotenvPostgresConnector, PostgresDB

from ..database import Database, SyncDatabase
from ..query import Operator, Query, SortOrder
from ..records import Record
from ..streaming import StreamConfig, StreamResult


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
        id = str(uuid.uuid4())
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

    def stream_read(
        self,
        query: Optional[Query] = None,
        config: Optional[StreamConfig] = None
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
        config: Optional[StreamConfig] = None
    ) -> StreamResult:
        """Stream records into PostgreSQL."""
        self._check_connection()
        config = config or StreamConfig()
        result = StreamResult()
        start_time = time.time()
        
        batch = []
        for record in records:
            batch.append(record)
            
            if len(batch) >= config.batch_size:
                # Write batch
                try:
                    self._write_batch(batch)
                    result.successful += len(batch)
                    result.total_processed += len(batch)
                except Exception as e:
                    result.failed += len(batch)
                    result.total_processed += len(batch)
                    if config.on_error:
                        for rec in batch:
                            if not config.on_error(e, rec):
                                result.add_error(None, e)
                                break
                    else:
                        result.add_error(None, e)
                
                batch = []
        
        # Write remaining batch
        if batch:
            try:
                self._write_batch(batch)
                result.successful += len(batch)
                result.total_processed += len(batch)
            except Exception as e:
                result.failed += len(batch)
                result.total_processed += len(batch)
                result.add_error(None, e)
        
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


class PostgresDatabase(Database, ConfigurableBase):
    """Asynchronous PostgreSQL database backend with proper connection management."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize async PostgreSQL database."""
        # Create sync database for delegation
        self._sync_db = SyncPostgresDatabase(config)
        super().__init__(config)
        self._connected = False
    
    @classmethod
    def from_config(cls, config: dict) -> "PostgresDatabase":
        """Create from config dictionary."""
        return cls(config)

    async def connect(self) -> None:
        """Connect to the database."""
        if self._connected:
            return
        
        # Run sync connect in executor
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._sync_db.connect)
        self._connected = True

    async def close(self) -> None:
        """Close the database connection."""
        if self._connected:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._sync_db.close)
            self._connected = False

    def _initialize(self) -> None:
        """Initialize is handled by sync database."""
        pass

    async def create(self, record: Record) -> str:
        """Create a new record asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_db.create, record)

    async def read(self, id: str) -> Record | None:
        """Read a record asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_db.read, id)

    async def update(self, id: str, record: Record) -> bool:
        """Update a record asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_db.update, id, record)

    async def delete(self, id: str) -> bool:
        """Delete a record asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_db.delete, id)

    async def exists(self, id: str) -> bool:
        """Check if a record exists asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_db.exists, id)
    
    async def upsert(self, id: str, record: Record) -> str:
        """Update or insert a record with a specific ID."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_db.upsert, id, record)

    async def search(self, query: Query) -> list[Record]:
        """Search for records asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_db.search, query)

    async def _count_all(self) -> int:
        """Count all records asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_db._count_all)

    async def clear(self) -> int:
        """Clear all records asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_db.clear)

    async def stream_read(
        self,
        query: Optional[Query] = None,
        config: Optional[StreamConfig] = None
    ) -> AsyncIterator[Record]:
        """Stream records from PostgreSQL asynchronously."""
        loop = asyncio.get_event_loop()
        
        # Get sync iterator in thread
        sync_iter = await loop.run_in_executor(
            None,
            self._sync_db.stream_read,
            query,
            config
        )
        
        # Convert to async iterator
        for record in sync_iter:
            yield record
            # Small yield to prevent blocking
            await asyncio.sleep(0)

    async def stream_write(
        self,
        records: AsyncIterator[Record],
        config: Optional[StreamConfig] = None
    ) -> StreamResult:
        """Stream records into PostgreSQL asynchronously."""
        config = config or StreamConfig()
        result = StreamResult()
        start_time = time.time()
        
        batch = []
        async for record in records:
            batch.append(record)
            
            if len(batch) >= config.batch_size:
                # Write batch in executor
                loop = asyncio.get_event_loop()
                try:
                    await loop.run_in_executor(None, self._sync_db._write_batch, batch)
                    result.successful += len(batch)
                    result.total_processed += len(batch)
                except Exception as e:
                    result.failed += len(batch)
                    result.total_processed += len(batch)
                    if config.on_error:
                        for rec in batch:
                            if not config.on_error(e, rec):
                                result.add_error(None, e)
                                break
                    else:
                        result.add_error(None, e)
                
                batch = []
        
        # Write remaining batch
        if batch:
            loop = asyncio.get_event_loop()
            try:
                await loop.run_in_executor(None, self._sync_db._write_batch, batch)
                result.successful += len(batch)
                result.total_processed += len(batch)
            except Exception as e:
                result.failed += len(batch)
                result.total_processed += len(batch)
                result.add_error(None, e)
        
        result.duration = time.time() - start_time
        return result