"""PostgreSQL backend implementation for the data package."""

import asyncio
import json
import uuid
from typing import Any

from dataknobs_config import ConfigurableBase
from dataknobs_utils.sql_utils import DotenvPostgresConnector, PostgresDB

from ..database import Database, SyncDatabase
from ..query import Operator, Query, SortOrder
from ..records import Record


class SyncPostgresDatabase(SyncDatabase, ConfigurableBase):
    """Synchronous PostgreSQL database backend."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize PostgreSQL database.

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
    
    @classmethod
    def from_config(cls, config: dict) -> "SyncPostgresDatabase":
        """Create from config dictionary."""
        return cls(config)

    def _initialize(self) -> None:
        """Initialize the PostgreSQL connection and table."""
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

    def _ensure_table(self) -> None:
        """Ensure the records table exists."""
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
        row = self._record_to_row(record, id)

        sql = f"""
        UPDATE {self.schema_name}.{self.table_name}
        SET data = %(data)s, 
            metadata = %(metadata)s,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = %(id)s
        """

        # Execute update and check if any rows were affected
        affected = self.db.execute(sql, row)
        return affected > 0

    def delete(self, id: str) -> bool:
        """Delete a record by ID."""
        sql = f"""
        DELETE FROM {self.schema_name}.{self.table_name}
        WHERE id = %(id)s
        """

        affected = self.db.execute(sql, {"id": id})
        return affected > 0

    def exists(self, id: str) -> bool:
        """Check if a record exists."""
        sql = f"""
        SELECT 1 FROM {self.schema_name}.{self.table_name}
        WHERE id = %(id)s
        LIMIT 1
        """
        df = self.db.query(sql, {"id": id})
        return not df.empty

    def search(self, query: Query) -> list[Record]:
        """Search for records matching a query."""
        # Build SQL query from Query object
        sql_parts = [f"SELECT id, data, metadata FROM {self.schema_name}.{self.table_name}"]
        where_clauses = []
        params = {}

        # Apply filters
        for i, filter_obj in enumerate(query.filters):
            param_name = f"filter_{i}"
            field_path = f"data->>'{filter_obj.field}'"

            if filter_obj.operator == Operator.EQ:
                where_clauses.append(f"{field_path} = %({param_name})s")
                # Handle boolean values correctly (JSON stores as lowercase)
                if isinstance(filter_obj.value, bool):
                    params[param_name] = str(filter_obj.value).lower()
                else:
                    params[param_name] = str(filter_obj.value)
            elif filter_obj.operator == Operator.NEQ:
                where_clauses.append(f"{field_path} != %({param_name})s")
                # Handle boolean values correctly (JSON stores as lowercase)
                if isinstance(filter_obj.value, bool):
                    params[param_name] = str(filter_obj.value).lower()
                else:
                    params[param_name] = str(filter_obj.value)
            elif filter_obj.operator == Operator.GT:
                where_clauses.append(f"CAST({field_path} AS NUMERIC) > %({param_name})s")
                params[param_name] = filter_obj.value
            elif filter_obj.operator == Operator.GTE:
                where_clauses.append(f"CAST({field_path} AS NUMERIC) >= %({param_name})s")
                params[param_name] = filter_obj.value
            elif filter_obj.operator == Operator.LT:
                where_clauses.append(f"CAST({field_path} AS NUMERIC) < %({param_name})s")
                params[param_name] = filter_obj.value
            elif filter_obj.operator == Operator.LTE:
                where_clauses.append(f"CAST({field_path} AS NUMERIC) <= %({param_name})s")
                params[param_name] = filter_obj.value
            elif filter_obj.operator == Operator.LIKE:
                where_clauses.append(f"{field_path} LIKE %({param_name})s")
                params[param_name] = filter_obj.value
            elif filter_obj.operator == Operator.IN:
                placeholders = ", ".join([f"%({param_name}_{j})s" for j in range(len(filter_obj.value))])
                where_clauses.append(f"{field_path} IN ({placeholders})")
                for j, val in enumerate(filter_obj.value):
                    params[f"{param_name}_{j}"] = str(val)
            elif filter_obj.operator == Operator.NOT_IN:
                placeholders = ", ".join([f"%({param_name}_{j})s" for j in range(len(filter_obj.value))])
                where_clauses.append(f"{field_path} NOT IN ({placeholders})")
                for j, val in enumerate(filter_obj.value):
                    params[f"{param_name}_{j}"] = str(val)
            elif filter_obj.operator == Operator.EXISTS:
                where_clauses.append(f"data ? '{filter_obj.field}'")
            elif filter_obj.operator == Operator.NOT_EXISTS:
                where_clauses.append(f"NOT (data ? '{filter_obj.field}')")
            elif filter_obj.operator == Operator.REGEX:
                where_clauses.append(f"{field_path} ~ %({param_name})s")
                params[param_name] = filter_obj.value

        if where_clauses:
            sql_parts.append("WHERE " + " AND ".join(where_clauses))

        # Apply sorting
        if query.sort_specs:
            order_parts = []
            for sort_spec in query.sort_specs:
                field_path = f"data->>'{sort_spec.field}'"
                order = "DESC" if sort_spec.order == SortOrder.DESC else "ASC"
                # Try to cast numeric fields for proper sorting
                # This will use numeric sorting if the field looks numeric, otherwise string
                order_parts.append(f"CASE WHEN data->>'{sort_spec.field}' ~ '^[0-9]+(\\.[0-9]+)?$' THEN CAST(data->>'{sort_spec.field}' AS NUMERIC) ELSE 0 END {order}, {field_path} {order}")
            sql_parts.append("ORDER BY " + ", ".join(order_parts))

        # Apply limit and offset
        if query.limit_value:
            sql_parts.append(f"LIMIT {query.limit_value}")
        if query.offset_value:
            sql_parts.append(f"OFFSET {query.offset_value}")

        sql = " ".join(sql_parts)
        df = self.db.query(sql, params)

        records = []
        for _, row in df.iterrows():
            records.append(self._row_to_record(row.to_dict()))

        return records

    def _count_all(self) -> int:
        """Count all records in the database."""
        sql = f"SELECT COUNT(*) as count FROM {self.schema_name}.{self.table_name}"
        df = self.db.query(sql)
        return int(df.iloc[0]["count"])
    
    def count(self, query: Query | None = None) -> int:
        """Count records matching a query using efficient SQL COUNT.
        
        Args:
            query: Optional search query (counts all if None)
            
        Returns:
            Number of matching records
        """
        if not query or not query.filters:
            return self._count_all()
        
        # Build SQL count query from Query object
        sql_parts = [f"SELECT COUNT(*) as count FROM {self.schema_name}.{self.table_name}"]
        where_clauses = []
        params = {}
        
        # Apply filters (same logic as search method)
        for i, filter_obj in enumerate(query.filters):
            param_name = f"filter_{i}"
            field_path = f"data->>'{filter_obj.field}'"
            
            if filter_obj.operator == Operator.EQ:
                where_clauses.append(f"{field_path} = %({param_name})s")
                # Handle boolean values correctly (JSON stores as lowercase)
                if isinstance(filter_obj.value, bool):
                    params[param_name] = str(filter_obj.value).lower()
                else:
                    params[param_name] = str(filter_obj.value)
            elif filter_obj.operator == Operator.NEQ:
                where_clauses.append(f"{field_path} != %({param_name})s")
                # Handle boolean values correctly (JSON stores as lowercase)
                if isinstance(filter_obj.value, bool):
                    params[param_name] = str(filter_obj.value).lower()
                else:
                    params[param_name] = str(filter_obj.value)
            elif filter_obj.operator == Operator.GT:
                where_clauses.append(f"CAST({field_path} AS NUMERIC) > %({param_name})s")
                params[param_name] = filter_obj.value
            elif filter_obj.operator == Operator.GTE:
                where_clauses.append(f"CAST({field_path} AS NUMERIC) >= %({param_name})s")
                params[param_name] = filter_obj.value
            elif filter_obj.operator == Operator.LT:
                where_clauses.append(f"CAST({field_path} AS NUMERIC) < %({param_name})s")
                params[param_name] = filter_obj.value
            elif filter_obj.operator == Operator.LTE:
                where_clauses.append(f"CAST({field_path} AS NUMERIC) <= %({param_name})s")
                params[param_name] = filter_obj.value
            elif filter_obj.operator == Operator.LIKE:
                where_clauses.append(f"{field_path} LIKE %({param_name})s")
                params[param_name] = filter_obj.value
            elif filter_obj.operator == Operator.IN:
                placeholders = ", ".join([f"%({param_name}_{j})s" for j in range(len(filter_obj.value))])
                where_clauses.append(f"{field_path} IN ({placeholders})")
                for j, val in enumerate(filter_obj.value):
                    params[f"{param_name}_{j}"] = str(val)
            elif filter_obj.operator == Operator.NOT_IN:
                placeholders = ", ".join([f"%({param_name}_{j})s" for j in range(len(filter_obj.value))])
                where_clauses.append(f"{field_path} NOT IN ({placeholders})")
                for j, val in enumerate(filter_obj.value):
                    params[f"{param_name}_{j}"] = str(val)
            elif filter_obj.operator == Operator.EXISTS:
                where_clauses.append(f"data ? '{filter_obj.field}'")
            elif filter_obj.operator == Operator.NOT_EXISTS:
                where_clauses.append(f"NOT (data ? '{filter_obj.field}')")
            elif filter_obj.operator == Operator.REGEX:
                where_clauses.append(f"{field_path} ~ %({param_name})s")
                params[param_name] = filter_obj.value
        
        if where_clauses:
            sql_parts.append("WHERE " + " AND ".join(where_clauses))
        
        sql = " ".join(sql_parts)
        df = self.db.query(sql, params)
        return int(df.iloc[0]["count"])

    def clear(self) -> int:
        """Clear all records from the database."""
        sql = f"""
        DELETE FROM {self.schema_name}.{self.table_name}
        """

        with self.db.get_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql)
                affected = cursor.rowcount
                conn.commit()

        return affected

    def close(self) -> None:
        """Close the database connection."""
        # PostgresDB manages its own connections via context managers
        pass


class PostgresDatabase(Database, ConfigurableBase):
    """Asynchronous PostgreSQL database backend."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize async PostgreSQL database."""
        # Create sync database for delegation
        self._sync_db = SyncPostgresDatabase(config)
        super().__init__(config)
    
    @classmethod
    def from_config(cls, config: dict) -> "PostgresDatabase":
        """Create from config dictionary."""
        return cls(config)

    def _initialize(self) -> None:
        """Initialize is handled by sync database."""
        pass

    async def create(self, record: Record) -> str:
        """Create a new record asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_db.create, record)

    async def read(self, id: str) -> Record | None:
        """Read a record by ID asynchronously."""
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

    async def search(self, query: Query) -> list[Record]:
        """Search for records asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_db.search, query)

    async def exists(self, id: str) -> bool:
        """Check if a record exists asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_db.exists, id)

    async def _count_all(self) -> int:
        """Count all records asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_db._count_all)
    
    async def count(self, query: Query | None = None) -> int:
        """Count records matching a query asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_db.count, query)

    async def clear(self) -> int:
        """Clear all records asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_db.clear)

    async def close(self) -> None:
        """Close the database connection asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_db.close)
