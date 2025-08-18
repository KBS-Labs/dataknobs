"""Base SQL functionality shared between SQL database backends."""

import json
import uuid
from datetime import datetime
from typing import Any

from ..query import Operator, Query, SortOrder
from ..query_logic import ComplexQuery
from ..records import Record


class SQLQueryBuilder:
    """Builds SQL queries from Query objects."""
    
    def __init__(self, table_name: str, schema_name: str | None = None, dialect: str = "standard"):
        """Initialize the SQL query builder.
        
        Args:
            table_name: Name of the database table
            schema_name: Optional schema name
            dialect: SQL dialect ('postgres', 'sqlite', 'standard')
        """
        self.table_name = table_name
        self.schema_name = schema_name
        self.dialect = dialect
        self.qualified_table = self._get_qualified_table_name()
    
    def _get_qualified_table_name(self) -> str:
        """Get the fully qualified table name."""
        if self.schema_name:
            return f"{self.schema_name}.{self.table_name}"
        return self.table_name
    
    def build_create_query(self, record: Record, record_id: str | None = None) -> tuple[str, list[Any]]:
        """Build an INSERT query for creating a record.
        
        Args:
            record: The record to insert
            record_id: Optional ID (will generate if not provided)
            
        Returns:
            Tuple of (SQL query, parameters)
        """
        record_id = record_id or str(uuid.uuid4())
        data = self._record_to_json(record)
        metadata = json.dumps(record.metadata) if record.metadata else None
        
        if self.dialect == "postgres":
            query = f"""
                INSERT INTO {self.qualified_table} (id, data, metadata, created_at, updated_at)
                VALUES ($1, $2, $3, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                RETURNING id
            """
            params = [record_id, data, metadata]
        else:  # sqlite, standard
            query = f"""
                INSERT INTO {self.qualified_table} (id, data, metadata, created_at, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """
            params = [record_id, data, metadata]
        
        return query, params
    
    def build_read_query(self, record_id: str) -> tuple[str, list[Any]]:
        """Build a SELECT query for reading a record by ID.
        
        Args:
            record_id: The record ID
            
        Returns:
            Tuple of (SQL query, parameters)
        """
        if self.dialect == "postgres":
            query = f"SELECT * FROM {self.qualified_table} WHERE id = $1"
        else:  # sqlite, standard
            query = f"SELECT * FROM {self.qualified_table} WHERE id = ?"
        
        return query, [record_id]
    
    def build_update_query(self, record_id: str, record: Record) -> tuple[str, list[Any]]:
        """Build an UPDATE query for updating a record.
        
        Args:
            record_id: The record ID
            record: The updated record
            
        Returns:
            Tuple of (SQL query, parameters)
        """
        data = self._record_to_json(record)
        metadata = json.dumps(record.metadata) if record.metadata else None
        
        if self.dialect == "postgres":
            query = f"""
                UPDATE {self.qualified_table}
                SET data = $2, metadata = $3, updated_at = CURRENT_TIMESTAMP
                WHERE id = $1
            """
            params = [record_id, data, metadata]
        else:  # sqlite, standard
            query = f"""
                UPDATE {self.qualified_table}
                SET data = ?, metadata = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """
            params = [data, metadata, record_id]
        
        return query, params
    
    def build_delete_query(self, record_id: str) -> tuple[str, list[Any]]:
        """Build a DELETE query for deleting a record.
        
        Args:
            record_id: The record ID
            
        Returns:
            Tuple of (SQL query, parameters)
        """
        if self.dialect == "postgres":
            query = f"DELETE FROM {self.qualified_table} WHERE id = $1"
        else:  # sqlite, standard
            query = f"DELETE FROM {self.qualified_table} WHERE id = ?"
        
        return query, [record_id]
    
    def build_exists_query(self, record_id: str) -> tuple[str, list[Any]]:
        """Build a query to check if a record exists.
        
        Args:
            record_id: The record ID
            
        Returns:
            Tuple of (SQL query, parameters)
        """
        if self.dialect == "postgres":
            query = f"SELECT 1 FROM {self.qualified_table} WHERE id = $1 LIMIT 1"
        else:  # sqlite, standard
            query = f"SELECT 1 FROM {self.qualified_table} WHERE id = ? LIMIT 1"
        
        return query, [record_id]
    
    def build_complex_search_query(self, query: ComplexQuery) -> tuple[str, list[Any]]:
        """Build a SELECT query from a ComplexQuery object with boolean logic.
        
        Args:
            query: The ComplexQuery object
            
        Returns:
            Tuple of (SQL query, parameters)
        """
        sql_parts = [f"SELECT * FROM {self.qualified_table}"]
        params = []
        
        # Build WHERE clause from complex conditions
        if query.condition:
            where_clause, where_params = self._build_complex_condition(query.condition, 1)
            if where_clause:
                sql_parts.append(f"WHERE {where_clause}")
                params.extend(where_params)
        
        # Add ORDER BY
        if query.sort_specs:
            order_parts = []
            for sort_spec in query.sort_specs:
                direction = "DESC" if sort_spec.order == SortOrder.DESC else "ASC"
                if self.dialect == "postgres":
                    order_parts.append(f"data->'{sort_spec.field}' {direction}")
                elif self.dialect == "sqlite":
                    order_parts.append(f"json_extract(data, '$.{sort_spec.field}') {direction}")
                else:
                    order_parts.append(f"data {direction}")
            sql_parts.append("ORDER BY " + ", ".join(order_parts))
        
        # Add LIMIT and OFFSET
        if query.limit_value:
            sql_parts.append(f"LIMIT {query.limit_value}")
        if query.offset_value:
            sql_parts.append(f"OFFSET {query.offset_value}")
        
        return " ".join(sql_parts), params
    
    def _build_complex_condition(self, condition: Any, param_start: int) -> tuple[str, list[Any]]:
        """Build WHERE clause for complex boolean logic conditions.
        
        Args:
            condition: The Condition object (LogicCondition or FilterCondition)
            param_start: Starting parameter number
            
        Returns:
            Tuple of (SQL clause, parameters)
        """
        from ..query_logic import FilterCondition, LogicCondition, LogicOperator
        
        params = []
        
        # Handle FilterCondition (leaf node)
        if isinstance(condition, FilterCondition):
            clause, filter_params = self._build_filter_clause(condition.filter, param_start)
            return clause, filter_params
        
        # Handle LogicCondition (branch node)
        elif isinstance(condition, LogicCondition):
            if condition.operator == LogicOperator.AND:
                clauses = []
                current_param = param_start
                for sub_condition in condition.conditions:
                    sub_clause, sub_params = self._build_complex_condition(sub_condition, current_param)
                    if sub_clause:
                        clauses.append(sub_clause)
                        params.extend(sub_params)
                        current_param += len(sub_params)
                return (f"({' AND '.join(clauses)})", params) if clauses else ("", [])
            
            elif condition.operator == LogicOperator.OR:
                clauses = []
                current_param = param_start
                for sub_condition in condition.conditions:
                    sub_clause, sub_params = self._build_complex_condition(sub_condition, current_param)
                    if sub_clause:
                        clauses.append(sub_clause)
                        params.extend(sub_params)
                        current_param += len(sub_params)
                return (f"({' OR '.join(clauses)})", params) if clauses else ("", [])
            
            elif condition.operator == LogicOperator.NOT:
                sub_clause, sub_params = self._build_complex_condition(condition.conditions[0], param_start)
                params.extend(sub_params)
                return (f"NOT ({sub_clause})", params) if sub_clause else ("", [])
        
        return ("", [])
    
    def build_search_query(self, query: Query) -> tuple[str, list[Any]]:
        """Build a SELECT query from a Query object.
        
        Args:
            query: The Query object
            
        Returns:
            Tuple of (SQL query, parameters)
        """
        sql_parts = [f"SELECT * FROM {self.qualified_table}"]
        params = []
        param_count = 0
        
        # Build WHERE clause
        where_clauses = []
        for filter_spec in query.filters:
            param_count += 1
            clause, new_params = self._build_filter_clause(filter_spec, param_count)
            where_clauses.append(clause)
            params.extend(new_params)
            param_count += len(new_params) - 1  # Adjust for multiple params
        
        if where_clauses:
            sql_parts.append("WHERE " + " AND ".join(where_clauses))
        
        # Add ORDER BY
        if query.sort_specs:
            order_parts = []
            for sort_spec in query.sort_specs:
                direction = "DESC" if sort_spec.order == SortOrder.DESC else "ASC"
                # Use JSON extraction based on dialect
                if self.dialect == "postgres":
                    order_parts.append(f"data->'{sort_spec.field}' {direction}")
                elif self.dialect == "sqlite":
                    order_parts.append(f"json_extract(data, '$.{sort_spec.field}') {direction}")
                else:
                    order_parts.append(f"data {direction}")  # Fallback
            sql_parts.append("ORDER BY " + ", ".join(order_parts))
        
        # Add LIMIT and OFFSET
        if query.limit_value:
            sql_parts.append(f"LIMIT {query.limit_value}")
        if query.offset_value:
            sql_parts.append(f"OFFSET {query.offset_value}")
        
        return " ".join(sql_parts), params
    
    def build_count_query(self, query: Query | None = None) -> tuple[str, list[Any]]:
        """Build a COUNT query.
        
        Args:
            query: Optional Query object for filtering
            
        Returns:
            Tuple of (SQL query, parameters)
        """
        if query and query.filters:
            search_query, params = self.build_search_query(query)
            # Replace SELECT * with SELECT COUNT(*)
            count_query = search_query.replace("SELECT *", "SELECT COUNT(*)", 1)
            # Remove ORDER BY, LIMIT, OFFSET clauses
            for clause in ["ORDER BY", "LIMIT", "OFFSET"]:
                if clause in count_query:
                    count_query = count_query[:count_query.index(clause)]
            return count_query.strip(), params
        else:
            return f"SELECT COUNT(*) FROM {self.qualified_table}", []
    
    def _build_filter_clause(self, filter_spec: Any, param_start: int) -> tuple[str, list[Any]]:
        """Build a WHERE clause for a filter.
        
        Args:
            filter_spec: The filter specification
            param_start: Starting parameter number
            
        Returns:
            Tuple of (SQL clause, parameters)
        """
        field = filter_spec.field
        op = filter_spec.operator
        value = filter_spec.value
        
        # JSON field extraction
        if self.dialect == "postgres":
            field_expr = f"data->>'{field}'"
            param_placeholder = f"${param_start}"
        elif self.dialect == "sqlite":
            field_expr = f"json_extract(data, '$.{field}')"
            param_placeholder = "?"
        else:
            field_expr = field
            param_placeholder = "?"
        
        # Build clause based on operator
        if op == Operator.EQ:
            return f"{field_expr} = {param_placeholder}", [value]
        elif op == Operator.NEQ:
            return f"{field_expr} != {param_placeholder}", [value]
        elif op == Operator.GT:
            return f"{field_expr} > {param_placeholder}", [value]
        elif op == Operator.GTE:
            return f"{field_expr} >= {param_placeholder}", [value]
        elif op == Operator.LT:
            return f"{field_expr} < {param_placeholder}", [value]
        elif op == Operator.LTE:
            return f"{field_expr} <= {param_placeholder}", [value]
        elif op == Operator.LIKE:
            return f"{field_expr} LIKE {param_placeholder}", [value]
        elif op == Operator.IN:
            if self.dialect == "postgres":
                placeholders = ", ".join([f"${i}" for i in range(param_start, param_start + len(value))])
            else:
                placeholders = ", ".join(["?" for _ in value])
            return f"{field_expr} IN ({placeholders})", list(value)
        elif op == Operator.NOT_IN:
            if self.dialect == "postgres":
                placeholders = ", ".join([f"${i}" for i in range(param_start, param_start + len(value))])
            else:
                placeholders = ", ".join(["?" for _ in value])
            return f"{field_expr} NOT IN ({placeholders})", list(value)
        elif op == Operator.BETWEEN:
            if self.dialect == "postgres":
                return f"{field_expr} BETWEEN ${param_start} AND ${param_start + 1}", list(value)
            else:
                return f"{field_expr} BETWEEN ? AND ?", list(value)
        elif op == Operator.EXISTS:
            return f"{field_expr} IS NOT NULL", []
        elif op == Operator.NOT_EXISTS:
            return f"{field_expr} IS NULL", []
        else:
            raise ValueError(f"Unsupported operator: {op}")
    
    def _record_to_json(self, record: Record) -> str:
        """Convert a Record to JSON string for storage."""
        data = {}
        for field_name, field_obj in record.fields.items():
            data[field_name] = field_obj.value
        return json.dumps(data)
    
    @staticmethod
    def row_to_record(row: dict[str, Any]) -> Record:
        """Convert a database row to a Record.
        
        Args:
            row: Database row as dictionary
            
        Returns:
            Record object
        """
        data = row.get("data", {})
        if isinstance(data, str):
            data = json.loads(data)
        
        metadata = row.get("metadata", {})
        if isinstance(metadata, str) and metadata:
            metadata = json.loads(metadata)
        elif not metadata:
            metadata = {}
        
        return Record(data=data, metadata=metadata)


class SQLTableManager:
    """Manages SQL table creation and schema."""
    
    def __init__(self, table_name: str, schema_name: str | None = None, dialect: str = "standard"):
        """Initialize the table manager.
        
        Args:
            table_name: Name of the database table
            schema_name: Optional schema name
            dialect: SQL dialect ('postgres', 'sqlite', 'standard')
        """
        self.table_name = table_name
        self.schema_name = schema_name
        self.dialect = dialect
        self.qualified_table = self._get_qualified_table_name()
    
    def _get_qualified_table_name(self) -> str:
        """Get the fully qualified table name."""
        if self.schema_name:
            return f"{self.schema_name}.{self.table_name}"
        return self.table_name
    
    def get_create_table_sql(self) -> str:
        """Get the CREATE TABLE SQL statement.
        
        Returns:
            SQL statement for creating the table
        """
        if self.dialect == "postgres":
            return f"""
            CREATE TABLE IF NOT EXISTS {self.qualified_table} (
                id VARCHAR(255) PRIMARY KEY,
                data JSONB NOT NULL,
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_{self.table_name}_data 
            ON {self.qualified_table} USING GIN (data);
            
            CREATE INDEX IF NOT EXISTS idx_{self.table_name}_metadata
            ON {self.qualified_table} USING GIN (metadata);
            """
        elif self.dialect == "sqlite":
            # SQLite doesn't have JSONB, uses TEXT for JSON storage
            return f"""
            CREATE TABLE IF NOT EXISTS {self.qualified_table} (
                id VARCHAR(255) PRIMARY KEY,
                data TEXT NOT NULL CHECK (json_valid(data)),
                metadata TEXT CHECK (metadata IS NULL OR json_valid(metadata)),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_{self.table_name}_created
            ON {self.qualified_table} (created_at);
            
            CREATE INDEX IF NOT EXISTS idx_{self.table_name}_updated
            ON {self.qualified_table} (updated_at);
            """
        else:
            # Generic SQL
            return f"""
            CREATE TABLE IF NOT EXISTS {self.qualified_table} (
                id VARCHAR(255) PRIMARY KEY,
                data TEXT NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
    
    def get_drop_table_sql(self) -> str:
        """Get the DROP TABLE SQL statement.
        
        Returns:
            SQL statement for dropping the table
        """
        return f"DROP TABLE IF EXISTS {self.qualified_table}"