"""Shared mixins for PostgreSQL database backends.

These mixins provide common functionality for both sync and async PostgreSQL implementations,
reducing code duplication and ensuring consistent behavior.
"""

import logging
import re
from typing import Any

from dataknobs_common import normalize_postgres_connection_config

from ..records import Record
from .sql_base import SQLTableManager
from .vector_config_mixin import VectorConfigMixin

logger = logging.getLogger(__name__)

# Valid PostgreSQL identifier pattern (unquoted identifiers)
_VALID_DB_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def validate_database_name(name: str) -> None:
    """Validate a database name to prevent SQL injection.

    Args:
        name: Database name to validate.

    Raises:
        ValueError: If the name contains invalid characters.
    """
    if not _VALID_DB_NAME_RE.match(name):
        raise ValueError(
            f"Invalid database name {name!r}: must start with a letter or "
            "underscore and contain only alphanumeric characters and underscores"
        )


def _validate_sql_identifier(name: str, label: str) -> str:
    """Raise ValueError if name is not a safe unquoted SQL identifier.

    Used to validate table_name and schema_name at construction time so that
    all downstream f-string DDL interpolations of those attributes are safe.
    """
    if not isinstance(name, str) or not _VALID_DB_NAME_RE.fullmatch(name):
        raise ValueError(f"Invalid SQL identifier for {label}: {name!r}")
    return name


class PostgresBaseConfig(VectorConfigMixin):
    """Shared configuration logic for PostgreSQL backends."""

    def _parse_postgres_config(
        self, config: dict[str, Any],
    ) -> tuple[str, str, dict, bool, bool]:
        """Extract table, schema, connection configuration, and DDL flags.

        Args:
            config: Configuration dictionary

        Returns:
            Tuple of (table_name, schema_name, connection_config,
            ensure_database, auto_create_table)
        """
        config = config.copy() if config else {}

        # Parse vector configuration using the mixin
        self._parse_vector_config(config)

        # Extract PostgreSQL-specific configuration
        table_name = config.pop("table", config.pop("table_name", "records"))
        schema_name = config.pop("schema", config.pop("schema_name", "public"))

        # Remove vector config parameters since they've been processed
        config.pop("vector_enabled", None)
        config.pop("vector_metric", None)

        # Extract and validate ensure_database as a proper boolean.
        # Delegate to the shared _coerce_bool (denylist) so all SQL backends
        # agree on what strings mean True vs False.
        raw_ensure = config.pop("ensure_database", True)
        ensure_database = SQLTableManager._coerce_bool(raw_ensure)

        # Extract and validate auto_create_table flag.
        raw_auto_create = config.pop("auto_create_table", True)
        auto_create_table = SQLTableManager._coerce_bool(raw_auto_create)

        # Normalize connection config via the shared helper so that every
        # downstream postgres site reads host/port/database/... from the
        # same canonical shape.
        #
        # ``require=False`` is intentional here — this is an internal
        # helper called from ``__init__``, where database-backend
        # contracts historically defer "is the connection resolvable"
        # to ``connect()``. Direct postgres entry points
        # (``PgVectorStore``, ``PostgresEventBus``) use ``require=True``
        # and fail at construction; backend ``__init__`` stays
        # permissive so consumers can construct, inspect, and swap
        # implementations without triggering config errors. Connection
        # failures surface at ``connect()`` time with asyncpg's native
        # errors.
        normalized = normalize_postgres_connection_config(
            config, require=False,
        )
        if normalized is not None:
            config.update(normalized)

        return table_name, schema_name, config, ensure_database, auto_create_table

    def _init_postgres_attributes(
        self,
        table_name: str,
        schema_name: str,
        ensure_database: bool = True,
        auto_create_table: bool = True,
    ) -> None:
        """Initialize common PostgreSQL attributes.

        Args:
            table_name: Name of the database table
            schema_name: Name of the database schema
            ensure_database: Auto-create database if missing (default: True)
            auto_create_table: Auto-create the records table if missing on
                connect (default: True). Set to False when an external
                migration tool (Alembic, Flyway, etc.) owns DDL — startup
                will then verify the table exists and raise a clear error
                if it doesn't.
        """
        self.table_name = _validate_sql_identifier(table_name, "table_name")
        self.schema_name = _validate_sql_identifier(schema_name, "schema_name")
        self._connected = False
        self._ensure_database_enabled = ensure_database
        self.auto_create_table = auto_create_table

        # Initialize vector state using the mixin
        self._init_vector_state()


class PostgresTableManager:
    """Shared table management SQL and logic."""

    @staticmethod
    def get_create_table_sql(schema_name: str, table_name: str) -> str:
        """Get SQL for creating the records table with indexes.
        
        Args:
            schema_name: Database schema name
            table_name: Database table name
            
        Returns:
            SQL string for table creation
        """
        return f"""
        CREATE TABLE IF NOT EXISTS {schema_name}.{table_name} (
            id TEXT PRIMARY KEY,
            data JSONB NOT NULL,
            metadata JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_{table_name}_data 
        ON {schema_name}.{table_name} USING GIN (data);
        
        CREATE INDEX IF NOT EXISTS idx_{table_name}_metadata
        ON {schema_name}.{table_name} USING GIN (metadata);
        """

class PostgresVectorSupport:
    """Shared vector support detection and management."""

    def _has_vector_fields(self, record: Record) -> bool:
        """Check if record has vector fields.
        
        Args:
            record: Record to check
            
        Returns:
            True if record has vector fields
        """
        from ..fields import VectorField
        return any(isinstance(field, VectorField)
                   for field in record.fields.values())

    def _extract_vector_dimensions(self, record: Record) -> dict[str, int]:
        """Extract dimensions from vector fields in a record.
        
        Args:
            record: Record containing potential vector fields
            
        Returns:
            Dictionary mapping field names to dimensions
        """
        from ..fields import VectorField
        dimensions = {}
        for name, field in record.fields.items():
            if isinstance(field, VectorField) and field.dimensions:
                dimensions[name] = field.dimensions
        return dimensions

    def _update_vector_dimensions(self, record: Record) -> None:
        """Update tracked vector dimensions from a record.
        
        Args:
            record: Record containing vector fields
        """
        if hasattr(self, '_vector_dimensions'):
            dimensions = self._extract_vector_dimensions(record)
            self._vector_dimensions.update(dimensions)


class PostgresErrorHandler:
    """Shared error handling logic for PostgreSQL operations."""

    @staticmethod
    def handle_connection_error(e: Exception) -> None:
        """Handle and log connection errors consistently.
        
        Args:
            e: The exception that occurred
            
        Raises:
            RuntimeError: With a user-friendly message
        """
        logger.error(f"PostgreSQL connection error: {e}")
        raise RuntimeError(f"Database connection failed: {e}")

    @staticmethod
    def handle_query_error(e: Exception, operation: str) -> None:
        """Handle and log query execution errors.
        
        Args:
            e: The exception that occurred
            operation: The operation that failed (e.g., "create", "update")
            
        Raises:
            RuntimeError: With a user-friendly message
        """
        logger.error(f"PostgreSQL {operation} error: {e}")
        raise RuntimeError(f"Database {operation} failed: {e}")

    @staticmethod
    def log_operation(operation: str, details: str = "") -> None:
        """Log a database operation for debugging.
        
        Args:
            operation: The operation being performed
            details: Additional details about the operation
        """
        if details:
            logger.debug(f"PostgreSQL {operation}: {details}")
        else:
            logger.debug(f"PostgreSQL {operation}")


class PostgresConnectionValidator:
    """Shared connection validation logic."""

    def _check_connection(self) -> None:
        """Check if database is connected.
        
        Raises:
            RuntimeError: If not connected
        """
        if not getattr(self, '_connected', False):
            raise RuntimeError("Database not connected. Call connect() first.")

    def _check_async_connection(self) -> None:
        """Check if async database is connected with pool.
        
        Raises:
            RuntimeError: If not connected or pool not initialized
        """
        if not getattr(self, '_connected', False) or not getattr(self, '_pool', None):
            raise RuntimeError("Database not connected. Call connect() first.")
