"""Shared mixins for PostgreSQL database backends.

These mixins provide common functionality for both sync and async PostgreSQL implementations,
reducing code duplication and ensuring consistent behavior.
"""

import logging
import re
from typing import Any

from dataknobs_common import normalize_postgres_connection_config
from dataknobs_common.exceptions import ConfigurationError
from dataknobs_utils.sql_utils import quote_ident

from ..records import Record
from .sql_base import SQLTableManager
from .vector_config_mixin import VectorConfigMixin

logger = logging.getLogger(__name__)

# Valid unquoted PostgreSQL identifier pattern: letters, digits,
# underscores; must start with a letter or underscore.  Used to
# validate database names, table names, and schema names — any
# config key that flows into ``quote_ident()`` and produces a SQL
# identifier.
_VALID_DB_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def validate_database_name(name: str) -> None:
    """Validate a database name to prevent SQL injection.

    Args:
        name: Database name to validate.

    Raises:
        ConfigurationError: If the name contains invalid characters.
    """
    if not _VALID_DB_NAME_RE.match(name):
        raise ConfigurationError(
            f"Invalid database name {name!r}: must start with a letter or "
            "underscore and contain only alphanumeric characters and underscores"
        )


def validate_pg_identifier(value: Any, key: str) -> str:
    """Validate that a config value is a safe Postgres identifier.

    Raises ``ConfigurationError`` with a clear message when the value
    is not a string (e.g. a ``DatabaseSchema`` object accidentally
    injected via a config-key collision) or has an unsupported
    identifier shape (e.g. embedded spaces or quotes).

    Args:
        value: Config value to validate.
        key: Config key name (``"table"`` / ``"schema"`` / etc.) for
            error messages.

    Returns:
        The validated identifier string.

    Raises:
        ConfigurationError: If the value is not a string or does not
            match the unquoted-identifier pattern.
    """
    if not isinstance(value, str):
        raise ConfigurationError(
            f"Postgres '{key}' must be a string identifier, got "
            f"{type(value).__name__}.  If you intended to pass a "
            "non-identifier payload, use a different config key."
        )
    if not _VALID_DB_NAME_RE.match(value):
        raise ConfigurationError(
            f"Invalid Postgres '{key}' identifier: {value!r}.  Must "
            f"match {_VALID_DB_NAME_RE.pattern}."
        )
    return value


class PostgresBaseConfig(VectorConfigMixin):
    """Shared configuration logic for PostgreSQL backends."""

    def _parse_postgres_config(
        self, config: dict[str, Any],
    ) -> tuple[str, str, dict, bool, bool]:
        """Extract table, schema, connection configuration, and boolean flags.

        Args:
            config: Configuration dictionary

        Returns:
            Tuple of (table_name, schema_name, connection_config,
            ensure_database, auto_create_table)
        """
        config = config.copy() if config else {}

        # Parse vector configuration using the mixin
        self._parse_vector_config(config)

        # Extract PostgreSQL-specific configuration.  Validate both
        # ``table`` and ``schema`` early to catch non-string or
        # malformed identifiers before they propagate to broken DDL
        # at first query.
        raw_table = config.pop("table", config.pop("table_name", "records"))
        raw_schema = config.pop("schema", config.pop("schema_name", "public"))
        table_name = validate_pg_identifier(raw_table, "table")
        schema_name = validate_pg_identifier(raw_schema, "schema")

        # Remove vector config parameters since they've been processed
        config.pop("vector_enabled", None)
        config.pop("vector_metric", None)

        # Extract and validate boolean flags via the shared coerce_bool helper so
        # that string values from YAML/env ("false", "0", "no") are handled
        # consistently across all backends (security.md §8 anti-pattern: raw
        # truthy check treats the string "false" as True).
        ensure_database = SQLTableManager.coerce_bool(
            config.pop("ensure_database", None), default=True
        )
        auto_create_table = SQLTableManager.coerce_bool(
            config.pop("auto_create_table", None), default=True
        )

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
            auto_create_table: Create the records table on connect if missing
                (default: True). Set to False when an external migration tool
                (Alembic, Flyway, etc.) owns DDL.
        """
        self.table_name = table_name
        self.schema_name = schema_name
        self._q_table = quote_ident(table_name)
        self._q_schema = quote_ident(schema_name)
        self._q_qualified = f"{self._q_schema}.{self._q_table}"
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
        q_schema = quote_ident(schema_name)
        q_table = quote_ident(table_name)
        q_idx_data = quote_ident(f"idx_{table_name}_data")
        q_idx_meta = quote_ident(f"idx_{table_name}_metadata")
        return f"""
        CREATE TABLE IF NOT EXISTS {q_schema}.{q_table} (
            id TEXT PRIMARY KEY,
            data JSONB NOT NULL,
            metadata JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS {q_idx_data}
        ON {q_schema}.{q_table} USING GIN (data);

        CREATE INDEX IF NOT EXISTS {q_idx_meta}
        ON {q_schema}.{q_table} USING GIN (metadata);
        """

    @staticmethod
    def get_table_exists_sql(schema_name: str, table_name: str) -> tuple[str, tuple[str, str]]:
        """Return ``(sql, params)`` to check if a table exists via parameterized query.

        Uses ``$1``/``$2`` positional binding for asyncpg.

        Args:
            schema_name: Database schema name
            table_name: Database table name

        Returns:
            Tuple of (sql_string, params_tuple) for asyncpg ``fetchval(sql, *params)``
        """
        sql = """
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_schema = $1
            AND table_name = $2
        )
        """
        return sql, (schema_name, table_name)

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
