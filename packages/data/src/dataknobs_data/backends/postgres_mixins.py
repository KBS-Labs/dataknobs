# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Shared mixins for PostgreSQL database backends.

These mixins provide common functionality for both sync and async PostgreSQL implementations,
reducing code duplication and ensuring consistent behavior.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, ClassVar

from dataknobs_common.exceptions import ConfigurationError
from dataknobs_utils.sql_utils import quote_ident

from ..records import Record
from .layout_backend import ColumnLayoutMixin
from .vector_config_mixin import VectorConfigMixin

if TYPE_CHECKING:
    from .config import PostgresDatabaseConfig

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
    """The vector configuration both PostgreSQL backends share.

    Configuration itself is read by
    :class:`~dataknobs_data.backends.config.PostgresDatabaseConfig`; this
    class carries the vector state ``_apply_vector_config`` sets from it.
    """


class PostgresTableManager:
    """Shared table management SQL and logic."""

    @staticmethod
    def get_create_table_sql(schema_name: str, table_name: str) -> str:
        """Get SQL for creating the records table with indexes.

        ``id`` is declared ``COLLATE "C"``, so the primary-key index serves the
        code-point range and sort ``SQLQueryBuilder`` renders. A table created
        without it answers the same, and scans for those two.

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
            id TEXT COLLATE "C" PRIMARY KEY,
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

        return any(isinstance(field, VectorField) for field in record.fields.values())

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
        if hasattr(self, "_vector_dimensions"):
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
        if not getattr(self, "_connected", False):
            raise RuntimeError("Database not connected. Call connect() first.")

    def _check_async_connection(self) -> None:
        """Check if async database is connected with pool.

        Raises:
            RuntimeError: If not connected or pool not initialized
        """
        if not getattr(self, "_connected", False) or not getattr(self, "_pool", None):
            raise RuntimeError("Database not connected. Call connect() first.")


class PostgresLayoutMixin(ColumnLayoutMixin):
    """What reading a table through a column layout means on Postgres, for both twins.

    Everything but two questions is :class:`ColumnLayoutMixin`'s. Postgres
    resolves a native table by name with ``to_regclass``, which finds every
    relation a ``SELECT`` reads, and names the grant a role is missing when
    that lookup is refused.
    """

    _DIALECT: ClassVar[str] = "postgres"
    _LOCATION_KEYS: ClassVar[str] = "the table name and `schema_name`"

    schema_name: str
    _q_qualified: str

    if TYPE_CHECKING:

        @property
        def config(self) -> PostgresDatabaseConfig:
            """The consumer's typed configuration."""

    def _namespace(self) -> str | None:
        return self.schema_name

    def _relation_exists_query(self) -> tuple[str, Any]:
        """The statement and parameters asking whether the table is there, in the driver's style.

        A native table may be any relation a ``SELECT`` reads -- a table, a
        view, a materialized view, a foreign table -- so it is resolved by name
        with ``to_regclass``, which needs no privilege on the relation itself.
        ``information_schema.tables`` lists no materialized view. The table
        this package creates is a table, and is looked up there as before.
        """
        if not self.native:
            return super()._relation_exists_query()
        placeholder = "%(relation)s" if self._PARAM_STYLE == "pyformat" else "$1"
        sql = f"SELECT to_regclass({placeholder}) IS NOT NULL"
        if self._PARAM_STYLE == "pyformat":
            return sql, {"relation": self._q_qualified}
        return sql, (self._q_qualified,)

    def _schema_usage_error(self) -> RuntimeError:
        """The refusal for a schema the connecting role cannot use.

        ``to_regclass`` answers NULL for a relation that is not there, but
        raises when the role has no ``USAGE`` on the schema it names -- even for
        a table that is there. Each twin catches its driver's privilege error
        around the existence check and raises this instead.
        """
        return RuntimeError(
            f"The connecting role has no USAGE on schema {self.schema_name}, so "
            f"table {self.schema_name}.{self.table_name} cannot be read. A table read "
            f"through `layout: native` belongs to someone else: ask its owner to "
            f"grant USAGE on the schema and SELECT on the declared columns."
        )
