"""Tests for Postgres config-key identifier validation.

The postgres backend config parser must reject a
non-string ``schema`` or ``table`` value at construction time with a
clear ``ConfigurationError``, rather than silently propagating it
through ``quote_ident()`` and producing broken DDL at first query.

Coverage extends across all three Postgres consumers that flow user
config into ``quote_ident()``:

- ``Async/SyncPostgresDatabase`` via ``PostgresBaseConfig._parse_postgres_config``
- ``PgVectorStore._parse_backend_config`` (the third call site that
  bypasses ``_parse_postgres_config`` entirely)

Both go through the shared ``validate_pg_identifier`` helper.
"""

from __future__ import annotations

import pytest

from dataknobs_common.exceptions import ConfigurationError
from dataknobs_common.testing import requires_package


class TestPostgresConfigValidation:
    """Validation of the ``schema`` / ``table`` config keys."""

    def test_non_string_schema_raises_configuration_error(self) -> None:
        """A non-string ``schema`` is rejected.

        Reproduces the failure surface where the FSM's
        ``DatabaseSchema`` object was injected as the ``schema`` key
        (a config-name collision with the PG schema-name key), the
        prior code passed it through ``quote_ident()`` and emitted
        ``CREATE TABLE "DatabaseSchema(fields=...)".records (...)``.
        Post-fix, the parser fails fast with a clear error.
        """
        from dataknobs_data.backends.postgres import AsyncPostgresDatabase

        with pytest.raises(ConfigurationError) as exc_info:
            AsyncPostgresDatabase({"schema": object()})

        msg = str(exc_info.value)
        assert "schema" in msg.lower()
        assert "string" in msg.lower()

    def test_non_string_table_raises_configuration_error(self) -> None:
        """Same validation on ``table``.

        Same shape, same hazard — a non-string ``table`` value
        would propagate to broken DDL identically.  Pin the
        parallel guard so the data backend is robust against
        either-key misuse.
        """
        from dataknobs_data.backends.postgres import AsyncPostgresDatabase

        with pytest.raises(ConfigurationError) as exc_info:
            AsyncPostgresDatabase({"table": object()})

        msg = str(exc_info.value)
        assert "table" in msg.lower()
        assert "string" in msg.lower()

    def test_invalid_identifier_shape_raises_configuration_error(
        self,
    ) -> None:
        """Identifier-shape validation: clear error for malformed names.

        ``quote_ident()`` already escapes embedded double quotes
        defensively, but an early identifier-shape check produces
        a more actionable error than waiting for the SQL emission
        to fail asymmetrically per backend.
        """
        from dataknobs_data.backends.postgres import AsyncPostgresDatabase

        with pytest.raises(ConfigurationError):
            AsyncPostgresDatabase({"schema": "bad name with spaces"})


@requires_package("asyncpg")
class TestPgVectorStoreConfigValidation:
    """Validation of the ``schema`` / ``table_name`` keys on ``PgVectorStore``.

    The third Postgres consumer.  ``PgVectorStore._parse_backend_config``
    reads ``schema`` and ``table_name`` from its config dict directly
    and passes them to ``quote_ident()`` — the same hazard as the
    records backend, on a separate code path.  These tests pin that
    the same ``validate_pg_identifier`` defense applies.
    """

    @staticmethod
    def _config(**overrides: object) -> dict[str, object]:
        """Minimal valid PgVectorStore config + overrides.

        Connection-string is syntactically valid so ``__init__``
        gets past the connection-resolution gate and reaches the
        identifier-validation step we want to exercise.
        """
        base: dict[str, object] = {
            "connection_string": "postgres://test:test@localhost:5432/test",
        }
        base.update(overrides)
        return base

    def test_non_string_schema_raises_configuration_error(self) -> None:
        from dataknobs_data.vector.stores.pgvector import PgVectorStore

        with pytest.raises(ConfigurationError) as exc_info:
            PgVectorStore(self._config(schema=object()))

        msg = str(exc_info.value)
        assert "schema" in msg.lower()
        assert "string" in msg.lower()

    def test_non_string_table_name_raises_configuration_error(self) -> None:
        from dataknobs_data.vector.stores.pgvector import PgVectorStore

        with pytest.raises(ConfigurationError) as exc_info:
            PgVectorStore(self._config(table_name=object()))

        msg = str(exc_info.value)
        assert "table_name" in msg.lower()
        assert "string" in msg.lower()

    def test_invalid_schema_shape_raises_configuration_error(self) -> None:
        from dataknobs_data.vector.stores.pgvector import PgVectorStore

        with pytest.raises(ConfigurationError):
            PgVectorStore(self._config(schema="bad name with spaces"))
