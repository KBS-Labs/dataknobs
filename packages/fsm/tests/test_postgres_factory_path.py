"""End-to-end regression tests for ``UnifiedDatabaseStorage`` factory
path against a real PostgreSQL backend.

Pre-fix, the only way to use ``UnifiedDatabaseStorage`` with PG was
to pre-build ``AsyncPostgresDatabase`` and inject via ``database=``;
the factory path silently downgraded to ``AsyncMemoryDatabase``, or,
if that was bypassed, crashed with ``PostgresSyntaxError`` on the
first ``CREATE TABLE``.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import pytest

from dataknobs_common.testing import requires_postgres
from dataknobs_fsm.core.data_modes import DataHandlingMode
from dataknobs_fsm.execution.history import ExecutionHistory
from dataknobs_fsm.storage.base import StorageBackend, StorageConfig
from dataknobs_fsm.storage.database import UnifiedDatabaseStorage


@pytest.fixture
def postgres_factory_path_config(
    make_postgres_test_db,
) -> Generator[StorageConfig, None, None]:
    """Yield a ``StorageConfig`` wired for the factory path against PG.

    No ``'type'`` key in ``connection_params`` — proves the canonical
    enum drives backend selection.  Uses the shared
    ``make_postgres_test_db`` fixture from
    ``dataknobs_common.testing`` (Item 106 pytest11 plugin) so the
    table is unique per test and dropped on teardown.
    """
    for pg in make_postgres_test_db("test_fsm_factory_"):
        yield StorageConfig(
            backend=StorageBackend.POSTGRES,
            connection_params={
                "host": pg["host"],
                "port": pg["port"],
                "database": pg["database"],
                "user": pg["user"],
                "password": pg["password"],
                "table": pg["table"],
            },
        )


@requires_postgres
class TestPostgresFactoryPath:
    """Factory-path integration tests for the PG backend."""

    @pytest.mark.asyncio
    async def test_factory_path_initializes_against_postgres(
        self, postgres_factory_path_config: StorageConfig,
    ) -> None:
        """Items 116 + 117: factory path produces ``AsyncPostgresDatabase``
        and ``initialize()`` succeeds against a real PG.

        Failing pre-fix for two reasons simultaneously:

        - Item 116: ``_db`` is ``AsyncMemoryDatabase`` because
          ``connection_params['type']`` is missing.
        - Item 117: even with the type-key workaround in place,
          ``initialize()`` would raise ``PostgresSyntaxError``
          because the FSM's ``DatabaseSchema`` object collided
          with the PG ``schema`` (schema-name) config key.
        """
        from dataknobs_data.backends.postgres import AsyncPostgresDatabase

        storage = UnifiedDatabaseStorage(postgres_factory_path_config)
        try:
            await storage.initialize()
            assert isinstance(storage._db, AsyncPostgresDatabase)
        finally:
            await storage.close()

    @pytest.mark.asyncio
    async def test_factory_path_save_and_query_round_trip(
        self, postgres_factory_path_config: StorageConfig,
    ) -> None:
        """Round-trip: factory-path PG storage actually persists.

        The user-visible bug pre-fix: hours of runtime with zero
        history rows persisted.  Pin a save → query round-trip via
        the factory path so a future regression cannot silently
        revert.

        The leading ``isinstance`` guard ensures this test fails
        pre-fix on the silent-fallback bug (Item 116) rather than
        passing trivially because the round-trip happened to
        succeed in memory.  Without the guard, this test would
        pass on the buggy code path because save → query succeeds
        in ``AsyncMemoryDatabase`` even though the consumer asked
        for PG.
        """
        from dataknobs_data.backends.postgres import AsyncPostgresDatabase

        storage = UnifiedDatabaseStorage(postgres_factory_path_config)
        try:
            await storage.initialize()
            assert isinstance(storage._db, AsyncPostgresDatabase), (
                "Factory path silently fell back to "
                f"{type(storage._db).__name__} instead of producing an "
                "AsyncPostgresDatabase — round-trip below would test "
                "the wrong backend (Item 116)."
            )

            history = ExecutionHistory(
                execution_id="test-exec-116-117",
                fsm_name="test_fsm",
                data_mode=DataHandlingMode.COPY,
            )
            history.end_time = 1002.0
            await storage.save_history(
                history, metadata={"source": "test_116_117"},
            )

            results: list[dict[str, Any]] = await storage.query_histories({})
            assert any(
                r.get("id") == "test-exec-116-117" for r in results
            ), "History saved via factory path did not persist"
        finally:
            await storage.close()
