"""Postgres integration tests for ``UnifiedDatabaseStorage.save_step``.

Phase 4 of Item 122 — pin the metadata-column routing contract on
PostgreSQL for the step-save path.  ``save_step(..., metadata=...)``
must populate the JSONB metadata column so downstream consumers can
filter via ``metadata.X`` dot-notation against the indexable column.

Pre-migration, ``save_step`` built ``Record({...})`` inline with no
metadata channel; every saved step had an empty metadata column.  The
:class:`AsyncKeyedRecordStore[_StepRecord]` composition forces the
metadata channel through the serializer signature, so the JSONB
column is populated by construction.

This file is the sibling of ``test_save_step_metadata.py`` (memory
backend).  Skipped automatically when PostgreSQL is unavailable.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import pytest
from dataknobs_common.testing import requires_postgres

from dataknobs_data.query import Query
from dataknobs_fsm.core.data_modes import DataHandlingMode
from dataknobs_fsm.execution.history import ExecutionStatus, ExecutionStep
from dataknobs_fsm.storage.base import StorageBackend, StorageConfig
from dataknobs_fsm.storage.database import UnifiedDatabaseStorage

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Generator

pytestmark = requires_postgres


def _make_step(step_id: str = "step-1") -> ExecutionStep:
    """Create a freshly completed step for testing."""
    step = ExecutionStep(
        step_id=step_id,
        state_name="state_a",
        network_name="main",
        timestamp=time.time(),
        data_mode=DataHandlingMode.COPY,
        status=ExecutionStatus.PENDING,
    )
    step.start()
    step.complete("arc_next")
    return step


@pytest.fixture
def postgres_storage_config(
    make_postgres_test_db,
) -> Generator[StorageConfig, None, None]:
    """Yield a ``StorageConfig`` wired for the factory path against PG.

    Uses the shared ``make_postgres_test_db`` fixture so the table is
    unique per test and dropped on teardown.
    """
    for pg in make_postgres_test_db("test_save_step_meta_"):
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


@pytest.fixture
async def postgres_storage(
    postgres_storage_config: StorageConfig,
) -> AsyncGenerator[UnifiedDatabaseStorage, None]:
    """Yield an initialized ``UnifiedDatabaseStorage`` against PG."""
    storage = UnifiedDatabaseStorage(postgres_storage_config)
    await storage.initialize()
    try:
        yield storage
    finally:
        await storage.close()


class TestSaveStepMetadataPostgres:
    """End-to-end metadata routing on PostgreSQL."""

    @pytest.mark.asyncio
    async def test_save_step_with_metadata_round_trip(
        self, postgres_storage: UnifiedDatabaseStorage,
    ) -> None:
        """``save_step(..., metadata=...)`` round-trips on PG.

        Reads the raw record back from the underlying database to
        confirm the metadata channel reached the JSONB column.
        """
        await postgres_storage.save_step(
            "exec-1",
            _make_step("step-1"),
            metadata={"tenant_id": "acme", "correlation_id": "corr-42"},
        )

        # Find the step record from the underlying DB.
        assert postgres_storage._steps_db is not None
        results: list[Any] = await postgres_storage._steps_db.search(
            Query().filter("step_data", "exists")
        )
        step_records = [r for r in results if r.get_value("step_data")]
        assert len(step_records) == 1
        rec = step_records[0]

        assert rec.metadata == {
            "tenant_id": "acme",
            "correlation_id": "corr-42",
        }
        # No leakage into the data column.
        assert "tenant_id" not in rec.data
        assert "correlation_id" not in rec.data
        # Structural fields stayed in data.
        assert rec.data["execution_id"] == "exec-1"
        assert rec.data["step_id"] == "step-1"

    @pytest.mark.asyncio
    async def test_default_metadata_is_empty_on_postgres(
        self, postgres_storage: UnifiedDatabaseStorage,
    ) -> None:
        """No ``metadata=`` kwarg ⇒ empty metadata column on PG."""
        await postgres_storage.save_step("exec-1", _make_step("step-1"))

        assert postgres_storage._steps_db is not None
        results: list[Any] = await postgres_storage._steps_db.search(
            Query().filter("step_data", "exists")
        )
        step_records = [r for r in results if r.get_value("step_data")]
        assert len(step_records) == 1
        rec = step_records[0]

        assert rec.metadata == {}
        # And nothing metadata-shaped leaked into the data column.
        assert "tenant_id" not in rec.data

    @pytest.mark.asyncio
    async def test_load_steps_after_save_with_metadata(
        self, postgres_storage: UnifiedDatabaseStorage,
    ) -> None:
        """``load_steps`` reconstructs steps written with metadata on PG.

        The deserializer reads from both the data and metadata channels.
        This pins that the round-trip works end-to-end through the
        public registry surface (not just the raw DB).
        """
        await postgres_storage.save_step(
            "exec-1",
            _make_step("step-1"),
            metadata={"tenant_id": "acme"},
        )
        await postgres_storage.save_step(
            "exec-1",
            _make_step("step-2"),
            metadata={"tenant_id": "globex"},
        )

        steps = await postgres_storage.load_steps("exec-1")
        assert {s.step_id for s in steps} == {"step-1", "step-2"}

    @pytest.mark.asyncio
    async def test_metadata_dot_notation_filter_postgres(
        self, postgres_storage: UnifiedDatabaseStorage,
    ) -> None:
        """``metadata.X`` dot-notation filter reaches the JSONB column on PG.

        Phase 7a added the ``filter_metadata=`` kwarg on ``load_steps``
        so production callers no longer need to drop to the raw
        database surface.  This test stays at the raw-``Query`` layer
        as a lower-level regression guard on the PG JSONB pushdown
        path; the higher-level ``load_steps(filter_metadata=...)``
        path is covered in ``test_save_step_metadata.py``.

        Pre-migration, this query returned an empty list because every
        step record had an empty metadata column.
        """
        await postgres_storage.save_step(
            "exec-1", _make_step("step-1"), metadata={"tenant_id": "acme"}
        )
        await postgres_storage.save_step(
            "exec-1", _make_step("step-2"), metadata={"tenant_id": "globex"}
        )

        assert postgres_storage._steps_db is not None
        acme_results: list[Any] = await postgres_storage._steps_db.search(
            Query()
            .filter("step_data", "exists")
            .filter("metadata.tenant_id", "=", "acme")
        )
        assert len(acme_results) == 1
        assert acme_results[0].data["step_id"] == "step-1"
        assert acme_results[0].metadata == {"tenant_id": "acme"}

    @pytest.mark.asyncio
    async def test_load_steps_filter_on_data_column_field_postgres(
        self, postgres_storage: UnifiedDatabaseStorage,
    ) -> None:
        """``load_steps(filters=...)`` continues to match on data-column fields on PG.

        Regression guard: the migration to ``AsyncKeyedRecordStore``
        rerouted the read path through ``store.search()`` (escape
        hatch).  This pins that the existing data-column filter
        contract is preserved end-to-end on PostgreSQL.
        """
        await postgres_storage.save_step("exec-1", _make_step("step-1"))
        other = _make_step("step-2")
        other.state_name = "state_b"
        await postgres_storage.save_step("exec-1", other)

        matched = await postgres_storage.load_steps(
            "exec-1", filters={"state_name": "state_a"}
        )
        assert [s.step_id for s in matched] == ["step-1"]

        none = await postgres_storage.load_steps(
            "exec-1", filters={"state_name": "missing"}
        )
        assert none == []
