"""Tests for metadata filtering in query_histories()."""

from __future__ import annotations

import logging
import os
import uuid
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from dataknobs_fsm.core.data_modes import DataHandlingMode
from dataknobs_fsm.execution.history import ExecutionHistory
from dataknobs_fsm.storage import InMemoryStorage, StorageBackend, StorageConfig
from dataknobs_fsm.storage.database import UnifiedDatabaseStorage


def _make_history(
    execution_id: str,
    fsm_name: str = "test_fsm",
    data_mode: DataHandlingMode = DataHandlingMode.COPY,
) -> ExecutionHistory:
    """Create a finalized ExecutionHistory for testing."""
    history = ExecutionHistory(
        fsm_name=fsm_name,
        execution_id=execution_id,
        data_mode=data_mode,
    )
    history.finalize()
    return history


@pytest.fixture()
def storage() -> InMemoryStorage:
    config = StorageConfig(backend=StorageBackend.MEMORY)
    return InMemoryStorage(config)


class TestMetadataFiltering:
    """Tests for metadata.* filter key support in query_histories()."""

    @pytest.mark.asyncio
    async def test_filter_by_single_metadata_key(
        self, storage: InMemoryStorage
    ) -> None:
        """Filtering by metadata.work_order_id returns only matching histories."""
        await storage.initialize()
        await storage.save_history(
            _make_history("exec_1"),
            metadata={"work_order_id": "WO-001"},
        )
        await storage.save_history(
            _make_history("exec_2"),
            metadata={"work_order_id": "WO-002"},
        )

        results = await storage.query_histories(
            {"metadata.work_order_id": "WO-001"}
        )

        assert len(results) == 1
        assert results[0]["id"] == "exec_1"
        assert results[0]["metadata"]["work_order_id"] == "WO-001"

    @pytest.mark.asyncio
    async def test_filter_by_multiple_metadata_keys(
        self, storage: InMemoryStorage
    ) -> None:
        """Multiple metadata.* filters use AND semantics."""
        await storage.initialize()
        await storage.save_history(
            _make_history("exec_1"),
            metadata={"work_order_id": "WO-001", "scope_id": "S-A"},
        )
        await storage.save_history(
            _make_history("exec_2"),
            metadata={"work_order_id": "WO-001", "scope_id": "S-B"},
        )
        await storage.save_history(
            _make_history("exec_3"),
            metadata={"work_order_id": "WO-002", "scope_id": "S-A"},
        )

        results = await storage.query_histories(
            {"metadata.work_order_id": "WO-001", "metadata.scope_id": "S-A"}
        )

        assert len(results) == 1
        assert results[0]["id"] == "exec_1"

    @pytest.mark.asyncio
    async def test_metadata_filter_combined_with_builtin_filter(
        self, storage: InMemoryStorage
    ) -> None:
        """metadata.* filters work alongside builtin filters like fsm_name."""
        await storage.initialize()
        await storage.save_history(
            _make_history("exec_1", fsm_name="alpha"),
            metadata={"work_order_id": "WO-001"},
        )
        await storage.save_history(
            _make_history("exec_2", fsm_name="beta"),
            metadata={"work_order_id": "WO-001"},
        )

        results = await storage.query_histories(
            {"fsm_name": "alpha", "metadata.work_order_id": "WO-001"}
        )

        assert len(results) == 1
        assert results[0]["id"] == "exec_1"
        assert results[0]["fsm_name"] == "alpha"

    @pytest.mark.asyncio
    async def test_metadata_filter_no_match_returns_empty(
        self, storage: InMemoryStorage
    ) -> None:
        """Filtering by a metadata value that doesn't exist returns empty list."""
        await storage.initialize()
        await storage.save_history(
            _make_history("exec_1"),
            metadata={"work_order_id": "WO-001"},
        )

        results = await storage.query_histories(
            {"metadata.work_order_id": "WO-999"}
        )

        assert results == []

    @pytest.mark.asyncio
    async def test_empty_metadata_filters_returns_all(
        self, storage: InMemoryStorage
    ) -> None:
        """No metadata.* keys in filters returns all histories (backward compat)."""
        await storage.initialize()
        await storage.save_history(
            _make_history("exec_1"),
            metadata={"work_order_id": "WO-001"},
        )
        await storage.save_history(
            _make_history("exec_2"),
            metadata={"work_order_id": "WO-002"},
        )

        results = await storage.query_histories({})

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_metadata_filter_respects_pagination(
        self, storage: InMemoryStorage
    ) -> None:
        """Query-level metadata filtering works correctly with limit."""
        await storage.initialize()
        for i in range(5):
            await storage.save_history(
                _make_history(f"match_{i}", fsm_name="target"),
                metadata={"group": "A"},
            )
        # Add non-matching histories
        for i in range(3):
            await storage.save_history(
                _make_history(f"other_{i}", fsm_name="target"),
                metadata={"group": "B"},
            )

        results = await storage.query_histories(
            {"metadata.group": "A"}, limit=2
        )

        assert len(results) == 2
        # All returned results should have group=A metadata
        for r in results:
            assert r["metadata"]["group"] == "A"


    @pytest.mark.asyncio
    async def test_metadata_filter_respects_offset(
        self, storage: InMemoryStorage
    ) -> None:
        """Offset applies to filtered results, not the pre-filter set."""
        await storage.initialize()
        # Use distinct start_times to make ordering deterministic
        for i in range(5):
            h = _make_history(f"match_{i}", fsm_name="target")
            h.start_time = 1_000_000.0 + i
            await storage.save_history(h, metadata={"group": "A"})
        # Non-matching histories should not affect offset
        for i in range(3):
            h = _make_history(f"other_{i}", fsm_name="target")
            h.start_time = 2_000_000.0 + i
            await storage.save_history(h, metadata={"group": "B"})

        results = await storage.query_histories(
            {"metadata.group": "A"}, limit=3, offset=2
        )

        assert len(results) == 3
        # sort_by('start_time', 'desc') → match_4, match_3, match_2, match_1, match_0
        # offset=2 skips match_4 and match_3
        assert results[0]["id"] == "match_2"
        assert results[1]["id"] == "match_1"
        assert results[2]["id"] == "match_0"
        for r in results:
            assert r["metadata"]["group"] == "A"


class TestFileBackendMetadataFiltering:
    """Tests for metadata filtering on the FILE backend."""

    @pytest.mark.asyncio
    async def test_metadata_filter_works_on_file_backend(
        self, tmp_path: Path
    ) -> None:
        """FILE backend supports metadata filtering via dot-notation."""
        from dataknobs_fsm.storage import FileStorage

        config = StorageConfig(
            backend=StorageBackend.FILE,
            connection_params={"path": str(tmp_path / "histories")},
        )
        storage = FileStorage(config)
        await storage.initialize()

        await storage.save_history(
            _make_history("exec_1"),
            metadata={"tenant": "A"},
        )
        await storage.save_history(
            _make_history("exec_2"),
            metadata={"tenant": "B"},
        )

        results = await storage.query_histories({"metadata.tenant": "A"})
        assert len(results) == 1
        assert results[0]["id"] == "exec_1"
        assert results[0]["metadata"]["tenant"] == "A"


class TestLegacyMetadataFallback:
    """Tests for backward compatibility with records that stored metadata in data column."""

    @pytest.mark.asyncio
    async def test_legacy_metadata_in_data_column_still_readable(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Records stored under the old schema should still return metadata."""
        from dataknobs_data.backends.memory import AsyncMemoryDatabase
        from dataknobs_data.records import Record

        db = AsyncMemoryDatabase()
        config = StorageConfig(backend=StorageBackend.MEMORY)
        storage = UnifiedDatabaseStorage(config, database=db)
        await storage.initialize()

        # Simulate a legacy record with metadata in the data column
        legacy_record = Record({
            'id': str(uuid.uuid4()),
            'execution_id': 'exec-legacy',
            'fsm_name': 'test_fsm',
            'data_mode': 'copy',
            'status': 'completed',
            'start_time': 1_000_000.0,
            'end_time': 1_000_001.0,
            'total_steps': 1,
            'failed_steps': 0,
            'skipped_steps': 0,
            'history_data': '{}',
            'created_at': 1_000_000.0,
            'updated_at': 1_000_000.0,
            'metadata': {'work_order_id': 'WO-LEGACY'},
        })
        await db.upsert(legacy_record)

        with caplog.at_level(logging.WARNING):
            results = await storage.query_histories({})

        # Legacy metadata should be readable via fallback
        legacy_results = [r for r in results if r['id'] == 'exec-legacy']
        assert len(legacy_results) == 1
        assert legacy_results[0]['metadata']['work_order_id'] == 'WO-LEGACY'

        # Should log a warning about legacy location
        assert any("legacy" in r.message.lower() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_new_metadata_takes_precedence_over_legacy(self) -> None:
        """New-style metadata in Record.metadata is used when present."""
        from dataknobs_data.backends.memory import AsyncMemoryDatabase

        db = AsyncMemoryDatabase()
        config = StorageConfig(backend=StorageBackend.MEMORY)
        storage = UnifiedDatabaseStorage(config, database=db)
        await storage.initialize()

        # Save with the new storage model
        await storage.save_history(
            _make_history("exec_new"),
            metadata={"work_order_id": "WO-NEW"},
        )

        results = await storage.query_histories({})
        new_results = [r for r in results if r['id'] == 'exec_new']
        assert len(new_results) == 1
        assert new_results[0]['metadata']['work_order_id'] == 'WO-NEW'


class TestMetadataNotPollutedByRecordId:
    """Tests that Record internals do not pollute caller metadata."""

    @pytest.mark.asyncio
    async def test_metadata_does_not_contain_internal_id(self) -> None:
        """query_histories metadata should not contain Record's internal 'id' key."""
        from dataknobs_data.backends.memory import AsyncMemoryDatabase

        db = AsyncMemoryDatabase()
        config = StorageConfig(backend=StorageBackend.MEMORY)
        storage = UnifiedDatabaseStorage(config, database=db)
        await storage.initialize()

        await storage.save_history(
            _make_history("exec_1"),
            metadata={"work_order_id": "WO-001"},
        )

        results = await storage.query_histories({})
        assert len(results) == 1
        # Metadata should contain only what was passed, not internal Record id
        assert results[0]["metadata"] == {"work_order_id": "WO-001"}

    @pytest.mark.asyncio
    async def test_empty_metadata_stays_empty(self) -> None:
        """Empty metadata should remain empty after save/load."""
        from dataknobs_data.backends.memory import AsyncMemoryDatabase

        db = AsyncMemoryDatabase()
        config = StorageConfig(backend=StorageBackend.MEMORY)
        storage = UnifiedDatabaseStorage(config, database=db)
        await storage.initialize()

        await storage.save_history(_make_history("exec_1"), metadata={})

        results = await storage.query_histories({})
        assert len(results) == 1
        assert results[0]["metadata"] == {}


class TestUnknownFilterWarning:
    """Tests for warning on unknown filter keys."""

    @pytest.mark.asyncio
    async def test_unknown_filter_key_logs_warning(
        self, storage: InMemoryStorage, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Unrecognized filter keys log a warning and are ignored."""
        await storage.initialize()
        await storage.save_history(
            _make_history("exec_1"),
            metadata={"work_order_id": "WO-001"},
        )

        with caplog.at_level(logging.WARNING):
            results = await storage.query_histories(
                {"fsm_name": "test_fsm", "bogus": "value"}
            )

        assert len(results) == 1  # fsm_name filter still works
        assert any("bogus" in record.message for record in caplog.records)

    @pytest.mark.asyncio
    async def test_builtin_and_metadata_filters_no_warning(
        self, storage: InMemoryStorage, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Recognized keys (builtin + metadata.*) produce no warnings."""
        await storage.initialize()
        await storage.save_history(
            _make_history("exec_1"),
            metadata={"work_order_id": "WO-001"},
        )

        with caplog.at_level(logging.WARNING):
            await storage.query_histories(
                {"fsm_name": "test_fsm", "metadata.work_order_id": "WO-001"}
            )

        warning_records = [
            r for r in caplog.records if r.levelno >= logging.WARNING
        ]
        assert len(warning_records) == 0


class TestMetadataFilterNoBackendRestriction:
    """Verify the backend restriction guard was removed.

    These tests use ``AsyncMemoryDatabase`` regardless of config backend type.
    SQL query generation is tested separately in
    ``test_sql_query_builder_dot_notation.py``.  Real SQL backend integration
    tests are in ``TestSQLiteMetadataFilterIntegration`` and
    ``TestPostgresMetadataFilterIntegration`` below.
    """

    @pytest.mark.asyncio
    async def test_metadata_filter_works_with_sqlite_config(self) -> None:
        """No NotImplementedError when config says sqlite (memory backend)."""
        from dataknobs_data.backends.memory import AsyncMemoryDatabase

        config = StorageConfig(backend=StorageBackend.SQLITE)
        db = AsyncMemoryDatabase()
        storage = UnifiedDatabaseStorage(config, database=db)
        await storage.initialize()

        await storage.save_history(
            _make_history("exec_1"),
            metadata={"work_order_id": "WO-001"},
        )

        results = await storage.query_histories(
            {"metadata.work_order_id": "WO-001"}
        )
        assert len(results) == 1
        assert results[0]["id"] == "exec_1"
        assert results[0]["metadata"]["work_order_id"] == "WO-001"

    @pytest.mark.asyncio
    async def test_builtin_and_metadata_filters_combined(self) -> None:
        """Builtin and metadata filters work together (memory backend)."""
        from dataknobs_data.backends.memory import AsyncMemoryDatabase

        config = StorageConfig(backend=StorageBackend.SQLITE)
        db = AsyncMemoryDatabase()
        storage = UnifiedDatabaseStorage(config, database=db)
        await storage.initialize()

        await storage.save_history(
            _make_history("exec_1", fsm_name="alpha"),
            metadata={"work_order_id": "WO-001"},
        )
        await storage.save_history(
            _make_history("exec_2", fsm_name="beta"),
            metadata={"work_order_id": "WO-001"},
        )

        results = await storage.query_histories(
            {"fsm_name": "alpha", "metadata.work_order_id": "WO-001"}
        )
        assert len(results) == 1
        assert results[0]["id"] == "exec_1"


class TestSQLiteMetadataFilterIntegration:
    """End-to-end metadata filtering through a real SQLite backend."""

    @pytest.mark.asyncio
    async def test_metadata_filter_through_sqlite(self, tmp_path: Path) -> None:
        """Save and query with metadata through a real AsyncSQLiteDatabase."""
        from dataknobs_data.backends.sqlite_async import AsyncSQLiteDatabase

        db = AsyncSQLiteDatabase({"path": str(tmp_path / "test.db")})
        await db.connect()
        try:
            config = StorageConfig(backend=StorageBackend.SQLITE)
            storage = UnifiedDatabaseStorage(config, database=db)
            await storage.initialize()

            await storage.save_history(
                _make_history("exec_1"),
                metadata={"tenant_id": "T-1"},
            )
            await storage.save_history(
                _make_history("exec_2"),
                metadata={"tenant_id": "T-2"},
            )

            results = await storage.query_histories(
                {"metadata.tenant_id": "T-1"}
            )
            assert len(results) == 1
            assert results[0]["id"] == "exec_1"
            assert results[0]["metadata"]["tenant_id"] == "T-1"
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_metadata_filter_combined_with_builtin_sqlite(
        self, tmp_path: Path,
    ) -> None:
        """Metadata + builtin filters work through real SQLite."""
        from dataknobs_data.backends.sqlite_async import AsyncSQLiteDatabase

        db = AsyncSQLiteDatabase({"path": str(tmp_path / "test.db")})
        await db.connect()
        try:
            config = StorageConfig(backend=StorageBackend.SQLITE)
            storage = UnifiedDatabaseStorage(config, database=db)
            await storage.initialize()

            await storage.save_history(
                _make_history("exec_1", fsm_name="alpha"),
                metadata={"work_order_id": "WO-001"},
            )
            await storage.save_history(
                _make_history("exec_2", fsm_name="beta"),
                metadata={"work_order_id": "WO-001"},
            )

            results = await storage.query_histories(
                {"fsm_name": "alpha", "metadata.work_order_id": "WO-001"}
            )
            assert len(results) == 1
            assert results[0]["id"] == "exec_1"
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_metadata_no_match_returns_empty_sqlite(
        self, tmp_path: Path,
    ) -> None:
        """Non-matching metadata filter returns empty through real SQLite."""
        from dataknobs_data.backends.sqlite_async import AsyncSQLiteDatabase

        db = AsyncSQLiteDatabase({"path": str(tmp_path / "test.db")})
        await db.connect()
        try:
            config = StorageConfig(backend=StorageBackend.SQLITE)
            storage = UnifiedDatabaseStorage(config, database=db)
            await storage.initialize()

            await storage.save_history(
                _make_history("exec_1"),
                metadata={"tenant_id": "T-1"},
            )

            results = await storage.query_histories(
                {"metadata.tenant_id": "T-999"}
            )
            assert results == []
        finally:
            await db.close()


_skip_no_postgres = pytest.mark.skipif(
    not os.environ.get("TEST_POSTGRES", "").lower() == "true",
    reason="PostgreSQL tests require TEST_POSTGRES=true and a running PostgreSQL instance",
)


@_skip_no_postgres
class TestPostgresMetadataFilterIntegration:
    """End-to-end metadata filtering through a real PostgreSQL backend."""

    @pytest.mark.asyncio
    async def test_metadata_filter_through_postgres(self) -> None:
        """Save and query with metadata through a real AsyncPostgresDatabase."""
        from dataknobs_data.backends.postgres import AsyncPostgresDatabase

        db = AsyncPostgresDatabase({
            "host": os.getenv("POSTGRES_HOST", "localhost"),
            "port": int(os.getenv("POSTGRES_PORT", "5432")),
            "database": os.getenv("POSTGRES_DB", "dataknobs_test"),
            "user": os.getenv("POSTGRES_USER", "postgres"),
            "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
            "table": f"test_histories_{uuid.uuid4().hex[:8]}",
        })
        await db.connect()
        try:
            config = StorageConfig(backend=StorageBackend.POSTGRES)
            storage = UnifiedDatabaseStorage(config, database=db)
            await storage.initialize()

            await storage.save_history(
                _make_history("exec_1"),
                metadata={"tenant_id": "T-1"},
            )
            await storage.save_history(
                _make_history("exec_2"),
                metadata={"tenant_id": "T-2"},
            )

            results = await storage.query_histories(
                {"metadata.tenant_id": "T-1"}
            )
            assert len(results) == 1
            assert results[0]["id"] == "exec_1"
            assert results[0]["metadata"]["tenant_id"] == "T-1"
        finally:
            await db.close()
