"""Tests for metadata filtering in query_histories()."""

from __future__ import annotations

import logging
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


class TestAllBackendsMetadataFilter:
    """Tests that metadata filtering works on all backend types.

    SQL backends now handle dot-notation natively via JSONB/JSON path
    extraction in SQLQueryBuilder, so no backend restriction exists.
    """

    @pytest.mark.asyncio
    async def test_metadata_filter_works_on_sqlite_backend(self) -> None:
        """Metadata filtering works even when config says sqlite.

        The memory database's search() supports dot-notation via
        Record.get_value(), and SQL backends now generate correct
        nested JSON path queries. Either way, no NotImplementedError.
        """
        from dataknobs_data.backends.memory import AsyncMemoryDatabase

        config = StorageConfig(backend=StorageBackend.SQLITE)
        db = AsyncMemoryDatabase()  # real db, config says sqlite
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
        """Builtin and metadata filters work together on any backend."""
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
