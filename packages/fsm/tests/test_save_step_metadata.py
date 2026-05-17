"""Metadata channel routes through ``record.metadata`` at save_step.

These tests exercise the structural prevention contract introduced by
the ``AsyncKeyedRecordStore[_StepRecord]`` migration of
:meth:`UnifiedDatabaseStorage.save_step`:

* Caller-supplied ``metadata`` lands in ``record.metadata``, not the
  data column.
* Defaulting to ``None`` produces an empty metadata column with no
  leakage into the data column.
* Existing save/load behavior round-trips unchanged.
* Metadata is indexable via the ``metadata.X`` dot-notation convention
  on backends that support it (memory backend resolves through
  ``Record.get_nested_value``).

These complement the existing ``save_history`` metadata tests in
``test_query_histories_metadata.py``.
"""

from __future__ import annotations

import time

import pytest

from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_fsm.core.data_modes import DataHandlingMode
from dataknobs_fsm.execution.history import ExecutionStatus, ExecutionStep
from dataknobs_fsm.storage import StorageBackend, StorageConfig
from dataknobs_fsm.storage.database import UnifiedDatabaseStorage


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


def _make_storage() -> tuple[UnifiedDatabaseStorage, AsyncMemoryDatabase]:
    """Build a storage backed by an injected memory database."""
    db = AsyncMemoryDatabase()
    config = StorageConfig(backend=StorageBackend.MEMORY)
    return UnifiedDatabaseStorage(config, database=db), db


class TestSaveStepMetadataRouting:
    """Caller-supplied ``metadata`` reaches the metadata column."""

    @pytest.mark.asyncio
    async def test_metadata_kwarg_routes_to_metadata_column(self) -> None:
        """``save_step(..., metadata=...)`` lands in ``record.metadata``."""
        storage, db = _make_storage()
        await storage.initialize()

        await storage.save_step(
            "exec-1",
            _make_step("step-1"),
            metadata={"tenant_id": "acme", "correlation_id": "corr-42"},
        )

        # Find the step record (data column contains step_data dict).
        all_records = await db.all()
        step_records = [r for r in all_records if r.get_value("step_data")]
        assert len(step_records) == 1
        rec = step_records[0]

        assert rec.metadata == {
            "tenant_id": "acme",
            "correlation_id": "corr-42",
        }
        # No leakage into the data column.
        assert "tenant_id" not in rec.data
        assert "correlation_id" not in rec.data

    @pytest.mark.asyncio
    async def test_default_metadata_is_empty_metadata_column(self) -> None:
        """``save_step`` without ``metadata`` writes ``{}`` to the metadata column."""
        storage, db = _make_storage()
        await storage.initialize()

        await storage.save_step("exec-1", _make_step("step-1"))

        all_records = await db.all()
        step_records = [r for r in all_records if r.get_value("step_data")]
        assert len(step_records) == 1
        rec = step_records[0]

        assert rec.metadata == {}
        # Data column carries structural fields, not metadata-shaped keys.
        assert "tenant_id" not in rec.data
        assert "correlation_id" not in rec.data

    @pytest.mark.asyncio
    async def test_structural_fields_remain_in_data_column(self) -> None:
        """Persisted shape preserves ``execution_id``, ``step_id``, etc. in ``data``."""
        storage, db = _make_storage()
        await storage.initialize()

        step = _make_step("step-42")
        await storage.save_step("exec-1", step, parent_id="parent-7")

        all_records = await db.all()
        step_records = [r for r in all_records if r.get_value("step_data")]
        rec = step_records[0]

        assert rec.data["execution_id"] == "exec-1"
        assert rec.data["step_id"] == "step-42"
        assert rec.data["parent_id"] == "parent-7"
        assert rec.data["state_name"] == "state_a"
        assert rec.data["network_name"] == "main"
        assert rec.data["status"] == ExecutionStatus.COMPLETED.value
        assert rec.data["record_type"] == "step"
        # ``step_data`` carries the full step.to_dict() blob for read-side reconstruction.
        assert rec.data["step_data"]["step_id"] == "step-42"
        assert rec.data["step_data"]["arc_taken"] == "arc_next"


class TestSaveStepMetadataRoundTrip:
    """Existing round-trip behavior is preserved post-migration."""

    @pytest.mark.asyncio
    async def test_load_steps_after_save_with_metadata(self) -> None:
        """``load_steps`` reconstructs the step even when metadata is supplied."""
        storage, _ = _make_storage()
        await storage.initialize()

        await storage.save_step(
            "exec-1",
            _make_step("step-1"),
            metadata={"tenant_id": "acme"},
        )
        await storage.save_step(
            "exec-1",
            _make_step("step-2"),
            metadata={"tenant_id": "globex"},
        )

        steps = await storage.load_steps("exec-1")
        assert {s.step_id for s in steps} == {"step-1", "step-2"}

    @pytest.mark.asyncio
    async def test_load_steps_filter_on_data_column_field(self) -> None:
        """``load_steps`` ``filters`` continues to match on data-column fields."""
        storage, _ = _make_storage()
        await storage.initialize()

        await storage.save_step("exec-1", _make_step("step-1"))
        # Second step with a different state_name
        other = _make_step("step-2")
        other.state_name = "state_b"
        await storage.save_step("exec-1", other)

        matched = await storage.load_steps("exec-1", filters={"state_name": "state_a"})
        assert [s.step_id for s in matched] == ["step-1"]

        none = await storage.load_steps("exec-1", filters={"state_name": "missing"})
        assert none == []


class TestSaveStepMetadataFiltering:
    """``metadata.X`` dot-notation queries reach the metadata column."""

    @pytest.mark.asyncio
    async def test_metadata_dot_notation_filter_via_raw_search(self) -> None:
        """Direct ``Query`` with ``metadata.tenant_id`` reaches the metadata column.

        Verifies the underlying structural contract: metadata written
        through ``save_step`` is filterable via the ``metadata.X``
        field-path convention that backends already honor.  The
        ``filter_metadata=`` kwarg on ``load_steps`` lets
        production callers avoid dropping to the raw database
        surface; the lower-level test stays here as a regression guard
        on the persisted shape.
        """
        from dataknobs_data.query import Query

        storage, db = _make_storage()
        await storage.initialize()

        await storage.save_step("exec-1", _make_step("step-1"), metadata={"tenant_id": "acme"})
        await storage.save_step("exec-1", _make_step("step-2"), metadata={"tenant_id": "globex"})

        # Memory backend honors metadata.X via Record.get_nested_value.
        acme = await db.search(
            Query().filter("step_data", "exists").filter("metadata.tenant_id", "=", "acme")
        )
        assert len(acme) == 1
        assert acme[0].data["step_id"] == "step-1"
        assert acme[0].metadata == {"tenant_id": "acme"}


class TestLoadStepsFilterSymmetry:
    """``load_steps`` filter/pagination kwargs mirror the registry layer."""

    @pytest.mark.asyncio
    async def test_filter_metadata_routes_to_metadata_column(self) -> None:
        """``load_steps(filter_metadata=...)`` filters by the metadata column."""
        storage, _ = _make_storage()
        await storage.initialize()

        await storage.save_step(
            "exec-1", _make_step("step-1"), metadata={"tenant_id": "acme"}
        )
        await storage.save_step(
            "exec-1", _make_step("step-2"), metadata={"tenant_id": "globex"}
        )

        matched = await storage.load_steps(
            "exec-1", filter_metadata={"tenant_id": "acme"}
        )
        assert [s.step_id for s in matched] == ["step-1"]

        none = await storage.load_steps(
            "exec-1", filter_metadata={"tenant_id": "missing"}
        )
        assert none == []

    @pytest.mark.asyncio
    async def test_filter_metadata_ands_with_data_filters(self) -> None:
        """``filters`` (data columns) and ``filter_metadata`` AND-combine."""
        storage, _ = _make_storage()
        await storage.initialize()

        a = _make_step("step-1")
        a.state_name = "state_a"
        await storage.save_step("exec-1", a, metadata={"tenant_id": "acme"})

        b = _make_step("step-2")
        b.state_name = "state_b"
        await storage.save_step("exec-1", b, metadata={"tenant_id": "acme"})

        c = _make_step("step-3")
        c.state_name = "state_a"
        await storage.save_step("exec-1", c, metadata={"tenant_id": "globex"})

        matched = await storage.load_steps(
            "exec-1",
            filters={"state_name": "state_a"},
            filter_metadata={"tenant_id": "acme"},
        )
        assert [s.step_id for s in matched] == ["step-1"]

    @pytest.mark.asyncio
    async def test_limit_and_offset_pushdown(self) -> None:
        """``limit`` / ``offset`` paginate the result set."""
        storage, _ = _make_storage()
        await storage.initialize()

        for i in range(5):
            step = _make_step(f"step-{i}")
            step.timestamp = 1_000_000.0 + i
            await storage.save_step("exec-1", step)

        first_two = await storage.load_steps("exec-1", limit=2)
        assert len(first_two) == 2

        next_two = await storage.load_steps("exec-1", limit=2, offset=2)
        assert len(next_two) == 2

        # limit=0 honors Python-slice semantics (empty result), matching
        # the dataknobs-data falsy-check fix.
        empty = await storage.load_steps("exec-1", limit=0)
        assert empty == []

    @pytest.mark.asyncio
    async def test_sort_pushdown(self) -> None:
        """``sort=`` controls ordering of returned steps."""
        from dataknobs_data import SortOrder, SortSpec

        storage, _ = _make_storage()
        await storage.initialize()

        for i, ts in enumerate([3.0, 1.0, 2.0]):
            step = _make_step(f"step-{i}")
            step.timestamp = ts
            await storage.save_step("exec-1", step)

        asc = await storage.load_steps(
            "exec-1", sort=[SortSpec(field="timestamp", order=SortOrder.ASC)]
        )
        assert [s.timestamp for s in asc] == [1.0, 2.0, 3.0]

        desc = await storage.load_steps(
            "exec-1", sort=[SortSpec(field="timestamp", order=SortOrder.DESC)]
        )
        assert [s.timestamp for s in desc] == [3.0, 2.0, 1.0]
