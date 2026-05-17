"""Tests for rubric registry."""

from __future__ import annotations

from typing import Any

from dataknobs_data import Record, SortOrder, SortSpec
from dataknobs_data.backends.memory import AsyncMemoryDatabase

from dataknobs_bots.rubrics.models import (
    Rubric,
    RubricCriterion,
    RubricLevel,
    ScoringMethod,
    ScoringType,
)
from dataknobs_bots.rubrics.registry import RubricRegistry


def _make_rubric(
    rubric_id: str = "rubric_001",
    version: str = "1.0.0",
    target_type: str = "content",
) -> Rubric:
    return Rubric(
        id=rubric_id,
        name=f"Rubric {rubric_id}",
        description="A test rubric",
        version=version,
        target_type=target_type,
        criteria=[
            RubricCriterion(
                id="c1",
                name="Criterion 1",
                description="Test criterion",
                weight=1.0,
                levels=[
                    RubricLevel(
                        id="fail", label="Fail", description="Fail", score=0.0
                    ),
                    RubricLevel(
                        id="pass", label="Pass", description="Pass", score=1.0
                    ),
                ],
                scoring_method=ScoringMethod(
                    type=ScoringType.DETERMINISTIC,
                    function_ref="test:func",
                ),
            ),
        ],
        pass_threshold=0.7,
        metadata={"author": "test"},
    )


class TestRubricRegistry:
    async def test_register_and_get(self) -> None:
        db = AsyncMemoryDatabase()
        registry = RubricRegistry(db)
        rubric = _make_rubric()

        rubric_id = await registry.register(rubric)

        assert rubric_id == "rubric_001"
        retrieved = await registry.get("rubric_001")
        assert retrieved is not None
        assert retrieved.id == "rubric_001"
        assert retrieved.name == "Rubric rubric_001"
        assert retrieved.version == "1.0.0"

    async def test_get_nonexistent(self) -> None:
        db = AsyncMemoryDatabase()
        registry = RubricRegistry(db)

        result = await registry.get("nonexistent")

        assert result is None

    async def test_version_management(self) -> None:
        db = AsyncMemoryDatabase()
        registry = RubricRegistry(db)

        v1 = _make_rubric(version="1.0.0")
        v2 = _make_rubric(version="2.0.0")
        v2.name = "Updated Rubric"

        await registry.register(v1)
        await registry.update(v2)

        # Latest should be v2
        latest = await registry.get("rubric_001")
        assert latest is not None
        assert latest.version == "2.0.0"
        assert latest.name == "Updated Rubric"

        # v1 should still be accessible
        old = await registry.get("rubric_001", version="1.0.0")
        assert old is not None
        assert old.version == "1.0.0"
        assert old.name == "Rubric rubric_001"

        # v2 also accessible by version
        specific = await registry.get("rubric_001", version="2.0.0")
        assert specific is not None
        assert specific.version == "2.0.0"

    async def test_get_for_target(self) -> None:
        db = AsyncMemoryDatabase()
        registry = RubricRegistry(db)

        await registry.register(_make_rubric("r1", target_type="content"))
        await registry.register(_make_rubric("r2", target_type="content"))
        await registry.register(_make_rubric("r3", target_type="rubric"))

        content_rubrics = await registry.get_for_target("content")
        rubric_rubrics = await registry.get_for_target("rubric")

        content_ids = {r.id for r in content_rubrics}
        assert "r1" in content_ids
        assert "r2" in content_ids
        assert "r3" not in content_ids

        rubric_ids = {r.id for r in rubric_rubrics}
        assert "r3" in rubric_ids

    async def test_get_for_target_no_matches(self) -> None:
        db = AsyncMemoryDatabase()
        registry = RubricRegistry(db)

        await registry.register(_make_rubric("r1", target_type="content"))

        result = await registry.get_for_target("nonexistent_type")

        assert result == []

    async def test_get_for_target_returns_latest_versions(self) -> None:
        """Regression: get_for_target previously deduped by id alone and
        could surface a stale versioned-snapshot row if it preceded the
        latest pointer in scan order.  After the iter_latest_records
        refactor, snapshots are dropped before dedup, so the latest
        pointer always wins.
        """
        db = AsyncMemoryDatabase()
        registry = RubricRegistry(db)

        v1 = _make_rubric("r1", version="1.0.0", target_type="content")
        v1.name = "Original Name"
        await registry.register(v1)

        v2 = _make_rubric("r1", version="2.0.0", target_type="content")
        v2.name = "Updated Name"
        await registry.update(v2)

        matches = await registry.get_for_target("content")

        r1_entries = [r for r in matches if r.id == "r1"]
        assert len(r1_entries) == 1
        assert r1_entries[0].version == "2.0.0"
        assert r1_entries[0].name == "Updated Name"

    async def test_delete(self) -> None:
        db = AsyncMemoryDatabase()
        registry = RubricRegistry(db)

        await registry.register(_make_rubric())

        deleted = await registry.delete("rubric_001")
        assert deleted is True

        result = await registry.get("rubric_001")
        assert result is None

    async def test_delete_nonexistent(self) -> None:
        db = AsyncMemoryDatabase()
        registry = RubricRegistry(db)

        deleted = await registry.delete("nonexistent")
        assert deleted is False

    async def test_list_all(self) -> None:
        db = AsyncMemoryDatabase()
        registry = RubricRegistry(db)

        await registry.register(_make_rubric("r1"))
        await registry.register(_make_rubric("r2"))
        await registry.register(_make_rubric("r3"))

        all_rubrics = await registry.list_all()

        ids = {r.id for r in all_rubrics}
        assert ids == {"r1", "r2", "r3"}

    async def test_list_all_returns_latest_versions(self) -> None:
        db = AsyncMemoryDatabase()
        registry = RubricRegistry(db)

        await registry.register(_make_rubric("r1", version="1.0.0"))
        v2 = _make_rubric("r1", version="2.0.0")
        v2.name = "Updated"
        await registry.update(v2)

        all_rubrics = await registry.list_all()

        r1_entries = [r for r in all_rubrics if r.id == "r1"]
        assert len(r1_entries) == 1
        assert r1_entries[0].version == "2.0.0"

    async def test_from_config(self) -> None:
        db = AsyncMemoryDatabase()
        rubric_data = _make_rubric("cfg_rubric").to_dict()
        config = {"rubrics": [rubric_data]}

        registry = await RubricRegistry.from_config(config, db)

        result = await registry.get("cfg_rubric")
        assert result is not None
        assert result.id == "cfg_rubric"

    async def test_from_config_empty(self) -> None:
        db = AsyncMemoryDatabase()

        registry = await RubricRegistry.from_config({}, db)

        all_rubrics = await registry.list_all()
        assert all_rubrics == []

    async def test_serialization_integrity(self) -> None:
        """Verify that storing and retrieving preserves all rubric data."""
        db = AsyncMemoryDatabase()
        registry = RubricRegistry(db)

        original = _make_rubric()
        await registry.register(original)
        retrieved = await registry.get("rubric_001")

        assert retrieved is not None
        assert retrieved.to_dict() == original.to_dict()


class TestRubricRegistryMetadata:
    """Metadata channel routes through ``record.metadata``."""

    async def test_metadata_round_trips_through_metadata_column(self) -> None:
        """``Rubric.metadata`` is stored in and read from the metadata column."""
        db = AsyncMemoryDatabase()
        registry = RubricRegistry(db)

        rubric = _make_rubric()
        rubric.metadata = {"tenant_id": "acme", "author": "alice"}
        await registry.register(rubric)

        # Sanity: the raw record has metadata in the metadata column,
        # not duplicated in the data column.
        raw = await db.read("rubric_001")
        assert raw is not None
        assert raw.metadata == {"tenant_id": "acme", "author": "alice"}
        assert "metadata" not in raw.data

        # Round-trip through the typed surface preserves it.
        retrieved = await registry.get("rubric_001")
        assert retrieved is not None
        assert retrieved.metadata == {"tenant_id": "acme", "author": "alice"}

    async def test_metadata_default_empty(self) -> None:
        """Rubrics with no metadata round-trip to an empty dict."""
        db = AsyncMemoryDatabase()
        registry = RubricRegistry(db)

        rubric = _make_rubric()
        rubric.metadata = {}
        await registry.register(rubric)

        retrieved = await registry.get("rubric_001")
        assert retrieved is not None
        assert retrieved.metadata == {}

    async def test_versioned_snapshot_preserves_metadata(self) -> None:
        """Both the latest pointer and versioned snapshot carry metadata."""
        db = AsyncMemoryDatabase()
        registry = RubricRegistry(db)

        v1 = _make_rubric(version="1.0.0")
        v1.metadata = {"tenant_id": "acme", "version_note": "initial"}
        await registry.register(v1)

        v2 = _make_rubric(version="2.0.0")
        v2.metadata = {"tenant_id": "acme", "version_note": "updated"}
        await registry.update(v2)

        latest = await registry.get("rubric_001")
        assert latest is not None
        assert latest.metadata == {
            "tenant_id": "acme",
            "version_note": "updated",
        }

        snapshot_v1 = await registry.get("rubric_001", version="1.0.0")
        assert snapshot_v1 is not None
        assert snapshot_v1.metadata == {
            "tenant_id": "acme",
            "version_note": "initial",
        }

    async def test_get_for_target_filter_metadata(self) -> None:
        """``filter_metadata`` AND-combines with target_type."""
        db = AsyncMemoryDatabase()
        registry = RubricRegistry(db)

        r1 = _make_rubric("r1", target_type="content")
        r1.metadata = {"tenant_id": "acme"}
        r2 = _make_rubric("r2", target_type="content")
        r2.metadata = {"tenant_id": "globex"}
        r3 = _make_rubric("r3", target_type="content")
        r3.metadata = {"tenant_id": "acme"}

        await registry.register(r1)
        await registry.register(r2)
        await registry.register(r3)

        acme_content = await registry.get_for_target(
            "content", filter_metadata={"tenant_id": "acme"}
        )
        ids = {r.id for r in acme_content}
        assert ids == {"r1", "r3"}

    async def test_get_for_target_empty_filter_metadata_is_no_filter(
        self,
    ) -> None:
        """``filter_metadata={}`` ≡ ``filter_metadata=None``."""
        db = AsyncMemoryDatabase()
        registry = RubricRegistry(db)

        r1 = _make_rubric("r1", target_type="content")
        r1.metadata = {"tenant_id": "acme"}
        r2 = _make_rubric("r2", target_type="content")
        await registry.register(r1)
        await registry.register(r2)

        with_empty = await registry.get_for_target("content", filter_metadata={})
        with_none = await registry.get_for_target("content")
        assert {r.id for r in with_empty} == {r.id for r in with_none}

    async def test_list_all_filter_metadata(self) -> None:
        """``list_all`` honors ``filter_metadata`` and de-duplicates by id."""
        db = AsyncMemoryDatabase()
        registry = RubricRegistry(db)

        r1 = _make_rubric("r1")
        r1.metadata = {"tenant_id": "acme"}
        r2 = _make_rubric("r2")
        r2.metadata = {"tenant_id": "globex"}
        await registry.register(r1)
        await registry.register(r2)

        acme = await registry.list_all(filter_metadata={"tenant_id": "acme"})
        assert {r.id for r in acme} == {"r1"}

    async def test_legacy_data_with_metadata_in_data_column(self) -> None:
        """Pre-migration records (metadata in data column) are still readable.

        Simulates a record written by the old implementation where the
        entire payload — including the model's ``metadata`` field —
        was routed into the data column via ``Record(data=...)``.
        The deserializer's legacy fallback must surface it correctly.
        """
        db = AsyncMemoryDatabase()
        registry = RubricRegistry(db)

        # Build a payload matching the pre-migration shape.
        legacy_rubric = _make_rubric()
        legacy_rubric.metadata = {"tenant_id": "legacy", "author": "old"}
        legacy_data: dict[str, Any] = legacy_rubric.to_dict()
        legacy_data["_version_key"] = (
            f"{legacy_rubric.id}:{legacy_rubric.version}"
        )
        # Old code: Record(data) — metadata column is empty.
        await db.upsert(legacy_rubric.id, Record(data=legacy_data))

        retrieved = await registry.get(legacy_rubric.id)
        assert retrieved is not None
        assert retrieved.metadata == {
            "tenant_id": "legacy",
            "author": "old",
        }


class TestRubricRegistryPagination:
    """``get_for_target``/``list_all`` sort/limit/offset and counts.

    Pins the post-dedup pagination contract: ``sort`` pushes to the
    database but ``limit``/``offset`` apply to the deduplicated rubric
    list — never to the pre-dedup row stream — because the dual-write
    storage shape (latest pointer + versioned snapshots) means N rows
    in the database can collapse to anywhere between ``ceil(N/2)`` and
    ``1`` rubric.
    """

    async def test_get_for_target_sort_by_name_asc(self) -> None:
        """Sort by name ascending returns rubrics in name order."""
        db = AsyncMemoryDatabase()
        registry = RubricRegistry(db)

        for rid in ("rC", "rA", "rB"):
            r = _make_rubric(rid, target_type="content")
            r.name = rid
            await registry.register(r)

        results = await registry.get_for_target(
            "content",
            sort=[SortSpec(field="name", order=SortOrder.ASC)],
        )
        assert [r.name for r in results] == ["rA", "rB", "rC"]

    async def test_get_for_target_sort_by_name_desc(self) -> None:
        db = AsyncMemoryDatabase()
        registry = RubricRegistry(db)

        for rid in ("rC", "rA", "rB"):
            r = _make_rubric(rid, target_type="content")
            r.name = rid
            await registry.register(r)

        results = await registry.get_for_target(
            "content",
            sort=[SortSpec(field="name", order=SortOrder.DESC)],
        )
        assert [r.name for r in results] == ["rC", "rB", "rA"]

    async def test_get_for_target_limit_applies_after_dedup(self) -> None:
        """``limit`` returns N deduplicated rubrics, not N database rows.

        Each register writes both a latest pointer and a versioned
        snapshot, so 3 rubrics produce 6 records.  ``limit=2`` must
        return 2 rubrics (not, for example, 2 records that may all be
        snapshots).
        """
        db = AsyncMemoryDatabase()
        registry = RubricRegistry(db)

        for rid in ("rA", "rB", "rC"):
            r = _make_rubric(rid, target_type="content")
            r.name = rid
            await registry.register(r)

        results = await registry.get_for_target(
            "content",
            sort=[SortSpec(field="name", order=SortOrder.ASC)],
            limit=2,
        )
        assert [r.name for r in results] == ["rA", "rB"]

    async def test_get_for_target_offset_applies_after_dedup(self) -> None:
        db = AsyncMemoryDatabase()
        registry = RubricRegistry(db)

        for rid in ("rA", "rB", "rC"):
            r = _make_rubric(rid, target_type="content")
            r.name = rid
            await registry.register(r)

        results = await registry.get_for_target(
            "content",
            sort=[SortSpec(field="name", order=SortOrder.ASC)],
            offset=1,
        )
        assert [r.name for r in results] == ["rB", "rC"]

    async def test_get_for_target_limit_and_offset_combine(self) -> None:
        """``offset`` first, then ``limit`` — standard pagination semantics."""
        db = AsyncMemoryDatabase()
        registry = RubricRegistry(db)

        for rid in ("rA", "rB", "rC", "rD", "rE"):
            r = _make_rubric(rid, target_type="content")
            r.name = rid
            await registry.register(r)

        page = await registry.get_for_target(
            "content",
            sort=[SortSpec(field="name", order=SortOrder.ASC)],
            offset=1,
            limit=2,
        )
        assert [r.name for r in page] == ["rB", "rC"]

    async def test_get_for_target_limit_with_versioned_rubrics(self) -> None:
        """Pre-existing versions don't leak through ``limit``.

        After two updates on ``rA``, the database has 4 rows for ``rA``
        (pointer + 3 snapshots).  A query with ``limit=2`` must still
        deduplicate first, then take 2 rubrics — not 2 rows — and ``rA``
        must surface its latest version.
        """
        db = AsyncMemoryDatabase()
        registry = RubricRegistry(db)

        v1 = _make_rubric("rA", version="1.0.0", target_type="content")
        v1.name = "A1"
        await registry.register(v1)

        v2 = _make_rubric("rA", version="2.0.0", target_type="content")
        v2.name = "A2"
        await registry.update(v2)

        v3 = _make_rubric("rA", version="3.0.0", target_type="content")
        v3.name = "A3"
        await registry.update(v3)

        rb = _make_rubric("rB", target_type="content")
        rb.name = "B1"
        await registry.register(rb)

        results = await registry.get_for_target(
            "content",
            sort=[SortSpec(field="name", order=SortOrder.ASC)],
            limit=2,
        )
        assert len(results) == 2
        assert {r.id for r in results} == {"rA", "rB"}
        latest_a = next(r for r in results if r.id == "rA")
        assert latest_a.version == "3.0.0"
        assert latest_a.name == "A3"

    async def test_get_for_target_sort_combines_with_filter_metadata(
        self,
    ) -> None:
        """Sort works alongside ``filter_metadata``."""
        db = AsyncMemoryDatabase()
        registry = RubricRegistry(db)

        for rid in ("rC", "rA", "rB", "rD"):
            r = _make_rubric(rid, target_type="content")
            r.name = rid
            r.metadata = {"tenant_id": "acme" if rid != "rD" else "globex"}
            await registry.register(r)

        results = await registry.get_for_target(
            "content",
            filter_metadata={"tenant_id": "acme"},
            sort=[SortSpec(field="name", order=SortOrder.ASC)],
        )
        assert [r.name for r in results] == ["rA", "rB", "rC"]

    async def test_get_for_target_limit_zero_returns_empty(self) -> None:
        db = AsyncMemoryDatabase()
        registry = RubricRegistry(db)
        await registry.register(_make_rubric("rA", target_type="content"))

        results = await registry.get_for_target("content", limit=0)
        assert results == []

    async def test_get_for_target_offset_beyond_count_returns_empty(
        self,
    ) -> None:
        db = AsyncMemoryDatabase()
        registry = RubricRegistry(db)
        await registry.register(_make_rubric("rA", target_type="content"))

        results = await registry.get_for_target("content", offset=10)
        assert results == []

    async def test_list_all_sort_limit_offset(self) -> None:
        """``list_all`` honors sort/limit/offset on the full rubric set."""
        db = AsyncMemoryDatabase()
        registry = RubricRegistry(db)

        for rid in ("rC", "rA", "rB", "rD", "rE"):
            r = _make_rubric(rid)
            r.name = rid
            await registry.register(r)

        page = await registry.list_all(
            sort=[SortSpec(field="name", order=SortOrder.ASC)],
            offset=1,
            limit=2,
        )
        assert [r.name for r in page] == ["rB", "rC"]

    async def test_list_all_limit_with_versioned_rubrics(self) -> None:
        """``list_all`` deduplicates versioned rows before applying limit."""
        db = AsyncMemoryDatabase()
        registry = RubricRegistry(db)

        v1 = _make_rubric("rA", version="1.0.0")
        v1.name = "A1"
        await registry.register(v1)
        v2 = _make_rubric("rA", version="2.0.0")
        v2.name = "A2"
        await registry.update(v2)
        rb = _make_rubric("rB")
        rb.name = "B1"
        await registry.register(rb)

        results = await registry.list_all(
            sort=[SortSpec(field="name", order=SortOrder.ASC)],
            limit=2,
        )
        assert len(results) == 2
        ids = {r.id for r in results}
        assert ids == {"rA", "rB"}
        latest_a = next(r for r in results if r.id == "rA")
        assert latest_a.version == "2.0.0"

    async def test_count_for_target_empty_returns_zero(self) -> None:
        db = AsyncMemoryDatabase()
        registry = RubricRegistry(db)
        assert await registry.count_for_target("content") == 0

    async def test_count_for_target_returns_distinct_rubric_count(self) -> None:
        db = AsyncMemoryDatabase()
        registry = RubricRegistry(db)

        await registry.register(_make_rubric("rA", target_type="content"))
        await registry.register(_make_rubric("rB", target_type="content"))
        await registry.register(_make_rubric("rC", target_type="rubric"))

        assert await registry.count_for_target("content") == 2
        assert await registry.count_for_target("rubric") == 1
        assert await registry.count_for_target("nonexistent") == 0

    async def test_count_for_target_with_versioned_rubrics_dedups(
        self,
    ) -> None:
        """Versioned snapshots are deduplicated in the count.

        After one update, rubric ``rA`` has 3 raw rows (pointer + 2
        snapshots).  ``count_for_target`` must return 1 for the single
        distinct rubric, not 3.
        """
        db = AsyncMemoryDatabase()
        registry = RubricRegistry(db)

        v1 = _make_rubric("rA", version="1.0.0", target_type="content")
        await registry.register(v1)
        v2 = _make_rubric("rA", version="2.0.0", target_type="content")
        await registry.update(v2)

        assert await registry.count_for_target("content") == 1

    async def test_count_for_target_with_filter_metadata(self) -> None:
        db = AsyncMemoryDatabase()
        registry = RubricRegistry(db)

        r1 = _make_rubric("r1", target_type="content")
        r1.metadata = {"tenant_id": "acme"}
        r2 = _make_rubric("r2", target_type="content")
        r2.metadata = {"tenant_id": "acme"}
        r3 = _make_rubric("r3", target_type="content")
        r3.metadata = {"tenant_id": "globex"}

        await registry.register(r1)
        await registry.register(r2)
        await registry.register(r3)

        assert (
            await registry.count_for_target(
                "content", filter_metadata={"tenant_id": "acme"}
            )
            == 2
        )
        assert (
            await registry.count_for_target(
                "content", filter_metadata={"tenant_id": "globex"}
            )
            == 1
        )
        assert (
            await registry.count_for_target(
                "content", filter_metadata={"tenant_id": "none"}
            )
            == 0
        )

    async def test_count_for_target_matches_get_for_target_length(
        self,
    ) -> None:
        """count_for_target(...) and len(get_for_target(...)) agree."""
        db = AsyncMemoryDatabase()
        registry = RubricRegistry(db)

        for rid in ("r1", "r2", "r3", "r4"):
            r = _make_rubric(rid, target_type="content")
            r.metadata = {"tenant_id": "acme"}
            await registry.register(r)
        r5 = _make_rubric("r5", target_type="content")
        r5.metadata = {"tenant_id": "globex"}
        await registry.register(r5)

        kwargs: dict[str, object] = {
            "filter_metadata": {"tenant_id": "acme"},
        }
        results = await registry.get_for_target("content", **kwargs)  # type: ignore[arg-type]
        count = await registry.count_for_target("content", **kwargs)  # type: ignore[arg-type]
        assert count == len(results) == 4

    async def test_count_all_empty_returns_zero(self) -> None:
        db = AsyncMemoryDatabase()
        registry = RubricRegistry(db)
        assert await registry.count_all() == 0

    async def test_count_all_returns_distinct_rubric_count(self) -> None:
        db = AsyncMemoryDatabase()
        registry = RubricRegistry(db)

        await registry.register(_make_rubric("rA"))
        await registry.register(_make_rubric("rB"))
        await registry.register(_make_rubric("rC"))

        assert await registry.count_all() == 3

    async def test_count_all_with_versioned_rubrics_dedups(self) -> None:
        db = AsyncMemoryDatabase()
        registry = RubricRegistry(db)

        v1 = _make_rubric("rA", version="1.0.0")
        await registry.register(v1)
        v2 = _make_rubric("rA", version="2.0.0")
        await registry.update(v2)

        assert await registry.count_all() == 1

    async def test_count_all_with_filter_metadata(self) -> None:
        db = AsyncMemoryDatabase()
        registry = RubricRegistry(db)

        r1 = _make_rubric("r1")
        r1.metadata = {"tenant_id": "acme"}
        r2 = _make_rubric("r2")
        r2.metadata = {"tenant_id": "globex"}
        await registry.register(r1)
        await registry.register(r2)

        assert (
            await registry.count_all(filter_metadata={"tenant_id": "acme"})
            == 1
        )
        assert (
            await registry.count_all(filter_metadata={"tenant_id": "globex"})
            == 1
        )

    async def test_count_all_matches_list_all_length(self) -> None:
        """count_all(...) and len(list_all(...)) agree for the same shape."""
        db = AsyncMemoryDatabase()
        registry = RubricRegistry(db)

        for rid in ("r1", "r2", "r3"):
            r = _make_rubric(rid)
            r.metadata = {"tenant_id": "acme"}
            await registry.register(r)
        other = _make_rubric("r4")
        other.metadata = {"tenant_id": "globex"}
        await registry.register(other)

        kwargs: dict[str, object] = {
            "filter_metadata": {"tenant_id": "acme"},
        }
        results = await registry.list_all(**kwargs)  # type: ignore[arg-type]
        count = await registry.count_all(**kwargs)  # type: ignore[arg-type]
        assert count == len(results) == 3
