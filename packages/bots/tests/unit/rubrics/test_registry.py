"""Tests for rubric registry."""

from __future__ import annotations

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
