"""Tests for artifact tools (async registry API)."""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_data.backends.memory import AsyncMemoryDatabase

from dataknobs_bots.artifacts.models import ArtifactStatus
from dataknobs_bots.artifacts.provenance import create_provenance
from dataknobs_bots.artifacts.registry import ArtifactRegistry
from dataknobs_bots.artifacts.tools import (
    CreateArtifactTool,
    GetArtifactTool,
    QueryArtifactsTool,
    SubmitForReviewTool,
    UpdateArtifactTool,
)
from dataknobs_llm.tools.context import ToolExecutionContext


# --- CreateArtifactTool Tests ---


class TestCreateArtifactTool:
    async def test_schema_has_required_fields(self) -> None:
        db = AsyncMemoryDatabase()
        registry = ArtifactRegistry(db)
        tool = CreateArtifactTool(artifact_registry=registry)

        assert tool.name == "create_artifact"
        assert "content" in tool.schema["required"]
        assert "name" in tool.schema["required"]

    async def test_creates_artifact(self) -> None:
        db = AsyncMemoryDatabase()
        registry = ArtifactRegistry(db)
        tool = CreateArtifactTool(artifact_registry=registry)
        context = ToolExecutionContext.empty()

        result = await tool.execute_with_context(
            context,
            content={"questions": ["Q1", "Q2"]},
            name="Test Questions",
        )

        assert "artifact_id" in result
        assert result["status"] == "draft"
        assert result["name"] == "Test Questions"

        artifact = await registry.get(result["artifact_id"])
        assert artifact is not None
        assert artifact.content["questions"] == ["Q1", "Q2"]

    async def test_creates_with_type_and_tags(self) -> None:
        db = AsyncMemoryDatabase()
        registry = ArtifactRegistry(db)
        tool = CreateArtifactTool(artifact_registry=registry)
        context = ToolExecutionContext.empty()

        result = await tool.execute_with_context(
            context,
            content={"data": "value"},
            name="Config",
            artifact_type="config",
            tags=["system"],
        )

        artifact = await registry.get(result["artifact_id"])
        assert artifact is not None
        assert artifact.type == "config"
        assert "system" in artifact.tags

    async def test_includes_user_provenance(self) -> None:
        db = AsyncMemoryDatabase()
        registry = ArtifactRegistry(db)
        tool = CreateArtifactTool(artifact_registry=registry)
        context = ToolExecutionContext(user_id="alice")

        result = await tool.execute_with_context(
            context,
            content={"data": "value"},
            name="User Artifact",
        )

        artifact = await registry.get(result["artifact_id"])
        assert artifact is not None
        assert artifact.provenance.created_by == "user:alice"


# --- UpdateArtifactTool Tests ---


class TestUpdateArtifactTool:
    async def test_schema_has_required_fields(self) -> None:
        db = AsyncMemoryDatabase()
        registry = ArtifactRegistry(db)
        tool = UpdateArtifactTool(artifact_registry=registry)

        assert tool.name == "update_artifact"
        assert "artifact_id" in tool.schema["required"]
        assert "content" in tool.schema["required"]

    async def test_updates_artifact(self) -> None:
        db = AsyncMemoryDatabase()
        registry = ArtifactRegistry(db)
        tool = UpdateArtifactTool(artifact_registry=registry)
        context = ToolExecutionContext.empty()

        # Create initial artifact
        artifact = await registry.create(
            artifact_type="content",
            name="Original",
            content={"v": 1},
        )

        result = await tool.execute_with_context(
            context,
            artifact_id=artifact.id,
            content={"v": 2},
            reason="Updated content",
        )

        assert result["artifact_id"] == artifact.id
        assert result["status"] == "draft"

        updated = await registry.get(artifact.id)
        assert updated is not None
        assert updated.content == {"v": 2}
        assert updated.version != "1.0.0"

    async def test_update_not_found(self) -> None:
        db = AsyncMemoryDatabase()
        registry = ArtifactRegistry(db)
        tool = UpdateArtifactTool(artifact_registry=registry)
        context = ToolExecutionContext.empty()

        result = await tool.execute_with_context(
            context,
            artifact_id="nonexistent",
            content={"v": 1},
        )

        assert "error" in result


# --- QueryArtifactsTool Tests ---


class TestQueryArtifactsTool:
    async def test_schema(self) -> None:
        db = AsyncMemoryDatabase()
        registry = ArtifactRegistry(db)
        tool = QueryArtifactsTool(artifact_registry=registry)

        assert tool.name == "query_artifacts"
        assert "status" in tool.schema["properties"]

    async def test_query_all(self) -> None:
        db = AsyncMemoryDatabase()
        registry = ArtifactRegistry(db)
        await registry.create(artifact_type="content", name="A", content={"a": 1})
        await registry.create(artifact_type="content", name="B", content={"b": 2})

        tool = QueryArtifactsTool(artifact_registry=registry)
        context = ToolExecutionContext.empty()

        result = await tool.execute_with_context(context)

        assert result["count"] == 2
        assert len(result["artifacts"]) == 2

    async def test_query_by_type(self) -> None:
        db = AsyncMemoryDatabase()
        registry = ArtifactRegistry(db)
        await registry.create(artifact_type="quiz", name="Q", content={"q": 1})
        await registry.create(artifact_type="config", name="C", content={"c": 1})

        tool = QueryArtifactsTool(artifact_registry=registry)
        context = ToolExecutionContext.empty()

        result = await tool.execute_with_context(
            context, artifact_type="quiz"
        )

        assert result["count"] == 1
        assert result["artifacts"][0]["type"] == "quiz"

    async def test_query_by_tags(self) -> None:
        db = AsyncMemoryDatabase()
        registry = ArtifactRegistry(db)
        await registry.create(
            artifact_type="content", name="Tagged", content={"x": 1},
            tags=["math"],
        )
        await registry.create(
            artifact_type="content", name="Untagged", content={"x": 2},
        )

        tool = QueryArtifactsTool(artifact_registry=registry)
        context = ToolExecutionContext.empty()

        result = await tool.execute_with_context(context, tags=["math"])

        assert result["count"] == 1
        assert result["artifacts"][0]["name"] == "Tagged"

    async def test_query_include_content(self) -> None:
        db = AsyncMemoryDatabase()
        registry = ArtifactRegistry(db)
        await registry.create(
            artifact_type="content", name="A", content={"data": "value"}
        )

        tool = QueryArtifactsTool(artifact_registry=registry)
        context = ToolExecutionContext.empty()

        result = await tool.execute_with_context(context, include_content=True)

        assert "content" in result["artifacts"][0]
        assert result["artifacts"][0]["content"]["data"] == "value"


# --- SubmitForReviewTool Tests ---


class TestSubmitForReviewTool:
    async def test_schema(self) -> None:
        db = AsyncMemoryDatabase()
        registry = ArtifactRegistry(db)
        tool = SubmitForReviewTool(artifact_registry=registry)

        assert tool.name == "submit_for_review"
        assert "artifact_id" in tool.schema["required"]

    async def test_submits_artifact(self) -> None:
        db = AsyncMemoryDatabase()
        registry = ArtifactRegistry(db)
        artifact = await registry.create(
            artifact_type="content", name="Test", content={"v": 1}
        )

        tool = SubmitForReviewTool(artifact_registry=registry)
        context = ToolExecutionContext.empty()

        result = await tool.execute_with_context(
            context, artifact_id=artifact.id
        )

        assert result["artifact_id"] == artifact.id
        assert "evaluations" in result

    async def test_submit_not_found(self) -> None:
        db = AsyncMemoryDatabase()
        registry = ArtifactRegistry(db)
        tool = SubmitForReviewTool(artifact_registry=registry)
        context = ToolExecutionContext.empty()

        result = await tool.execute_with_context(
            context, artifact_id="nonexistent"
        )

        assert "error" in result


# --- GetArtifactTool Tests ---


class TestGetArtifactTool:
    async def test_schema(self) -> None:
        db = AsyncMemoryDatabase()
        registry = ArtifactRegistry(db)
        tool = GetArtifactTool(artifact_registry=registry)

        assert tool.name == "get_artifact"
        assert "artifact_id" in tool.schema["required"]

    async def test_gets_artifact(self) -> None:
        db = AsyncMemoryDatabase()
        registry = ArtifactRegistry(db)
        artifact = await registry.create(
            artifact_type="content",
            name="Test Doc",
            content={"body": "hello"},
            tags=["doc"],
        )

        tool = GetArtifactTool(artifact_registry=registry)
        context = ToolExecutionContext.empty()

        result = await tool.execute_with_context(
            context, artifact_id=artifact.id
        )

        assert result["id"] == artifact.id
        assert result["name"] == "Test Doc"
        assert result["content"]["body"] == "hello"
        assert result["status"] == "draft"
        assert "provenance" in result
        assert result["tags"] == ["doc"]

    async def test_get_not_found(self) -> None:
        db = AsyncMemoryDatabase()
        registry = ArtifactRegistry(db)
        tool = GetArtifactTool(artifact_registry=registry)
        context = ToolExecutionContext.empty()

        result = await tool.execute_with_context(
            context, artifact_id="nonexistent"
        )

        assert "error" in result

    async def test_get_with_evaluations(self) -> None:
        db = AsyncMemoryDatabase()
        registry = ArtifactRegistry(db)
        artifact = await registry.create(
            artifact_type="content",
            name="Test",
            content={"v": 1},
        )

        tool = GetArtifactTool(artifact_registry=registry)
        context = ToolExecutionContext.empty()

        result = await tool.execute_with_context(
            context, artifact_id=artifact.id, include_evaluations=True
        )

        assert "evaluations" in result
        assert isinstance(result["evaluations"], list)
