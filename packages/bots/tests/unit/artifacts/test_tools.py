"""Tests for artifact tools."""

import asyncio
from unittest.mock import MagicMock

import pytest

from dataknobs_bots.artifacts.models import Artifact
from dataknobs_bots.artifacts.registry import ArtifactRegistry
from dataknobs_bots.artifacts.tools import (
    CreateArtifactTool,
    GetArtifactTool,
    QueryArtifactsTool,
    SubmitForReviewTool,
    UpdateArtifactTool,
)
from dataknobs_llm.tools.context import ToolExecutionContext, WizardStateSnapshot


class TestCreateArtifactTool:
    """Tests for CreateArtifactTool."""

    def test_init(self) -> None:
        """Test tool initialization."""
        registry = ArtifactRegistry()
        tool = CreateArtifactTool(artifact_registry=registry)

        assert tool.name == "create_artifact"
        assert "Create" in tool.description or "create" in tool.description

    def test_schema(self) -> None:
        """Test tool schema."""
        registry = ArtifactRegistry()
        tool = CreateArtifactTool(artifact_registry=registry)

        schema = tool.schema
        assert schema["type"] == "object"
        assert "content" in schema["properties"]
        assert "name" in schema["properties"]
        assert "content" in schema["required"]
        assert "name" in schema["required"]

    def test_execute_basic(self) -> None:
        """Test basic artifact creation."""
        registry = ArtifactRegistry()
        tool = CreateArtifactTool(artifact_registry=registry)
        context = ToolExecutionContext.empty()

        result = asyncio.run(
            tool.execute_with_context(
                context,
                content={"questions": ["Q1", "Q2"]},
                name="Test Questions",
            )
        )

        assert "artifact_id" in result
        assert result["status"] == "draft"
        assert result["name"] == "Test Questions"

    def test_execute_with_wizard_context(self) -> None:
        """Test artifact creation with wizard context."""
        registry = ArtifactRegistry()
        tool = CreateArtifactTool(artifact_registry=registry)

        wizard_state = WizardStateSnapshot(
            current_stage="build_questions",
            collected_data={"subject": "math"},
        )
        context = ToolExecutionContext(wizard_state=wizard_state)

        result = asyncio.run(
            tool.execute_with_context(
                context,
                content={"questions": ["Q1"]},
                name="Math Questions",
            )
        )

        assert result["artifact_id"]
        # Check artifact was created with stage
        artifact = registry.get(result["artifact_id"])
        assert artifact is not None
        assert artifact.metadata.stage == "build_questions"

    def test_execute_with_definition_id(self) -> None:
        """Test artifact creation with definition ID."""
        registry = ArtifactRegistry()
        tool = CreateArtifactTool(artifact_registry=registry)
        context = ToolExecutionContext.empty()

        result = asyncio.run(
            tool.execute_with_context(
                context,
                content={"data": "value"},
                name="Test",
                definition_id="test_def",
                purpose="Testing purpose",
            )
        )

        artifact = registry.get(result["artifact_id"])
        assert artifact is not None
        assert artifact.definition_id == "test_def"
        assert artifact.metadata.purpose == "Testing purpose"


class TestUpdateArtifactTool:
    """Tests for UpdateArtifactTool."""

    def test_init(self) -> None:
        """Test tool initialization."""
        registry = ArtifactRegistry()
        tool = UpdateArtifactTool(artifact_registry=registry)

        assert tool.name == "update_artifact"

    def test_execute_update(self) -> None:
        """Test artifact update."""
        registry = ArtifactRegistry()
        artifact = registry.create(content={"v": 1}, name="Test")
        tool = UpdateArtifactTool(artifact_registry=registry)
        context = ToolExecutionContext.empty()

        result = asyncio.run(
            tool.execute_with_context(
                context,
                artifact_id=artifact.id,
                content={"v": 2},
            )
        )

        assert result["artifact_id"] != artifact.id  # New version
        assert result["previous_id"] == artifact.id
        assert result["version"] == 2

    def test_execute_not_found(self) -> None:
        """Test update with non-existent artifact."""
        registry = ArtifactRegistry()
        tool = UpdateArtifactTool(artifact_registry=registry)
        context = ToolExecutionContext.empty()

        result = asyncio.run(
            tool.execute_with_context(
                context,
                artifact_id="nonexistent",
                content={"v": 1},
            )
        )

        assert "error" in result


class TestQueryArtifactsTool:
    """Tests for QueryArtifactsTool."""

    def test_init(self) -> None:
        """Test tool initialization."""
        registry = ArtifactRegistry()
        tool = QueryArtifactsTool(artifact_registry=registry)

        assert tool.name == "query_artifacts"

    def test_execute_no_filter(self) -> None:
        """Test query with no filter."""
        registry = ArtifactRegistry()
        registry.create(content={"v": 1}, name="Test 1")
        registry.create(content={"v": 2}, name="Test 2")
        tool = QueryArtifactsTool(artifact_registry=registry)
        context = ToolExecutionContext.empty()

        result = asyncio.run(tool.execute_with_context(context))

        assert result["count"] == 2
        assert len(result["artifacts"]) == 2

    def test_execute_filter_by_status(self) -> None:
        """Test query filtering by status."""
        registry = ArtifactRegistry()
        a1 = registry.create(content={"v": 1}, name="Test 1")
        registry.create(content={"v": 2}, name="Test 2")
        registry.submit_for_review(a1.id)

        tool = QueryArtifactsTool(artifact_registry=registry)
        context = ToolExecutionContext.empty()

        result = asyncio.run(
            tool.execute_with_context(context, status="pending_review")
        )

        assert result["count"] == 1
        assert result["artifacts"][0]["status"] == "pending_review"

    def test_execute_filter_by_stage(self) -> None:
        """Test query filtering by stage."""
        registry = ArtifactRegistry()
        registry.create(content={"v": 1}, name="Test 1", stage="stage_a")
        registry.create(content={"v": 2}, name="Test 2", stage="stage_b")

        tool = QueryArtifactsTool(artifact_registry=registry)
        context = ToolExecutionContext.empty()

        result = asyncio.run(
            tool.execute_with_context(context, stage="stage_a")
        )

        assert result["count"] == 1
        assert result["artifacts"][0]["stage"] == "stage_a"

    def test_execute_include_content(self) -> None:
        """Test query with content included."""
        registry = ArtifactRegistry()
        registry.create(content={"data": "value"}, name="Test")

        tool = QueryArtifactsTool(artifact_registry=registry)
        context = ToolExecutionContext.empty()

        result = asyncio.run(
            tool.execute_with_context(context, include_content=True)
        )

        assert "content" in result["artifacts"][0]
        assert result["artifacts"][0]["content"] == {"data": "value"}


class TestSubmitForReviewTool:
    """Tests for SubmitForReviewTool."""

    def test_init(self) -> None:
        """Test tool initialization."""
        registry = ArtifactRegistry()
        tool = SubmitForReviewTool(artifact_registry=registry)

        assert tool.name == "submit_for_review"

    def test_execute_submit(self) -> None:
        """Test submitting artifact for review."""
        registry = ArtifactRegistry()
        artifact = registry.create(content={"v": 1}, name="Test")
        tool = SubmitForReviewTool(artifact_registry=registry)
        context = ToolExecutionContext.empty()

        result = asyncio.run(
            tool.execute_with_context(context, artifact_id=artifact.id)
        )

        assert result["status"] == "pending_review"
        assert registry.get(artifact.id).status == "pending_review"

    def test_execute_not_found(self) -> None:
        """Test submit with non-existent artifact."""
        registry = ArtifactRegistry()
        tool = SubmitForReviewTool(artifact_registry=registry)
        context = ToolExecutionContext.empty()

        result = asyncio.run(
            tool.execute_with_context(context, artifact_id="nonexistent")
        )

        assert "error" in result


class TestGetArtifactTool:
    """Tests for GetArtifactTool."""

    def test_init(self) -> None:
        """Test tool initialization."""
        registry = ArtifactRegistry()
        tool = GetArtifactTool(artifact_registry=registry)

        assert tool.name == "get_artifact"

    def test_execute_get(self) -> None:
        """Test getting an artifact."""
        registry = ArtifactRegistry()
        artifact = registry.create(
            content={"data": "value"},
            name="Test",
            purpose="Testing",
        )
        tool = GetArtifactTool(artifact_registry=registry)
        context = ToolExecutionContext.empty()

        result = asyncio.run(
            tool.execute_with_context(context, artifact_id=artifact.id)
        )

        assert result["id"] == artifact.id
        assert result["name"] == "Test"
        assert result["content"] == {"data": "value"}
        assert result["metadata"]["purpose"] == "Testing"

    def test_execute_not_found(self) -> None:
        """Test getting non-existent artifact."""
        registry = ArtifactRegistry()
        tool = GetArtifactTool(artifact_registry=registry)
        context = ToolExecutionContext.empty()

        result = asyncio.run(
            tool.execute_with_context(context, artifact_id="nonexistent")
        )

        assert "error" in result

    def test_execute_with_reviews(self) -> None:
        """Test getting artifact with reviews."""
        from dataknobs_bots.artifacts.models import ArtifactReview

        registry = ArtifactRegistry()
        artifact = registry.create(content={"v": 1}, name="Test")
        review = ArtifactReview(
            artifact_id=artifact.id,
            reviewer="adversarial",
            passed=True,
            score=0.85,
        )
        registry.add_review(artifact.id, review)

        tool = GetArtifactTool(artifact_registry=registry)
        context = ToolExecutionContext.empty()

        result = asyncio.run(
            tool.execute_with_context(
                context,
                artifact_id=artifact.id,
                include_reviews=True,
            )
        )

        assert "reviews" in result
        assert len(result["reviews"]) == 1
        assert result["reviews"][0]["passed"] is True
        assert result["reviews"][0]["score"] == 0.85
