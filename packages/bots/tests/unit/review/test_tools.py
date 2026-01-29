"""Tests for review tools."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from dataknobs_bots.artifacts.models import Artifact, ArtifactDefinition, ArtifactReview
from dataknobs_bots.artifacts.registry import ArtifactRegistry
from dataknobs_bots.review.executor import ReviewExecutor
from dataknobs_bots.review.protocol import ReviewProtocolDefinition
from dataknobs_bots.review.tools import (
    GetReviewResultsTool,
    ReviewArtifactTool,
    RunAllReviewsTool,
)
from dataknobs_llm.tools.context import ToolExecutionContext


class TestReviewArtifactTool:
    """Tests for ReviewArtifactTool."""

    def test_init(self) -> None:
        """Test tool initialization."""
        registry = ArtifactRegistry()
        executor = ReviewExecutor()
        tool = ReviewArtifactTool(
            artifact_registry=registry,
            review_executor=executor,
        )

        assert tool.name == "review_artifact"

    def test_schema(self) -> None:
        """Test tool schema."""
        registry = ArtifactRegistry()
        executor = ReviewExecutor()
        tool = ReviewArtifactTool(
            artifact_registry=registry,
            review_executor=executor,
        )

        schema = tool.schema
        assert schema["type"] == "object"
        assert "artifact_id" in schema["properties"]
        assert "protocol_id" in schema["properties"]

    def test_execute_schema_review(self) -> None:
        """Test running a schema review."""
        registry = ArtifactRegistry()
        artifact = registry.create(
            content={"name": "test"},
            name="Test Artifact",
        )

        protocol = ReviewProtocolDefinition(
            id="schema_test",
            type="schema",
            schema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        )
        executor = ReviewExecutor(protocols={"schema_test": protocol})

        tool = ReviewArtifactTool(
            artifact_registry=registry,
            review_executor=executor,
        )
        context = ToolExecutionContext.empty()

        result = asyncio.run(
            tool.execute_with_context(
                context,
                artifact_id=artifact.id,
                protocol_id="schema_test",
            )
        )

        assert result["passed"] is True
        assert result["score"] == 1.0
        assert result["review_type"] == "schema"

    def test_execute_artifact_not_found(self) -> None:
        """Test review with non-existent artifact."""
        registry = ArtifactRegistry()
        executor = ReviewExecutor()
        tool = ReviewArtifactTool(
            artifact_registry=registry,
            review_executor=executor,
        )
        context = ToolExecutionContext.empty()

        result = asyncio.run(
            tool.execute_with_context(
                context,
                artifact_id="nonexistent",
                protocol_id="test",
            )
        )

        assert "error" in result

    def test_execute_protocol_not_found(self) -> None:
        """Test review with non-existent protocol."""
        registry = ArtifactRegistry()
        artifact = registry.create(content={"v": 1}, name="Test")
        executor = ReviewExecutor()

        tool = ReviewArtifactTool(
            artifact_registry=registry,
            review_executor=executor,
        )
        context = ToolExecutionContext.empty()

        result = asyncio.run(
            tool.execute_with_context(
                context,
                artifact_id=artifact.id,
                protocol_id="nonexistent",
            )
        )

        assert "error" in result

    def test_execute_custom_review(self) -> None:
        """Test running a custom review."""
        registry = ArtifactRegistry()
        artifact = registry.create(content={"v": 1}, name="Test")

        def custom_validator(art: Artifact) -> dict:
            return {"passed": True, "score": 0.9, "feedback": ["Good"]}

        protocol = ReviewProtocolDefinition(id="custom", type="custom")
        executor = ReviewExecutor(protocols={"custom": protocol})
        executor.register_function("custom", custom_validator)

        tool = ReviewArtifactTool(
            artifact_registry=registry,
            review_executor=executor,
        )
        context = ToolExecutionContext.empty()

        result = asyncio.run(
            tool.execute_with_context(
                context,
                artifact_id=artifact.id,
                protocol_id="custom",
            )
        )

        assert result["passed"] is True
        assert result["score"] == 0.9

        # Check review was added to artifact
        updated = registry.get(artifact.id)
        assert len(updated.reviews) == 1


class TestRunAllReviewsTool:
    """Tests for RunAllReviewsTool."""

    def test_init(self) -> None:
        """Test tool initialization."""
        registry = ArtifactRegistry()
        executor = ReviewExecutor()
        tool = RunAllReviewsTool(
            artifact_registry=registry,
            review_executor=executor,
        )

        assert tool.name == "run_all_reviews"

    def test_execute_no_reviews_configured(self) -> None:
        """Test run all reviews with no reviews configured."""
        registry = ArtifactRegistry()
        artifact = registry.create(content={"v": 1}, name="Test")
        executor = ReviewExecutor()

        tool = RunAllReviewsTool(
            artifact_registry=registry,
            review_executor=executor,
        )
        context = ToolExecutionContext.empty()

        result = asyncio.run(
            tool.execute_with_context(context, artifact_id=artifact.id)
        )

        assert result["all_passed"] is True
        assert result["review_count"] == 0

    def test_execute_with_definition(self) -> None:
        """Test run all reviews with artifact definition."""
        # Set up definition with reviews
        definition = ArtifactDefinition(
            id="test_def",
            reviews=["review1", "review2"],
        )

        registry = ArtifactRegistry()
        registry.register_definition(definition)
        artifact = registry.create(
            content={"v": 1},
            name="Test",
            definition_id="test_def",
        )

        # Set up custom validators
        def validator1(art: Artifact) -> dict:
            return {"passed": True, "score": 0.9}

        def validator2(art: Artifact) -> dict:
            return {"passed": True, "score": 0.8}

        protocols = {
            "review1": ReviewProtocolDefinition(id="review1", type="custom"),
            "review2": ReviewProtocolDefinition(id="review2", type="custom"),
        }
        executor = ReviewExecutor(protocols=protocols)
        executor.register_function("review1", validator1)
        executor.register_function("review2", validator2)

        tool = RunAllReviewsTool(
            artifact_registry=registry,
            review_executor=executor,
        )
        context = ToolExecutionContext.empty()

        result = asyncio.run(
            tool.execute_with_context(context, artifact_id=artifact.id)
        )

        assert result["all_passed"] is True
        assert result["review_count"] == 2
        assert result["average_score"] == 0.85

    def test_execute_artifact_not_found(self) -> None:
        """Test run all reviews with non-existent artifact."""
        registry = ArtifactRegistry()
        executor = ReviewExecutor()
        tool = RunAllReviewsTool(
            artifact_registry=registry,
            review_executor=executor,
        )
        context = ToolExecutionContext.empty()

        result = asyncio.run(
            tool.execute_with_context(context, artifact_id="nonexistent")
        )

        assert "error" in result


class TestGetReviewResultsTool:
    """Tests for GetReviewResultsTool."""

    def test_init(self) -> None:
        """Test tool initialization."""
        registry = ArtifactRegistry()
        tool = GetReviewResultsTool(artifact_registry=registry)

        assert tool.name == "get_review_results"

    def test_execute_no_reviews(self) -> None:
        """Test get reviews with no reviews."""
        registry = ArtifactRegistry()
        artifact = registry.create(content={"v": 1}, name="Test")
        tool = GetReviewResultsTool(artifact_registry=registry)
        context = ToolExecutionContext.empty()

        result = asyncio.run(
            tool.execute_with_context(context, artifact_id=artifact.id)
        )

        assert result["review_count"] == 0
        assert result["all_passed"] is True
        assert result["reviews"] == []

    def test_execute_with_reviews(self) -> None:
        """Test get reviews with existing reviews."""
        registry = ArtifactRegistry()
        artifact = registry.create(content={"v": 1}, name="Test")

        review1 = ArtifactReview(
            artifact_id=artifact.id,
            reviewer="adversarial",
            passed=True,
            score=0.85,
            issues=[],
        )
        review2 = ArtifactReview(
            artifact_id=artifact.id,
            reviewer="skeptical",
            passed=False,
            score=0.6,
            issues=["Issue 1"],
        )
        registry.add_review(artifact.id, review1)
        registry.add_review(artifact.id, review2)

        tool = GetReviewResultsTool(artifact_registry=registry)
        context = ToolExecutionContext.empty()

        result = asyncio.run(
            tool.execute_with_context(context, artifact_id=artifact.id)
        )

        assert result["review_count"] == 2
        assert result["all_passed"] is False
        assert result["average_score"] == 0.72  # (0.85 + 0.6) / 2 = 0.725, rounded to 0.72

    def test_execute_artifact_not_found(self) -> None:
        """Test get reviews with non-existent artifact."""
        registry = ArtifactRegistry()
        tool = GetReviewResultsTool(artifact_registry=registry)
        context = ToolExecutionContext.empty()

        result = asyncio.run(
            tool.execute_with_context(context, artifact_id="nonexistent")
        )

        assert "error" in result
