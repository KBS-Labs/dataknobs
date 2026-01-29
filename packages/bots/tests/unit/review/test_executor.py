"""Tests for review executor."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from dataknobs_bots.artifacts.models import Artifact, ArtifactDefinition, ArtifactMetadata
from dataknobs_bots.review.executor import ReviewExecutor
from dataknobs_bots.review.personas import ReviewPersona
from dataknobs_bots.review.protocol import ReviewProtocolDefinition


class TestReviewExecutorBasic:
    """Basic tests for ReviewExecutor."""

    def test_init_empty(self) -> None:
        """Test empty initialization."""
        executor = ReviewExecutor()
        # Should have built-in personas
        assert "adversarial" in executor.get_available_personas()
        assert "skeptical" in executor.get_available_personas()
        # No protocols yet
        assert executor.get_available_protocols() == []

    def test_init_with_protocols(self) -> None:
        """Test initialization with protocols."""
        protocols = {
            "test": ReviewProtocolDefinition(
                id="test",
                type="persona",
                persona_id="adversarial",
            )
        }
        executor = ReviewExecutor(protocols=protocols)
        assert "test" in executor.get_available_protocols()

    def test_from_config(self) -> None:
        """Test creation from config dict."""
        config = {
            "review_protocols": {
                "adversarial": {
                    "persona": "adversarial",
                    "score_threshold": 0.8,
                },
                "validation": {
                    "type": "schema",
                    "schema": {"type": "object"},
                },
            }
        }
        executor = ReviewExecutor.from_config(config)
        assert "adversarial" in executor.get_available_protocols()
        assert "validation" in executor.get_available_protocols()


class TestReviewExecutorRegistration:
    """Tests for runtime registration."""

    def test_register_persona(self) -> None:
        """Test registering a custom persona."""
        executor = ReviewExecutor()

        custom_persona = ReviewPersona(
            id="custom",
            name="Custom Reviewer",
            focus="custom focus",
            prompt_template="Custom template: {artifact_content}",
        )
        executor.register_persona(custom_persona)

        assert "custom" in executor.get_available_personas()
        assert executor.get_persona("custom") == custom_persona

    def test_register_protocol(self) -> None:
        """Test registering a custom protocol."""
        executor = ReviewExecutor()

        protocol = ReviewProtocolDefinition(
            id="custom_protocol",
            type="persona",
            persona_id="adversarial",
        )
        executor.register_protocol(protocol)

        assert "custom_protocol" in executor.get_available_protocols()
        assert executor.get_protocol("custom_protocol") == protocol

    def test_register_function(self) -> None:
        """Test registering a custom validation function."""
        executor = ReviewExecutor()

        def custom_validator(artifact: Artifact) -> dict:
            return {"passed": True, "score": 1.0}

        protocol = ReviewProtocolDefinition(
            id="custom_func",
            type="custom",
        )
        executor.register_protocol(protocol)
        executor.register_function("custom_func", custom_validator)

        assert "custom_func" in executor.get_available_protocols()


class TestReviewExecutorSchemaReview:
    """Tests for schema validation reviews."""

    def test_schema_review_pass(self) -> None:
        """Test passing schema validation."""
        protocol = ReviewProtocolDefinition(
            id="schema_test",
            type="schema",
            schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                },
                "required": ["name"],
            },
        )
        executor = ReviewExecutor(protocols={"schema_test": protocol})

        artifact = Artifact(
            content={"name": "test"},
            name="Test Artifact",
        )

        import asyncio
        review = asyncio.run(executor.run_review(artifact, "schema_test"))

        assert review.passed is True
        assert review.score == 1.0
        assert review.review_type == "schema"

    def test_schema_review_fail(self) -> None:
        """Test failing schema validation."""
        protocol = ReviewProtocolDefinition(
            id="schema_test",
            type="schema",
            schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                },
                "required": ["name"],
            },
        )
        executor = ReviewExecutor(protocols={"schema_test": protocol})

        artifact = Artifact(
            content={"other": "value"},  # Missing required "name"
            name="Test Artifact",
        )

        import asyncio
        review = asyncio.run(executor.run_review(artifact, "schema_test"))

        assert review.passed is False
        assert review.score == 0.0
        assert len(review.issues) > 0

    def test_schema_review_no_schema(self) -> None:
        """Test schema review with no schema defined (should pass)."""
        protocol = ReviewProtocolDefinition(
            id="no_schema",
            type="schema",
            schema=None,
        )
        executor = ReviewExecutor(protocols={"no_schema": protocol})

        artifact = Artifact(content={"any": "content"}, name="Test")

        import asyncio
        review = asyncio.run(executor.run_review(artifact, "no_schema"))

        assert review.passed is True
        assert "skipped" in review.feedback[0].lower()


class TestReviewExecutorCustomReview:
    """Tests for custom validation function reviews."""

    def test_custom_review_sync_pass(self) -> None:
        """Test passing custom validation (sync function)."""
        def validator(artifact: Artifact) -> dict:
            return {
                "passed": True,
                "score": 0.95,
                "feedback": ["Looks good"],
            }

        protocol = ReviewProtocolDefinition(
            id="custom_test",
            type="custom",
        )
        executor = ReviewExecutor(protocols={"custom_test": protocol})
        executor.register_function("custom_test", validator)

        artifact = Artifact(content={"data": "test"}, name="Test")

        import asyncio
        review = asyncio.run(executor.run_review(artifact, "custom_test"))

        assert review.passed is True
        assert review.score == 0.95
        assert review.review_type == "custom"

    def test_custom_review_async(self) -> None:
        """Test custom validation with async function."""
        async def validator(artifact: Artifact) -> dict:
            return {"passed": True, "score": 1.0}

        protocol = ReviewProtocolDefinition(
            id="async_test",
            type="custom",
        )
        executor = ReviewExecutor(protocols={"async_test": protocol})
        executor.register_function("async_test", validator)

        artifact = Artifact(content={"data": "test"}, name="Test")

        import asyncio
        review = asyncio.run(executor.run_review(artifact, "async_test"))

        assert review.passed is True

    def test_custom_review_boolean_result(self) -> None:
        """Test custom validation returning boolean."""
        def validator(artifact: Artifact) -> bool:
            return True

        protocol = ReviewProtocolDefinition(
            id="bool_test",
            type="custom",
        )
        executor = ReviewExecutor(protocols={"bool_test": protocol})
        executor.register_function("bool_test", validator)

        artifact = Artifact(content={"data": "test"}, name="Test")

        import asyncio
        review = asyncio.run(executor.run_review(artifact, "bool_test"))

        assert review.passed is True
        assert review.score == 1.0

    def test_custom_review_no_function(self) -> None:
        """Test custom review without registered function."""
        protocol = ReviewProtocolDefinition(
            id="missing_func",
            type="custom",
        )
        executor = ReviewExecutor(protocols={"missing_func": protocol})

        artifact = Artifact(content={"data": "test"}, name="Test")

        import asyncio
        review = asyncio.run(executor.run_review(artifact, "missing_func"))

        assert review.passed is False
        assert "not registered" in review.issues[0].lower()

    def test_custom_review_exception(self) -> None:
        """Test custom review handling exceptions."""
        def validator(artifact: Artifact) -> dict:
            raise ValueError("Validation error")

        protocol = ReviewProtocolDefinition(
            id="error_test",
            type="custom",
        )
        executor = ReviewExecutor(protocols={"error_test": protocol})
        executor.register_function("error_test", validator)

        artifact = Artifact(content={"data": "test"}, name="Test")

        import asyncio
        review = asyncio.run(executor.run_review(artifact, "error_test"))

        assert review.passed is False
        assert "failed" in review.issues[0].lower()


class TestReviewExecutorPersonaReview:
    """Tests for persona-based reviews."""

    def test_persona_review_no_llm(self) -> None:
        """Test persona review without LLM provider."""
        protocol = ReviewProtocolDefinition(
            id="persona_test",
            type="persona",
            persona_id="adversarial",
        )
        executor = ReviewExecutor(protocols={"persona_test": protocol}, llm=None)

        artifact = Artifact(content={"data": "test"}, name="Test")

        import asyncio
        review = asyncio.run(executor.run_review(artifact, "persona_test"))

        assert review.passed is False
        assert "llm" in review.issues[0].lower()

    @pytest.mark.asyncio
    async def test_persona_review_with_mock_llm(self) -> None:
        """Test persona review with mocked LLM."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '{"passed": true, "score": 0.85, "issues": [], "suggestions": ["Good job"], "feedback": ["Well done"]}'
        mock_llm.complete = AsyncMock(return_value=mock_response)

        protocol = ReviewProtocolDefinition(
            id="persona_test",
            type="persona",
            persona_id="adversarial",
        )
        executor = ReviewExecutor(protocols={"persona_test": protocol}, llm=mock_llm)

        artifact = Artifact(
            content={"data": "test"},
            name="Test Artifact",
            metadata=ArtifactMetadata(purpose="Testing"),
        )

        review = await executor.run_review(artifact, "persona_test")

        assert review.passed is True
        assert review.score == 0.85
        assert review.suggestions == ["Good job"]
        assert review.feedback == ["Well done"]
        mock_llm.complete.assert_called_once()

    def test_persona_review_missing_persona(self) -> None:
        """Test persona review with missing persona reference."""
        protocol = ReviewProtocolDefinition(
            id="missing_persona",
            type="persona",
            persona_id="nonexistent",
        )
        executor = ReviewExecutor(
            protocols={"missing_persona": protocol},
            llm=MagicMock(),
        )

        artifact = Artifact(content={"data": "test"}, name="Test")

        import asyncio
        review = asyncio.run(executor.run_review(artifact, "missing_persona"))

        assert review.passed is False
        assert "not found" in review.issues[0].lower()


class TestReviewExecutorDisabledProtocol:
    """Tests for disabled protocol handling."""

    def test_disabled_protocol_skipped(self) -> None:
        """Test that disabled protocols are skipped."""
        protocol = ReviewProtocolDefinition(
            id="disabled_test",
            type="persona",
            persona_id="adversarial",
            enabled=False,
        )
        executor = ReviewExecutor(protocols={"disabled_test": protocol})

        artifact = Artifact(content={"data": "test"}, name="Test")

        import asyncio
        review = asyncio.run(executor.run_review(artifact, "disabled_test"))

        assert review.passed is True
        assert review.score == 1.0
        assert "disabled" in review.feedback[0].lower()


class TestReviewExecutorProtocolNotFound:
    """Tests for protocol not found error."""

    def test_protocol_not_found(self) -> None:
        """Test that missing protocol raises KeyError."""
        executor = ReviewExecutor()
        artifact = Artifact(content={"data": "test"}, name="Test")

        import asyncio
        with pytest.raises(KeyError):
            asyncio.run(executor.run_review(artifact, "nonexistent"))


class TestReviewExecutorArtifactReviews:
    """Tests for running all configured reviews."""

    @pytest.mark.asyncio
    async def test_run_artifact_reviews_with_definition(self) -> None:
        """Test running all reviews for an artifact with definition."""
        # Set up protocols
        def validator1(artifact: Artifact) -> dict:
            return {"passed": True, "score": 0.9}

        def validator2(artifact: Artifact) -> dict:
            return {"passed": True, "score": 0.8}

        protocols = {
            "review1": ReviewProtocolDefinition(id="review1", type="custom"),
            "review2": ReviewProtocolDefinition(id="review2", type="custom"),
        }
        executor = ReviewExecutor(protocols=protocols)
        executor.register_function("review1", validator1)
        executor.register_function("review2", validator2)

        artifact = Artifact(content={"data": "test"}, name="Test")
        definition = ArtifactDefinition(
            id="test_def",
            reviews=["review1", "review2"],
        )

        reviews = await executor.run_artifact_reviews(artifact, definition)

        assert len(reviews) == 2
        assert all(r.passed for r in reviews)

    @pytest.mark.asyncio
    async def test_run_artifact_reviews_no_reviews_configured(self) -> None:
        """Test running reviews when none are configured."""
        executor = ReviewExecutor()
        artifact = Artifact(content={"data": "test"}, name="Test")
        definition = ArtifactDefinition(id="test_def", reviews=[])

        reviews = await executor.run_artifact_reviews(artifact, definition)

        assert len(reviews) == 0

    @pytest.mark.asyncio
    async def test_run_artifact_reviews_missing_protocol(self) -> None:
        """Test running reviews when protocol is missing."""
        executor = ReviewExecutor()
        artifact = Artifact(content={"data": "test"}, name="Test")
        definition = ArtifactDefinition(
            id="test_def",
            reviews=["nonexistent_protocol"],
        )

        # Should not raise, just skip missing protocol
        reviews = await executor.run_artifact_reviews(artifact, definition)
        assert len(reviews) == 0
