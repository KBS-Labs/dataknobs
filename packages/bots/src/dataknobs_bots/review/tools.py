"""Review tools for artifact evaluation.

This module provides context-aware tools for running reviews against
artifacts. These tools integrate with the ReviewExecutor to apply
persona-based, schema-based, or custom validation reviews.

Tools:
- ReviewArtifactTool: Run a single review protocol against an artifact
- RunAllReviewsTool: Run all configured reviews for an artifact

Example:
    >>> executor = ReviewExecutor(llm=llm, protocols=protocols)
    >>> registry = ArtifactRegistry()
    >>> review_tool = ReviewArtifactTool(
    ...     artifact_registry=registry,
    ...     review_executor=executor,
    ... )
    >>> result = await review_tool.execute(
    ...     artifact_id="art_123",
    ...     protocol_id="adversarial",
    ...     _context=context,
    ... )
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from dataknobs_llm.tools.context import ToolExecutionContext
from dataknobs_llm.tools.context_aware import ContextAwareTool

if TYPE_CHECKING:
    from ..artifacts.registry import ArtifactRegistry
    from .executor import ReviewExecutor

logger = logging.getLogger(__name__)


class ReviewArtifactTool(ContextAwareTool):
    """Tool for running a single review against an artifact.

    Executes a specific review protocol against an artifact and
    records the result.

    Attributes:
        _registry: ArtifactRegistry for artifact lookup
        _executor: ReviewExecutor for running reviews
    """

    def __init__(
        self,
        artifact_registry: ArtifactRegistry,
        review_executor: ReviewExecutor,
    ) -> None:
        """Initialize the tool.

        Args:
            artifact_registry: Registry for artifact lookup
            review_executor: Executor for running reviews
        """
        super().__init__(
            name="review_artifact",
            description=(
                "Run a specific review protocol against an artifact. "
                "Reviews can be persona-based (LLM evaluation), schema-based "
                "(JSON validation), or custom validation functions."
            ),
        )
        self._registry = artifact_registry
        self._executor = review_executor

    @property
    def schema(self) -> dict[str, Any]:
        """Return JSON Schema for tool parameters."""
        return {
            "type": "object",
            "properties": {
                "artifact_id": {
                    "type": "string",
                    "description": "ID of the artifact to review",
                },
                "protocol_id": {
                    "type": "string",
                    "description": (
                        "ID of the review protocol to use (e.g., 'adversarial', "
                        "'skeptical', 'validation')"
                    ),
                },
            },
            "required": ["artifact_id", "protocol_id"],
        }

    async def execute_with_context(
        self,
        context: ToolExecutionContext,
        artifact_id: str,
        protocol_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run a review against an artifact.

        Args:
            context: Execution context
            artifact_id: ID of artifact to review
            protocol_id: ID of review protocol

        Returns:
            Dict with review results
        """
        # Get artifact
        artifact = self._registry.get(artifact_id)
        if not artifact:
            return {
                "error": f"Artifact not found: {artifact_id}",
                "message": f"Artifact '{artifact_id}' not found",
            }

        # Run review
        try:
            review = await self._executor.run_review(artifact, protocol_id)

            # Add review to artifact
            self._registry.add_review(artifact_id, review)

            logger.info(
                "Completed review %s for artifact %s (passed=%s, score=%.2f)",
                protocol_id,
                artifact_id,
                review.passed,
                review.score,
            )

            return {
                "artifact_id": artifact_id,
                "protocol_id": protocol_id,
                "passed": review.passed,
                "score": review.score,
                "review_type": review.review_type,
                "issues": review.issues,
                "suggestions": review.suggestions,
                "feedback": review.feedback,
                "message": (
                    f"Review '{protocol_id}' "
                    f"{'passed' if review.passed else 'failed'} "
                    f"(score: {review.score:.2f})"
                ),
            }
        except KeyError as e:
            return {
                "error": str(e),
                "message": f"Review protocol not found: {protocol_id}",
            }
        except Exception as e:
            logger.error(
                "Review %s failed for artifact %s: %s",
                protocol_id,
                artifact_id,
                e,
            )
            return {
                "error": str(e),
                "message": f"Review failed: {e}",
            }


class RunAllReviewsTool(ContextAwareTool):
    """Tool for running all configured reviews for an artifact.

    Executes all review protocols configured for the artifact's
    definition type.

    Attributes:
        _registry: ArtifactRegistry for artifact lookup
        _executor: ReviewExecutor for running reviews
    """

    def __init__(
        self,
        artifact_registry: ArtifactRegistry,
        review_executor: ReviewExecutor,
    ) -> None:
        """Initialize the tool.

        Args:
            artifact_registry: Registry for artifact lookup
            review_executor: Executor for running reviews
        """
        super().__init__(
            name="run_all_reviews",
            description=(
                "Run all configured reviews for an artifact based on its "
                "definition. Returns whether all reviews passed."
            ),
        )
        self._registry = artifact_registry
        self._executor = review_executor

    @property
    def schema(self) -> dict[str, Any]:
        """Return JSON Schema for tool parameters."""
        return {
            "type": "object",
            "properties": {
                "artifact_id": {
                    "type": "string",
                    "description": "ID of the artifact to review",
                },
            },
            "required": ["artifact_id"],
        }

    async def execute_with_context(
        self,
        context: ToolExecutionContext,
        artifact_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run all reviews for an artifact.

        Args:
            context: Execution context
            artifact_id: ID of artifact to review

        Returns:
            Dict with all review results and overall status
        """
        # Get artifact
        artifact = self._registry.get(artifact_id)
        if not artifact:
            return {
                "error": f"Artifact not found: {artifact_id}",
                "message": f"Artifact '{artifact_id}' not found",
            }

        # Get artifact definition if available
        definition = None
        if artifact.definition_id:
            definition = self._registry.get_definition(artifact.definition_id)

        try:
            # Run all reviews
            reviews = await self._executor.run_artifact_reviews(artifact, definition)

            # Add reviews to artifact
            for review in reviews:
                self._registry.add_review(artifact_id, review)

            # Calculate summary
            all_passed = all(r.passed for r in reviews) if reviews else True
            total_score = sum(r.score for r in reviews) / len(reviews) if reviews else 1.0

            logger.info(
                "Completed %d reviews for artifact %s (all_passed=%s)",
                len(reviews),
                artifact_id,
                all_passed,
            )

            return {
                "artifact_id": artifact_id,
                "all_passed": all_passed,
                "average_score": round(total_score, 2),
                "review_count": len(reviews),
                "reviews": [
                    {
                        "protocol_id": r.reviewer,
                        "passed": r.passed,
                        "score": r.score,
                        "issues": r.issues,
                    }
                    for r in reviews
                ],
                "message": (
                    f"Ran {len(reviews)} reviews: "
                    f"{'all passed' if all_passed else 'some failed'}"
                ),
            }
        except Exception as e:
            logger.error("Failed to run reviews for artifact %s: %s", artifact_id, e)
            return {
                "error": str(e),
                "message": f"Failed to run reviews: {e}",
            }


class GetReviewResultsTool(ContextAwareTool):
    """Tool for retrieving review results for an artifact.

    Returns all reviews that have been conducted on an artifact.

    Attributes:
        _registry: ArtifactRegistry for artifact lookup
    """

    def __init__(self, artifact_registry: ArtifactRegistry) -> None:
        """Initialize the tool.

        Args:
            artifact_registry: Registry for artifact lookup
        """
        super().__init__(
            name="get_review_results",
            description=(
                "Get all review results for an artifact, including "
                "pass/fail status, scores, and feedback."
            ),
        )
        self._registry = artifact_registry

    @property
    def schema(self) -> dict[str, Any]:
        """Return JSON Schema for tool parameters."""
        return {
            "type": "object",
            "properties": {
                "artifact_id": {
                    "type": "string",
                    "description": "ID of the artifact to get reviews for",
                },
            },
            "required": ["artifact_id"],
        }

    async def execute_with_context(
        self,
        context: ToolExecutionContext,
        artifact_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Get review results for an artifact.

        Args:
            context: Execution context
            artifact_id: ID of artifact

        Returns:
            Dict with review results
        """
        artifact = self._registry.get(artifact_id)
        if not artifact:
            return {
                "error": f"Artifact not found: {artifact_id}",
                "message": f"Artifact '{artifact_id}' not found",
            }

        reviews = artifact.reviews or []
        all_passed = all(r.passed for r in reviews) if reviews else True
        total_score = sum(r.score for r in reviews) / len(reviews) if reviews else 1.0

        return {
            "artifact_id": artifact_id,
            "artifact_name": artifact.name,
            "artifact_status": artifact.status,
            "all_passed": all_passed,
            "average_score": round(total_score, 2),
            "review_count": len(reviews),
            "reviews": [
                {
                    "id": r.id,
                    "reviewer": r.reviewer,
                    "review_type": r.review_type,
                    "passed": r.passed,
                    "score": r.score,
                    "issues": r.issues,
                    "suggestions": r.suggestions,
                    "feedback": r.feedback,
                    "timestamp": r.timestamp,
                }
                for r in reviews
            ],
        }
