"""Artifact management tools for conversational workflows.

This module provides context-aware tools for managing artifacts within
conversations. These tools integrate with the ArtifactRegistry to provide
CRUD operations for artifacts.

Tools:
- CreateArtifactTool: Create new artifacts
- UpdateArtifactTool: Update existing artifact content
- QueryArtifactsTool: Query artifacts by status, type, or stage
- SubmitForReviewTool: Submit artifact for review

Example:
    >>> registry = ArtifactRegistry()
    >>> create_tool = CreateArtifactTool(artifact_registry=registry)
    >>> result = await create_tool.execute(
    ...     definition_id="assessment_questions",
    ...     content={"questions": [...]},
    ...     name="Unit 1 Questions",
    ...     _context=context,
    ... )
    >>> print(result["artifact_id"])
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from dataknobs_llm.tools.context import ToolExecutionContext
from dataknobs_llm.tools.context_aware import ContextAwareTool

if TYPE_CHECKING:
    from .registry import ArtifactRegistry

logger = logging.getLogger(__name__)


class CreateArtifactTool(ContextAwareTool):
    """Tool for creating new artifacts.

    Creates artifacts via the ArtifactRegistry, automatically capturing
    the conversation ID and current wizard stage from context.

    Attributes:
        _registry: ArtifactRegistry for artifact storage
    """

    def __init__(self, artifact_registry: ArtifactRegistry) -> None:
        """Initialize the tool.

        Args:
            artifact_registry: Registry for artifact storage
        """
        super().__init__(
            name="create_artifact",
            description=(
                "Create a new artifact (document, configuration, or content) "
                "that will be tracked and optionally reviewed. Use this when "
                "producing outputs that should be preserved and validated."
            ),
        )
        self._registry = artifact_registry

    @property
    def schema(self) -> dict[str, Any]:
        """Return JSON Schema for tool parameters."""
        return {
            "type": "object",
            "properties": {
                "definition_id": {
                    "type": "string",
                    "description": (
                        "ID of the artifact definition to use. This determines "
                        "the artifact type and which reviews will be applied."
                    ),
                },
                "content": {
                    "type": ["object", "array", "string"],
                    "description": "The artifact content (structured data or text)",
                },
                "name": {
                    "type": "string",
                    "description": "Human-readable name for the artifact",
                },
                "purpose": {
                    "type": "string",
                    "description": "Purpose or description of this artifact",
                },
            },
            "required": ["content", "name"],
        }

    async def execute_with_context(
        self,
        context: ToolExecutionContext,
        content: Any,
        name: str,
        definition_id: str | None = None,
        purpose: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Create a new artifact.

        Args:
            context: Execution context with conversation state
            content: Artifact content
            name: Artifact name
            definition_id: Optional definition ID
            purpose: Optional purpose description

        Returns:
            Dict with artifact_id and status
        """
        # Extract stage from wizard state if available
        stage: str | None = None
        if context.wizard_state and context.wizard_state.current_stage:
            stage = context.wizard_state.current_stage

        try:
            artifact = self._registry.create(
                content=content,
                name=name,
                definition_id=definition_id,
                stage=stage,
                purpose=purpose,
            )

            logger.info(
                "Created artifact %s (name=%s, stage=%s)",
                artifact.id,
                name,
                stage,
            )

            return {
                "artifact_id": artifact.id,
                "status": artifact.status,
                "name": artifact.name,
                "version": artifact.lineage.version if artifact.lineage else 1,
                "message": f"Created artifact '{name}'",
            }
        except Exception as e:
            logger.error("Failed to create artifact: %s", e)
            return {
                "error": str(e),
                "message": f"Failed to create artifact: {e}",
            }


class UpdateArtifactTool(ContextAwareTool):
    """Tool for updating existing artifacts.

    Updates artifact content, creating a new version. The original
    artifact is preserved in version history.

    Attributes:
        _registry: ArtifactRegistry for artifact storage
    """

    def __init__(self, artifact_registry: ArtifactRegistry) -> None:
        """Initialize the tool.

        Args:
            artifact_registry: Registry for artifact storage
        """
        super().__init__(
            name="update_artifact",
            description=(
                "Update an existing artifact with new content. This creates "
                "a new version while preserving the original."
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
                    "description": "ID of the artifact to update",
                },
                "content": {
                    "type": ["object", "array", "string"],
                    "description": "New content for the artifact",
                },
                "reason": {
                    "type": "string",
                    "description": "Reason for the update (for audit trail)",
                },
            },
            "required": ["artifact_id", "content"],
        }

    async def execute_with_context(
        self,
        context: ToolExecutionContext,
        artifact_id: str,
        content: Any,
        reason: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Update an artifact.

        Args:
            context: Execution context
            artifact_id: ID of artifact to update
            content: New content
            reason: Optional reason for update

        Returns:
            Dict with new artifact_id, version, and status
        """
        try:
            updated = self._registry.update(
                artifact_id=artifact_id,
                content=content,
                derived_from=reason,
            )

            logger.info(
                "Updated artifact %s -> %s (version %d)",
                artifact_id,
                updated.id,
                updated.lineage.version if updated.lineage else 1,
            )

            return {
                "artifact_id": updated.id,
                "previous_id": artifact_id,
                "version": updated.lineage.version if updated.lineage else 1,
                "status": updated.status,
                "message": f"Updated artifact to version {updated.lineage.version if updated.lineage else 1}",
            }
        except KeyError:
            return {
                "error": f"Artifact not found: {artifact_id}",
                "message": f"Artifact '{artifact_id}' not found",
            }
        except Exception as e:
            logger.error("Failed to update artifact %s: %s", artifact_id, e)
            return {
                "error": str(e),
                "message": f"Failed to update artifact: {e}",
            }


class QueryArtifactsTool(ContextAwareTool):
    """Tool for querying artifacts.

    Queries artifacts by various criteria including status, type,
    definition ID, and wizard stage.

    Attributes:
        _registry: ArtifactRegistry for artifact storage
    """

    def __init__(self, artifact_registry: ArtifactRegistry) -> None:
        """Initialize the tool.

        Args:
            artifact_registry: Registry for artifact storage
        """
        super().__init__(
            name="query_artifacts",
            description=(
                "Query artifacts by status, type, or stage. Use this to find "
                "artifacts that need review, are pending, or match other criteria."
            ),
        )
        self._registry = artifact_registry

    @property
    def schema(self) -> dict[str, Any]:
        """Return JSON Schema for tool parameters."""
        return {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": [
                        "draft",
                        "pending_review",
                        "approved",
                        "rejected",
                        "superseded",
                    ],
                    "description": "Filter by artifact status",
                },
                "artifact_type": {
                    "type": "string",
                    "description": "Filter by artifact type (e.g., 'content', 'config')",
                },
                "definition_id": {
                    "type": "string",
                    "description": "Filter by artifact definition ID",
                },
                "stage": {
                    "type": "string",
                    "description": "Filter by wizard stage where artifact was created",
                },
                "include_content": {
                    "type": "boolean",
                    "description": "Whether to include artifact content in results",
                    "default": False,
                },
            },
        }

    async def execute_with_context(
        self,
        context: ToolExecutionContext,
        status: str | None = None,
        artifact_type: str | None = None,
        definition_id: str | None = None,
        stage: str | None = None,
        include_content: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Query artifacts.

        Args:
            context: Execution context
            status: Optional status filter
            artifact_type: Optional type filter
            definition_id: Optional definition ID filter
            stage: Optional stage filter
            include_content: Whether to include content

        Returns:
            Dict with list of matching artifacts
        """
        artifacts: list[dict[str, Any]] = []

        # Get all artifacts and filter
        # Use specific query methods when available
        if status:
            found = self._registry.get_by_status(status)
        elif artifact_type:
            found = self._registry.get_by_type(artifact_type)
        elif definition_id:
            found = self._registry.get_by_definition(definition_id)
        elif stage:
            found = self._registry.get_by_stage(stage)
        else:
            # Get all artifacts
            found = list(self._registry._artifacts.values())

        # Apply additional filters
        for artifact in found:
            if status and artifact.status != status:
                continue
            if artifact_type and artifact.type != artifact_type:
                continue
            if definition_id and artifact.definition_id != definition_id:
                continue
            artifact_stage = artifact.metadata.stage if artifact.metadata else None
            if stage and artifact_stage != stage:
                continue

            artifact_data: dict[str, Any] = {
                "id": artifact.id,
                "name": artifact.name,
                "status": artifact.status,
                "type": artifact.type,
                "definition_id": artifact.definition_id,
                "stage": artifact_stage,
                "version": artifact.lineage.version if artifact.lineage else 1,
                "created_at": artifact.metadata.created_at if artifact.metadata else None,
            }
            if include_content:
                artifact_data["content"] = artifact.content

            artifacts.append(artifact_data)

        return {
            "artifacts": artifacts,
            "count": len(artifacts),
            "filters": {
                "status": status,
                "artifact_type": artifact_type,
                "definition_id": definition_id,
                "stage": stage,
            },
        }


class SubmitForReviewTool(ContextAwareTool):
    """Tool for submitting artifacts for review.

    Changes artifact status to pending_review and prepares it
    for review execution.

    Attributes:
        _registry: ArtifactRegistry for artifact storage
    """

    def __init__(self, artifact_registry: ArtifactRegistry) -> None:
        """Initialize the tool.

        Args:
            artifact_registry: Registry for artifact storage
        """
        super().__init__(
            name="submit_for_review",
            description=(
                "Submit an artifact for review. This changes the artifact status "
                "to pending_review and triggers any configured review protocols."
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
                    "description": "ID of the artifact to submit for review",
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
        """Submit artifact for review.

        Args:
            context: Execution context
            artifact_id: ID of artifact to submit

        Returns:
            Dict with updated status
        """
        try:
            artifact = self._registry.submit_for_review(artifact_id)

            logger.info("Submitted artifact %s for review", artifact_id)

            return {
                "artifact_id": artifact.id,
                "status": artifact.status,
                "message": f"Artifact '{artifact.name}' submitted for review",
            }
        except KeyError:
            return {
                "error": f"Artifact not found: {artifact_id}",
                "message": f"Artifact '{artifact_id}' not found",
            }
        except ValueError as e:
            # Invalid status transition
            return {
                "error": str(e),
                "message": f"Cannot submit artifact for review: {e}",
            }
        except Exception as e:
            logger.error("Failed to submit artifact %s for review: %s", artifact_id, e)
            return {
                "error": str(e),
                "message": f"Failed to submit artifact for review: {e}",
            }


class GetArtifactTool(ContextAwareTool):
    """Tool for retrieving a specific artifact.

    Retrieves an artifact by ID, including its content, metadata,
    and review history.

    Attributes:
        _registry: ArtifactRegistry for artifact storage
    """

    def __init__(self, artifact_registry: ArtifactRegistry) -> None:
        """Initialize the tool.

        Args:
            artifact_registry: Registry for artifact storage
        """
        super().__init__(
            name="get_artifact",
            description=(
                "Get a specific artifact by ID, including its content, "
                "status, and review history."
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
                    "description": "ID of the artifact to retrieve",
                },
                "include_reviews": {
                    "type": "boolean",
                    "description": "Whether to include review history",
                    "default": True,
                },
            },
            "required": ["artifact_id"],
        }

    async def execute_with_context(
        self,
        context: ToolExecutionContext,
        artifact_id: str,
        include_reviews: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Get an artifact.

        Args:
            context: Execution context
            artifact_id: ID of artifact to retrieve
            include_reviews: Whether to include reviews

        Returns:
            Dict with artifact data
        """
        artifact = self._registry.get(artifact_id)
        if not artifact:
            return {
                "error": f"Artifact not found: {artifact_id}",
                "message": f"Artifact '{artifact_id}' not found",
            }

        result: dict[str, Any] = {
            "id": artifact.id,
            "name": artifact.name,
            "status": artifact.status,
            "type": artifact.type,
            "content": artifact.content,
            "definition_id": artifact.definition_id,
            "stage": artifact.metadata.stage if artifact.metadata else None,
            "created_at": artifact.metadata.created_at if artifact.metadata else None,
        }

        if artifact.lineage:
            result["lineage"] = {
                "version": artifact.lineage.version,
                "parent_id": artifact.lineage.parent_id,
            }

        if artifact.metadata:
            result["metadata"] = {
                "purpose": artifact.metadata.purpose,
                "stage": artifact.metadata.stage,
                "tags": artifact.metadata.tags,
            }

        if include_reviews and artifact.reviews:
            result["reviews"] = [
                {
                    "id": r.id,
                    "reviewer": r.reviewer,
                    "passed": r.passed,
                    "score": r.score,
                    "review_type": r.review_type,
                    "issues": r.issues,
                    "suggestions": r.suggestions,
                    "timestamp": r.timestamp,
                }
                for r in artifact.reviews
            ]

        return result
