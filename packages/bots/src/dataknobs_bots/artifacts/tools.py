"""Artifact management tools for conversational workflows.

Context-aware tools for managing artifacts within conversations.
These tools integrate with the async ArtifactRegistry to provide
CRUD operations, review submission, and querying.

Tools:
- CreateArtifactTool: Create new artifacts with provenance
- UpdateArtifactTool: Revise existing artifact content
- QueryArtifactsTool: Query artifacts by status, type, or tags
- SubmitForReviewTool: Submit artifact for rubric-based review
- GetArtifactTool: Retrieve a specific artifact with details

Example:
    >>> from dataknobs_data.backends.memory import AsyncMemoryDatabase
    >>> registry = ArtifactRegistry(AsyncMemoryDatabase())
    >>> create_tool = CreateArtifactTool(artifact_registry=registry)
    >>> result = await create_tool.execute_with_context(
    ...     context=ToolExecutionContext.empty(),
    ...     content={"questions": [...]},
    ...     name="Unit 1 Questions",
    ... )
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from dataknobs_llm.tools.context import ToolExecutionContext
from dataknobs_llm.tools.context_aware import ContextAwareTool

from .models import ArtifactStatus
from .provenance import create_provenance

if TYPE_CHECKING:
    from .registry import ArtifactRegistry

logger = logging.getLogger(__name__)


class CreateArtifactTool(ContextAwareTool):
    """Tool for creating new artifacts.

    Creates artifacts via the async ArtifactRegistry, automatically
    building provenance from the execution context.
    """

    def __init__(self, artifact_registry: ArtifactRegistry) -> None:
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
                "artifact_type": {
                    "type": "string",
                    "description": "Type of artifact (e.g., 'content', 'config')",
                },
                "content": {
                    "type": "object",
                    "description": "The artifact content as structured data",
                },
                "name": {
                    "type": "string",
                    "description": "Human-readable name for the artifact",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags for categorization and search",
                },
            },
            "required": ["content", "name"],
        }

    async def execute_with_context(
        self,
        context: ToolExecutionContext,
        content: dict[str, Any],
        name: str,
        artifact_type: str = "content",
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Create a new artifact.

        Args:
            context: Execution context with conversation state.
            content: Artifact content dictionary.
            name: Human-readable artifact name.
            artifact_type: Type identifier (default: "content").
            tags: Optional tags for categorization.

        Returns:
            Dict with artifact_id, status, name, version, and message.
        """
        try:
            created_by = f"user:{context.user_id}" if context.user_id else "system:tool"
            provenance = create_provenance(
                created_by=created_by,
                creation_method="tool",
                creation_context={"tool": "create_artifact"},
            )

            artifact = await self._registry.create(
                artifact_type=artifact_type,
                name=name,
                content=content,
                provenance=provenance,
                tags=tags,
            )

            logger.info(
                "Created artifact %s (name=%s, type=%s)",
                artifact.id,
                name,
                artifact_type,
            )

            return {
                "artifact_id": artifact.id,
                "status": artifact.status.value,
                "name": artifact.name,
                "version": artifact.version,
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

    Creates a new version of an artifact via ``registry.revise()``.
    The original artifact is marked as superseded.
    """

    def __init__(self, artifact_registry: ArtifactRegistry) -> None:
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
                    "type": "object",
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
        content: dict[str, Any],
        reason: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Update an artifact by creating a new version.

        Args:
            context: Execution context.
            artifact_id: ID of artifact to update.
            content: New content dictionary.
            reason: Optional reason for the update.

        Returns:
            Dict with artifact_id, previous version, and status.
        """
        try:
            triggered_by = f"user:{context.user_id}" if context.user_id else "system:tool"
            updated = await self._registry.revise(
                artifact_id=artifact_id,
                new_content=content,
                reason=reason or "Updated via tool",
                triggered_by=triggered_by,
            )

            logger.info(
                "Updated artifact %s to v%s",
                artifact_id,
                updated.version,
            )

            return {
                "artifact_id": updated.id,
                "version": updated.version,
                "status": updated.status.value,
                "message": f"Updated artifact to version {updated.version}",
            }
        except ValueError as e:
            return {
                "error": str(e),
                "message": f"Artifact not found: {e}",
            }
        except Exception as e:
            logger.error("Failed to update artifact %s: %s", artifact_id, e)
            return {
                "error": str(e),
                "message": f"Failed to update artifact: {e}",
            }


class QueryArtifactsTool(ContextAwareTool):
    """Tool for querying artifacts.

    Queries artifacts by type, status, and/or tags using the async
    ``registry.query()`` method.
    """

    def __init__(self, artifact_registry: ArtifactRegistry) -> None:
        super().__init__(
            name="query_artifacts",
            description=(
                "Query artifacts by status, type, or tags. Use this to find "
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
                    "enum": [s.value for s in ArtifactStatus],
                    "description": "Filter by artifact status",
                },
                "artifact_type": {
                    "type": "string",
                    "description": "Filter by artifact type",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter by tags (artifact must have all)",
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
        tags: list[str] | None = None,
        include_content: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Query artifacts by criteria.

        Args:
            context: Execution context.
            status: Optional status filter.
            artifact_type: Optional type filter.
            tags: Optional tag filter (all must match).
            include_content: Whether to include content in results.

        Returns:
            Dict with matching artifacts list and count.
        """
        status_enum = ArtifactStatus(status) if status else None
        found = await self._registry.query(
            artifact_type=artifact_type,
            status=status_enum,
            tags=tags,
        )

        artifacts: list[dict[str, Any]] = []
        for artifact in found:
            artifact_data: dict[str, Any] = {
                "id": artifact.id,
                "name": artifact.name,
                "status": artifact.status.value,
                "type": artifact.type,
                "version": artifact.version,
                "tags": artifact.tags,
                "created_at": artifact.created_at,
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
                "tags": tags,
            },
        }


class SubmitForReviewTool(ContextAwareTool):
    """Tool for submitting artifacts for rubric-based review.

    Submits an artifact for evaluation via ``registry.submit_for_review()``,
    which transitions the artifact through the review lifecycle and
    returns evaluation results.
    """

    def __init__(self, artifact_registry: ArtifactRegistry) -> None:
        super().__init__(
            name="submit_for_review",
            description=(
                "Submit an artifact for review. This triggers rubric-based "
                "evaluation and updates the artifact status based on results."
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
            context: Execution context.
            artifact_id: ID of the artifact to submit.

        Returns:
            Dict with artifact_id, status, evaluations, and message.
        """
        try:
            evaluations = await self._registry.submit_for_review(artifact_id)

            # Get updated artifact to report final status
            artifact = await self._registry.get(artifact_id)
            status = artifact.status.value if artifact else "unknown"

            logger.info(
                "Submitted artifact %s for review, status=%s",
                artifact_id,
                status,
            )

            return {
                "artifact_id": artifact_id,
                "status": status,
                "evaluations": evaluations,
                "message": f"Review complete, status: {status}",
            }
        except ValueError as e:
            return {
                "error": str(e),
                "message": f"Cannot submit artifact for review: {e}",
            }
        except Exception as e:
            logger.error(
                "Failed to submit artifact %s for review: %s",
                artifact_id,
                e,
            )
            return {
                "error": str(e),
                "message": f"Failed to submit artifact for review: {e}",
            }


class GetArtifactTool(ContextAwareTool):
    """Tool for retrieving a specific artifact.

    Retrieves an artifact by ID with its content, provenance summary,
    and optional evaluation results.
    """

    def __init__(self, artifact_registry: ArtifactRegistry) -> None:
        super().__init__(
            name="get_artifact",
            description=(
                "Get a specific artifact by ID, including its content, "
                "status, and provenance information."
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
                "include_evaluations": {
                    "type": "boolean",
                    "description": "Whether to include evaluation results",
                    "default": False,
                },
            },
            "required": ["artifact_id"],
        }

    async def execute_with_context(
        self,
        context: ToolExecutionContext,
        artifact_id: str,
        include_evaluations: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Get an artifact by ID.

        Args:
            context: Execution context.
            artifact_id: ID of the artifact to retrieve.
            include_evaluations: Whether to include evaluation results.

        Returns:
            Dict with full artifact data.
        """
        artifact = await self._registry.get(artifact_id)
        if not artifact:
            return {
                "error": f"Artifact not found: {artifact_id}",
                "message": f"Artifact '{artifact_id}' not found",
            }

        result: dict[str, Any] = {
            "id": artifact.id,
            "name": artifact.name,
            "status": artifact.status.value,
            "type": artifact.type,
            "version": artifact.version,
            "content": artifact.content,
            "tags": artifact.tags,
            "created_at": artifact.created_at,
            "updated_at": artifact.updated_at,
            "provenance": {
                "created_by": artifact.provenance.created_by,
                "creation_method": artifact.provenance.creation_method,
                "created_at": artifact.provenance.created_at,
            },
        }

        if include_evaluations:
            result["evaluations"] = await self._registry.get_evaluations(
                artifact_id
            )

        return result
