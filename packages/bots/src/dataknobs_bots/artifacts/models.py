"""Artifact data models for tracking work products in conversational workflows.

This module provides the core data structures for:
- Artifacts: Versioned work products with provenance tracking
- Status management: Enum-based lifecycle status
- Type definitions: Configuration-driven artifact type specifications

Example:
    >>> from dataknobs_bots.artifacts.provenance import create_provenance
    >>> artifact = Artifact(
    ...     type="content",
    ...     name="Assessment Questions",
    ...     content={"questions": [...]},
    ...     provenance=create_provenance(
    ...         created_by="bot:edubot",
    ...         creation_method="generator",
    ...     ),
    ... )
    >>> artifact.status
    <ArtifactStatus.DRAFT: 'draft'>
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from .provenance import ProvenanceRecord


class ArtifactStatus(str, Enum):
    """Lifecycle status of an artifact."""

    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    IN_REVIEW = "in_review"
    NEEDS_REVISION = "needs_revision"
    APPROVED = "approved"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"


def _generate_artifact_id() -> str:
    """Generate a unique artifact ID."""
    return f"art_{uuid.uuid4().hex[:12]}"


def _now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Artifact:
    """A versioned work product with provenance tracking.

    Artifacts represent intermediate or final outputs from conversational
    workflows. Each artifact carries full provenance, can be evaluated
    by rubrics, and follows a defined lifecycle.

    Attributes:
        id: Unique artifact identifier.
        type: Extensible type string (e.g., "content", "config", "assessment").
        name: Human-readable name.
        version: Semantic version string (e.g., "1.0.0").
        status: Current lifecycle status.
        content: The artifact content as a dictionary.
        content_schema: JSON Schema ID for content validation.
        provenance: Full provenance record.
        tags: Searchable tags for categorization.
        rubric_ids: IDs of rubrics applicable to this artifact type.
        evaluation_ids: IDs of completed RubricEvaluation results.
        created_at: ISO 8601 timestamp of creation.
        updated_at: ISO 8601 timestamp of last update.
    """

    id: str = field(default_factory=_generate_artifact_id)
    type: str = "content"
    name: str = ""
    version: str = "1.0.0"
    status: ArtifactStatus = ArtifactStatus.DRAFT
    content: dict[str, Any] = field(default_factory=dict)
    content_schema: str | None = None
    provenance: ProvenanceRecord = field(default_factory=ProvenanceRecord)
    tags: list[str] = field(default_factory=list)
    rubric_ids: list[str] = field(default_factory=list)
    evaluation_ids: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)

    @property
    def is_approved(self) -> bool:
        """Check if artifact is approved."""
        return self.status == ArtifactStatus.APPROVED

    @property
    def is_reviewable(self) -> bool:
        """Check if artifact can be submitted for review."""
        return self.status in (
            ArtifactStatus.DRAFT,
            ArtifactStatus.PENDING_REVIEW,
            ArtifactStatus.NEEDS_REVISION,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        return {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "version": self.version,
            "status": self.status.value,
            "content": self.content,
            "content_schema": self.content_schema,
            "provenance": self.provenance.to_dict(),
            "tags": self.tags,
            "rubric_ids": self.rubric_ids,
            "evaluation_ids": self.evaluation_ids,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Artifact:
        """Deserialize from a dictionary."""
        return cls(
            id=data.get("id", _generate_artifact_id()),
            type=data.get("type", "content"),
            name=data.get("name", ""),
            version=data.get("version", "1.0.0"),
            status=ArtifactStatus(data.get("status", "draft")),
            content=data.get("content", {}),
            content_schema=data.get("content_schema"),
            provenance=ProvenanceRecord.from_dict(
                data.get("provenance", {})
            ),
            tags=data.get("tags", []),
            rubric_ids=data.get("rubric_ids", []),
            evaluation_ids=data.get("evaluation_ids", []),
            created_at=data.get("created_at", _now_iso()),
            updated_at=data.get("updated_at", _now_iso()),
        )


@dataclass
class ArtifactTypeDefinition:
    """Configuration-driven artifact type specification.

    Defines default behaviors and constraints for artifacts of a
    given type, including which rubrics to apply and approval requirements.

    Attributes:
        id: Type identifier (e.g., "assessment_questions", "lesson_plan").
        description: Human-readable description of this type.
        content_schema: JSON Schema reference for content validation.
        rubrics: Default rubric IDs to apply for evaluation.
        auto_review: Automatically evaluate on creation.
        requires_approval: Must pass rubric evaluation to proceed.
        approval_threshold: Minimum score for approval (0.0 to 1.0).
        tags: Default tags for artifacts of this type.
    """

    id: str
    description: str = ""
    content_schema: str | None = None
    rubrics: list[str] = field(default_factory=list)
    auto_review: bool = False
    requires_approval: bool = False
    approval_threshold: float = 0.7
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_config(
        cls, type_id: str, config: dict[str, Any]
    ) -> ArtifactTypeDefinition:
        """Create from a configuration dictionary.

        Args:
            type_id: The type identifier.
            config: Configuration dictionary.

        Returns:
            An ArtifactTypeDefinition instance.
        """
        return cls(
            id=type_id,
            description=config.get("description", ""),
            content_schema=config.get("content_schema"),
            rubrics=config.get("rubrics", []),
            auto_review=config.get("auto_review", False),
            requires_approval=config.get("requires_approval", False),
            approval_threshold=config.get("approval_threshold", 0.7),
            tags=config.get("tags", []),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        return {
            "id": self.id,
            "description": self.description,
            "content_schema": self.content_schema,
            "rubrics": self.rubrics,
            "auto_review": self.auto_review,
            "requires_approval": self.requires_approval,
            "approval_threshold": self.approval_threshold,
            "tags": self.tags,
        }
