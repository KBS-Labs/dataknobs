"""Artifact data models for tracking work products in conversational workflows.

This module provides the core data structures for:
- Artifacts: Work products (documents, configurations, content) produced during bot workflows
- Reviews: Results of reviewing artifacts using various protocols
- Definitions: Configuration-defined artifact type specifications

Example:
    >>> artifact = Artifact(
    ...     type="content",
    ...     name="Assessment Questions",
    ...     content={"questions": [...]},
    ...     metadata=ArtifactMetadata(
    ...         stage="build_questions",
    ...         purpose="Questions for math assessment",
    ...     ),
    ... )
    >>> artifact.status
    'draft'
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal


# Type aliases for artifact status and type literals
ArtifactStatus = Literal[
    "draft",           # Initial creation
    "pending_review",  # Submitted for review
    "in_review",       # Review in progress
    "needs_revision",  # Review found issues
    "approved",        # Passed all required reviews
    "rejected",        # Failed review, won't proceed
    "superseded",      # Replaced by newer version
]

ArtifactType = Literal[
    "planning",        # Plans, outlines, task breakdowns
    "content",         # Text content, documents
    "data",            # Structured data (JSON, YAML)
    "config",          # Configuration files
    "code",            # Code snippets, scripts
    "composite",       # Artifact containing other artifacts
]


def _generate_artifact_id() -> str:
    """Generate a unique artifact ID."""
    return f"art_{uuid.uuid4().hex[:12]}"


def _generate_review_id() -> str:
    """Generate a unique review ID."""
    return f"rev_{uuid.uuid4().hex[:12]}"


@dataclass
class ArtifactMetadata:
    """Metadata about artifact creation and context.

    Attributes:
        created_at: Unix timestamp of creation
        created_by: Who/what created this (bot_id, tool_name, etc.)
        stage: Wizard stage that produced this artifact
        task_id: Task that produced this artifact
        purpose: Description of why this artifact was created
        tags: Searchable tags for categorization
        custom: Additional custom metadata
    """

    created_at: float = field(default_factory=time.time)
    created_by: str | None = None
    stage: str | None = None
    task_id: str | None = None
    purpose: str | None = None
    tags: list[str] = field(default_factory=list)
    custom: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "created_at": self.created_at,
            "created_by": self.created_by,
            "stage": self.stage,
            "task_id": self.task_id,
            "purpose": self.purpose,
            "tags": self.tags,
            "custom": self.custom,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArtifactMetadata:
        """Deserialize from dictionary."""
        return cls(
            created_at=data.get("created_at", time.time()),
            created_by=data.get("created_by"),
            stage=data.get("stage"),
            task_id=data.get("task_id"),
            purpose=data.get("purpose"),
            tags=data.get("tags", []),
            custom=data.get("custom", {}),
        )


@dataclass
class ArtifactLineage:
    """Tracks artifact derivation and relationships.

    Attributes:
        parent_id: ID of artifact this was derived from (for versions)
        source_ids: IDs of artifacts used as input to create this
        version: Version number (1, 2, 3, ...)
        derived_from: Description of how this was created
    """

    parent_id: str | None = None
    source_ids: list[str] = field(default_factory=list)
    version: int = 1
    derived_from: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "parent_id": self.parent_id,
            "source_ids": self.source_ids,
            "version": self.version,
            "derived_from": self.derived_from,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArtifactLineage:
        """Deserialize from dictionary."""
        return cls(
            parent_id=data.get("parent_id"),
            source_ids=data.get("source_ids", []),
            version=data.get("version", 1),
            derived_from=data.get("derived_from"),
        )


@dataclass
class ArtifactReview:
    """Result of reviewing an artifact.

    Attributes:
        id: Unique review identifier
        artifact_id: ID of reviewed artifact
        reviewer: Reviewer identifier (persona name or validator)
        review_type: Type of review (persona, schema, custom)
        passed: Whether the review passed
        score: Optional numeric score (0.0 to 1.0)
        feedback: List of feedback items
        suggestions: List of improvement suggestions
        issues: List of specific issues found
        timestamp: When the review was performed
        metadata: Additional review metadata
    """

    id: str = field(default_factory=_generate_review_id)
    artifact_id: str = ""
    reviewer: str = ""
    review_type: str = "persona"  # "persona", "schema", "custom"
    passed: bool = False
    score: float | None = None
    feedback: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "artifact_id": self.artifact_id,
            "reviewer": self.reviewer,
            "review_type": self.review_type,
            "passed": self.passed,
            "score": self.score,
            "feedback": self.feedback,
            "suggestions": self.suggestions,
            "issues": self.issues,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArtifactReview:
        """Deserialize from dictionary."""
        return cls(
            id=data.get("id", _generate_review_id()),
            artifact_id=data.get("artifact_id", ""),
            reviewer=data.get("reviewer", ""),
            review_type=data.get("review_type", "persona"),
            passed=data.get("passed", False),
            score=data.get("score"),
            feedback=data.get("feedback", []),
            suggestions=data.get("suggestions", []),
            issues=data.get("issues", []),
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Artifact:
    """A work product from a conversational workflow.

    Artifacts represent any intermediate or final output that needs
    to be tracked, reviewed, or used in subsequent steps.

    Attributes:
        id: Unique identifier
        type: Type of artifact (planning, content, config, etc.)
        name: Human-readable name
        content: The actual artifact content
        content_type: MIME type or format (application/json, text/plain, etc.)
        status: Current lifecycle status
        schema_id: Optional reference to validation schema
        metadata: Creation and context metadata
        lineage: Version and derivation tracking
        reviews: List of review results
        definition_id: Reference to artifact definition in config

    Example:
        >>> artifact = Artifact(
        ...     type="content",
        ...     name="Assessment Questions",
        ...     content={"questions": [...]},
        ...     content_type="application/json",
        ...     metadata=ArtifactMetadata(
        ...         stage="build_questions",
        ...         task_id="generate_questions",
        ...         purpose="Questions for math assessment",
        ...     ),
        ... )
    """

    id: str = field(default_factory=_generate_artifact_id)
    type: ArtifactType = "content"
    name: str = ""
    content: Any = None
    content_type: str = "application/json"
    status: ArtifactStatus = "draft"
    schema_id: str | None = None
    metadata: ArtifactMetadata = field(default_factory=ArtifactMetadata)
    lineage: ArtifactLineage = field(default_factory=ArtifactLineage)
    reviews: list[ArtifactReview] = field(default_factory=list)
    definition_id: str | None = None  # Reference to config definition

    @property
    def is_approved(self) -> bool:
        """Check if artifact is approved."""
        return self.status == "approved"

    @property
    def is_reviewable(self) -> bool:
        """Check if artifact can be reviewed."""
        return self.status in ("draft", "pending_review", "needs_revision")

    @property
    def latest_review(self) -> ArtifactReview | None:
        """Get most recent review."""
        return self.reviews[-1] if self.reviews else None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "content": self.content,
            "content_type": self.content_type,
            "status": self.status,
            "schema_id": self.schema_id,
            "metadata": self.metadata.to_dict(),
            "lineage": self.lineage.to_dict(),
            "reviews": [r.to_dict() for r in self.reviews],
            "definition_id": self.definition_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Artifact:
        """Deserialize from dictionary."""
        metadata_data = data.get("metadata", {})
        lineage_data = data.get("lineage", {})
        reviews_data = data.get("reviews", [])

        return cls(
            id=data.get("id", _generate_artifact_id()),
            type=data.get("type", "content"),
            name=data.get("name", ""),
            content=data.get("content"),
            content_type=data.get("content_type", "application/json"),
            status=data.get("status", "draft"),
            schema_id=data.get("schema_id"),
            metadata=ArtifactMetadata.from_dict(metadata_data),
            lineage=ArtifactLineage.from_dict(lineage_data),
            reviews=[ArtifactReview.from_dict(r) for r in reviews_data],
            definition_id=data.get("definition_id"),
        )


@dataclass
class ArtifactDefinition:
    """Configuration-defined artifact type.

    This is parsed from bot configuration and defines:
    - What type of artifact this is
    - What schema validates the content
    - Which review protocols apply
    - Approval requirements

    Attributes:
        id: Definition identifier (matches config key)
        type: Artifact type
        name: Human-readable name
        description: Description of this artifact type
        schema: JSON Schema for content validation
        schema_ref: Reference to external schema
        reviews: List of review protocol IDs to apply
        approval_threshold: Score threshold for approval (0.0-1.0)
        require_all_reviews: Whether all reviews must pass
        auto_submit_for_review: Auto-submit on creation
        tags: Default tags for artifacts of this type
    """

    id: str
    type: ArtifactType = "content"
    name: str = ""
    description: str = ""
    schema: dict[str, Any] | None = None
    schema_ref: str | None = None
    reviews: list[str] = field(default_factory=list)
    approval_threshold: float = 0.8
    require_all_reviews: bool = True
    auto_submit_for_review: bool = False
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_config(cls, definition_id: str, config: dict[str, Any]) -> ArtifactDefinition:
        """Create from configuration dict.

        Args:
            definition_id: The definition ID (config key)
            config: Configuration dictionary

        Returns:
            ArtifactDefinition instance
        """
        return cls(
            id=definition_id,
            type=config.get("type", "content"),
            name=config.get("name", definition_id),
            description=config.get("description", ""),
            schema=config.get("schema"),
            schema_ref=config.get("schema_ref"),
            reviews=config.get("reviews", []),
            approval_threshold=config.get("approval_threshold", 0.8),
            require_all_reviews=config.get("require_all_reviews", True),
            auto_submit_for_review=config.get("auto_submit_for_review", False),
            tags=config.get("tags", []),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "description": self.description,
            "schema": self.schema,
            "schema_ref": self.schema_ref,
            "reviews": self.reviews,
            "approval_threshold": self.approval_threshold,
            "require_all_reviews": self.require_all_reviews,
            "auto_submit_for_review": self.auto_submit_for_review,
            "tags": self.tags,
        }
