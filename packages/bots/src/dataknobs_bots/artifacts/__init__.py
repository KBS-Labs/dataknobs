"""Artifact management for conversational workflows.

This package provides infrastructure for tracking work products (artifacts)
produced during bot workflows, including:

- Artifact creation and versioning
- Lifecycle status management
- Review tracking
- Configuration-driven artifact definitions

Example:
    >>> from dataknobs_bots.artifacts import (
    ...     Artifact,
    ...     ArtifactRegistry,
    ...     ArtifactDefinition,
    ... )
    >>>
    >>> # Create a registry
    >>> registry = ArtifactRegistry()
    >>>
    >>> # Create an artifact
    >>> artifact = registry.create(
    ...     content={"questions": [...]},
    ...     name="Assessment Questions",
    ...     stage="build_questions",
    ... )
    >>>
    >>> # Query artifacts
    >>> pending = registry.get_pending_review()
"""

from .models import (
    Artifact,
    ArtifactDefinition,
    ArtifactLineage,
    ArtifactMetadata,
    ArtifactReview,
    ArtifactStatus,
    ArtifactType,
)
from .registry import ArtifactRegistry
from .tools import (
    CreateArtifactTool,
    GetArtifactTool,
    QueryArtifactsTool,
    SubmitForReviewTool,
    UpdateArtifactTool,
)

__all__ = [
    # Models
    "Artifact",
    "ArtifactDefinition",
    "ArtifactLineage",
    "ArtifactMetadata",
    "ArtifactReview",
    # Type literals
    "ArtifactStatus",
    "ArtifactType",
    # Registry
    "ArtifactRegistry",
    # Tools
    "CreateArtifactTool",
    "GetArtifactTool",
    "QueryArtifactsTool",
    "SubmitForReviewTool",
    "UpdateArtifactTool",
]
