"""Artifact management for conversational workflows.

This package provides infrastructure for tracking work products (artifacts)
produced during bot workflows, including:

- Artifact creation with provenance tracking
- Lifecycle status management with transition enforcement
- Rubric-based evaluation
- Configuration-driven artifact type definitions

Example:
    >>> from dataknobs_bots.artifacts import Artifact, ArtifactStatus
    >>> from dataknobs_bots.artifacts.provenance import create_provenance
    >>>
    >>> artifact = Artifact(
    ...     type="content",
    ...     name="Assessment Questions",
    ...     content={"questions": [...]},
    ...     provenance=create_provenance(
    ...         created_by="bot:edubot",
    ...         creation_method="generator",
    ...     ),
    ... )
"""

from .models import (
    Artifact,
    ArtifactStatus,
    ArtifactTypeDefinition,
)
from .provenance import (
    LLMInvocation,
    ProvenanceRecord,
    RevisionRecord,
    SourceReference,
    ToolInvocation,
    create_provenance,
)
from .registry import ArtifactRegistry
from .transitions import ARTIFACT_STATUS, validate_transition

__all__ = [
    # Models
    "Artifact",
    "ArtifactStatus",
    "ArtifactTypeDefinition",
    # Provenance
    "LLMInvocation",
    "ProvenanceRecord",
    "RevisionRecord",
    "SourceReference",
    "ToolInvocation",
    "create_provenance",
    # Registry
    "ArtifactRegistry",
    # Transitions
    "ARTIFACT_STATUS",
    "validate_transition",
]
