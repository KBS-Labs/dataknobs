"""Artifact management for conversational workflows.

This package provides infrastructure for tracking work products (artifacts)
produced during bot workflows, including:

- Artifact creation with provenance tracking
- Lifecycle status management with transition enforcement
- Rubric-based evaluation
- Configuration-driven artifact type definitions
- Wizard transforms for artifact lifecycle operations
- Display helpers for rendering evaluation data
- Assessment session tracking

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

from .assessment import (
    AssessmentSession,
    CumulativePerformance,
    StudentResponse,
    finalize_assessment,
    record_response,
    start_assessment_session,
)
from .display import (
    format_comparison,
    format_criterion_detail,
    format_evaluation_summary,
    format_provenance_chain,
)
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
from .transforms import (
    TransformContext,
    approve_artifact,
    create_artifact,
    revise_artifact,
    save_artifact_draft,
    submit_for_review,
)
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
    # Transforms
    "TransformContext",
    "approve_artifact",
    "create_artifact",
    "revise_artifact",
    "save_artifact_draft",
    "submit_for_review",
    # Display
    "format_comparison",
    "format_criterion_detail",
    "format_evaluation_summary",
    "format_provenance_chain",
    # Assessment
    "AssessmentSession",
    "CumulativePerformance",
    "StudentResponse",
    "finalize_assessment",
    "record_response",
    "start_assessment_session",
]
