"""Rubric data models for structured evaluation of artifacts and content.

This module provides the core data structures for:
- Rubrics: Structured evaluation instruments with criteria, levels, and scoring methods
- Criteria: Individual dimensions of evaluation with quality levels
- Evaluations: Results of applying a rubric to a target

Rubrics support three scoring methods:
- DETERMINISTIC: Python functions evaluate against criteria
- SCHEMA: JSON Schema validation for structural checks
- LLM_DECODE: Narrow LLM classification (Phase 3)

Example:
    >>> rubric = Rubric(
    ...     id="rubric_001",
    ...     name="Content Quality",
    ...     description="Evaluates content quality",
    ...     version="1.0.0",
    ...     target_type="content",
    ...     criteria=[...],
    ...     pass_threshold=0.7,
    ... )
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class ScoringType(str, Enum):
    """How a rubric criterion is evaluated."""

    DETERMINISTIC = "deterministic"
    SCHEMA = "schema"
    LLM_DECODE = "llm_decode"


def _generate_id(prefix: str) -> str:
    """Generate a unique ID with the given prefix."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ScoringMethod:
    """Configuration for how a criterion is scored.

    Attributes:
        type: The scoring approach (deterministic, schema, or llm_decode).
        function_ref: For deterministic scoring, a dotted path to a callable
            in the format ``"module.path:function"``.
        schema_ref: For schema scoring, a reference to an external JSON Schema.
        schema: For schema scoring, an inline JSON Schema dict.
        decode_prompt: For llm_decode scoring, the prompt template (Phase 3).
        decode_output_schema: For llm_decode scoring, the expected output
            schema (Phase 3).
    """

    type: ScoringType
    function_ref: str | None = None
    schema_ref: str | None = None
    schema: dict[str, Any] | None = None
    decode_prompt: str | None = None
    decode_output_schema: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        result: dict[str, Any] = {"type": self.type.value}
        if self.function_ref is not None:
            result["function_ref"] = self.function_ref
        if self.schema_ref is not None:
            result["schema_ref"] = self.schema_ref
        if self.schema is not None:
            result["schema"] = self.schema
        if self.decode_prompt is not None:
            result["decode_prompt"] = self.decode_prompt
        if self.decode_output_schema is not None:
            result["decode_output_schema"] = self.decode_output_schema
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScoringMethod:
        """Deserialize from a dictionary."""
        return cls(
            type=ScoringType(data["type"]),
            function_ref=data.get("function_ref"),
            schema_ref=data.get("schema_ref"),
            schema=data.get("schema"),
            decode_prompt=data.get("decode_prompt"),
            decode_output_schema=data.get("decode_output_schema"),
        )


@dataclass
class RubricLevel:
    """A quality level within a rubric criterion.

    Levels represent distinct quality tiers (e.g., "excellent", "pass", "fail")
    with associated numeric scores for aggregation.

    Attributes:
        id: Level identifier (e.g., "excellent", "pass", "fail").
        label: Human-readable display label.
        description: What qualifies content for this level.
        score: Numeric score for this level (0.0 to 1.0).
        indicators: Observable indicators for this quality level.
    """

    id: str
    label: str
    description: str
    score: float
    indicators: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        return {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "score": self.score,
            "indicators": self.indicators,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RubricLevel:
        """Deserialize from a dictionary."""
        return cls(
            id=data["id"],
            label=data["label"],
            description=data["description"],
            score=data["score"],
            indicators=data.get("indicators", []),
        )


@dataclass
class RubricCriterion:
    """A single dimension of evaluation within a rubric.

    Each criterion defines what is being measured, how it is scored,
    and the possible quality levels.

    Attributes:
        id: Unique identifier for this criterion.
        name: Human-readable name.
        description: What this criterion measures.
        weight: Relative importance (0.0 to 1.0). Weights across all
            criteria in a rubric should sum to 1.0.
        levels: Quality levels ordered from lowest to highest score.
        scoring_method: How to evaluate this criterion.
        required: Whether this criterion must be evaluated (default True).
    """

    id: str
    name: str
    description: str
    weight: float
    levels: list[RubricLevel]
    scoring_method: ScoringMethod
    required: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "weight": self.weight,
            "levels": [level.to_dict() for level in self.levels],
            "scoring_method": self.scoring_method.to_dict(),
            "required": self.required,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RubricCriterion:
        """Deserialize from a dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            weight=data["weight"],
            levels=[RubricLevel.from_dict(level) for level in data["levels"]],
            scoring_method=ScoringMethod.from_dict(data["scoring_method"]),
            required=data.get("required", True),
        )


@dataclass
class Rubric:
    """A structured evaluation instrument with criteria and scoring.

    A rubric defines how to evaluate a target (artifact, content, etc.)
    across multiple criteria, each with defined quality levels and
    scoring methods.

    Attributes:
        id: Unique rubric identifier.
        name: Human-readable name.
        description: What this rubric evaluates.
        version: Semantic version string (e.g., "1.0.0").
        target_type: What type of target this rubric evaluates
            (e.g., "content", "rubric", "assessment").
        criteria: List of evaluation criteria.
        pass_threshold: Minimum weighted score to pass (0.0 to 1.0).
        metadata: Additional metadata (tags, author, etc.).
    """

    id: str
    name: str
    description: str
    version: str
    target_type: str
    criteria: list[RubricCriterion]
    pass_threshold: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "target_type": self.target_type,
            "criteria": [c.to_dict() for c in self.criteria],
            "pass_threshold": self.pass_threshold,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Rubric:
        """Deserialize from a dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            version=data["version"],
            target_type=data["target_type"],
            criteria=[RubricCriterion.from_dict(c) for c in data["criteria"]],
            pass_threshold=data["pass_threshold"],
            metadata=data.get("metadata", {}),
        )


@dataclass
class CriterionResult:
    """The result of evaluating a single criterion.

    Attributes:
        criterion_id: ID of the criterion that was evaluated.
        level_id: Which quality level was assigned.
        score: Numeric score for this criterion.
        evidence: Supporting evidence for the assigned level.
        notes: Additional context or explanation.
        scoring_method_used: Which scoring approach was used.
    """

    criterion_id: str
    level_id: str
    score: float
    evidence: list[str] = field(default_factory=list)
    notes: str = ""
    scoring_method_used: ScoringType = ScoringType.DETERMINISTIC

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        return {
            "criterion_id": self.criterion_id,
            "level_id": self.level_id,
            "score": self.score,
            "evidence": self.evidence,
            "notes": self.notes,
            "scoring_method_used": self.scoring_method_used.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CriterionResult:
        """Deserialize from a dictionary."""
        return cls(
            criterion_id=data["criterion_id"],
            level_id=data["level_id"],
            score=data["score"],
            evidence=data.get("evidence", []),
            notes=data.get("notes", ""),
            scoring_method_used=ScoringType(data.get(
                "scoring_method_used", "deterministic"
            )),
        )


@dataclass
class RubricEvaluation:
    """The complete result of evaluating a target against a rubric.

    Attributes:
        id: Unique evaluation identifier.
        rubric_id: ID of the rubric used.
        rubric_version: Version of the rubric used.
        target_id: ID of the evaluated target.
        target_type: Type of the evaluated target.
        criterion_results: Results for each criterion.
        weighted_score: Aggregated weighted score (0.0 to 1.0).
        passed: Whether the target met the pass_threshold.
        feedback_summary: Human-readable summary of the evaluation.
        evaluated_at: ISO 8601 timestamp of evaluation.
        evaluated_by: Who or what performed the evaluation
            (e.g., "system", "user:xxx").
    """

    id: str = field(default_factory=lambda: _generate_id("eval"))
    rubric_id: str = ""
    rubric_version: str = ""
    target_id: str = ""
    target_type: str = ""
    criterion_results: list[CriterionResult] = field(default_factory=list)
    weighted_score: float = 0.0
    passed: bool = False
    feedback_summary: str = ""
    evaluated_at: str = field(default_factory=_now_iso)
    evaluated_by: str = "system"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        return {
            "id": self.id,
            "rubric_id": self.rubric_id,
            "rubric_version": self.rubric_version,
            "target_id": self.target_id,
            "target_type": self.target_type,
            "criterion_results": [r.to_dict() for r in self.criterion_results],
            "weighted_score": self.weighted_score,
            "passed": self.passed,
            "feedback_summary": self.feedback_summary,
            "evaluated_at": self.evaluated_at,
            "evaluated_by": self.evaluated_by,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RubricEvaluation:
        """Deserialize from a dictionary."""
        return cls(
            id=data["id"],
            rubric_id=data["rubric_id"],
            rubric_version=data["rubric_version"],
            target_id=data["target_id"],
            target_type=data["target_type"],
            criterion_results=[
                CriterionResult.from_dict(r) for r in data["criterion_results"]
            ],
            weighted_score=data["weighted_score"],
            passed=data["passed"],
            feedback_summary=data.get("feedback_summary", ""),
            evaluated_at=data["evaluated_at"],
            evaluated_by=data.get("evaluated_by", "system"),
        )
