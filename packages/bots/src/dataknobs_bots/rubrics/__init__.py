"""Rubric-based evaluation system for structured content assessment.

This package provides:
- **Models**: Data structures for rubrics, criteria, levels, and evaluations
- **Executor**: Evaluation engine supporting deterministic, schema, and LLM decode scoring
- **Feedback**: Summary generation (LLM-enhanced or deterministic fallback)
- **Registry**: Persistent rubric storage backed by AsyncDatabase
- **Meta-rubric**: Built-in validation for rubric structural quality
"""

from .executor import FunctionRegistry, RubricExecutor
from .feedback import (
    generate_criterion_feedback,
    generate_deterministic_summary,
    generate_feedback_summary,
)
from .meta import build_meta_rubric
from .models import (
    CriterionResult,
    Rubric,
    RubricCriterion,
    RubricEvaluation,
    RubricLevel,
    ScoringMethod,
    ScoringType,
)
from .registry import RubricRegistry

__all__ = [
    "CriterionResult",
    "FunctionRegistry",
    "Rubric",
    "RubricCriterion",
    "RubricEvaluation",
    "RubricExecutor",
    "RubricLevel",
    "RubricRegistry",
    "ScoringMethod",
    "ScoringType",
    "build_meta_rubric",
    "generate_criterion_feedback",
    "generate_deterministic_summary",
    "generate_feedback_summary",
]
