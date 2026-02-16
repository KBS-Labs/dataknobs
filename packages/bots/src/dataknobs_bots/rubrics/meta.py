"""Meta-rubric: deterministic validation functions for evaluating rubrics.

This module provides pure, stateless functions that validate the structural
quality of rubric definitions. These functions are designed to be used as
deterministic scoring functions within a meta-rubric — a rubric that
evaluates other rubrics.

Each function follows the scoring function contract:
    ``(dict[str, Any]) -> str`` — receives rubric content, returns a level_id.

Example:
    >>> meta_rubric = build_meta_rubric()
    >>> # Evaluate another rubric using the meta-rubric
    >>> evaluation = await executor.evaluate(meta_rubric, some_rubric.to_dict())
"""

from __future__ import annotations

from itertools import pairwise
from typing import Any

from .models import (
    Rubric,
    RubricCriterion,
    RubricLevel,
    ScoringMethod,
    ScoringType,
)


def check_criteria_coverage(content: dict[str, Any]) -> str:
    """Check that a rubric has at least one criterion.

    Returns:
        ``"pass"`` if criteria list is non-empty, ``"fail"`` if empty.
    """
    criteria = content.get("criteria", [])
    if len(criteria) >= 1:
        return "pass"
    return "fail"


def check_criteria_independence(content: dict[str, Any]) -> str:
    """Check that no two criteria have identical names or descriptions.

    Returns:
        ``"pass"`` if all names and descriptions are unique.
        ``"partial"`` if names are unique but descriptions overlap.
        ``"fail"`` if any criterion names are duplicated.
    """
    criteria = content.get("criteria", [])
    if len(criteria) <= 1:
        return "pass"

    names = [c.get("name", "") for c in criteria]
    descriptions = [c.get("description", "") for c in criteria]

    if len(names) != len(set(names)):
        return "fail"

    if len(descriptions) != len(set(descriptions)):
        return "partial"

    return "pass"


def check_weight_distribution(content: dict[str, Any]) -> str:
    """Check that criterion weights sum to approximately 1.0.

    Uses a tolerance of 0.01.

    Returns:
        ``"pass"`` if the weight sum is within tolerance of 1.0,
        ``"fail"`` otherwise.
    """
    criteria = content.get("criteria", [])
    if not criteria:
        return "fail"

    total = sum(c.get("weight", 0.0) for c in criteria)
    if abs(total - 1.0) <= 0.01:
        return "pass"
    return "fail"


def check_threshold(content: dict[str, Any]) -> str:
    """Check that pass_threshold is achievable and non-trivial.

    Computes the maximum possible score (all criteria at highest level)
    and the minimum possible score (all criteria at lowest level),
    then checks whether the threshold falls in a reasonable range.

    Returns:
        ``"pass"`` if the threshold is reasonable.
        ``"too_high"`` if the threshold exceeds the maximum possible score.
        ``"too_low"`` if the threshold is at or below the minimum possible score.
    """
    criteria = content.get("criteria", [])
    threshold = content.get("pass_threshold", 0.0)

    if not criteria:
        return "fail"

    total_weight = sum(c.get("weight", 0.0) for c in criteria)
    if total_weight == 0:
        return "fail"

    max_weighted = 0.0
    min_weighted = 0.0
    for c in criteria:
        weight = c.get("weight", 0.0)
        levels = c.get("levels", [])
        if not levels:
            continue
        scores = [lev.get("score", 0.0) for lev in levels]
        max_weighted += weight * max(scores)
        min_weighted += weight * min(scores)

    max_score = max_weighted / total_weight
    min_score = min_weighted / total_weight

    if threshold > max_score:
        return "too_high"
    if threshold <= min_score:
        return "too_low"
    return "pass"


def check_level_ordering(content: dict[str, Any]) -> str:
    """Check that levels within each criterion are ordered by score.

    Levels should be consistently ordered (ascending or descending)
    across all criteria.

    Returns:
        ``"pass"`` if all criteria have consistently ordered levels.
        ``"fail"`` if any criterion has unordered levels.
    """
    criteria = content.get("criteria", [])
    if not criteria:
        return "pass"

    for c in criteria:
        levels = c.get("levels", [])
        if len(levels) <= 1:
            continue
        scores = [lev.get("score", 0.0) for lev in levels]
        is_ascending = all(a <= b for a, b in pairwise(scores))
        is_descending = all(a >= b for a, b in pairwise(scores))
        if not (is_ascending or is_descending):
            return "fail"

    return "pass"


# Module-level reference paths for use in the meta-rubric
_META_FUNCTION_REFS = {
    "criteria_coverage": "dataknobs_bots.rubrics.meta:check_criteria_coverage",
    "criteria_independence": "dataknobs_bots.rubrics.meta:check_criteria_independence",
    "weight_distribution": "dataknobs_bots.rubrics.meta:check_weight_distribution",
    "threshold": "dataknobs_bots.rubrics.meta:check_threshold",
    "level_ordering": "dataknobs_bots.rubrics.meta:check_level_ordering",
}

_PASS_FAIL_LEVELS = [
    RubricLevel(id="fail", label="Fail", description="Does not meet", score=0.0),
    RubricLevel(id="pass", label="Pass", description="Meets criteria", score=1.0),
]

_THRESHOLD_LEVELS = [
    RubricLevel(id="too_low", label="Too Low", description="Trivially passing", score=0.0),
    RubricLevel(id="too_high", label="Too High", description="Unachievable", score=0.0),
    RubricLevel(id="pass", label="Pass", description="Reasonable threshold", score=1.0),
]

_INDEPENDENCE_LEVELS = [
    RubricLevel(id="fail", label="Fail", description="Duplicate criteria names", score=0.0),
    RubricLevel(
        id="partial",
        label="Partial",
        description="Unique names but overlapping descriptions",
        score=0.5,
    ),
    RubricLevel(id="pass", label="Pass", description="All criteria independent", score=1.0),
]


def build_meta_rubric() -> Rubric:
    """Create a rubric for evaluating other rubrics.

    The meta-rubric checks structural quality of rubric definitions using
    five deterministic criteria: criteria coverage, criteria independence,
    weight distribution, threshold reasonableness, and level ordering.

    Returns:
        A Rubric configured for evaluating rubric definitions.
    """
    return Rubric(
        id="meta_rubric",
        name="Meta-Rubric",
        description="Validates the structural quality of rubric definitions",
        version="1.0.0",
        target_type="rubric",
        criteria=[
            RubricCriterion(
                id="criteria_coverage",
                name="Criteria Coverage",
                description="Rubric has at least one criterion",
                weight=0.2,
                levels=list(_PASS_FAIL_LEVELS),
                scoring_method=ScoringMethod(
                    type=ScoringType.DETERMINISTIC,
                    function_ref=_META_FUNCTION_REFS["criteria_coverage"],
                ),
            ),
            RubricCriterion(
                id="criteria_independence",
                name="Criteria Independence",
                description="No duplicate criterion names or descriptions",
                weight=0.2,
                levels=list(_INDEPENDENCE_LEVELS),
                scoring_method=ScoringMethod(
                    type=ScoringType.DETERMINISTIC,
                    function_ref=_META_FUNCTION_REFS["criteria_independence"],
                ),
            ),
            RubricCriterion(
                id="weight_distribution",
                name="Weight Distribution",
                description="Criterion weights sum to approximately 1.0",
                weight=0.2,
                levels=list(_PASS_FAIL_LEVELS),
                scoring_method=ScoringMethod(
                    type=ScoringType.DETERMINISTIC,
                    function_ref=_META_FUNCTION_REFS["weight_distribution"],
                ),
            ),
            RubricCriterion(
                id="threshold",
                name="Threshold Reasonableness",
                description="Pass threshold is achievable and non-trivial",
                weight=0.2,
                levels=list(_THRESHOLD_LEVELS),
                scoring_method=ScoringMethod(
                    type=ScoringType.DETERMINISTIC,
                    function_ref=_META_FUNCTION_REFS["threshold"],
                ),
            ),
            RubricCriterion(
                id="level_ordering",
                name="Level Ordering",
                description="Levels within each criterion are ordered by score",
                weight=0.2,
                levels=list(_PASS_FAIL_LEVELS),
                scoring_method=ScoringMethod(
                    type=ScoringType.DETERMINISTIC,
                    function_ref=_META_FUNCTION_REFS["level_ordering"],
                ),
            ),
        ],
        pass_threshold=0.7,
        metadata={"type": "meta", "built_in": True},
    )
