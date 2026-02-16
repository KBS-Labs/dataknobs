"""Feedback summary generation for rubric evaluations.

This module provides functions to generate human-readable feedback summaries
from rubric evaluation results:
- ``generate_feedback_summary``: LLM-enhanced summary with deterministic fallback
- ``generate_deterministic_summary``: Pure template-based summary (no LLM)
- ``generate_criterion_feedback``: Per-criterion feedback string
"""

from __future__ import annotations

import logging
from typing import Any

from dataknobs_llm.llm.base import AsyncLLMProvider, LLMMessage

from .models import (
    CriterionResult,
    Rubric,
    RubricCriterion,
    RubricEvaluation,
)

logger = logging.getLogger(__name__)


async def generate_feedback_summary(
    rubric: Rubric,
    evaluation: RubricEvaluation,
    llm: AsyncLLMProvider | None = None,
) -> str:
    """Generate a feedback summary for a rubric evaluation.

    If an LLM provider is available, produces a natural language summary by
    encoding the structured evaluation results through the LLM. Falls back
    to a deterministic template-based summary if no LLM is provided or if
    the LLM call fails.

    Args:
        rubric: The rubric used for evaluation.
        evaluation: The completed evaluation results.
        llm: Optional LLM provider for enhanced summaries.

    Returns:
        A human-readable feedback summary string.
    """
    if llm is not None:
        try:
            return await _generate_llm_summary(rubric, evaluation, llm)
        except Exception as e:
            logger.warning(
                "LLM feedback generation failed, falling back to deterministic: %s",
                e,
            )

    return generate_deterministic_summary(rubric, evaluation)


def generate_deterministic_summary(
    rubric: Rubric,
    evaluation: RubricEvaluation,
) -> str:
    """Generate a deterministic template-based feedback summary.

    Produces a structured summary with:
    - Overall pass/fail status and weighted score
    - Per-criterion results with level and score
    - Issues found (criteria below threshold)
    - Suggestions for improvement

    Args:
        rubric: The rubric used for evaluation.
        evaluation: The completed evaluation results.

    Returns:
        A structured feedback summary string.
    """
    status = "PASSED" if evaluation.passed else "FAILED"
    lines = [f"Evaluation {status} for rubric '{rubric.name}'."]
    lines.append(f"Overall score: {evaluation.weighted_score:.0%}")
    lines.append("")

    result_by_id = {r.criterion_id: r for r in evaluation.criterion_results}
    criterion_by_id = {c.id: c for c in rubric.criteria}

    issues: list[str] = []
    for criterion in rubric.criteria:
        result = result_by_id.get(criterion.id)
        if result is None:
            continue
        feedback = generate_criterion_feedback(criterion, result)
        lines.append(f"- {feedback}")

        if result.score < rubric.pass_threshold and result.level_id != "error":
            _add_improvement_suggestion(criterion, result, criterion_by_id, issues)

    if issues:
        lines.append("")
        lines.append("Suggestions for improvement:")
        for issue in issues:
            lines.append(f"  - {issue}")

    return "\n".join(lines)


def generate_criterion_feedback(
    criterion: RubricCriterion,
    result: CriterionResult,
) -> str:
    """Generate a single-line feedback string for one criterion result.

    Args:
        criterion: The criterion definition.
        result: The evaluation result for this criterion.

    Returns:
        A formatted feedback string like "Criterion Name: pass (score: 0.70)".
    """
    parts = [f"{criterion.name}: {result.level_id} (score: {result.score:.2f})"]

    if result.level_id == "error":
        parts.append(f" [{result.notes}]")
    elif result.level_id == "unable_to_evaluate":
        parts.append(" [LLM evaluation inconclusive]")

    return "".join(parts)


def _add_improvement_suggestion(
    criterion: RubricCriterion,
    result: CriterionResult,
    criterion_by_id: dict[str, RubricCriterion],
    issues: list[str],
) -> None:
    """Add improvement suggestions for criteria scoring below threshold."""
    sorted_levels = sorted(criterion.levels, key=lambda level: level.score)
    current_idx = next(
        (i for i, level in enumerate(sorted_levels) if level.id == result.level_id),
        -1,
    )
    if current_idx >= 0 and current_idx < len(sorted_levels) - 1:
        next_level = sorted_levels[current_idx + 1]
        issues.append(
            f"{criterion.name}: aim for '{next_level.label}' — {next_level.description}"
        )
    elif result.level_id == "unable_to_evaluate":
        issues.append(f"{criterion.name}: could not be evaluated by LLM")


async def _generate_llm_summary(
    rubric: Rubric,
    evaluation: RubricEvaluation,
    llm: AsyncLLMProvider,
) -> str:
    """Generate an LLM-enhanced natural language summary."""
    criterion_by_id = {c.id: c for c in rubric.criteria}

    results_text = _build_results_text(rubric, evaluation, criterion_by_id)

    system_message = (
        "You are a feedback summarizer for content evaluation. "
        "Given structured evaluation results, produce a concise, helpful "
        "natural language summary. Focus on actionable feedback. "
        "Keep the summary to 3-5 sentences."
    )

    user_message = (
        f"Rubric: {rubric.name}\n"
        f"Description: {rubric.description}\n"
        f"Overall: {'PASSED' if evaluation.passed else 'FAILED'} "
        f"(score: {evaluation.weighted_score:.0%})\n\n"
        f"Criteria results:\n{results_text}"
    )

    messages = [
        LLMMessage(role="system", content=system_message),
        LLMMessage(role="user", content=user_message),
    ]

    response = await llm.complete(messages)
    return response.content.strip()


def _build_results_text(
    rubric: Rubric,
    evaluation: RubricEvaluation,
    criterion_by_id: dict[str, Any],
) -> str:
    """Build a text representation of evaluation results for the LLM prompt."""
    result_by_id = {r.criterion_id: r for r in evaluation.criterion_results}
    lines: list[str] = []

    for criterion in rubric.criteria:
        result = result_by_id.get(criterion.id)
        if result is None:
            continue
        lines.append(
            f"- {criterion.name} (weight: {criterion.weight:.0%}): "
            f"level={result.level_id}, score={result.score:.2f}"
        )
        if result.evidence:
            for evidence in result.evidence[:3]:
                lines.append(f"  Evidence: {evidence[:200]}")

    return "\n".join(lines)
