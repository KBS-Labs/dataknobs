"""Display helpers for rendering artifact and evaluation data as markdown.

Pure functions that format rubric evaluations, criterion details,
evaluation comparisons, and provenance chains into human-readable
markdown strings. Suitable for use in wizard templates, chat responses,
or reports.

Example:
    >>> from dataknobs_bots.rubrics.models import RubricEvaluation
    >>> summary = format_evaluation_summary(evaluation)
    >>> print(summary)
    ## Evaluation Summary
    ...
"""

from __future__ import annotations


from dataknobs_bots.rubrics.models import CriterionResult, RubricCriterion, RubricEvaluation

from .provenance import ProvenanceRecord


def format_evaluation_summary(evaluation: RubricEvaluation) -> str:
    """Format a rubric evaluation as a markdown summary.

    Includes overall score, pass/fail status, and per-criterion results
    with level assignments and scores.

    Args:
        evaluation: The evaluation to format.

    Returns:
        Markdown-formatted evaluation summary.
    """
    status = "PASSED" if evaluation.passed else "FAILED"
    lines: list[str] = [
        "## Evaluation Summary",
        "",
        f"**Rubric:** {evaluation.rubric_id} (v{evaluation.rubric_version})",
        f"**Score:** {evaluation.weighted_score:.1%}",
        f"**Result:** {status}",
        "",
    ]

    if evaluation.criterion_results:
        lines.append("### Criteria Results")
        lines.append("")
        lines.append("| Criterion | Level | Score |")
        lines.append("|-----------|-------|-------|")
        for result in evaluation.criterion_results:
            lines.append(
                f"| {result.criterion_id} | {result.level_id} "
                f"| {result.score:.1%} |"
            )
        lines.append("")

    if evaluation.feedback_summary:
        lines.append("### Feedback")
        lines.append("")
        lines.append(evaluation.feedback_summary)
        lines.append("")

    return "\n".join(lines)


def format_criterion_detail(
    result: CriterionResult,
    criterion: RubricCriterion | None = None,
) -> str:
    """Format a single criterion result with full detail.

    Includes the assigned level, score, evidence, and notes. When the
    criterion definition is provided, includes level description and
    indicators.

    Args:
        result: The criterion evaluation result.
        criterion: Optional criterion definition for additional context.

    Returns:
        Markdown-formatted criterion detail.
    """
    lines: list[str] = []

    header = criterion.name if criterion else result.criterion_id
    lines.append(f"### {header}")
    lines.append("")
    lines.append(f"**Level:** {result.level_id}")
    lines.append(f"**Score:** {result.score:.1%}")
    lines.append("")

    # Add level description if criterion is provided
    if criterion:
        for level in criterion.levels:
            if level.id == result.level_id:
                lines.append(f"*{level.description}*")
                lines.append("")
                if level.indicators:
                    lines.append("**Indicators:**")
                    for indicator in level.indicators:
                        lines.append(f"- {indicator}")
                    lines.append("")
                break

    if result.evidence:
        lines.append("**Evidence:**")
        for item in result.evidence:
            lines.append(f"- {item}")
        lines.append("")

    if result.notes:
        lines.append(f"**Notes:** {result.notes}")
        lines.append("")

    return "\n".join(lines)


def format_comparison(evaluations: list[RubricEvaluation]) -> str:
    """Format a side-by-side comparison of multiple evaluations.

    Shows a table comparing scores across evaluations, with score
    change indicators when there are exactly two evaluations.

    Args:
        evaluations: List of evaluations to compare.

    Returns:
        Markdown-formatted comparison table.
    """
    if not evaluations:
        return "No evaluations to compare."

    if len(evaluations) == 1:
        return format_evaluation_summary(evaluations[0])

    lines: list[str] = [
        "## Evaluation Comparison",
        "",
    ]

    # Overall scores
    lines.append("### Overall Scores")
    lines.append("")
    header = "| Metric |"
    separator = "|--------|"
    for i, _eval in enumerate(evaluations):
        header += f" v{i + 1} |"
        separator += "------|"
    lines.append(header)
    lines.append(separator)

    score_row = "| Score |"
    for evaluation in evaluations:
        score_row += f" {evaluation.weighted_score:.1%} |"
    lines.append(score_row)

    result_row = "| Result |"
    for evaluation in evaluations:
        result_row += f" {'Pass' if evaluation.passed else 'Fail'} |"
    lines.append(result_row)
    lines.append("")

    # Per-criterion comparison
    all_criteria: list[str] = []
    for evaluation in evaluations:
        for result in evaluation.criterion_results:
            if result.criterion_id not in all_criteria:
                all_criteria.append(result.criterion_id)

    if all_criteria:
        lines.append("### Per-Criterion Scores")
        lines.append("")
        header = "| Criterion |"
        separator = "|-----------|"
        for i, _eval in enumerate(evaluations):
            header += f" v{i + 1} |"
            separator += "------|"

        if len(evaluations) == 2:
            header += " Change |"
            separator += "--------|"

        lines.append(header)
        lines.append(separator)

        for criterion_id in all_criteria:
            row = f"| {criterion_id} |"
            scores: list[float | None] = []
            for evaluation in evaluations:
                score = _get_criterion_score(evaluation, criterion_id)
                scores.append(score)
                row += f" {score:.1%} |" if score is not None else " - |"

            if len(evaluations) == 2 and scores[0] is not None and scores[1] is not None:
                diff = scores[1] - scores[0]
                if diff > 0:
                    row += f" +{diff:.1%} |"
                elif diff < 0:
                    row += f" {diff:.1%} |"
                else:
                    row += " = |"

            lines.append(row)
        lines.append("")

    return "\n".join(lines)


def format_provenance_chain(provenance: ProvenanceRecord) -> str:
    """Format a provenance record as a timeline view.

    Shows creation context, source references, tool invocations,
    LLM invocations, and revision history.

    Args:
        provenance: The provenance record to format.

    Returns:
        Markdown-formatted provenance timeline.
    """
    lines: list[str] = [
        "## Provenance",
        "",
        f"**Created by:** {provenance.created_by}",
        f"**Method:** {provenance.creation_method}",
        f"**Created at:** {provenance.created_at}",
        "",
    ]

    if provenance.creation_context:
        lines.append("### Creation Context")
        lines.append("")
        for key, value in provenance.creation_context.items():
            lines.append(f"- **{key}:** {value}")
        lines.append("")

    if provenance.sources:
        lines.append("### Sources")
        lines.append("")
        for source in provenance.sources:
            location = f" ({source.source_location})" if source.source_location else ""
            lines.append(f"- **{source.source_type}:** {source.source_id}{location}")
            if source.relevance:
                lines.append(f"  Relevance: {source.relevance}")
            if source.excerpt:
                lines.append(f"  > {source.excerpt}")
        lines.append("")

    if provenance.tool_chain:
        lines.append("### Tool Chain")
        lines.append("")
        for tool in provenance.tool_chain:
            version = f" v{tool.tool_version}" if tool.tool_version else ""
            lines.append(f"- **{tool.tool_name}**{version} ({tool.timestamp})")
        lines.append("")

    if provenance.llm_invocations:
        lines.append("### LLM Invocations")
        lines.append("")
        for invocation in provenance.llm_invocations:
            model = f" ({invocation.model})" if invocation.model else ""
            lines.append(
                f"- **{invocation.purpose}**{model} ({invocation.timestamp})"
            )
        lines.append("")

    if provenance.revision_history:
        lines.append("### Revision History")
        lines.append("")
        for revision in provenance.revision_history:
            lines.append(
                f"- **v{revision.previous_version}** -> revision "
                f"({revision.timestamp})"
            )
            lines.append(f"  Reason: {revision.reason}")
            lines.append(f"  Triggered by: {revision.triggered_by}")
        lines.append("")

    return "\n".join(lines)


def _get_criterion_score(
    evaluation: RubricEvaluation, criterion_id: str
) -> float | None:
    """Get the score for a specific criterion from an evaluation."""
    for result in evaluation.criterion_results:
        if result.criterion_id == criterion_id:
            return result.score
    return None
