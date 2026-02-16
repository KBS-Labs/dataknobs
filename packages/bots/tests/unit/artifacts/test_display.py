"""Tests for artifact display helpers."""

from __future__ import annotations

from dataknobs_bots.artifacts.display import (
    format_comparison,
    format_criterion_detail,
    format_evaluation_summary,
    format_provenance_chain,
)
from dataknobs_bots.artifacts.provenance import (
    LLMInvocation,
    ProvenanceRecord,
    RevisionRecord,
    SourceReference,
    ToolInvocation,
    create_provenance,
)
from dataknobs_bots.rubrics.models import (
    CriterionResult,
    RubricCriterion,
    RubricEvaluation,
    RubricLevel,
    ScoringMethod,
    ScoringType,
)


# --- Helpers ---


def _make_evaluation(
    passed: bool = True,
    score: float = 0.85,
    rubric_id: str = "rubric_001",
    criterion_results: list[CriterionResult] | None = None,
) -> RubricEvaluation:
    return RubricEvaluation(
        id="eval_test",
        rubric_id=rubric_id,
        rubric_version="1.0.0",
        target_id="art_001",
        target_type="content",
        criterion_results=criterion_results or [
            CriterionResult(
                criterion_id="clarity",
                level_id="excellent",
                score=0.9,
                evidence=["Clear language"],
            ),
            CriterionResult(
                criterion_id="accuracy",
                level_id="pass",
                score=0.75,
                notes="Minor issues",
            ),
        ],
        weighted_score=score,
        passed=passed,
        feedback_summary="Good overall quality." if passed else "Needs improvement.",
    )


def _make_criterion() -> RubricCriterion:
    return RubricCriterion(
        id="clarity",
        name="Clarity",
        description="How clear the content is",
        weight=0.5,
        levels=[
            RubricLevel(
                id="excellent",
                label="Excellent",
                description="Exceptionally clear and well-organized",
                score=1.0,
                indicators=["Easy to follow", "Logical flow"],
            ),
            RubricLevel(
                id="pass",
                label="Pass",
                description="Adequately clear",
                score=0.7,
            ),
            RubricLevel(
                id="fail",
                label="Fail",
                description="Unclear or disorganized",
                score=0.0,
            ),
        ],
        scoring_method=ScoringMethod(type=ScoringType.DETERMINISTIC),
    )


# --- format_evaluation_summary Tests ---


class TestFormatEvaluationSummary:
    def test_passed_evaluation(self) -> None:
        evaluation = _make_evaluation(passed=True, score=0.85)
        result = format_evaluation_summary(evaluation)

        assert "## Evaluation Summary" in result
        assert "PASSED" in result
        assert "85.0%" in result
        assert "rubric_001" in result
        assert "clarity" in result
        assert "accuracy" in result

    def test_failed_evaluation(self) -> None:
        evaluation = _make_evaluation(passed=False, score=0.4)
        result = format_evaluation_summary(evaluation)

        assert "FAILED" in result
        assert "40.0%" in result

    def test_includes_feedback(self) -> None:
        evaluation = _make_evaluation(passed=True)
        result = format_evaluation_summary(evaluation)

        assert "### Feedback" in result
        assert "Good overall quality." in result

    def test_empty_criterion_results(self) -> None:
        evaluation = RubricEvaluation(
            rubric_id="rubric_001",
            rubric_version="1.0.0",
            weighted_score=0.0,
            passed=False,
            criterion_results=[],
        )
        result = format_evaluation_summary(evaluation)

        assert "## Evaluation Summary" in result
        assert "Criteria Results" not in result


# --- format_criterion_detail Tests ---


class TestFormatCriterionDetail:
    def test_basic_criterion(self) -> None:
        result_data = CriterionResult(
            criterion_id="clarity",
            level_id="excellent",
            score=0.9,
            evidence=["Clear language", "Good structure"],
            notes="Well done",
        )
        result = format_criterion_detail(result_data)

        assert "### clarity" in result
        assert "**Level:** excellent" in result
        assert "90.0%" in result
        assert "Clear language" in result
        assert "Good structure" in result
        assert "Well done" in result

    def test_with_criterion_definition(self) -> None:
        result_data = CriterionResult(
            criterion_id="clarity",
            level_id="excellent",
            score=0.9,
        )
        criterion = _make_criterion()
        result = format_criterion_detail(result_data, criterion)

        assert "### Clarity" in result  # Uses criterion.name
        assert "Exceptionally clear" in result
        assert "Easy to follow" in result
        assert "Logical flow" in result

    def test_without_evidence(self) -> None:
        result_data = CriterionResult(
            criterion_id="test",
            level_id="pass",
            score=0.7,
        )
        result = format_criterion_detail(result_data)

        assert "### test" in result
        assert "**Evidence:**" not in result

    def test_without_notes(self) -> None:
        result_data = CriterionResult(
            criterion_id="test",
            level_id="pass",
            score=0.7,
        )
        result = format_criterion_detail(result_data)

        assert "**Notes:**" not in result


# --- format_comparison Tests ---


class TestFormatComparison:
    def test_two_evaluations(self) -> None:
        eval1 = _make_evaluation(passed=False, score=0.5, criterion_results=[
            CriterionResult(criterion_id="clarity", level_id="fail", score=0.3),
            CriterionResult(criterion_id="accuracy", level_id="pass", score=0.7),
        ])
        eval2 = _make_evaluation(passed=True, score=0.85, criterion_results=[
            CriterionResult(criterion_id="clarity", level_id="excellent", score=0.9),
            CriterionResult(criterion_id="accuracy", level_id="pass", score=0.8),
        ])
        result = format_comparison([eval1, eval2])

        assert "## Evaluation Comparison" in result
        assert "50.0%" in result
        assert "85.0%" in result
        assert "Change" in result
        assert "clarity" in result
        assert "accuracy" in result

    def test_single_evaluation_delegates(self) -> None:
        evaluation = _make_evaluation()
        result = format_comparison([evaluation])

        assert "## Evaluation Summary" in result

    def test_empty_list(self) -> None:
        result = format_comparison([])
        assert "No evaluations to compare" in result

    def test_three_evaluations_no_change_column(self) -> None:
        evals = [
            _make_evaluation(score=0.5, criterion_results=[
                CriterionResult(criterion_id="c1", level_id="pass", score=0.5),
            ]),
            _make_evaluation(score=0.7, criterion_results=[
                CriterionResult(criterion_id="c1", level_id="pass", score=0.7),
            ]),
            _make_evaluation(score=0.9, criterion_results=[
                CriterionResult(criterion_id="c1", level_id="excellent", score=0.9),
            ]),
        ]
        result = format_comparison(evals)

        assert "## Evaluation Comparison" in result
        # Change column only for 2 evaluations
        assert "Change" not in result


# --- format_provenance_chain Tests ---


class TestFormatProvenanceChain:
    def test_basic_provenance(self) -> None:
        provenance = create_provenance(
            created_by="user:alice",
            creation_method="wizard",
        )
        result = format_provenance_chain(provenance)

        assert "## Provenance" in result
        assert "user:alice" in result
        assert "wizard" in result

    def test_with_creation_context(self) -> None:
        provenance = create_provenance(
            created_by="system",
            creation_method="generator",
            creation_context={"generator_id": "quiz_gen", "version": "1.0.0"},
        )
        result = format_provenance_chain(provenance)

        assert "### Creation Context" in result
        assert "generator_id" in result
        assert "quiz_gen" in result

    def test_with_sources(self) -> None:
        provenance = ProvenanceRecord(
            created_by="system",
            creation_method="derived",
            sources=[
                SourceReference(
                    source_id="doc_001",
                    source_type="document",
                    source_location="/docs/source.md",
                    relevance="Primary reference",
                    excerpt="Key passage...",
                ),
            ],
        )
        result = format_provenance_chain(provenance)

        assert "### Sources" in result
        assert "doc_001" in result
        assert "document" in result
        assert "/docs/source.md" in result
        assert "Primary reference" in result
        assert "Key passage..." in result

    def test_with_tool_chain(self) -> None:
        provenance = ProvenanceRecord(
            created_by="system",
            creation_method="generator",
            tool_chain=[
                ToolInvocation(
                    tool_name="generator:quiz_gen",
                    tool_version="1.0.0",
                    parameters={"count": 5},
                ),
            ],
        )
        result = format_provenance_chain(provenance)

        assert "### Tool Chain" in result
        assert "generator:quiz_gen" in result
        assert "v1.0.0" in result

    def test_with_revision_history(self) -> None:
        provenance = ProvenanceRecord(
            created_by="system",
            creation_method="manual",
            revision_history=[
                RevisionRecord(
                    previous_version="1.0.0",
                    reason="Fixed typos",
                    changes_summary="Minor corrections",
                    triggered_by="user:bob",
                ),
            ],
        )
        result = format_provenance_chain(provenance)

        assert "### Revision History" in result
        assert "v1.0.0" in result
        assert "Fixed typos" in result
        assert "user:bob" in result

    def test_with_llm_invocations(self) -> None:
        provenance = ProvenanceRecord(
            created_by="system",
            creation_method="llm",
            llm_invocations=[
                LLMInvocation(
                    purpose="generate_content",
                    model="llama3.2",
                ),
            ],
        )
        result = format_provenance_chain(provenance)

        assert "### LLM Invocations" in result
        assert "generate_content" in result
        assert "llama3.2" in result

    def test_empty_provenance(self) -> None:
        provenance = ProvenanceRecord()
        result = format_provenance_chain(provenance)

        assert "## Provenance" in result
        # Should not have subsections for empty lists
        assert "### Sources" not in result
        assert "### Tool Chain" not in result
        assert "### Revision History" not in result
