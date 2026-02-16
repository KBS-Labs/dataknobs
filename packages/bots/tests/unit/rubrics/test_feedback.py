"""Tests for rubric feedback summary generation."""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_llm import EchoProvider

from dataknobs_bots.rubrics.feedback import (
    generate_criterion_feedback,
    generate_deterministic_summary,
    generate_feedback_summary,
)
from dataknobs_bots.rubrics.models import (
    CriterionResult,
    Rubric,
    RubricCriterion,
    RubricEvaluation,
    RubricLevel,
    ScoringMethod,
    ScoringType,
)

# --- Fixtures ---

FAIL_LEVEL = RubricLevel(
    id="fail", label="Fail", description="Does not meet criteria", score=0.0
)
PASS_LEVEL = RubricLevel(
    id="pass", label="Pass", description="Meets minimum criteria", score=0.7
)
EXCELLENT_LEVEL = RubricLevel(
    id="excellent", label="Excellent", description="Exceeds expectations", score=1.0
)

STANDARD_LEVELS = [FAIL_LEVEL, PASS_LEVEL, EXCELLENT_LEVEL]


def _make_rubric(
    criteria: list[RubricCriterion],
    pass_threshold: float = 0.7,
) -> Rubric:
    return Rubric(
        id="test_rubric",
        name="Test Rubric",
        description="A test rubric for content quality",
        version="1.0.0",
        target_type="content",
        criteria=criteria,
        pass_threshold=pass_threshold,
    )


def _make_criterion(
    criterion_id: str,
    name: str = "",
    weight: float = 1.0,
) -> RubricCriterion:
    return RubricCriterion(
        id=criterion_id,
        name=name or f"Criterion {criterion_id}",
        description=f"Test criterion {criterion_id}",
        weight=weight,
        levels=list(STANDARD_LEVELS),
        scoring_method=ScoringMethod(type=ScoringType.DETERMINISTIC, function_ref="test:func"),
    )


def _make_evaluation(
    rubric: Rubric,
    results: list[CriterionResult],
    weighted_score: float,
    passed: bool,
) -> RubricEvaluation:
    return RubricEvaluation(
        id="eval_test",
        rubric_id=rubric.id,
        rubric_version=rubric.version,
        target_id="target_1",
        target_type="content",
        criterion_results=results,
        weighted_score=weighted_score,
        passed=passed,
    )


# --- Deterministic Summary Tests ---


class TestDeterministicSummary:
    def test_passed_evaluation(self) -> None:
        criterion = _make_criterion("c1", name="Title Check")
        rubric = _make_rubric([criterion])
        result = CriterionResult(
            criterion_id="c1", level_id="pass", score=0.7,
            scoring_method_used=ScoringType.DETERMINISTIC,
        )
        evaluation = _make_evaluation(rubric, [result], 0.7, passed=True)

        summary = generate_deterministic_summary(rubric, evaluation)

        assert "PASSED" in summary
        assert "70%" in summary
        assert "Title Check" in summary
        assert "pass" in summary
        assert "0.70" in summary

    def test_failed_evaluation(self) -> None:
        criterion = _make_criterion("c1", name="Title Check")
        rubric = _make_rubric([criterion])
        result = CriterionResult(
            criterion_id="c1", level_id="fail", score=0.0,
            scoring_method_used=ScoringType.DETERMINISTIC,
        )
        evaluation = _make_evaluation(rubric, [result], 0.0, passed=False)

        summary = generate_deterministic_summary(rubric, evaluation)

        assert "FAILED" in summary
        assert "0%" in summary

    def test_includes_improvement_suggestions(self) -> None:
        criterion = _make_criterion("c1", name="Word Count")
        rubric = _make_rubric([criterion])
        result = CriterionResult(
            criterion_id="c1", level_id="fail", score=0.0,
            scoring_method_used=ScoringType.DETERMINISTIC,
        )
        evaluation = _make_evaluation(rubric, [result], 0.0, passed=False)

        summary = generate_deterministic_summary(rubric, evaluation)

        assert "Suggestions for improvement" in summary
        assert "aim for 'Pass'" in summary

    def test_no_suggestions_when_all_pass(self) -> None:
        criterion = _make_criterion("c1", name="Quality")
        rubric = _make_rubric([criterion])
        result = CriterionResult(
            criterion_id="c1", level_id="excellent", score=1.0,
            scoring_method_used=ScoringType.DETERMINISTIC,
        )
        evaluation = _make_evaluation(rubric, [result], 1.0, passed=True)

        summary = generate_deterministic_summary(rubric, evaluation)

        assert "Suggestions for improvement" not in summary

    def test_multiple_criteria(self) -> None:
        c1 = _make_criterion("c1", name="Title", weight=0.5)
        c2 = _make_criterion("c2", name="Content Length", weight=0.5)
        rubric = _make_rubric([c1, c2])
        results = [
            CriterionResult(criterion_id="c1", level_id="pass", score=0.7),
            CriterionResult(criterion_id="c2", level_id="fail", score=0.0),
        ]
        evaluation = _make_evaluation(rubric, results, 0.35, passed=False)

        summary = generate_deterministic_summary(rubric, evaluation)

        assert "Title" in summary
        assert "Content Length" in summary
        assert "FAILED" in summary

    def test_single_criterion(self) -> None:
        criterion = _make_criterion("c1", name="Check")
        rubric = _make_rubric([criterion])
        result = CriterionResult(criterion_id="c1", level_id="pass", score=0.7)
        evaluation = _make_evaluation(rubric, [result], 0.7, passed=True)

        summary = generate_deterministic_summary(rubric, evaluation)

        assert "Check" in summary
        assert "PASSED" in summary


# --- Criterion Feedback Tests ---


class TestCriterionFeedback:
    def test_normal_criterion(self) -> None:
        criterion = _make_criterion("c1", name="Quality Check")
        result = CriterionResult(criterion_id="c1", level_id="pass", score=0.7)

        feedback = generate_criterion_feedback(criterion, result)

        assert feedback == "Quality Check: pass (score: 0.70)"

    def test_error_criterion(self) -> None:
        criterion = _make_criterion("c1", name="Quality Check")
        result = CriterionResult(
            criterion_id="c1", level_id="error", score=0.0,
            notes="Something went wrong",
        )

        feedback = generate_criterion_feedback(criterion, result)

        assert "Quality Check" in feedback
        assert "error" in feedback
        assert "Something went wrong" in feedback

    def test_unable_to_evaluate(self) -> None:
        criterion = _make_criterion("c1", name="LLM Check")
        result = CriterionResult(
            criterion_id="c1", level_id="unable_to_evaluate", score=0.0,
        )

        feedback = generate_criterion_feedback(criterion, result)

        assert "LLM evaluation inconclusive" in feedback


# --- LLM-Enhanced Summary Tests ---


class TestLLMFeedbackSummary:
    async def test_llm_enhanced_summary(self) -> None:
        """With LLM provider, feedback uses LLM-generated text."""
        provider = EchoProvider({"provider": "echo", "model": "test"})
        provider.set_responses(["This content passes all quality checks with a strong score."])

        criterion = _make_criterion("c1", name="Quality")
        rubric = _make_rubric([criterion])
        result = CriterionResult(criterion_id="c1", level_id="pass", score=0.7)
        evaluation = _make_evaluation(rubric, [result], 0.7, passed=True)

        summary = await generate_feedback_summary(rubric, evaluation, llm=provider)

        assert provider.call_count == 1
        assert len(summary) > 0
        # The echo provider returns a predictable response
        assert "quality checks" in summary.lower()

    async def test_fallback_to_deterministic_when_no_llm(self) -> None:
        """Without LLM provider, falls back to deterministic summary."""
        criterion = _make_criterion("c1", name="Quality")
        rubric = _make_rubric([criterion])
        result = CriterionResult(criterion_id="c1", level_id="pass", score=0.7)
        evaluation = _make_evaluation(rubric, [result], 0.7, passed=True)

        summary = await generate_feedback_summary(rubric, evaluation, llm=None)

        assert "PASSED" in summary
        assert "Quality" in summary

    async def test_fallback_to_deterministic_on_llm_failure(self) -> None:
        """When LLM call fails, falls back to deterministic summary."""
        provider = EchoProvider({"provider": "echo", "model": "test"})
        # Set response function that raises an exception
        provider.set_response_function(
            lambda msgs: (_ for _ in ()).throw(RuntimeError("LLM unavailable"))
        )

        criterion = _make_criterion("c1", name="Quality")
        rubric = _make_rubric([criterion])
        result = CriterionResult(criterion_id="c1", level_id="pass", score=0.7)
        evaluation = _make_evaluation(rubric, [result], 0.7, passed=True)

        summary = await generate_feedback_summary(rubric, evaluation, llm=provider)

        # Should fall back to deterministic
        assert "PASSED" in summary
        assert "Quality" in summary

    async def test_llm_summary_receives_evaluation_context(self) -> None:
        """LLM receives structured evaluation data in its prompt."""
        provider = EchoProvider({"provider": "echo", "model": "test"})
        provider.set_responses(["Summary text"])

        c1 = _make_criterion("c1", name="Title Quality", weight=0.6)
        c2 = _make_criterion("c2", name="Content Depth", weight=0.4)
        rubric = _make_rubric([c1, c2])
        results = [
            CriterionResult(criterion_id="c1", level_id="pass", score=0.7),
            CriterionResult(criterion_id="c2", level_id="excellent", score=1.0),
        ]
        evaluation = _make_evaluation(rubric, results, 0.82, passed=True)

        await generate_feedback_summary(rubric, evaluation, llm=provider)

        # Check what was sent to the LLM
        call = provider.get_call(0)
        user_msg = call["messages"][1].content
        assert "Title Quality" in user_msg
        assert "Content Depth" in user_msg
        assert "PASSED" in user_msg
