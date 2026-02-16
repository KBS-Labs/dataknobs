"""Tests for rubric executor."""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_bots.rubrics.executor import FunctionRegistry, RubricExecutor
from dataknobs_bots.rubrics.models import (
    Rubric,
    RubricCriterion,
    RubricLevel,
    ScoringMethod,
    ScoringType,
)


# --- Test scoring functions ---


def score_has_title(target: dict[str, Any]) -> str:
    """Check if target has a non-empty title."""
    if target.get("title") and len(target["title"]) > 0:
        return "pass"
    return "fail"


def score_word_count(target: dict[str, Any]) -> str:
    """Check content word count."""
    content = target.get("content", "")
    words = len(content.split()) if content else 0
    if words >= 100:
        return "excellent"
    if words >= 50:
        return "pass"
    return "fail"


def score_always_raises(target: dict[str, Any]) -> str:
    """A scoring function that always raises."""
    raise RuntimeError("Intentional test error")


# --- Fixtures ---

FAIL_LEVEL = RubricLevel(
    id="fail", label="Fail", description="Does not meet", score=0.0
)
PASS_LEVEL = RubricLevel(
    id="pass", label="Pass", description="Meets", score=0.7
)
EXCELLENT_LEVEL = RubricLevel(
    id="excellent", label="Excellent", description="Exceeds", score=1.0
)

STANDARD_LEVELS = [FAIL_LEVEL, PASS_LEVEL, EXCELLENT_LEVEL]


def _make_deterministic_criterion(
    criterion_id: str,
    function_ref: str,
    weight: float = 1.0,
    levels: list[RubricLevel] | None = None,
) -> RubricCriterion:
    return RubricCriterion(
        id=criterion_id,
        name=f"Criterion {criterion_id}",
        description=f"Test criterion {criterion_id}",
        weight=weight,
        levels=levels or list(STANDARD_LEVELS),
        scoring_method=ScoringMethod(
            type=ScoringType.DETERMINISTIC,
            function_ref=function_ref,
        ),
    )


def _make_schema_criterion(
    criterion_id: str,
    schema: dict[str, Any],
    weight: float = 1.0,
) -> RubricCriterion:
    return RubricCriterion(
        id=criterion_id,
        name=f"Schema Criterion {criterion_id}",
        description=f"Schema check {criterion_id}",
        weight=weight,
        levels=list(STANDARD_LEVELS),
        scoring_method=ScoringMethod(
            type=ScoringType.SCHEMA,
            schema=schema,
        ),
    )


def _make_rubric(
    criteria: list[RubricCriterion],
    pass_threshold: float = 0.7,
) -> Rubric:
    return Rubric(
        id="test_rubric",
        name="Test Rubric",
        description="A test rubric",
        version="1.0.0",
        target_type="content",
        criteria=criteria,
        pass_threshold=pass_threshold,
    )


def _make_executor(
    functions: dict[str, Any] | None = None,
) -> RubricExecutor:
    registry = FunctionRegistry()
    if functions:
        for ref, func in functions.items():
            registry.register(ref, func)
    return RubricExecutor(function_registry=registry)


# --- FunctionRegistry Tests ---


class TestFunctionRegistry:
    def test_register_and_get(self) -> None:
        reg = FunctionRegistry()
        reg.register("test:func", score_has_title)
        assert reg.get("test:func") is score_has_title

    def test_get_unregistered_invalid_format(self) -> None:
        reg = FunctionRegistry()
        with pytest.raises(KeyError, match="expected 'module.path:function_name'"):
            reg.get("no_colon_here")

    def test_get_unregistered_bad_module(self) -> None:
        reg = FunctionRegistry()
        with pytest.raises(KeyError, match="Cannot import module"):
            reg.get("nonexistent.module:func")

    def test_dynamic_import_caches(self) -> None:
        reg = FunctionRegistry()
        # Use a real importable function
        reg.register("os.path:exists", None)  # Pre-register to test cache
        result = reg.get("os.path:exists")
        assert result is None  # Returns registered value, not import

    def test_dynamic_import_real_module(self) -> None:
        reg = FunctionRegistry()
        func = reg.get("os.path:join")
        import os.path

        assert func is os.path.join


# --- RubricExecutor Tests ---


class TestRubricExecutorDeterministic:
    async def test_single_criterion_pass(self) -> None:
        executor = _make_executor({"test:has_title": score_has_title})
        rubric = _make_rubric([
            _make_deterministic_criterion("c1", "test:has_title"),
        ])
        target = {"title": "My Document"}

        evaluation = await executor.evaluate(rubric, target, target_id="t1")

        assert evaluation.passed is True
        assert evaluation.weighted_score == pytest.approx(0.7)
        assert len(evaluation.criterion_results) == 1
        assert evaluation.criterion_results[0].level_id == "pass"

    async def test_single_criterion_fail(self) -> None:
        executor = _make_executor({"test:has_title": score_has_title})
        rubric = _make_rubric([
            _make_deterministic_criterion("c1", "test:has_title"),
        ])
        target = {"title": ""}

        evaluation = await executor.evaluate(rubric, target)

        assert evaluation.passed is False
        assert evaluation.weighted_score == pytest.approx(0.0)
        assert evaluation.criterion_results[0].level_id == "fail"

    async def test_multi_criteria_weighted(self) -> None:
        executor = _make_executor({
            "test:has_title": score_has_title,
            "test:word_count": score_word_count,
        })
        rubric = _make_rubric([
            _make_deterministic_criterion("c1", "test:has_title", weight=0.4),
            _make_deterministic_criterion("c2", "test:word_count", weight=0.6),
        ])
        target = {
            "title": "My Doc",
            "content": " ".join(["word"] * 120),
        }

        evaluation = await executor.evaluate(rubric, target)

        # c1: pass (0.7) * 0.4 = 0.28
        # c2: excellent (1.0) * 0.6 = 0.60
        # total = 0.88 / 1.0 = 0.88
        assert evaluation.weighted_score == pytest.approx(0.88)
        assert evaluation.passed is True

    async def test_multi_criteria_mixed_results(self) -> None:
        executor = _make_executor({
            "test:has_title": score_has_title,
            "test:word_count": score_word_count,
        })
        rubric = _make_rubric(
            [
                _make_deterministic_criterion(
                    "c1", "test:has_title", weight=0.5
                ),
                _make_deterministic_criterion(
                    "c2", "test:word_count", weight=0.5
                ),
            ],
            pass_threshold=0.5,
        )
        # title pass (0.7), word count fail (0.0)
        target = {"title": "Title", "content": "short"}

        evaluation = await executor.evaluate(rubric, target)

        # (0.7 * 0.5 + 0.0 * 0.5) / 1.0 = 0.35
        assert evaluation.weighted_score == pytest.approx(0.35)
        assert evaluation.passed is False

    async def test_threshold_edge_case_exact(self) -> None:
        executor = _make_executor({"test:has_title": score_has_title})
        rubric = _make_rubric(
            [_make_deterministic_criterion("c1", "test:has_title")],
            pass_threshold=0.7,
        )
        target = {"title": "Present"}

        evaluation = await executor.evaluate(rubric, target)

        assert evaluation.weighted_score == pytest.approx(0.7)
        assert evaluation.passed is True

    async def test_scoring_function_raises(self) -> None:
        executor = _make_executor({"test:raises": score_always_raises})
        rubric = _make_rubric([
            _make_deterministic_criterion("c1", "test:raises"),
        ])

        evaluation = await executor.evaluate(rubric, {})

        assert evaluation.criterion_results[0].level_id == "error"
        assert evaluation.criterion_results[0].score == 0.0
        assert "Intentional test error" in evaluation.criterion_results[0].notes

    async def test_missing_function_ref(self) -> None:
        executor = _make_executor({})
        criterion = RubricCriterion(
            id="c1",
            name="Test",
            description="Test",
            weight=1.0,
            levels=list(STANDARD_LEVELS),
            scoring_method=ScoringMethod(type=ScoringType.DETERMINISTIC),
        )
        rubric = _make_rubric([criterion])

        evaluation = await executor.evaluate(rubric, {})

        assert evaluation.criterion_results[0].level_id == "error"
        assert "no function_ref" in evaluation.criterion_results[0].notes

    async def test_unknown_level_id_returns_zero(self) -> None:
        def returns_unknown(target: dict[str, Any]) -> str:
            return "nonexistent_level"

        executor = _make_executor({"test:unknown": returns_unknown})
        rubric = _make_rubric([
            _make_deterministic_criterion("c1", "test:unknown"),
        ])

        evaluation = await executor.evaluate(rubric, {})

        assert evaluation.criterion_results[0].score == 0.0

    async def test_zero_weight_criterion(self) -> None:
        executor = _make_executor({
            "test:has_title": score_has_title,
            "test:word_count": score_word_count,
        })
        rubric = _make_rubric([
            _make_deterministic_criterion("c1", "test:has_title", weight=1.0),
            _make_deterministic_criterion("c2", "test:word_count", weight=0.0),
        ])
        target = {"title": "Title", "content": "short"}

        evaluation = await executor.evaluate(rubric, target)

        # c1: 0.7 * 1.0 = 0.7, c2: 0.0 * 0.0 = 0.0, total weight = 1.0
        assert evaluation.weighted_score == pytest.approx(0.7)

    async def test_evaluation_metadata(self) -> None:
        executor = _make_executor({"test:has_title": score_has_title})
        rubric = _make_rubric([
            _make_deterministic_criterion("c1", "test:has_title"),
        ])

        evaluation = await executor.evaluate(
            rubric, {"title": "T"}, target_id="art_123", target_type="doc"
        )

        assert evaluation.rubric_id == "test_rubric"
        assert evaluation.rubric_version == "1.0.0"
        assert evaluation.target_id == "art_123"
        assert evaluation.target_type == "doc"
        assert evaluation.id.startswith("eval_")
        assert evaluation.evaluated_at != ""

    async def test_default_target_type_from_rubric(self) -> None:
        executor = _make_executor({"test:has_title": score_has_title})
        rubric = _make_rubric([
            _make_deterministic_criterion("c1", "test:has_title"),
        ])

        evaluation = await executor.evaluate(rubric, {"title": "T"})

        assert evaluation.target_type == "content"


class TestRubricExecutorSchema:
    async def test_schema_validation_passes(self) -> None:
        schema = {
            "type": "object",
            "required": ["title"],
            "properties": {"title": {"type": "string"}},
        }
        executor = _make_executor()
        rubric = _make_rubric([_make_schema_criterion("c1", schema)])

        evaluation = await executor.evaluate(rubric, {"title": "Hello"})

        assert evaluation.criterion_results[0].level_id == "excellent"
        assert evaluation.criterion_results[0].score == 1.0
        assert evaluation.passed is True

    async def test_schema_validation_fails(self) -> None:
        schema = {
            "type": "object",
            "required": ["title"],
            "properties": {"title": {"type": "string"}},
        }
        executor = _make_executor()
        rubric = _make_rubric([_make_schema_criterion("c1", schema)])

        evaluation = await executor.evaluate(rubric, {"not_title": "Hello"})

        assert evaluation.criterion_results[0].level_id == "fail"
        assert evaluation.criterion_results[0].score == 0.0
        assert len(evaluation.criterion_results[0].evidence) > 0

    async def test_schema_missing_schema_definition(self) -> None:
        criterion = RubricCriterion(
            id="c1",
            name="Schema Check",
            description="Test",
            weight=1.0,
            levels=list(STANDARD_LEVELS),
            scoring_method=ScoringMethod(type=ScoringType.SCHEMA),
        )
        executor = _make_executor()
        rubric = _make_rubric([criterion])

        evaluation = await executor.evaluate(rubric, {})

        assert evaluation.criterion_results[0].level_id == "error"
        assert "no schema defined" in evaluation.criterion_results[0].notes


class TestRubricExecutorLLMDecode:
    async def test_llm_decode_not_implemented(self) -> None:
        criterion = RubricCriterion(
            id="c1",
            name="LLM Check",
            description="Test",
            weight=1.0,
            levels=list(STANDARD_LEVELS),
            scoring_method=ScoringMethod(type=ScoringType.LLM_DECODE),
        )
        executor = _make_executor()
        rubric = _make_rubric([criterion])

        evaluation = await executor.evaluate(rubric, {})

        assert evaluation.criterion_results[0].level_id == "error"
        assert "not yet implemented" in evaluation.criterion_results[0].notes


class TestRubricExecutorSummary:
    async def test_summary_contains_status(self) -> None:
        executor = _make_executor({"test:has_title": score_has_title})
        rubric = _make_rubric([
            _make_deterministic_criterion("c1", "test:has_title"),
        ])

        eval_pass = await executor.evaluate(rubric, {"title": "T"})
        assert "PASSED" in eval_pass.feedback_summary

        eval_fail = await executor.evaluate(rubric, {"title": ""})
        assert "FAILED" in eval_fail.feedback_summary

    async def test_summary_contains_criterion_results(self) -> None:
        executor = _make_executor({"test:has_title": score_has_title})
        rubric = _make_rubric([
            _make_deterministic_criterion("c1", "test:has_title"),
        ])

        evaluation = await executor.evaluate(rubric, {"title": "T"})

        assert "Criterion c1" in evaluation.feedback_summary
        assert "pass" in evaluation.feedback_summary
