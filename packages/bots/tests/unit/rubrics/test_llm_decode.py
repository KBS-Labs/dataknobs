"""Tests for LLM decode scoring in the rubric executor."""

from __future__ import annotations

import json
from typing import Any

import pytest

from dataknobs_llm import EchoProvider

from dataknobs_bots.rubrics.executor import FunctionRegistry, RubricExecutor
from dataknobs_bots.rubrics.models import (
    Rubric,
    RubricCriterion,
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


def _make_llm_decode_criterion(
    criterion_id: str = "c1",
    decode_prompt: str = "Evaluate the quality of: {{ content }}",
    weight: float = 1.0,
    levels: list[RubricLevel] | None = None,
    decode_output_schema: dict[str, Any] | None = None,
) -> RubricCriterion:
    return RubricCriterion(
        id=criterion_id,
        name=f"LLM Criterion {criterion_id}",
        description=f"LLM-evaluated criterion {criterion_id}",
        weight=weight,
        levels=levels or list(STANDARD_LEVELS),
        scoring_method=ScoringMethod(
            type=ScoringType.LLM_DECODE,
            decode_prompt=decode_prompt,
            decode_output_schema=decode_output_schema,
        ),
    )


def _make_deterministic_criterion(
    criterion_id: str,
    function_ref: str,
    weight: float = 1.0,
) -> RubricCriterion:
    return RubricCriterion(
        id=criterion_id,
        name=f"Criterion {criterion_id}",
        description=f"Deterministic criterion {criterion_id}",
        weight=weight,
        levels=list(STANDARD_LEVELS),
        scoring_method=ScoringMethod(
            type=ScoringType.DETERMINISTIC,
            function_ref=function_ref,
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


def score_has_title(target: dict[str, Any]) -> str:
    if target.get("title") and len(target["title"]) > 0:
        return "pass"
    return "fail"


# --- LLM Decode Tests ---


class TestLLMDecodeScoring:
    async def test_llm_decode_json_response_matching_level(self) -> None:
        """LLM returns JSON with a valid level_id."""
        provider = EchoProvider({"provider": "echo", "model": "test"})
        provider.set_responses([json.dumps({"level_id": "pass"})])

        executor = RubricExecutor(
            function_registry=FunctionRegistry(), llm=provider
        )
        rubric = _make_rubric([_make_llm_decode_criterion()])
        target = {"content": "This is some test content."}

        evaluation = await executor.evaluate(rubric, target)

        assert evaluation.criterion_results[0].level_id == "pass"
        assert evaluation.criterion_results[0].score == pytest.approx(0.7)
        assert evaluation.criterion_results[0].scoring_method_used == ScoringType.LLM_DECODE
        assert evaluation.criterion_results[0].llm_invocation is not None
        assert "model" in evaluation.criterion_results[0].llm_invocation
        assert "prompt_hash" in evaluation.criterion_results[0].llm_invocation
        assert provider.call_count == 2  # 1 for decode + 1 for feedback

    async def test_llm_decode_plain_text_level_id(self) -> None:
        """LLM returns plain text matching a level_id."""
        provider = EchoProvider({"provider": "echo", "model": "test"})
        provider.set_responses(["excellent", "feedback summary"])

        executor = RubricExecutor(
            function_registry=FunctionRegistry(), llm=provider
        )
        rubric = _make_rubric([_make_llm_decode_criterion()])
        target = {"content": "Outstanding content."}

        evaluation = await executor.evaluate(rubric, target)

        assert evaluation.criterion_results[0].level_id == "excellent"
        assert evaluation.criterion_results[0].score == pytest.approx(1.0)

    async def test_llm_decode_case_insensitive_matching(self) -> None:
        """LLM response matching is case-insensitive for plain text."""
        provider = EchoProvider({"provider": "echo", "model": "test"})
        provider.set_responses(["PASS", "feedback summary"])

        executor = RubricExecutor(
            function_registry=FunctionRegistry(), llm=provider
        )
        rubric = _make_rubric([_make_llm_decode_criterion()])
        target = {"content": "Some content."}

        evaluation = await executor.evaluate(rubric, target)

        assert evaluation.criterion_results[0].level_id == "pass"

    async def test_llm_decode_no_matching_level(self) -> None:
        """LLM returns text not matching any level → unable_to_evaluate."""
        provider = EchoProvider({"provider": "echo", "model": "test"})
        provider.set_responses(["something completely unrelated", "feedback summary"])

        executor = RubricExecutor(
            function_registry=FunctionRegistry(), llm=provider
        )
        rubric = _make_rubric([_make_llm_decode_criterion()])
        target = {"content": "Test content."}

        evaluation = await executor.evaluate(rubric, target)

        assert evaluation.criterion_results[0].level_id == "unable_to_evaluate"
        assert evaluation.criterion_results[0].score == 0.0
        assert evaluation.criterion_results[0].llm_invocation is not None

    async def test_llm_not_configured_raises_error(self) -> None:
        """LLM_DECODE without LLM provider configured → error result."""
        executor = RubricExecutor(
            function_registry=FunctionRegistry(), llm=None
        )
        rubric = _make_rubric([_make_llm_decode_criterion()])

        evaluation = await executor.evaluate(rubric, {})

        assert evaluation.criterion_results[0].level_id == "error"
        assert "no LLM provider" in evaluation.criterion_results[0].notes

    async def test_llm_decode_missing_decode_prompt(self) -> None:
        """LLM_DECODE without decode_prompt → error result."""
        provider = EchoProvider({"provider": "echo", "model": "test"})
        criterion = RubricCriterion(
            id="c1",
            name="LLM Check",
            description="Test",
            weight=1.0,
            levels=list(STANDARD_LEVELS),
            scoring_method=ScoringMethod(type=ScoringType.LLM_DECODE),
        )
        executor = RubricExecutor(
            function_registry=FunctionRegistry(), llm=provider
        )
        rubric = _make_rubric([criterion])

        evaluation = await executor.evaluate(rubric, {})

        assert evaluation.criterion_results[0].level_id == "error"
        assert "no decode_prompt" in evaluation.criterion_results[0].notes


class TestLLMDecodeTemplateRendering:
    async def test_template_renders_target_data(self) -> None:
        """Jinja2 template receives and renders target data."""
        provider = EchoProvider({"provider": "echo", "model": "test"})
        provider.set_responses([json.dumps({"level_id": "pass"}), "feedback"])

        criterion = _make_llm_decode_criterion(
            decode_prompt="Title: {{ title }}, Words: {{ word_count }}",
        )
        executor = RubricExecutor(
            function_registry=FunctionRegistry(), llm=provider
        )
        rubric = _make_rubric([criterion])
        target = {"title": "My Doc", "word_count": 150}

        evaluation = await executor.evaluate(rubric, target)

        assert evaluation.criterion_results[0].level_id == "pass"
        # Verify the prompt was rendered by checking call history
        call = provider.get_call(0)
        user_msg = call["messages"][1]
        assert "My Doc" in user_msg.content
        assert "150" in user_msg.content

    async def test_template_missing_variable_causes_error(self) -> None:
        """Jinja2 StrictUndefined causes error on missing template variables."""
        provider = EchoProvider({"provider": "echo", "model": "test"})
        provider.set_responses(["pass", "feedback"])

        criterion = _make_llm_decode_criterion(
            decode_prompt="Content: {{ missing_var }}",
        )
        executor = RubricExecutor(
            function_registry=FunctionRegistry(), llm=provider
        )
        rubric = _make_rubric([criterion])

        evaluation = await executor.evaluate(rubric, {"content": "test"})

        assert evaluation.criterion_results[0].level_id == "error"
        assert "missing_var" in evaluation.criterion_results[0].notes


class TestLLMDecodeOutputSchema:
    async def test_valid_output_schema(self) -> None:
        """Response matching decode_output_schema is accepted."""
        output_schema = {
            "type": "object",
            "required": ["level_id"],
            "properties": {
                "level_id": {"type": "string"},
            },
        }
        provider = EchoProvider({"provider": "echo", "model": "test"})
        provider.set_responses([
            json.dumps({"level_id": "excellent"}),
            "feedback",
        ])

        criterion = _make_llm_decode_criterion(
            decode_output_schema=output_schema,
        )
        executor = RubricExecutor(
            function_registry=FunctionRegistry(), llm=provider
        )
        rubric = _make_rubric([criterion])

        evaluation = await executor.evaluate(rubric, {"content": "test"})

        assert evaluation.criterion_results[0].level_id == "excellent"
        assert evaluation.criterion_results[0].score == pytest.approx(1.0)


class TestLLMDecodeMixedCriteria:
    async def test_mixed_deterministic_and_llm_decode(self) -> None:
        """Rubric with both deterministic and LLM decode criteria."""
        provider = EchoProvider({"provider": "echo", "model": "test"})
        provider.set_responses([
            json.dumps({"level_id": "excellent"}),
            "feedback",
        ])

        registry = FunctionRegistry()
        registry.register("test:has_title", score_has_title)

        det_criterion = _make_deterministic_criterion(
            "c1", "test:has_title", weight=0.5
        )
        llm_criterion = _make_llm_decode_criterion(
            criterion_id="c2", weight=0.5
        )
        rubric = _make_rubric([det_criterion, llm_criterion])

        executor = RubricExecutor(function_registry=registry, llm=provider)
        target = {"title": "My Title", "content": "Great content."}

        evaluation = await executor.evaluate(rubric, target)

        assert len(evaluation.criterion_results) == 2
        # c1: deterministic, pass (0.7)
        assert evaluation.criterion_results[0].level_id == "pass"
        assert evaluation.criterion_results[0].scoring_method_used == ScoringType.DETERMINISTIC
        assert evaluation.criterion_results[0].llm_invocation is None
        # c2: llm_decode, excellent (1.0)
        assert evaluation.criterion_results[1].level_id == "excellent"
        assert evaluation.criterion_results[1].scoring_method_used == ScoringType.LLM_DECODE
        assert evaluation.criterion_results[1].llm_invocation is not None

        # weighted: (0.7 * 0.5 + 1.0 * 0.5) / 1.0 = 0.85
        assert evaluation.weighted_score == pytest.approx(0.85)
        assert evaluation.passed is True


class TestLLMDecodeInvocationTracking:
    async def test_llm_invocation_recorded(self) -> None:
        """LLM invocation details are recorded in criterion result."""
        provider = EchoProvider({"provider": "echo", "model": "test"})
        provider.set_responses([
            json.dumps({"level_id": "pass"}),
            "feedback",
        ])

        executor = RubricExecutor(
            function_registry=FunctionRegistry(), llm=provider
        )
        rubric = _make_rubric([_make_llm_decode_criterion()])

        evaluation = await executor.evaluate(rubric, {"content": "test"})

        invocation = evaluation.criterion_results[0].llm_invocation
        assert invocation is not None
        assert "model" in invocation
        assert "prompt_hash" in invocation
        assert "response_hash" in invocation
        assert "timestamp" in invocation

    async def test_llm_invocation_serialization(self) -> None:
        """LLM invocation is included in to_dict/from_dict round-trip."""
        provider = EchoProvider({"provider": "echo", "model": "test"})
        provider.set_responses([
            json.dumps({"level_id": "pass"}),
            "feedback",
        ])

        executor = RubricExecutor(
            function_registry=FunctionRegistry(), llm=provider
        )
        rubric = _make_rubric([_make_llm_decode_criterion()])

        evaluation = await executor.evaluate(rubric, {"content": "test"})

        result = evaluation.criterion_results[0]
        result_dict = result.to_dict()
        assert "llm_invocation" in result_dict

        from dataknobs_bots.rubrics.models import CriterionResult
        restored = CriterionResult.from_dict(result_dict)
        assert restored.llm_invocation is not None
        assert restored.llm_invocation["prompt_hash"] == result.llm_invocation["prompt_hash"]

    async def test_deterministic_result_no_llm_invocation_in_dict(self) -> None:
        """Deterministic results do not include llm_invocation in serialization."""
        from dataknobs_bots.rubrics.models import CriterionResult

        result = CriterionResult(
            criterion_id="c1",
            level_id="pass",
            score=0.7,
            scoring_method_used=ScoringType.DETERMINISTIC,
        )
        result_dict = result.to_dict()
        assert "llm_invocation" not in result_dict
