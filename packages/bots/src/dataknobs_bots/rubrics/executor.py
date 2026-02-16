"""Rubric evaluation executor for deterministic, schema, and LLM decode scoring.

This module provides the evaluation engine that applies rubrics to targets:
- FunctionRegistry: Maps function references to callables for deterministic scoring
- RubricExecutor: Evaluates targets against rubrics, computing weighted scores

Scoring is dispatched by ScoringType:
- DETERMINISTIC: Calls registered Python functions
- SCHEMA: Validates against JSON Schema
- LLM_DECODE: Narrowly-scoped LLM classification via Jinja2 decode prompts

Example:
    >>> registry = FunctionRegistry()
    >>> registry.register("mymodule:check_quality", check_quality_func)
    >>> executor = RubricExecutor(function_registry=registry)
    >>> evaluation = await executor.evaluate(rubric, target, target_id="art_001")
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
from typing import Any

from dataknobs_llm.llm.base import AsyncLLMProvider, LLMMessage

from .feedback import generate_feedback_summary
from .models import (
    CriterionResult,
    Rubric,
    RubricCriterion,
    RubricEvaluation,
    ScoringMethod,
    ScoringType,
    _generate_id,
    _now_iso,
)

logger = logging.getLogger(__name__)


class FunctionRegistry:
    """Registry for scoring functions, supporting both direct registration
    and dynamic import from dotted-path references.

    Function references use the format ``"module.path:function_name"``.
    Functions receive a ``dict[str, Any]`` target and return a ``str`` level_id.
    """

    def __init__(self) -> None:
        self._functions: dict[str, Any] = {}

    def register(self, ref: str, func: Any) -> None:
        """Register a callable under the given reference string."""
        self._functions[ref] = func

    def get(self, ref: str) -> Any:
        """Get a registered function, resolving via dynamic import if needed.

        Raises:
            KeyError: If the function reference cannot be resolved.
        """
        if ref in self._functions:
            return self._functions[ref]

        func = self._resolve_import(ref)
        self._functions[ref] = func
        return func

    def _resolve_import(self, ref: str) -> Any:
        """Dynamically import a function from a ``"module:function"`` reference.

        Raises:
            KeyError: If the module or function cannot be imported.
        """
        if ":" not in ref:
            raise KeyError(
                f"Invalid function reference '{ref}': "
                "expected 'module.path:function_name' format"
            )

        module_path, func_name = ref.rsplit(":", 1)
        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            raise KeyError(
                f"Cannot import module '{module_path}' "
                f"from reference '{ref}': {e}"
            ) from e

        if not hasattr(module, func_name):
            raise KeyError(
                f"Module '{module_path}' has no attribute '{func_name}'"
            )

        return getattr(module, func_name)


class RubricExecutor:
    """Evaluates targets against rubrics using registered scoring functions.

    The executor dispatches criterion evaluation based on each criterion's
    ``ScoringType``, computes weighted aggregate scores, and determines
    pass/fail based on the rubric's threshold.

    Args:
        function_registry: Registry of scoring functions for deterministic criteria.
        llm: Optional LLM provider for LLM_DECODE scoring and feedback generation.
    """

    def __init__(
        self,
        function_registry: FunctionRegistry,
        llm: AsyncLLMProvider | None = None,
    ) -> None:
        self._function_registry = function_registry
        self._llm = llm

    async def evaluate(
        self,
        rubric: Rubric,
        target: dict[str, Any],
        target_id: str = "",
        target_type: str = "",
    ) -> RubricEvaluation:
        """Evaluate a target against a rubric.

        Args:
            rubric: The rubric to evaluate against.
            target: The target content as a dictionary.
            target_id: Identifier of the target being evaluated.
            target_type: Type of the target (defaults to rubric's target_type).

        Returns:
            A RubricEvaluation with scored results and pass/fail determination.
        """
        effective_target_type = target_type or rubric.target_type
        criterion_results: list[CriterionResult] = []

        for criterion in rubric.criteria:
            result = await self._evaluate_criterion(criterion, target)
            criterion_results.append(result)

        weighted_score = self._compute_weighted_score(
            rubric.criteria, criterion_results
        )
        passed = weighted_score >= rubric.pass_threshold

        evaluation = RubricEvaluation(
            id=_generate_id("eval"),
            rubric_id=rubric.id,
            rubric_version=rubric.version,
            target_id=target_id,
            target_type=effective_target_type,
            criterion_results=criterion_results,
            weighted_score=weighted_score,
            passed=passed,
            evaluated_at=_now_iso(),
        )

        evaluation.feedback_summary = await generate_feedback_summary(
            rubric, evaluation, self._llm
        )

        return evaluation

    async def _evaluate_criterion(
        self,
        criterion: RubricCriterion,
        target: dict[str, Any],
    ) -> CriterionResult:
        """Evaluate a single criterion against the target.

        Dispatches to the appropriate scoring method. On error, returns
        a result with level_id="error" and the exception details in notes.
        """
        scoring_type = criterion.scoring_method.type

        try:
            if scoring_type == ScoringType.DETERMINISTIC:
                return self._evaluate_deterministic(criterion, target)
            elif scoring_type == ScoringType.SCHEMA:
                return self._evaluate_schema(criterion, target)
            elif scoring_type == ScoringType.LLM_DECODE:
                return await self._evaluate_llm_decode(criterion, target)
            else:
                raise ValueError(f"Unknown scoring type: {scoring_type}")
        except Exception as e:
            logger.warning(
                "Criterion '%s' evaluation failed: %s",
                criterion.id,
                e,
            )
            return CriterionResult(
                criterion_id=criterion.id,
                level_id="error",
                score=0.0,
                notes=f"Evaluation error: {e}",
                scoring_method_used=scoring_type,
            )

    def _evaluate_deterministic(
        self,
        criterion: RubricCriterion,
        target: dict[str, Any],
    ) -> CriterionResult:
        """Evaluate using a registered Python function.

        The function receives the target dict and returns a level_id string.
        """
        ref = criterion.scoring_method.function_ref
        if ref is None:
            raise ValueError(
                f"Criterion '{criterion.id}' has DETERMINISTIC scoring "
                "but no function_ref"
            )

        func = self._function_registry.get(ref)
        level_id = func(target)

        score = self._level_id_to_score(criterion, level_id)

        return CriterionResult(
            criterion_id=criterion.id,
            level_id=level_id,
            score=score,
            scoring_method_used=ScoringType.DETERMINISTIC,
        )

    def _evaluate_schema(
        self,
        criterion: RubricCriterion,
        target: dict[str, Any],
    ) -> CriterionResult:
        """Evaluate using JSON Schema validation.

        Validates the target against the criterion's schema. If validation
        passes, assigns the highest-scoring level; if it fails, assigns
        the lowest-scoring level.
        """
        import jsonschema

        schema = criterion.scoring_method.schema
        if schema is None:
            raise ValueError(
                f"Criterion '{criterion.id}' has SCHEMA scoring "
                "but no schema defined"
            )

        errors: list[str] = []
        try:
            jsonschema.validate(target, schema)
        except jsonschema.ValidationError as e:
            errors.append(str(e.message))
        except jsonschema.SchemaError as e:
            raise ValueError(
                f"Invalid JSON Schema for criterion '{criterion.id}': {e}"
            ) from e

        sorted_levels = sorted(criterion.levels, key=lambda level: level.score)

        if errors:
            level = sorted_levels[0]
            return CriterionResult(
                criterion_id=criterion.id,
                level_id=level.id,
                score=level.score,
                evidence=errors,
                notes="Schema validation failed",
                scoring_method_used=ScoringType.SCHEMA,
            )
        else:
            level = sorted_levels[-1]
            return CriterionResult(
                criterion_id=criterion.id,
                level_id=level.id,
                score=level.score,
                evidence=["Schema validation passed"],
                scoring_method_used=ScoringType.SCHEMA,
            )

    async def _evaluate_llm_decode(
        self,
        criterion: RubricCriterion,
        target: dict[str, Any],
    ) -> CriterionResult:
        """Evaluate using narrowly-scoped LLM classification.

        Renders the criterion's decode prompt as a Jinja2 template with target
        data, sends it to the LLM, and parses the response to extract a level_id.
        """
        if self._llm is None:
            raise ValueError(
                f"Criterion '{criterion.id}' requires LLM_DECODE scoring "
                "but no LLM provider is configured"
            )

        method = criterion.scoring_method
        if method.decode_prompt is None:
            raise ValueError(
                f"Criterion '{criterion.id}' has LLM_DECODE scoring "
                "but no decode_prompt defined"
            )

        rendered_prompt = self._render_template(method.decode_prompt, target)
        valid_level_ids = [level.id for level in criterion.levels]

        system_message = self._build_decode_system_message(criterion)
        messages = [
            LLMMessage(role="system", content=system_message),
            LLMMessage(role="user", content=rendered_prompt),
        ]

        prompt_hash = hashlib.sha256(rendered_prompt.encode()).hexdigest()[:16]

        try:
            response = await self._llm.complete(messages)
            response_text = response.content.strip()
            response_hash = hashlib.sha256(response_text.encode()).hexdigest()[:16]

            llm_invocation: dict[str, Any] = {
                "model": response.model,
                "prompt_hash": prompt_hash,
                "response_hash": response_hash,
                "timestamp": _now_iso(),
            }
            if response.usage:
                llm_invocation["usage"] = response.usage

            level_id = self._parse_llm_level_response(
                response_text, valid_level_ids, method
            )

        except Exception as e:
            logger.warning(
                "LLM decode failed for criterion '%s': %s",
                criterion.id,
                e,
            )
            return CriterionResult(
                criterion_id=criterion.id,
                level_id="unable_to_evaluate",
                score=0.0,
                notes=f"LLM decode failed: {e}",
                scoring_method_used=ScoringType.LLM_DECODE,
                llm_invocation={
                    "prompt_hash": prompt_hash,
                    "timestamp": _now_iso(),
                    "error": str(e),
                },
            )

        score = self._level_id_to_score(criterion, level_id)

        return CriterionResult(
            criterion_id=criterion.id,
            level_id=level_id,
            score=score,
            evidence=[f"LLM response: {response_text}"],
            scoring_method_used=ScoringType.LLM_DECODE,
            llm_invocation=llm_invocation,
        )

    def _render_template(self, template_str: str, context: dict[str, Any]) -> str:
        """Render a Jinja2 template string with the given context."""
        import jinja2

        template = jinja2.Template(template_str, undefined=jinja2.StrictUndefined)
        return template.render(**context)

    def _build_decode_system_message(self, criterion: RubricCriterion) -> str:
        """Build the system message for LLM decode classification."""
        level_descriptions = "\n".join(
            f"- {level.id}: {level.description}"
            for level in criterion.levels
        )
        valid_ids = ", ".join(f'"{level.id}"' for level in criterion.levels)

        return (
            f"You are a classification evaluator for the criterion: {criterion.name}\n"
            f"Description: {criterion.description}\n\n"
            f"Classify the content into exactly one of these levels:\n"
            f"{level_descriptions}\n\n"
            f"Respond with a JSON object containing a single field \"level_id\" "
            f"set to one of: {valid_ids}\n"
            f"Example: {{\"level_id\": \"{criterion.levels[0].id}\"}}\n"
            f"Do not include any other text."
        )

    def _parse_llm_level_response(
        self,
        response_text: str,
        valid_level_ids: list[str],
        method: ScoringMethod,
    ) -> str:
        """Parse LLM response to extract a level_id.

        Tries JSON parsing first, then falls back to matching response text
        against known level IDs. Returns "unable_to_evaluate" if no match.
        """
        level_id = self._try_parse_json_level(response_text)
        if level_id is not None and level_id in valid_level_ids:
            if method.decode_output_schema is not None:
                self._validate_output_schema(response_text, method.decode_output_schema)
            return level_id

        cleaned = response_text.strip().strip('"').strip("'").lower()
        for valid_id in valid_level_ids:
            if cleaned == valid_id.lower():
                return valid_id

        logger.warning(
            "LLM response '%s' does not match any valid level: %s",
            response_text[:100],
            valid_level_ids,
        )
        return "unable_to_evaluate"

    def _try_parse_json_level(self, text: str) -> str | None:
        """Attempt to parse a JSON response and extract level_id."""
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "level_id" in data:
                return str(data["level_id"])
        except (json.JSONDecodeError, TypeError):
            pass
        return None

    def _validate_output_schema(
        self, response_text: str, schema: dict[str, Any]
    ) -> None:
        """Validate the LLM response against the decode_output_schema."""
        import jsonschema

        try:
            data = json.loads(response_text)
            jsonschema.validate(data, schema)
        except (json.JSONDecodeError, jsonschema.ValidationError) as e:
            logger.warning("LLM decode output schema validation failed: %s", e)

    def _level_id_to_score(
        self, criterion: RubricCriterion, level_id: str
    ) -> float:
        """Look up the numeric score for a level_id within a criterion.

        Returns 0.0 if the level_id is not found.
        """
        for level in criterion.levels:
            if level.id == level_id:
                return level.score
        logger.warning(
            "Level '%s' not found in criterion '%s', defaulting to 0.0",
            level_id,
            criterion.id,
        )
        return 0.0

    def _compute_weighted_score(
        self,
        criteria: list[RubricCriterion],
        results: list[CriterionResult],
    ) -> float:
        """Compute the weighted aggregate score across all criteria.

        Each criterion's score is multiplied by its weight. The total
        weight is used as the divisor to handle cases where weights
        don't sum to exactly 1.0.
        """
        total_weight = sum(c.weight for c in criteria)
        if total_weight == 0:
            return 0.0

        weighted_sum = 0.0
        result_by_id = {r.criterion_id: r for r in results}

        for criterion in criteria:
            result = result_by_id.get(criterion.id)
            if result is not None:
                weighted_sum += criterion.weight * result.score

        return weighted_sum / total_weight

