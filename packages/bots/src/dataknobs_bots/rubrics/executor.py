"""Rubric evaluation executor for deterministic and schema-based scoring.

This module provides the evaluation engine that applies rubrics to targets:
- FunctionRegistry: Maps function references to callables for deterministic scoring
- RubricExecutor: Evaluates targets against rubrics, computing weighted scores

Scoring is dispatched by ScoringType:
- DETERMINISTIC: Calls registered Python functions
- SCHEMA: Validates against JSON Schema
- LLM_DECODE: Deferred to Phase 3

Example:
    >>> registry = FunctionRegistry()
    >>> registry.register("mymodule:check_quality", check_quality_func)
    >>> executor = RubricExecutor(function_registry=registry)
    >>> evaluation = await executor.evaluate(rubric, target, target_id="art_001")
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

from dataknobs_llm.llm.base import AsyncLLMProvider

from .models import (
    CriterionResult,
    Rubric,
    RubricCriterion,
    RubricEvaluation,
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
        llm: Optional LLM provider for llm_decode scoring (Phase 3).
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
        summary = self._generate_deterministic_summary(
            rubric, criterion_results, passed
        )

        return RubricEvaluation(
            id=_generate_id("eval"),
            rubric_id=rubric.id,
            rubric_version=rubric.version,
            target_id=target_id,
            target_type=effective_target_type,
            criterion_results=criterion_results,
            weighted_score=weighted_score,
            passed=passed,
            feedback_summary=summary,
            evaluated_at=_now_iso(),
        )

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
                raise NotImplementedError(
                    "LLM_DECODE scoring is not yet implemented (Phase 3)"
                )
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

    def _generate_deterministic_summary(
        self,
        rubric: Rubric,
        results: list[CriterionResult],
        passed: bool,
    ) -> str:
        """Generate a template-based feedback summary.

        This deterministic summary is used in Phases 1-2. Phase 3 adds
        LLM-enhanced summaries.
        """
        status = "PASSED" if passed else "FAILED"
        lines = [f"Evaluation {status} for rubric '{rubric.name}'."]

        result_by_id = {r.criterion_id: r for r in results}
        for criterion in rubric.criteria:
            result = result_by_id.get(criterion.id)
            if result is not None:
                lines.append(
                    f"- {criterion.name}: {result.level_id} "
                    f"(score: {result.score:.2f})"
                )

        return "\n".join(lines)
