"""Tests for meta-rubric validation functions."""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_bots.rubrics.executor import FunctionRegistry, RubricExecutor
from dataknobs_bots.rubrics.meta import (
    build_meta_rubric,
    check_criteria_coverage,
    check_criteria_independence,
    check_level_ordering,
    check_threshold,
    check_weight_distribution,
)
from dataknobs_bots.rubrics.models import (
    RubricCriterion,
    RubricLevel,
    ScoringMethod,
    ScoringType,
)


def _rubric_content(
    criteria: list[dict[str, Any]] | None = None,
    pass_threshold: float = 0.7,
) -> dict[str, Any]:
    """Helper to build a rubric content dict for testing meta-functions."""
    if criteria is None:
        criteria = [
            {
                "id": "c1",
                "name": "C1",
                "description": "Criterion 1",
                "weight": 1.0,
                "levels": [
                    {"id": "fail", "score": 0.0},
                    {"id": "pass", "score": 1.0},
                ],
            },
        ]
    return {
        "id": "test_rubric",
        "name": "Test",
        "description": "Test",
        "version": "1.0.0",
        "target_type": "content",
        "criteria": criteria,
        "pass_threshold": pass_threshold,
    }


class TestCheckCriteriaCoverage:
    def test_pass_with_criteria(self) -> None:
        assert check_criteria_coverage(_rubric_content()) == "pass"

    def test_fail_empty_criteria(self) -> None:
        assert check_criteria_coverage(_rubric_content(criteria=[])) == "fail"

    def test_fail_missing_criteria_key(self) -> None:
        assert check_criteria_coverage({}) == "fail"


class TestCheckCriteriaIndependence:
    def test_pass_unique(self) -> None:
        criteria = [
            {"name": "A", "description": "Desc A"},
            {"name": "B", "description": "Desc B"},
        ]
        assert check_criteria_independence(
            _rubric_content(criteria=criteria)
        ) == "pass"

    def test_fail_duplicate_names(self) -> None:
        criteria = [
            {"name": "Same", "description": "Desc A"},
            {"name": "Same", "description": "Desc B"},
        ]
        assert check_criteria_independence(
            _rubric_content(criteria=criteria)
        ) == "fail"

    def test_partial_duplicate_descriptions(self) -> None:
        criteria = [
            {"name": "A", "description": "Same desc"},
            {"name": "B", "description": "Same desc"},
        ]
        assert check_criteria_independence(
            _rubric_content(criteria=criteria)
        ) == "partial"

    def test_pass_single_criterion(self) -> None:
        criteria = [{"name": "Only", "description": "Only desc"}]
        assert check_criteria_independence(
            _rubric_content(criteria=criteria)
        ) == "pass"


class TestCheckWeightDistribution:
    def test_pass_weights_sum_to_one(self) -> None:
        criteria = [
            {"weight": 0.6},
            {"weight": 0.4},
        ]
        assert check_weight_distribution(
            _rubric_content(criteria=criteria)
        ) == "pass"

    def test_pass_within_tolerance(self) -> None:
        criteria = [
            {"weight": 0.33},
            {"weight": 0.33},
            {"weight": 0.34},
        ]
        assert check_weight_distribution(
            _rubric_content(criteria=criteria)
        ) == "pass"

    def test_fail_weights_dont_sum(self) -> None:
        criteria = [
            {"weight": 0.5},
            {"weight": 0.3},
        ]
        assert check_weight_distribution(
            _rubric_content(criteria=criteria)
        ) == "fail"

    def test_fail_empty_criteria(self) -> None:
        assert check_weight_distribution(
            _rubric_content(criteria=[])
        ) == "fail"


class TestCheckThreshold:
    def test_pass_reasonable_threshold(self) -> None:
        criteria = [
            {
                "weight": 1.0,
                "levels": [
                    {"score": 0.0},
                    {"score": 0.5},
                    {"score": 1.0},
                ],
            },
        ]
        assert check_threshold(
            _rubric_content(criteria=criteria, pass_threshold=0.5)
        ) == "pass"

    def test_too_high_unachievable(self) -> None:
        criteria = [
            {
                "weight": 1.0,
                "levels": [
                    {"score": 0.0},
                    {"score": 0.5},
                ],
            },
        ]
        assert check_threshold(
            _rubric_content(criteria=criteria, pass_threshold=0.8)
        ) == "too_high"

    def test_too_low_trivial(self) -> None:
        criteria = [
            {
                "weight": 1.0,
                "levels": [
                    {"score": 0.5},
                    {"score": 1.0},
                ],
            },
        ]
        # Threshold of 0.5 == min possible (0.5), so too_low
        assert check_threshold(
            _rubric_content(criteria=criteria, pass_threshold=0.5)
        ) == "too_low"

    def test_fail_empty_criteria(self) -> None:
        assert check_threshold(
            _rubric_content(criteria=[], pass_threshold=0.5)
        ) == "fail"

    def test_fail_zero_weight(self) -> None:
        criteria = [{"weight": 0.0, "levels": [{"score": 0.5}]}]
        assert check_threshold(
            _rubric_content(criteria=criteria, pass_threshold=0.5)
        ) == "fail"


class TestCheckLevelOrdering:
    def test_pass_ascending(self) -> None:
        criteria = [
            {
                "levels": [
                    {"score": 0.0},
                    {"score": 0.5},
                    {"score": 1.0},
                ],
            },
        ]
        assert check_level_ordering(
            _rubric_content(criteria=criteria)
        ) == "pass"

    def test_pass_descending(self) -> None:
        criteria = [
            {
                "levels": [
                    {"score": 1.0},
                    {"score": 0.5},
                    {"score": 0.0},
                ],
            },
        ]
        assert check_level_ordering(
            _rubric_content(criteria=criteria)
        ) == "pass"

    def test_fail_unordered(self) -> None:
        criteria = [
            {
                "levels": [
                    {"score": 0.0},
                    {"score": 1.0},
                    {"score": 0.5},
                ],
            },
        ]
        assert check_level_ordering(
            _rubric_content(criteria=criteria)
        ) == "fail"

    def test_pass_single_level(self) -> None:
        criteria = [{"levels": [{"score": 0.5}]}]
        assert check_level_ordering(
            _rubric_content(criteria=criteria)
        ) == "pass"

    def test_pass_empty_criteria(self) -> None:
        assert check_level_ordering(_rubric_content(criteria=[])) == "pass"


class TestBuildMetaRubric:
    def test_produces_valid_rubric(self) -> None:
        meta = build_meta_rubric()
        assert meta.id == "meta_rubric"
        assert meta.target_type == "rubric"
        assert len(meta.criteria) == 5
        assert meta.pass_threshold == 0.7

    def test_all_criteria_are_deterministic(self) -> None:
        meta = build_meta_rubric()
        for c in meta.criteria:
            assert c.scoring_method.type == ScoringType.DETERMINISTIC
            assert c.scoring_method.function_ref is not None

    def test_weights_sum_to_one(self) -> None:
        meta = build_meta_rubric()
        total = sum(c.weight for c in meta.criteria)
        assert total == pytest.approx(1.0)

    def test_serialization_round_trip(self) -> None:
        meta = build_meta_rubric()
        from dataknobs_bots.rubrics.models import Rubric

        restored = Rubric.from_dict(meta.to_dict())
        assert restored.id == meta.id
        assert len(restored.criteria) == len(meta.criteria)


class TestMetaRubricEvaluation:
    """Integration tests: evaluate rubrics using the meta-rubric."""

    def _build_executor(self) -> RubricExecutor:
        """Build an executor with meta-rubric functions registered."""
        from dataknobs_bots.rubrics.meta import (
            _META_FUNCTION_REFS,
            check_criteria_coverage,
            check_criteria_independence,
            check_level_ordering,
            check_threshold,
            check_weight_distribution,
        )

        registry = FunctionRegistry()
        funcs = {
            _META_FUNCTION_REFS["criteria_coverage"]: check_criteria_coverage,
            _META_FUNCTION_REFS["criteria_independence"]: check_criteria_independence,
            _META_FUNCTION_REFS["weight_distribution"]: check_weight_distribution,
            _META_FUNCTION_REFS["threshold"]: check_threshold,
            _META_FUNCTION_REFS["level_ordering"]: check_level_ordering,
        }
        for ref, func in funcs.items():
            registry.register(ref, func)
        return RubricExecutor(function_registry=registry)

    async def test_good_rubric_passes(self) -> None:
        executor = self._build_executor()
        meta = build_meta_rubric()

        good_rubric_content = {
            "id": "test_r",
            "criteria": [
                {
                    "id": "c1",
                    "name": "Coverage",
                    "description": "Content coverage",
                    "weight": 0.5,
                    "levels": [
                        {"id": "fail", "score": 0.0},
                        {"id": "pass", "score": 1.0},
                    ],
                },
                {
                    "id": "c2",
                    "name": "Quality",
                    "description": "Content quality",
                    "weight": 0.5,
                    "levels": [
                        {"id": "fail", "score": 0.0},
                        {"id": "pass", "score": 1.0},
                    ],
                },
            ],
            "pass_threshold": 0.5,
        }

        evaluation = await executor.evaluate(
            meta, good_rubric_content, target_id="test_r"
        )

        assert evaluation.passed is True
        assert evaluation.weighted_score == pytest.approx(1.0)

    async def test_empty_rubric_fails(self) -> None:
        executor = self._build_executor()
        meta = build_meta_rubric()

        empty_rubric = {
            "criteria": [],
            "pass_threshold": 0.5,
        }

        evaluation = await executor.evaluate(meta, empty_rubric)

        assert evaluation.passed is False

    async def test_bad_weights_reduces_score(self) -> None:
        executor = self._build_executor()
        meta = build_meta_rubric()

        bad_weights_rubric = {
            "criteria": [
                {
                    "id": "c1",
                    "name": "A",
                    "description": "Desc A",
                    "weight": 0.3,
                    "levels": [
                        {"id": "fail", "score": 0.0},
                        {"id": "pass", "score": 1.0},
                    ],
                },
                {
                    "id": "c2",
                    "name": "B",
                    "description": "Desc B",
                    "weight": 0.3,
                    "levels": [
                        {"id": "fail", "score": 0.0},
                        {"id": "pass", "score": 1.0},
                    ],
                },
            ],
            "pass_threshold": 0.5,
        }

        evaluation = await executor.evaluate(meta, bad_weights_rubric)

        # weight_distribution should fail (0.3 + 0.3 = 0.6 != 1.0)
        weight_result = next(
            r for r in evaluation.criterion_results
            if r.criterion_id == "weight_distribution"
        )
        assert weight_result.level_id == "fail"
        # But other criteria may still pass, so overall may still pass
        assert evaluation.weighted_score < 1.0
