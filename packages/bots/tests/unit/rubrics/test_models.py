"""Tests for rubric data models."""

from __future__ import annotations

from dataknobs_bots.rubrics.models import (
    CriterionResult,
    Rubric,
    RubricCriterion,
    RubricEvaluation,
    RubricLevel,
    ScoringMethod,
    ScoringType,
)


def _make_level(
    level_id: str = "pass",
    label: str = "Pass",
    description: str = "Meets expectations",
    score: float = 0.7,
) -> RubricLevel:
    return RubricLevel(
        id=level_id,
        label=label,
        description=description,
        score=score,
        indicators=["indicator_a"],
    )


def _make_criterion(
    criterion_id: str = "c1",
    weight: float = 1.0,
    scoring_type: ScoringType = ScoringType.DETERMINISTIC,
) -> RubricCriterion:
    return RubricCriterion(
        id=criterion_id,
        name="Test Criterion",
        description="A test criterion",
        weight=weight,
        levels=[
            _make_level("fail", "Fail", "Does not meet expectations", 0.0),
            _make_level("pass", "Pass", "Meets expectations", 0.7),
            _make_level("excellent", "Excellent", "Exceeds expectations", 1.0),
        ],
        scoring_method=ScoringMethod(
            type=scoring_type,
            function_ref="tests.scoring:check_quality"
            if scoring_type == ScoringType.DETERMINISTIC
            else None,
        ),
    )


def _make_rubric(
    rubric_id: str = "rubric_001",
    criteria: list[RubricCriterion] | None = None,
) -> Rubric:
    return Rubric(
        id=rubric_id,
        name="Test Rubric",
        description="A test rubric",
        version="1.0.0",
        target_type="content",
        criteria=criteria if criteria is not None else [_make_criterion()],
        pass_threshold=0.7,
        metadata={"author": "test"},
    )


class TestScoringType:
    def test_enum_values(self) -> None:
        assert ScoringType.DETERMINISTIC.value == "deterministic"
        assert ScoringType.SCHEMA.value == "schema"
        assert ScoringType.LLM_DECODE.value == "llm_decode"

    def test_str_enum(self) -> None:
        assert str(ScoringType.DETERMINISTIC) == "ScoringType.DETERMINISTIC"
        assert ScoringType("deterministic") == ScoringType.DETERMINISTIC


class TestScoringMethod:
    def test_deterministic_method(self) -> None:
        method = ScoringMethod(
            type=ScoringType.DETERMINISTIC,
            function_ref="mymodule:my_func",
        )
        assert method.type == ScoringType.DETERMINISTIC
        assert method.function_ref == "mymodule:my_func"
        assert method.schema is None

    def test_schema_method(self) -> None:
        schema = {"type": "object", "required": ["title"]}
        method = ScoringMethod(
            type=ScoringType.SCHEMA,
            schema=schema,
        )
        assert method.type == ScoringType.SCHEMA
        assert method.schema == schema
        assert method.function_ref is None

    def test_to_dict_omits_none_fields(self) -> None:
        method = ScoringMethod(
            type=ScoringType.DETERMINISTIC,
            function_ref="mod:func",
        )
        d = method.to_dict()
        assert d == {"type": "deterministic", "function_ref": "mod:func"}
        assert "schema" not in d
        assert "decode_prompt" not in d

    def test_serialization_round_trip(self) -> None:
        method = ScoringMethod(
            type=ScoringType.SCHEMA,
            schema={"type": "object"},
            schema_ref="schemas/content.json",
        )
        restored = ScoringMethod.from_dict(method.to_dict())
        assert restored.type == method.type
        assert restored.schema == method.schema
        assert restored.schema_ref == method.schema_ref


class TestRubricLevel:
    def test_creation(self) -> None:
        level = _make_level()
        assert level.id == "pass"
        assert level.score == 0.7
        assert level.indicators == ["indicator_a"]

    def test_default_indicators(self) -> None:
        level = RubricLevel(
            id="pass", label="Pass", description="OK", score=0.7
        )
        assert level.indicators == []

    def test_serialization_round_trip(self) -> None:
        level = _make_level()
        restored = RubricLevel.from_dict(level.to_dict())
        assert restored.id == level.id
        assert restored.label == level.label
        assert restored.score == level.score
        assert restored.indicators == level.indicators


class TestRubricCriterion:
    def test_creation(self) -> None:
        criterion = _make_criterion()
        assert criterion.id == "c1"
        assert criterion.weight == 1.0
        assert len(criterion.levels) == 3
        assert criterion.required is True

    def test_optional_required(self) -> None:
        criterion = _make_criterion()
        criterion.required = False
        assert criterion.required is False

    def test_serialization_round_trip(self) -> None:
        criterion = _make_criterion()
        restored = RubricCriterion.from_dict(criterion.to_dict())
        assert restored.id == criterion.id
        assert restored.weight == criterion.weight
        assert len(restored.levels) == len(criterion.levels)
        assert restored.scoring_method.type == criterion.scoring_method.type
        assert (
            restored.scoring_method.function_ref
            == criterion.scoring_method.function_ref
        )


class TestRubric:
    def test_creation(self) -> None:
        rubric = _make_rubric()
        assert rubric.id == "rubric_001"
        assert rubric.version == "1.0.0"
        assert rubric.target_type == "content"
        assert rubric.pass_threshold == 0.7
        assert len(rubric.criteria) == 1

    def test_default_metadata(self) -> None:
        rubric = Rubric(
            id="r1",
            name="R",
            description="D",
            version="1.0.0",
            target_type="content",
            criteria=[],
            pass_threshold=0.7,
        )
        assert rubric.metadata == {}

    def test_serialization_round_trip(self) -> None:
        rubric = _make_rubric(
            criteria=[_make_criterion("c1", 0.6), _make_criterion("c2", 0.4)]
        )
        d = rubric.to_dict()
        restored = Rubric.from_dict(d)
        assert restored.id == rubric.id
        assert restored.version == rubric.version
        assert restored.pass_threshold == rubric.pass_threshold
        assert len(restored.criteria) == 2
        assert restored.criteria[0].weight == 0.6
        assert restored.criteria[1].weight == 0.4
        assert restored.metadata == {"author": "test"}

    def test_multi_criteria(self) -> None:
        rubric = _make_rubric(
            criteria=[
                _make_criterion("c1", 0.5),
                _make_criterion("c2", 0.3),
                _make_criterion(
                    "c3", 0.2, scoring_type=ScoringType.SCHEMA
                ),
            ]
        )
        assert len(rubric.criteria) == 3
        assert rubric.criteria[2].scoring_method.type == ScoringType.SCHEMA


class TestCriterionResult:
    def test_creation(self) -> None:
        result = CriterionResult(
            criterion_id="c1",
            level_id="pass",
            score=0.7,
            evidence=["Found required fields"],
            notes="All checks passed",
        )
        assert result.criterion_id == "c1"
        assert result.score == 0.7
        assert result.scoring_method_used == ScoringType.DETERMINISTIC

    def test_defaults(self) -> None:
        result = CriterionResult(
            criterion_id="c1", level_id="pass", score=0.7
        )
        assert result.evidence == []
        assert result.notes == ""
        assert result.scoring_method_used == ScoringType.DETERMINISTIC

    def test_serialization_round_trip(self) -> None:
        result = CriterionResult(
            criterion_id="c1",
            level_id="excellent",
            score=1.0,
            evidence=["ev1", "ev2"],
            notes="Great",
            scoring_method_used=ScoringType.SCHEMA,
        )
        restored = CriterionResult.from_dict(result.to_dict())
        assert restored.criterion_id == result.criterion_id
        assert restored.level_id == result.level_id
        assert restored.score == result.score
        assert restored.evidence == result.evidence
        assert restored.scoring_method_used == ScoringType.SCHEMA


class TestRubricEvaluation:
    def test_creation_with_defaults(self) -> None:
        evaluation = RubricEvaluation()
        assert evaluation.id.startswith("eval_")
        assert evaluation.rubric_id == ""
        assert evaluation.weighted_score == 0.0
        assert evaluation.passed is False
        assert evaluation.evaluated_by == "system"
        assert evaluation.evaluated_at != ""

    def test_creation_with_values(self) -> None:
        results = [
            CriterionResult(criterion_id="c1", level_id="pass", score=0.7),
            CriterionResult(
                criterion_id="c2", level_id="excellent", score=1.0
            ),
        ]
        evaluation = RubricEvaluation(
            id="eval_test123",
            rubric_id="rubric_001",
            rubric_version="1.0.0",
            target_id="art_abc",
            target_type="content",
            criterion_results=results,
            weighted_score=0.85,
            passed=True,
            feedback_summary="Good quality content",
            evaluated_by="system",
        )
        assert evaluation.id == "eval_test123"
        assert evaluation.weighted_score == 0.85
        assert evaluation.passed is True
        assert len(evaluation.criterion_results) == 2

    def test_serialization_round_trip(self) -> None:
        results = [
            CriterionResult(
                criterion_id="c1",
                level_id="pass",
                score=0.7,
                evidence=["ev1"],
            ),
        ]
        evaluation = RubricEvaluation(
            id="eval_test456",
            rubric_id="rubric_001",
            rubric_version="1.0.0",
            target_id="art_xyz",
            target_type="content",
            criterion_results=results,
            weighted_score=0.7,
            passed=True,
            feedback_summary="Passed",
            evaluated_at="2026-02-16T12:00:00+00:00",
            evaluated_by="user:test",
        )
        d = evaluation.to_dict()
        restored = RubricEvaluation.from_dict(d)
        assert restored.id == evaluation.id
        assert restored.rubric_id == evaluation.rubric_id
        assert restored.weighted_score == evaluation.weighted_score
        assert restored.passed == evaluation.passed
        assert len(restored.criterion_results) == 1
        assert restored.criterion_results[0].level_id == "pass"
        assert restored.evaluated_by == "user:test"

    def test_unique_ids_generated(self) -> None:
        eval1 = RubricEvaluation()
        eval2 = RubricEvaluation()
        assert eval1.id != eval2.id
