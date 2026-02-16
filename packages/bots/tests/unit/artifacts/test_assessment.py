"""Tests for assessment session tracking."""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_bots.artifacts.assessment import (
    AssessmentSession,
    CumulativePerformance,
    StudentResponse,
    finalize_assessment,
    record_response,
    start_assessment_session,
)


# --- StudentResponse Tests ---


class TestStudentResponse:
    def test_create_basic_response(self) -> None:
        response = StudentResponse(
            question_id="q1",
            response="Paris",
            correct=True,
        )
        assert response.question_id == "q1"
        assert response.correct is True
        assert response.attempt_number == 1

    def test_serialization_roundtrip(self) -> None:
        response = StudentResponse(
            question_id="q2",
            response=2,
            correct=False,
            rubric_score=0.6,
            time_taken_ms=5000,
            attempt_number=2,
        )
        data = response.to_dict()
        restored = StudentResponse.from_dict(data)

        assert restored.question_id == "q2"
        assert restored.response == 2
        assert restored.correct is False
        assert restored.rubric_score == 0.6
        assert restored.time_taken_ms == 5000
        assert restored.attempt_number == 2

    def test_optional_fields_not_serialized_when_none(self) -> None:
        response = StudentResponse(question_id="q1", response="answer")
        data = response.to_dict()

        assert "correct" not in data
        assert "rubric_score" not in data
        assert "time_taken_ms" not in data

    def test_from_dict_defaults(self) -> None:
        response = StudentResponse.from_dict({"question_id": "q1"})
        assert response.response is None
        assert response.correct is None
        assert response.attempt_number == 1


# --- AssessmentSession Tests ---


class TestAssessmentSession:
    def test_create_session(self) -> None:
        session = AssessmentSession(
            student_id="student_001",
            assessment_artifact_id="art_quiz_123",
            assessment_version="1.0.0",
        )
        assert session.student_id == "student_001"
        assert session.is_complete is False
        assert session.total_responses == 0
        assert session.correct_count == 0

    def test_is_complete(self) -> None:
        session = AssessmentSession()
        assert session.is_complete is False

        session.completed_at = "2026-02-16T00:00:00Z"
        assert session.is_complete is True

    def test_correct_count(self) -> None:
        session = AssessmentSession(
            responses=[
                StudentResponse(question_id="q1", response="A", correct=True),
                StudentResponse(question_id="q2", response="B", correct=False),
                StudentResponse(question_id="q3", response="C", correct=True),
            ],
        )
        assert session.correct_count == 2
        assert session.total_responses == 3

    def test_serialization_roundtrip(self) -> None:
        session = AssessmentSession(
            id="sess_test",
            student_id="student_001",
            assessment_artifact_id="art_123",
            assessment_version="2.0.0",
            responses=[
                StudentResponse(question_id="q1", response="A", correct=True),
            ],
            score=1.0,
            rubric_evaluation_id="eval_456",
        )
        session.completed_at = "2026-02-16T00:00:00Z"

        data = session.to_dict()
        restored = AssessmentSession.from_dict(data)

        assert restored.id == "sess_test"
        assert restored.student_id == "student_001"
        assert restored.assessment_artifact_id == "art_123"
        assert restored.assessment_version == "2.0.0"
        assert restored.completed_at == "2026-02-16T00:00:00Z"
        assert len(restored.responses) == 1
        assert restored.responses[0].correct is True
        assert restored.score == 1.0
        assert restored.rubric_evaluation_id == "eval_456"

    def test_optional_fields_not_serialized_when_none(self) -> None:
        session = AssessmentSession(student_id="s1")
        data = session.to_dict()

        assert "completed_at" not in data
        assert "score" not in data
        assert "rubric_evaluation_id" not in data


# --- CumulativePerformance Tests ---


class TestCumulativePerformance:
    def test_create_empty(self) -> None:
        perf = CumulativePerformance(
            student_id="student_001",
            topic="math",
        )
        assert perf.total_sessions == 0
        assert perf.average_score == 0.0

    def test_update_from_session(self) -> None:
        perf = CumulativePerformance(
            student_id="student_001",
            topic="math",
        )
        session = AssessmentSession(
            student_id="student_001",
            responses=[
                StudentResponse(question_id="q1", response="A", correct=True),
                StudentResponse(question_id="q2", response="B", correct=False),
                StudentResponse(question_id="q3", response="C", correct=True),
            ],
            score=0.67,
            completed_at="2026-02-16T00:00:00Z",
        )

        perf.update_from_session(session)

        assert perf.total_sessions == 1
        assert perf.total_questions_attempted == 3
        assert perf.correct_count == 2
        assert perf.average_score == pytest.approx(0.67)
        assert perf.mastery_estimate == pytest.approx(2 / 3)
        assert perf.last_session_at == "2026-02-16T00:00:00Z"

    def test_update_from_multiple_sessions(self) -> None:
        perf = CumulativePerformance(student_id="s1", topic="math")

        session1 = AssessmentSession(
            responses=[
                StudentResponse(question_id="q1", response="A", correct=True),
                StudentResponse(question_id="q2", response="B", correct=True),
            ],
            score=1.0,
            completed_at="2026-02-16T01:00:00Z",
        )
        session2 = AssessmentSession(
            responses=[
                StudentResponse(question_id="q3", response="C", correct=False),
                StudentResponse(question_id="q4", response="D", correct=True),
            ],
            score=0.5,
            completed_at="2026-02-16T02:00:00Z",
        )

        perf.update_from_session(session1)
        perf.update_from_session(session2)

        assert perf.total_sessions == 2
        assert perf.total_questions_attempted == 4
        assert perf.correct_count == 3
        assert perf.average_score == pytest.approx(0.75)
        assert perf.mastery_estimate == pytest.approx(0.75)

    def test_serialization_roundtrip(self) -> None:
        perf = CumulativePerformance(
            student_id="s1",
            topic="science",
            total_sessions=5,
            total_questions_attempted=50,
            correct_count=40,
            average_score=0.8,
            mastery_estimate=0.8,
            last_session_at="2026-02-16T00:00:00Z",
        )
        data = perf.to_dict()
        restored = CumulativePerformance.from_dict(data)

        assert restored.student_id == "s1"
        assert restored.topic == "science"
        assert restored.total_sessions == 5
        assert restored.correct_count == 40
        assert restored.average_score == 0.8


# --- Transform Function Tests ---


class TestStartAssessmentSession:
    async def test_creates_session(self) -> None:
        data: dict[str, Any] = {}

        session = await start_assessment_session(
            data,
            assessment_artifact_id="art_quiz_001",
            student_id="student_001",
            question_ids=["q1", "q2", "q3"],
        )

        assert "_session" in data
        assert "_session_id" in data
        assert data["_current_question_index"] == 0
        assert data["_total_questions"] == 3
        assert data["_question_ids"] == ["q1", "q2", "q3"]
        assert session.student_id == "student_001"
        assert session.assessment_artifact_id == "art_quiz_001"

    async def test_without_question_ids(self) -> None:
        data: dict[str, Any] = {}

        session = await start_assessment_session(
            data,
            assessment_artifact_id="art_001",
            student_id="s1",
        )

        assert data["_total_questions"] == 0
        assert data["_question_ids"] == []


class TestRecordResponse:
    async def test_records_correct_response(self) -> None:
        data: dict[str, Any] = {}
        await start_assessment_session(
            data, assessment_artifact_id="art_001", student_id="s1",
            question_ids=["q1", "q2"],
        )

        response = await record_response(
            data,
            question_id="q1",
            response="Paris",
            correct=True,
            time_taken_ms=3000,
        )

        assert response.correct is True
        assert response.time_taken_ms == 3000
        assert data["_current_question_index"] == 1

        # Verify in session
        session = AssessmentSession.from_dict(data["_session"])
        assert len(session.responses) == 1
        assert session.responses[0].question_id == "q1"

    async def test_records_incorrect_response(self) -> None:
        data: dict[str, Any] = {}
        await start_assessment_session(
            data, assessment_artifact_id="art_001", student_id="s1",
            question_ids=["q1"],
        )

        response = await record_response(
            data, question_id="q1", response="London", correct=False,
        )

        assert response.correct is False

    async def test_tracks_attempt_number(self) -> None:
        data: dict[str, Any] = {}
        await start_assessment_session(
            data, assessment_artifact_id="art_001", student_id="s1",
            question_ids=["q1"],
        )

        await record_response(data, question_id="q1", response="A", correct=False)
        response2 = await record_response(data, question_id="q1", response="B", correct=True)

        assert response2.attempt_number == 2

    async def test_raises_without_session(self) -> None:
        data: dict[str, Any] = {}

        with pytest.raises(ValueError, match="No active session"):
            await record_response(data, question_id="q1", response="A")


class TestFinalizeAssessment:
    async def test_finalizes_with_score(self) -> None:
        data: dict[str, Any] = {}
        await start_assessment_session(
            data, assessment_artifact_id="art_001", student_id="s1",
            question_ids=["q1", "q2", "q3"],
        )
        await record_response(data, question_id="q1", response="A", correct=True)
        await record_response(data, question_id="q2", response="B", correct=False)
        await record_response(data, question_id="q3", response="C", correct=True)

        session = await finalize_assessment(data)

        assert session.is_complete is True
        assert session.score == pytest.approx(2 / 3)
        assert data["_assessment_score"] == pytest.approx(2 / 3)
        assert data["_assessment_complete"] is True

    async def test_finalizes_perfect_score(self) -> None:
        data: dict[str, Any] = {}
        await start_assessment_session(
            data, assessment_artifact_id="art_001", student_id="s1",
            question_ids=["q1", "q2"],
        )
        await record_response(data, question_id="q1", response="A", correct=True)
        await record_response(data, question_id="q2", response="B", correct=True)

        session = await finalize_assessment(data)

        assert session.score == 1.0

    async def test_finalizes_empty_session(self) -> None:
        data: dict[str, Any] = {}
        await start_assessment_session(
            data, assessment_artifact_id="art_001", student_id="s1",
        )

        session = await finalize_assessment(data)

        assert session.score == 0.0

    async def test_raises_without_session(self) -> None:
        data: dict[str, Any] = {}

        with pytest.raises(ValueError, match="No active session"):
            await finalize_assessment(data)
