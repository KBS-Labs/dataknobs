"""Assessment session tracking for quiz and evaluation workflows.

Models and transform functions for creating assessment sessions,
recording student responses, and calculating scores. Designed for
integration with wizard workflows and artifact registry.

Example:
    >>> session = AssessmentSession(
    ...     student_id="student_001",
    ...     assessment_artifact_id="art_quiz_123",
    ...     assessment_version="1.0.0",
    ... )
    >>> session.responses.append(StudentResponse(
    ...     question_id="q1",
    ...     response="Paris",
    ...     correct=True,
    ... ))
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


def _generate_session_id() -> str:
    """Generate a unique session ID."""
    return f"sess_{uuid.uuid4().hex[:12]}"


def _now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class StudentResponse:
    """A single student response to an assessment question.

    Attributes:
        question_id: Identifier of the question answered.
        response: The student's response (string for open-ended,
            index for multiple choice).
        correct: For deterministic scoring (e.g., MC), whether the
            response was correct. None if not yet evaluated.
        rubric_score: For rubric-scored responses, the evaluation score
            (0.0 to 1.0). None if not yet evaluated.
        time_taken_ms: Time spent on this question in milliseconds.
        attempt_number: Which attempt this is (1-based).
    """

    question_id: str = ""
    response: Any = None
    correct: bool | None = None
    rubric_score: float | None = None
    time_taken_ms: int | None = None
    attempt_number: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        result: dict[str, Any] = {
            "question_id": self.question_id,
            "response": self.response,
            "attempt_number": self.attempt_number,
        }
        if self.correct is not None:
            result["correct"] = self.correct
        if self.rubric_score is not None:
            result["rubric_score"] = self.rubric_score
        if self.time_taken_ms is not None:
            result["time_taken_ms"] = self.time_taken_ms
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StudentResponse:
        """Deserialize from a dictionary."""
        return cls(
            question_id=data.get("question_id", ""),
            response=data.get("response"),
            correct=data.get("correct"),
            rubric_score=data.get("rubric_score"),
            time_taken_ms=data.get("time_taken_ms"),
            attempt_number=data.get("attempt_number", 1),
        )


@dataclass
class AssessmentSession:
    """A single assessment session tracking questions and responses.

    Represents one student's attempt at a quiz or assessment, tracking
    their responses, scores, and timing.

    Attributes:
        id: Unique session identifier.
        student_id: Identifier of the student taking the assessment.
        assessment_artifact_id: ID of the quiz/assessment artifact.
        assessment_version: Version of the assessment being taken.
        started_at: ISO 8601 timestamp when the session started.
        completed_at: ISO 8601 timestamp when completed (None if ongoing).
        responses: List of student responses.
        score: Final calculated score (None if not yet finalized).
        rubric_evaluation_id: ID of the rubric evaluation (if applicable).
    """

    id: str = field(default_factory=_generate_session_id)
    student_id: str = ""
    assessment_artifact_id: str = ""
    assessment_version: str = ""
    started_at: str = field(default_factory=_now_iso)
    completed_at: str | None = None
    responses: list[StudentResponse] = field(default_factory=list)
    score: float | None = None
    rubric_evaluation_id: str | None = None

    @property
    def is_complete(self) -> bool:
        """Check if the session has been completed."""
        return self.completed_at is not None

    @property
    def correct_count(self) -> int:
        """Count of responses marked as correct."""
        return sum(1 for r in self.responses if r.correct is True)

    @property
    def total_responses(self) -> int:
        """Total number of responses recorded."""
        return len(self.responses)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        result: dict[str, Any] = {
            "id": self.id,
            "student_id": self.student_id,
            "assessment_artifact_id": self.assessment_artifact_id,
            "assessment_version": self.assessment_version,
            "started_at": self.started_at,
            "responses": [r.to_dict() for r in self.responses],
        }
        if self.completed_at is not None:
            result["completed_at"] = self.completed_at
        if self.score is not None:
            result["score"] = self.score
        if self.rubric_evaluation_id is not None:
            result["rubric_evaluation_id"] = self.rubric_evaluation_id
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AssessmentSession:
        """Deserialize from a dictionary."""
        return cls(
            id=data.get("id", _generate_session_id()),
            student_id=data.get("student_id", ""),
            assessment_artifact_id=data.get("assessment_artifact_id", ""),
            assessment_version=data.get("assessment_version", ""),
            started_at=data.get("started_at", _now_iso()),
            completed_at=data.get("completed_at"),
            responses=[
                StudentResponse.from_dict(r) for r in data.get("responses", [])
            ],
            score=data.get("score"),
            rubric_evaluation_id=data.get("rubric_evaluation_id"),
        )


@dataclass
class CumulativePerformance:
    """Cumulative performance tracking for a student on a topic.

    Aggregates across multiple assessment sessions to track learning
    progress over time.

    Attributes:
        student_id: Identifier of the student.
        topic: Topic or subject area.
        total_sessions: Number of sessions completed.
        total_questions_attempted: Total questions across all sessions.
        correct_count: Total correct responses across all sessions.
        average_score: Running average score (0.0 to 1.0).
        mastery_estimate: Estimated mastery level (0.0 to 1.0).
        last_session_at: ISO 8601 timestamp of last session.
    """

    student_id: str = ""
    topic: str = ""
    total_sessions: int = 0
    total_questions_attempted: int = 0
    correct_count: int = 0
    average_score: float = 0.0
    mastery_estimate: float = 0.0
    last_session_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        return {
            "student_id": self.student_id,
            "topic": self.topic,
            "total_sessions": self.total_sessions,
            "total_questions_attempted": self.total_questions_attempted,
            "correct_count": self.correct_count,
            "average_score": self.average_score,
            "mastery_estimate": self.mastery_estimate,
            "last_session_at": self.last_session_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CumulativePerformance:
        """Deserialize from a dictionary."""
        return cls(
            student_id=data.get("student_id", ""),
            topic=data.get("topic", ""),
            total_sessions=data.get("total_sessions", 0),
            total_questions_attempted=data.get("total_questions_attempted", 0),
            correct_count=data.get("correct_count", 0),
            average_score=data.get("average_score", 0.0),
            mastery_estimate=data.get("mastery_estimate", 0.0),
            last_session_at=data.get("last_session_at", ""),
        )

    def update_from_session(self, session: AssessmentSession) -> None:
        """Update cumulative performance from a completed session.

        Args:
            session: The completed assessment session.
        """
        self.total_sessions += 1
        self.total_questions_attempted += session.total_responses
        self.correct_count += session.correct_count
        self.last_session_at = session.completed_at or _now_iso()

        if session.score is not None:
            # Running average
            self.average_score = (
                (self.average_score * (self.total_sessions - 1) + session.score)
                / self.total_sessions
            )

        # Simple mastery estimate based on correct rate
        if self.total_questions_attempted > 0:
            self.mastery_estimate = (
                self.correct_count / self.total_questions_attempted
            )


async def start_assessment_session(
    data: dict[str, Any],
    assessment_artifact_id: str,
    student_id: str,
    question_ids: list[str] | None = None,
) -> AssessmentSession:
    """Create a new assessment session.

    Creates a session from a quiz artifact. If ``question_ids`` is not
    provided, all questions from the artifact should be included.

    Sets ``data["_session"]`` with the session dict and
    ``data["_current_question_index"]`` to 0.

    Args:
        data: Wizard data dictionary (modified in place).
        assessment_artifact_id: ID of the quiz artifact.
        student_id: ID of the student.
        question_ids: Optional subset of question IDs to include.

    Returns:
        The created AssessmentSession.
    """
    session = AssessmentSession(
        student_id=student_id,
        assessment_artifact_id=assessment_artifact_id,
        assessment_version=data.get("_assessment_version", "1.0.0"),
    )

    data["_session"] = session.to_dict()
    data["_session_id"] = session.id
    data["_current_question_index"] = 0
    data["_total_questions"] = len(question_ids) if question_ids else 0
    data["_question_ids"] = question_ids or []

    logger.info(
        "Started assessment session %s for student %s",
        session.id,
        student_id,
    )

    return session


async def record_response(
    data: dict[str, Any],
    question_id: str,
    response: Any,
    correct: bool | None = None,
    time_taken_ms: int | None = None,
) -> StudentResponse:
    """Record a student's response to a question.

    Appends the response to the session and advances the question index.
    For multiple-choice questions, ``correct`` should be provided for
    deterministic scoring.

    Args:
        data: Wizard data dictionary (modified in place).
        question_id: ID of the question answered.
        response: The student's answer.
        correct: Whether the answer is correct (for MC).
        time_taken_ms: Time spent in milliseconds.

    Returns:
        The recorded StudentResponse.

    Raises:
        ValueError: If no active session exists.
    """
    session_data = data.get("_session")
    if not session_data:
        raise ValueError("No active session in data['_session']")

    session = AssessmentSession.from_dict(session_data)

    # Count existing responses for this question to determine attempt number
    attempt = sum(
        1 for r in session.responses if r.question_id == question_id
    ) + 1

    student_response = StudentResponse(
        question_id=question_id,
        response=response,
        correct=correct,
        time_taken_ms=time_taken_ms,
        attempt_number=attempt,
    )
    session.responses.append(student_response)

    # Update session in data
    data["_session"] = session.to_dict()

    # Advance question index
    current_index = data.get("_current_question_index", 0)
    data["_current_question_index"] = current_index + 1

    logger.info(
        "Recorded response for question %s (attempt %d, correct=%s)",
        question_id,
        attempt,
        correct,
    )

    return student_response


async def finalize_assessment(
    data: dict[str, Any],
) -> AssessmentSession:
    """Finalize an assessment session and calculate the score.

    Marks the session as completed, calculates the final score based
    on correct responses, and updates the data dict.

    Args:
        data: Wizard data dictionary (modified in place).

    Returns:
        The finalized AssessmentSession.

    Raises:
        ValueError: If no active session exists.
    """
    session_data = data.get("_session")
    if not session_data:
        raise ValueError("No active session in data['_session']")

    session = AssessmentSession.from_dict(session_data)
    session.completed_at = _now_iso()

    # Calculate score from correct count
    if session.total_responses > 0:
        session.score = session.correct_count / session.total_responses
    else:
        session.score = 0.0

    data["_session"] = session.to_dict()
    data["_assessment_score"] = session.score
    data["_assessment_complete"] = True

    logger.info(
        "Finalized assessment session %s: score=%.1f%% (%d/%d correct)",
        session.id,
        session.score * 100,
        session.correct_count,
        session.total_responses,
    )

    return session
