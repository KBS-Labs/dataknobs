"""Tests for focus guard system."""

import pytest

from dataknobs_bots.reasoning.focus_guard import (
    FocusContext,
    FocusEvaluation,
    FocusGuard,
)


class TestFocusContext:
    """Tests for FocusContext."""

    def test_basic_creation(self) -> None:
        """Test basic context creation."""
        ctx = FocusContext(
            primary_goal="Help user configure bot",
            current_task="Collect bot name",
        )

        assert ctx.primary_goal == "Help user configure bot"
        assert ctx.current_task == "Collect bot name"
        assert ctx.tangent_count == 0
        assert ctx.max_tangent_depth == 2

    def test_with_keywords(self) -> None:
        """Test context with keywords."""
        ctx = FocusContext(
            primary_goal="Configure bot",
            topic_keywords=["bot", "name", "configure"],
            off_topic_keywords=["weather", "sports"],
        )

        assert "bot" in ctx.topic_keywords
        assert "weather" in ctx.off_topic_keywords

    def test_is_at_tangent_limit(self) -> None:
        """Test tangent limit check."""
        ctx = FocusContext(
            primary_goal="Configure bot",
            tangent_count=0,
            max_tangent_depth=2,
        )
        assert ctx.is_at_tangent_limit is False

        ctx.tangent_count = 2
        assert ctx.is_at_tangent_limit is True

    def test_to_dict_from_dict(self) -> None:
        """Test serialization round-trip."""
        ctx = FocusContext(
            primary_goal="Configure bot",
            current_task="Get name",
            collected_data={"domain": "education"},
            tangent_count=1,
        )

        data = ctx.to_dict()
        restored = FocusContext.from_dict(data)

        assert restored.primary_goal == ctx.primary_goal
        assert restored.current_task == ctx.current_task
        assert restored.collected_data == ctx.collected_data
        assert restored.tangent_count == ctx.tangent_count


class TestFocusEvaluation:
    """Tests for FocusEvaluation."""

    def test_not_drifting(self) -> None:
        """Test evaluation for on-topic response."""
        evaluation = FocusEvaluation(
            is_drifting=False,
            drift_severity=0.0,
        )

        assert evaluation.is_drifting is False
        assert evaluation.needs_correction is False

    def test_drifting_low_severity(self) -> None:
        """Test evaluation for mild drift."""
        evaluation = FocusEvaluation(
            is_drifting=True,
            drift_severity=0.3,
        )

        assert evaluation.is_drifting is True
        assert evaluation.needs_correction is False  # Severity below threshold

    def test_drifting_high_severity(self) -> None:
        """Test evaluation for significant drift."""
        evaluation = FocusEvaluation(
            is_drifting=True,
            drift_severity=0.7,
            detected_topic="chatbot history",
            reason="Off-topic discussion",
            suggested_redirect="bot configuration",
        )

        assert evaluation.is_drifting is True
        assert evaluation.needs_correction is True
        assert evaluation.detected_topic == "chatbot history"


class TestFocusGuard:
    """Tests for FocusGuard."""

    def test_init_defaults(self) -> None:
        """Test default initialization."""
        guard = FocusGuard()

        assert guard.max_tangent_depth == 2
        assert guard.drift_threshold == 0.5
        assert guard.use_keyword_detection is True

    def test_init_custom(self) -> None:
        """Test custom initialization."""
        guard = FocusGuard(
            max_tangent_depth=3,
            drift_threshold=0.7,
        )

        assert guard.max_tangent_depth == 3
        assert guard.drift_threshold == 0.7

    def test_build_context(self) -> None:
        """Test building focus context."""
        guard = FocusGuard(max_tangent_depth=3)

        ctx = guard.build_context(
            primary_goal="Configure the bot",
            current_task="Get the bot name",
            collected_data={"domain": "education"},
            required_fields=["name", "description"],
            topic_keywords=["bot", "name", "configure"],
        )

        assert ctx.primary_goal == "Configure the bot"
        assert ctx.current_task == "Get the bot name"
        assert ctx.max_tangent_depth == 3
        assert "name" in ctx.required_fields
        assert "bot" in ctx.topic_keywords

    def test_get_focus_prompt(self) -> None:
        """Test generating focus prompt."""
        guard = FocusGuard()

        ctx = FocusContext(
            primary_goal="Help configure bot",
            current_task="Collect bot name",
            collected_data={"domain": "education"},
            required_fields=["name"],
        )

        prompt = guard.get_focus_prompt(ctx)

        assert "Focus Guidance" in prompt
        assert "Help configure bot" in prompt
        assert "Collect bot name" in prompt
        assert "name" in prompt  # Required field
        assert "domain" in prompt  # Collected data

    def test_evaluate_response_on_topic(self) -> None:
        """Test evaluating an on-topic response."""
        guard = FocusGuard()

        ctx = FocusContext(
            primary_goal="Configure the bot",
            current_task="Get bot name",
            topic_keywords=["bot", "name"],
        )

        response = "Let me help you configure your bot. What name would you like?"
        evaluation = guard.evaluate_response(response, ctx)

        assert evaluation.is_drifting is False

    def test_evaluate_response_off_topic(self) -> None:
        """Test evaluating an off-topic response."""
        guard = FocusGuard()

        ctx = FocusContext(
            primary_goal="Configure the bot",
            current_task="Get bot name",
            topic_keywords=["bot", "name", "configure"],
            off_topic_keywords=["weather", "sports", "history"],
        )

        response = "Let me tell you about the weather today and some sports news."
        evaluation = guard.evaluate_response(response, ctx)

        assert evaluation.is_drifting is True
        assert evaluation.tangent_count == 1

    def test_evaluate_empty_response(self) -> None:
        """Test evaluating empty response."""
        guard = FocusGuard()
        ctx = FocusContext(primary_goal="Configure bot")

        evaluation = guard.evaluate_response("", ctx)

        assert evaluation.is_drifting is False

    def test_get_correction_prompt(self) -> None:
        """Test generating correction prompt."""
        guard = FocusGuard()

        evaluation = FocusEvaluation(
            is_drifting=True,
            drift_severity=0.7,
            reason="Discussing off-topic content",
            suggested_redirect="bot configuration",
            tangent_count=1,
        )

        prompt = guard.get_correction_prompt(evaluation)

        assert "Focus Correction" in prompt
        assert "off-topic" in prompt
        assert "bot configuration" in prompt

    def test_get_correction_prompt_at_limit(self) -> None:
        """Test correction prompt when at tangent limit."""
        guard = FocusGuard(max_tangent_depth=2)

        evaluation = FocusEvaluation(
            is_drifting=True,
            drift_severity=0.8,
            tangent_count=2,
        )

        prompt = guard.get_correction_prompt(evaluation)

        assert "IMPORTANT" in prompt
        assert "2 turns" in prompt

    def test_update_context_after_evaluation(self) -> None:
        """Test updating context after evaluation."""
        guard = FocusGuard()

        ctx = FocusContext(
            primary_goal="Configure bot",
            tangent_count=0,
        )

        evaluation = FocusEvaluation(
            is_drifting=True,
            tangent_count=1,
        )

        updated = guard.update_context_after_evaluation(ctx, evaluation)

        assert updated.tangent_count == 1
        assert updated.primary_goal == ctx.primary_goal

    def test_from_config(self) -> None:
        """Test creating guard from config."""
        config = {
            "max_tangent_depth": 4,
            "drift_threshold": 0.6,
            "use_keyword_detection": True,
        }

        guard = FocusGuard.from_config(config)

        assert guard.max_tangent_depth == 4
        assert guard.drift_threshold == 0.6
        assert guard.use_keyword_detection is True

    def test_keyword_evaluation_no_keywords(self) -> None:
        """Test keyword evaluation with no keywords defined."""
        guard = FocusGuard()

        ctx = FocusContext(
            primary_goal="Help user",
            # No topic or off-topic keywords
        )

        # Should still work based on goal keywords
        response = "Let me help you with that."
        evaluation = guard.evaluate_response(response, ctx)

        # Can't determine drift without keywords
        assert evaluation.is_drifting is False

    def test_tangent_count_resets_on_topic(self) -> None:
        """Test that tangent count resets when back on topic."""
        guard = FocusGuard()

        ctx = FocusContext(
            primary_goal="Configure bot",
            topic_keywords=["bot", "configure"],
            tangent_count=2,  # Was off-topic before
        )

        # On-topic response
        response = "Let's configure your bot settings."
        evaluation = guard.evaluate_response(response, ctx)

        assert evaluation.is_drifting is False
        assert evaluation.tangent_count == 0  # Reset
