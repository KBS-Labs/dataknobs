"""Tests for wizard session extraction functionality.

Tests the ability to extract data from the entire wizard session conversation,
not just the current message. This allows the wizard to "remember" information
provided in earlier messages.
"""

import pytest

from dataknobs_bots.reasoning.wizard import WizardReasoning, WizardState
from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader

from .conftest import WizardTestManager


class TestBuildWizardContext:
    """Tests for _build_wizard_context method."""

    def test_no_previous_messages(self, wizard_reasoning: WizardReasoning) -> None:
        """Test with only the current message (no context to build)."""
        manager = WizardTestManager()
        manager.messages = [{"role": "user", "content": "Current message"}]

        wizard_state = WizardState(current_stage="welcome")

        context = wizard_reasoning._build_wizard_context(manager, wizard_state)

        # Should return empty string when there are no previous messages
        assert context == ""

    def test_single_previous_message(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Test with one previous user message plus current."""
        manager = WizardTestManager()
        manager.messages = [
            {"role": "user", "content": "I want to create a math tutor"},
            {"role": "assistant", "content": "Great choice!"},
            {"role": "user", "content": "Call it Algebra Ace"},
        ]

        wizard_state = WizardState(current_stage="configure_identity")

        context = wizard_reasoning._build_wizard_context(manager, wizard_state)

        assert "Previous conversation:" in context
        assert "I want to create a math tutor" in context
        # Current message should NOT be in context
        assert "Call it Algebra Ace" not in context

    def test_multiple_previous_messages(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Test with multiple previous user messages."""
        manager = WizardTestManager()
        manager.messages = [
            {"role": "user", "content": "Create a tutor for algebra"},
            {"role": "assistant", "content": "Great!"},
            {"role": "user", "content": "Call it Algebra Ace with ID algebra-ace"},
            {"role": "assistant", "content": "Got it!"},
            {"role": "user", "content": "Use Claude for the LLM"},
        ]

        wizard_state = WizardState(current_stage="configure_llm")

        context = wizard_reasoning._build_wizard_context(manager, wizard_state)

        assert "Previous conversation:" in context
        assert "Create a tutor for algebra" in context
        assert "Call it Algebra Ace with ID algebra-ace" in context
        # Current message should NOT be included
        assert "Use Claude for the LLM" not in context

    def test_truncates_long_messages(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Test that very long messages are truncated."""
        manager = WizardTestManager()
        long_message = "A" * 600  # Longer than 500 char limit
        manager.messages = [
            {"role": "user", "content": long_message},
            {"role": "user", "content": "Current message"},
        ]

        wizard_state = WizardState(current_stage="welcome")

        context = wizard_reasoning._build_wizard_context(manager, wizard_state)

        # Should be truncated to 500 + "..."
        assert "..." in context
        # Should not contain the full long message
        assert long_message not in context

    def test_handles_structured_content(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Test handling of structured message content."""
        manager = WizardTestManager()
        manager.messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Structured message content"}],
            },
            {"role": "user", "content": "Current message"},
        ]

        wizard_state = WizardState(current_stage="welcome")

        context = wizard_reasoning._build_wizard_context(manager, wizard_state)

        assert "Structured message content" in context

    def test_excludes_assistant_messages(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Test that only user messages are included in context."""
        manager = WizardTestManager()
        manager.messages = [
            {"role": "user", "content": "User message 1"},
            {"role": "assistant", "content": "Assistant should not appear"},
            {"role": "system", "content": "System should not appear"},
            {"role": "user", "content": "Current message"},
        ]

        wizard_state = WizardState(current_stage="welcome")

        context = wizard_reasoning._build_wizard_context(manager, wizard_state)

        assert "User message 1" in context
        assert "Assistant should not appear" not in context
        assert "System should not appear" not in context


class TestDetectConflicts:
    """Tests for _detect_conflicts method."""

    def test_no_conflicts(self, wizard_reasoning: WizardReasoning) -> None:
        """Test when there are no conflicting values."""
        existing = {"name": "Test Bot", "subject": "math"}
        new = {"llm_provider": "anthropic"}

        conflicts = wizard_reasoning._detect_conflicts(existing, new)

        assert conflicts == []

    def test_detects_simple_conflict(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Test detecting a simple value conflict."""
        existing = {"name": "Old Name"}
        new = {"name": "New Name"}

        conflicts = wizard_reasoning._detect_conflicts(existing, new)

        assert len(conflicts) == 1
        assert conflicts[0]["field"] == "name"
        assert conflicts[0]["previous"] == "Old Name"
        assert conflicts[0]["new"] == "New Name"

    def test_detects_multiple_conflicts(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Test detecting multiple conflicting values."""
        existing = {"name": "Old Name", "subject": "math", "llm": "openai"}
        new = {"name": "New Name", "llm": "anthropic"}

        conflicts = wizard_reasoning._detect_conflicts(existing, new)

        assert len(conflicts) == 2
        field_names = [c["field"] for c in conflicts]
        assert "name" in field_names
        assert "llm" in field_names

    def test_ignores_none_values(self, wizard_reasoning: WizardReasoning) -> None:
        """Test that None values don't count as conflicts."""
        existing = {"name": "Test Bot"}
        new = {"name": None}

        conflicts = wizard_reasoning._detect_conflicts(existing, new)

        assert conflicts == []

    def test_ignores_internal_fields(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Test that fields starting with _ are ignored."""
        existing = {"_internal": "old"}
        new = {"_internal": "new"}

        conflicts = wizard_reasoning._detect_conflicts(existing, new)

        assert conflicts == []

    def test_no_conflict_when_values_match(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Test that identical values don't count as conflicts."""
        existing = {"name": "Same Name"}
        new = {"name": "Same Name"}

        conflicts = wizard_reasoning._detect_conflicts(existing, new)

        assert conflicts == []

    def test_conflict_with_different_types(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Test conflict detection with different value types."""
        existing = {"enabled": True}
        new = {"enabled": False}

        conflicts = wizard_reasoning._detect_conflicts(existing, new)

        assert len(conflicts) == 1
        assert conflicts[0]["field"] == "enabled"


class TestExtractionScopeConfiguration:
    """Tests for extraction scope configuration."""

    def test_default_extraction_scope(
        self, simple_wizard_config: dict
    ) -> None:
        """Test that default extraction scope is wizard_session."""
        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(simple_wizard_config)
        reasoning = WizardReasoning(wizard_fsm=wizard_fsm)

        assert reasoning._extraction_scope == "wizard_session"

    def test_custom_extraction_scope(
        self, simple_wizard_config: dict
    ) -> None:
        """Test setting custom extraction scope."""
        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(simple_wizard_config)
        reasoning = WizardReasoning(
            wizard_fsm=wizard_fsm,
            extraction_scope="current_message",
        )

        assert reasoning._extraction_scope == "current_message"

    def test_default_conflict_strategy(
        self, simple_wizard_config: dict
    ) -> None:
        """Test that default conflict strategy is latest_wins."""
        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(simple_wizard_config)
        reasoning = WizardReasoning(wizard_fsm=wizard_fsm)

        assert reasoning._conflict_strategy == "latest_wins"

    def test_default_log_conflicts(self, simple_wizard_config: dict) -> None:
        """Test that conflict logging is enabled by default."""
        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(simple_wizard_config)
        reasoning = WizardReasoning(wizard_fsm=wizard_fsm)

        assert reasoning._log_conflicts is True


class TestExtractionScopeFromSettings:
    """Tests for loading extraction scope from wizard settings."""

    def test_loads_extraction_scope_from_settings(self) -> None:
        """Test that extraction scope is loaded from wizard settings."""
        config = {
            "name": "test-wizard",
            "version": "1.0",
            "settings": {
                "extraction_scope": "current_message",
                "conflict_strategy": "first_wins",
                "log_conflicts": False,
            },
            "stages": [
                {
                    "name": "welcome",
                    "is_start": True,
                    "is_end": True,
                    "prompt": "Hello",
                },
            ],
        }

        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(config)

        # Create reasoning using from_config style settings access
        reasoning = WizardReasoning(
            wizard_fsm=wizard_fsm,
            extraction_scope=wizard_fsm.settings.get(
                "extraction_scope", "wizard_session"
            ),
            conflict_strategy=wizard_fsm.settings.get(
                "conflict_strategy", "latest_wins"
            ),
            log_conflicts=wizard_fsm.settings.get("log_conflicts", True),
        )

        assert reasoning._extraction_scope == "current_message"
        assert reasoning._conflict_strategy == "first_wins"
        assert reasoning._log_conflicts is False


class TestWizardSessionExtractionIntegration:
    """Integration tests for wizard session extraction."""

    @pytest.mark.asyncio
    async def test_extraction_with_wizard_session_scope(
        self, simple_wizard_config: dict
    ) -> None:
        """Test that extraction uses wizard session context."""
        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(simple_wizard_config)
        reasoning = WizardReasoning(
            wizard_fsm=wizard_fsm,
            strict_validation=False,
            extraction_scope="wizard_session",
        )

        manager = WizardTestManager()
        # Simulate a conversation with multiple user messages
        manager.messages = [
            {"role": "user", "content": "I want to create a math tutor"},
            {"role": "assistant", "content": "Great choice!"},
            {"role": "user", "content": "Current question"},
        ]
        manager.metadata = {
            "wizard": {
                "fsm_state": {
                    "current_stage": "welcome",
                    "history": ["welcome"],
                    "data": {},
                    "completed": False,
                    "clarification_attempts": 0,
                }
            }
        }

        manager.echo_provider.set_responses(["Processing your request."])

        # This should use wizard session context for extraction
        await reasoning.generate(manager, llm=None)

        # Verify the wizard state was updated
        assert "wizard" in manager.metadata

    @pytest.mark.asyncio
    async def test_extraction_with_current_message_scope(
        self, simple_wizard_config: dict
    ) -> None:
        """Test extraction with current_message scope (original behavior)."""
        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(simple_wizard_config)
        reasoning = WizardReasoning(
            wizard_fsm=wizard_fsm,
            strict_validation=False,
            extraction_scope="current_message",
        )

        manager = WizardTestManager()
        manager.messages = [
            {"role": "user", "content": "Previous message with info"},
            {"role": "user", "content": "Current message only"},
        ]
        manager.metadata = {}

        manager.echo_provider.set_responses(["Response"])

        await reasoning.generate(manager, llm=None)

        # Should still work with original behavior
        assert "wizard" in manager.metadata


class TestConflictLogging:
    """Tests for conflict logging behavior."""

    def test_detect_conflicts_returns_all_info(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Test that conflict detection returns complete information."""
        existing = {"name": "Old", "subject": "Math"}
        new = {"name": "New", "subject": "Science"}

        conflicts = wizard_reasoning._detect_conflicts(existing, new)

        assert len(conflicts) == 2
        for conflict in conflicts:
            assert "field" in conflict
            assert "previous" in conflict
            assert "new" in conflict

    def test_empty_existing_data_no_conflicts(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Test that empty existing data produces no conflicts."""
        existing: dict = {}
        new = {"name": "New Value"}

        conflicts = wizard_reasoning._detect_conflicts(existing, new)

        assert conflicts == []

    def test_empty_new_data_no_conflicts(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Test that empty new data produces no conflicts."""
        existing = {"name": "Existing Value"}
        new: dict = {}

        conflicts = wizard_reasoning._detect_conflicts(existing, new)

        assert conflicts == []
