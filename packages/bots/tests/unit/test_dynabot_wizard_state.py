"""Tests for DynaBot wizard state API (enhancement 2g).

Tests the get_wizard_state() and _normalize_wizard_state() methods
added to DynaBot for public wizard state access.
"""

from unittest.mock import MagicMock

import pytest

from dataknobs_bots import DynaBot


class TestNormalizeWizardState:
    """Tests for DynaBot._normalize_wizard_state() method."""

    @pytest.fixture
    def bot(self) -> DynaBot:
        """Create a minimal DynaBot for testing normalization."""
        return DynaBot(
            llm=MagicMock(),
            prompt_builder=MagicMock(),
            conversation_storage=MagicMock(),
        )

    def test_normalize_flat_format(self, bot: DynaBot) -> None:
        """Verify normalization of flat (new) wizard state format."""
        wizard_meta = {
            "current_stage": "configure",
            "stage_index": 1,
            "total_stages": 3,
            "progress": 0.33,
            "completed": False,
            "data": {"name": "test"},
            "can_skip": True,
            "can_go_back": True,
            "suggestions": ["Skip", "Continue"],
            "history": ["welcome", "configure"],
        }

        result = bot._normalize_wizard_state(wizard_meta)

        assert result["current_stage"] == "configure"
        assert result["stage_index"] == 1
        assert result["total_stages"] == 3
        assert result["progress"] == 0.33
        assert result["completed"] is False
        assert result["data"] == {"name": "test"}
        assert result["can_skip"] is True
        assert result["can_go_back"] is True
        assert result["suggestions"] == ["Skip", "Continue"]
        assert result["history"] == ["welcome", "configure"]

    def test_normalize_nested_fsm_state_format(self, bot: DynaBot) -> None:
        """Verify normalization of nested (legacy) fsm_state format."""
        wizard_meta = {
            "fsm_state": {
                "current_stage": "step_2",
                "stage_index": 2,
                "data": {"field": "value"},
                "history": ["step_1", "step_2"],
            }
        }

        result = bot._normalize_wizard_state(wizard_meta)

        assert result["current_stage"] == "step_2"
        assert result["stage_index"] == 2
        assert result["data"] == {"field": "value"}
        assert result["history"] == ["step_1", "step_2"]

    def test_normalize_prefers_direct_over_nested(self, bot: DynaBot) -> None:
        """Verify direct fields take precedence over fsm_state."""
        wizard_meta = {
            "current_stage": "direct_stage",
            "data": {"direct": True},
            "fsm_state": {
                "current_stage": "nested_stage",
                "data": {"nested": True},
            },
        }

        result = bot._normalize_wizard_state(wizard_meta)

        assert result["current_stage"] == "direct_stage"
        assert result["data"] == {"direct": True}

    def test_normalize_stage_fallback_to_old_format(self, bot: DynaBot) -> None:
        """Verify 'stage' field is used as fallback for current_stage."""
        wizard_meta = {
            "stage": "old_format_stage",
            "data": {},
        }

        result = bot._normalize_wizard_state(wizard_meta)

        assert result["current_stage"] == "old_format_stage"

    def test_normalize_defaults(self, bot: DynaBot) -> None:
        """Verify default values are applied for missing fields."""
        wizard_meta = {}

        result = bot._normalize_wizard_state(wizard_meta)

        assert result["current_stage"] is None
        assert result["stage_index"] == 0
        assert result["total_stages"] == 0
        assert result["progress"] == 0.0
        assert result["completed"] is False
        assert result["data"] == {}
        assert result["can_skip"] is False
        assert result["can_go_back"] is True
        assert result["suggestions"] == []
        assert result["history"] == []

    def test_normalize_canonical_schema(self, bot: DynaBot) -> None:
        """Verify all canonical fields are present in output."""
        wizard_meta = {"current_stage": "test"}

        result = bot._normalize_wizard_state(wizard_meta)

        expected_keys = {
            "current_stage",
            "stage_index",
            "total_stages",
            "progress",
            "completed",
            "data",
            "can_skip",
            "can_go_back",
            "suggestions",
            "history",
        }
        assert set(result.keys()) == expected_keys


class TestGetWizardState:
    """Tests for DynaBot.get_wizard_state() method."""

    @pytest.fixture
    def bot(self) -> DynaBot:
        """Create a minimal DynaBot for testing."""
        return DynaBot(
            llm=MagicMock(),
            prompt_builder=MagicMock(),
            conversation_storage=MagicMock(),
        )

    def test_returns_none_for_unknown_conversation(self, bot: DynaBot) -> None:
        """Verify None is returned for non-existent conversation."""
        result = bot.get_wizard_state("unknown-conv-id")
        assert result is None

    def test_returns_none_when_no_wizard_metadata(self, bot: DynaBot) -> None:
        """Verify None is returned when conversation has no wizard metadata."""
        # Create a mock manager with no wizard metadata
        mock_manager = MagicMock()
        mock_manager.metadata = {"other": "data"}
        bot._conversation_managers["conv-123"] = mock_manager

        result = bot.get_wizard_state("conv-123")
        assert result is None

    def test_returns_none_when_metadata_empty(self, bot: DynaBot) -> None:
        """Verify None is returned when metadata is empty."""
        mock_manager = MagicMock()
        mock_manager.metadata = {}
        bot._conversation_managers["conv-123"] = mock_manager

        result = bot.get_wizard_state("conv-123")
        assert result is None

    def test_returns_none_when_metadata_is_none(self, bot: DynaBot) -> None:
        """Verify None is returned when metadata is None."""
        mock_manager = MagicMock()
        mock_manager.metadata = None
        bot._conversation_managers["conv-123"] = mock_manager

        result = bot.get_wizard_state("conv-123")
        assert result is None

    def test_returns_normalized_state(self, bot: DynaBot) -> None:
        """Verify wizard state is returned normalized."""
        mock_manager = MagicMock()
        mock_manager.metadata = {
            "wizard": {
                "current_stage": "step_1",
                "stage_index": 0,
                "total_stages": 3,
                "progress": 0.0,
                "completed": False,
                "data": {"user_input": "hello"},
                "can_skip": True,
                "can_go_back": False,
                "suggestions": ["Continue"],
                "history": ["step_1"],
            }
        }
        bot._conversation_managers["conv-123"] = mock_manager

        result = bot.get_wizard_state("conv-123")

        assert result is not None
        assert result["current_stage"] == "step_1"
        assert result["data"] == {"user_input": "hello"}
        assert result["can_skip"] is True
        assert result["can_go_back"] is False

    def test_returns_normalized_state_from_nested_format(self, bot: DynaBot) -> None:
        """Verify legacy nested format is normalized correctly."""
        mock_manager = MagicMock()
        mock_manager.metadata = {
            "wizard": {
                "fsm_state": {
                    "current_stage": "configure",
                    "stage_index": 1,
                    "data": {"config": "value"},
                    "history": ["welcome", "configure"],
                }
            }
        }
        bot._conversation_managers["conv-123"] = mock_manager

        result = bot.get_wizard_state("conv-123")

        assert result is not None
        assert result["current_stage"] == "configure"
        assert result["stage_index"] == 1
        assert result["data"] == {"config": "value"}
        assert result["history"] == ["welcome", "configure"]


class TestCanonicalSchema:
    """Tests verifying the canonical wizard state schema."""

    @pytest.fixture
    def bot(self) -> DynaBot:
        """Create a minimal DynaBot for testing."""
        return DynaBot(
            llm=MagicMock(),
            prompt_builder=MagicMock(),
            conversation_storage=MagicMock(),
        )

    def test_all_canonical_fields_present(self, bot: DynaBot) -> None:
        """Verify normalized state always has all canonical fields."""
        # Even with minimal input, all fields should be present
        wizard_meta = {"current_stage": "test"}

        result = bot._normalize_wizard_state(wizard_meta)

        # All canonical fields must be present
        assert "current_stage" in result
        assert "stage_index" in result
        assert "total_stages" in result
        assert "progress" in result
        assert "completed" in result
        assert "data" in result
        assert "can_skip" in result
        assert "can_go_back" in result
        assert "suggestions" in result
        assert "history" in result

    def test_field_types(self, bot: DynaBot) -> None:
        """Verify canonical fields have correct types."""
        wizard_meta = {
            "current_stage": "test",
            "stage_index": 1,
            "total_stages": 3,
            "progress": 0.5,
            "completed": False,
            "data": {"key": "value"},
            "can_skip": True,
            "can_go_back": False,
            "suggestions": ["option1", "option2"],
            "history": ["stage1", "stage2"],
        }

        result = bot._normalize_wizard_state(wizard_meta)

        assert isinstance(result["current_stage"], str)
        assert isinstance(result["stage_index"], int)
        assert isinstance(result["total_stages"], int)
        assert isinstance(result["progress"], float)
        assert isinstance(result["completed"], bool)
        assert isinstance(result["data"], dict)
        assert isinstance(result["can_skip"], bool)
        assert isinstance(result["can_go_back"], bool)
        assert isinstance(result["suggestions"], list)
        assert isinstance(result["history"], list)
