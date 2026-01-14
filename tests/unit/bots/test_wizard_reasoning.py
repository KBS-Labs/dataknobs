"""Tests for WizardReasoning strategy."""

import pytest

from dataknobs_bots.reasoning.wizard import (
    WizardReasoning,
    WizardStageContext,
    WizardState,
)
from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader

from .conftest import WizardTestManager


class TestWizardStageContext:
    """Tests for WizardStageContext dataclass."""

    def test_default_values(self) -> None:
        """Test default values for WizardStageContext."""
        ctx = WizardStageContext(name="test", prompt="Test prompt")

        assert ctx.name == "test"
        assert ctx.prompt == "Test prompt"
        assert ctx.schema is None
        assert ctx.suggestions == []
        assert ctx.help_text is None
        assert ctx.can_skip is False
        assert ctx.can_go_back is True
        assert ctx.tools == []

    def test_all_values(self) -> None:
        """Test WizardStageContext with all values set."""
        ctx = WizardStageContext(
            name="test",
            prompt="Test prompt",
            schema={"type": "object"},
            suggestions=["Option 1", "Option 2"],
            help_text="Some help",
            can_skip=True,
            can_go_back=False,
            tools=["tool1", "tool2"],
        )

        assert ctx.schema == {"type": "object"}
        assert ctx.suggestions == ["Option 1", "Option 2"]
        assert ctx.help_text == "Some help"
        assert ctx.can_skip is True
        assert ctx.can_go_back is False
        assert ctx.tools == ["tool1", "tool2"]


class TestWizardState:
    """Tests for WizardState dataclass."""

    def test_default_values(self) -> None:
        """Test default values for WizardState."""
        state = WizardState(current_stage="start")

        assert state.current_stage == "start"
        assert state.data == {}
        assert state.history == []
        assert state.completed is False
        assert state.clarification_attempts == 0

    def test_all_values(self) -> None:
        """Test WizardState with all values set."""
        state = WizardState(
            current_stage="middle",
            data={"key": "value"},
            history=["start", "middle"],
            completed=True,
            clarification_attempts=2,
        )

        assert state.current_stage == "middle"
        assert state.data == {"key": "value"}
        assert state.history == ["start", "middle"]
        assert state.completed is True
        assert state.clarification_attempts == 2


class TestWizardReasoning:
    """Tests for WizardReasoning."""

    @pytest.mark.asyncio
    async def test_generate_initial_state(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Test generate with initial state."""
        manager = WizardTestManager()
        manager.messages = [{"role": "user", "content": "Hello"}]
        manager.metadata = {}

        # Script a specific response for this test
        manager.echo_provider.set_responses(["Welcome to the wizard!"])

        response = await wizard_reasoning.generate(manager, llm=None)

        assert response is not None
        assert response.content == "Welcome to the wizard!"
        # State should be saved
        assert "wizard" in manager.metadata

    @pytest.mark.asyncio
    async def test_generate_with_existing_state(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Test generate with existing wizard state."""
        manager = WizardTestManager()
        manager.messages = [{"role": "user", "content": "Create"}]
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

        # Script response for existing state
        manager.echo_provider.set_responses(["Let's configure your settings."])

        response = await wizard_reasoning.generate(manager, llm=None)

        assert response is not None
        assert response.content == "Let's configure your settings."
        # Verify state was updated
        assert manager.metadata["wizard"]["fsm_state"]["history"]

    @pytest.mark.asyncio
    async def test_navigation_back(
        self, simple_wizard_config: dict
    ) -> None:
        """Test back navigation command."""
        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(simple_wizard_config)
        reasoning = WizardReasoning(wizard_fsm=wizard_fsm, strict_validation=False)

        manager = WizardTestManager()
        manager.messages = [{"role": "user", "content": "back"}]
        # Already at second stage with history
        manager.metadata = {
            "wizard": {
                "fsm_state": {
                    "current_stage": "configure",
                    "history": ["welcome", "configure"],
                    "data": {"intent": "test"},
                    "completed": False,
                    "clarification_attempts": 0,
                }
            }
        }

        # Script response for going back
        manager.echo_provider.set_responses(["Going back to the previous step."])

        response = await reasoning.generate(manager, llm=None)

        assert response is not None
        assert response.content == "Going back to the previous step."
        # Should have gone back
        state = manager.metadata["wizard"]["fsm_state"]
        assert state["current_stage"] == "welcome"
        assert len(state["history"]) == 1

    @pytest.mark.asyncio
    async def test_navigation_restart(
        self, simple_wizard_config: dict
    ) -> None:
        """Test restart navigation command."""
        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(simple_wizard_config)
        reasoning = WizardReasoning(wizard_fsm=wizard_fsm, strict_validation=False)

        manager = WizardTestManager()
        manager.messages = [{"role": "user", "content": "restart"}]
        manager.metadata = {
            "wizard": {
                "fsm_state": {
                    "current_stage": "configure",
                    "history": ["welcome", "configure"],
                    "data": {"intent": "test", "config": "value"},
                    "completed": False,
                    "clarification_attempts": 0,
                }
            }
        }

        # Script response for restart
        manager.echo_provider.set_responses(["Starting over from the beginning."])

        response = await reasoning.generate(manager, llm=None)

        assert response is not None
        assert response.content == "Starting over from the beginning."
        state = manager.metadata["wizard"]["fsm_state"]
        # Should have restarted
        assert state["current_stage"] == "welcome"
        assert state["data"] == {}
        assert len(state["history"]) == 1

    @pytest.mark.asyncio
    async def test_navigation_skip(
        self, simple_wizard_config: dict
    ) -> None:
        """Test skip navigation command on skippable stage."""
        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(simple_wizard_config)
        reasoning = WizardReasoning(wizard_fsm=wizard_fsm, strict_validation=False)

        manager = WizardTestManager()
        manager.messages = [{"role": "user", "content": "skip"}]
        # At configure stage which is skippable
        manager.metadata = {
            "wizard": {
                "fsm_state": {
                    "current_stage": "configure",
                    "history": ["welcome", "configure"],
                    "data": {"intent": "test"},
                    "completed": False,
                    "clarification_attempts": 0,
                }
            }
        }

        # Script response for skip
        manager.echo_provider.set_responses(["Skipping this step."])

        response = await reasoning.generate(manager, llm=None)

        assert response is not None
        assert response.content == "Skipping this step."
        state = manager.metadata["wizard"]["fsm_state"]
        # Should have skipped marker in data
        assert "_skipped_configure" in state["data"]

    @pytest.mark.asyncio
    async def test_response_metadata(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Test that response contains wizard metadata."""
        manager = WizardTestManager()
        manager.messages = [{"role": "user", "content": "Hello"}]
        manager.metadata = {}

        # Script response
        manager.echo_provider.set_responses(["Welcome!"])

        response = await wizard_reasoning.generate(manager, llm=None)

        # The WizardReasoning adds metadata to the response
        # Check that wizard metadata was added
        assert response.metadata is not None
        assert "wizard" in response.metadata
        wizard_meta = response.metadata["wizard"]
        assert "stage" in wizard_meta
        assert "progress" in wizard_meta
        assert "completed" in wizard_meta
        assert "suggestions" in wizard_meta

    @pytest.mark.asyncio
    async def test_complete_calls_tracked(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Test that complete() calls are tracked for verification."""
        manager = WizardTestManager()
        manager.messages = [{"role": "user", "content": "Hello"}]
        manager.metadata = {}

        manager.echo_provider.set_responses(["Test response"])

        await wizard_reasoning.generate(manager, llm=None)

        # Verify complete was called with expected parameters
        assert len(manager.complete_calls) == 1
        call = manager.complete_calls[0]
        assert "system_prompt_override" in call
        # System prompt override should contain wizard context
        # (either stage context or clarification context depending on extraction result)
        system_prompt = call["system_prompt_override"]
        assert (
            "## Current Wizard Stage" in system_prompt
            or "## Clarification Needed" in system_prompt
        )

    @pytest.mark.asyncio
    async def test_echo_provider_receives_messages(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Test that EchoProvider receives the conversation messages."""
        manager = WizardTestManager()
        manager.messages = [{"role": "user", "content": "Test input message"}]
        manager.metadata = {}

        manager.echo_provider.set_responses(["Scripted response"])

        await wizard_reasoning.generate(manager, llm=None)

        # Verify EchoProvider received the messages
        assert manager.echo_provider.call_count == 1
        last_message = manager.echo_provider.get_last_user_message()
        assert last_message == "Test input message"

    def test_validate_data_required_fields(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Test data validation for required fields."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }

        # Missing required field
        errors = wizard_reasoning._validate_data({}, schema)
        assert len(errors) == 1
        assert "name" in errors[0]

        # With required field
        errors = wizard_reasoning._validate_data({"name": "test"}, schema)
        assert len(errors) == 0

    def test_validate_data_enum_constraint(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Test data validation for enum constraints."""
        schema = {
            "type": "object",
            "properties": {"choice": {"type": "string", "enum": ["a", "b", "c"]}},
        }

        # Invalid enum value
        errors = wizard_reasoning._validate_data({"choice": "invalid"}, schema)
        assert len(errors) == 1
        assert "choice" in errors[0]

        # Valid enum value
        errors = wizard_reasoning._validate_data({"choice": "a"}, schema)
        assert len(errors) == 0

    def test_calculate_progress(
        self, simple_wizard_config: dict
    ) -> None:
        """Test progress calculation.

        Progress is calculated as: visited_stages / (total_stages - 1)
        For a 3-stage wizard (welcome, configure, complete):
        - 1 visited = 1/2 = 0.5
        - 2 visited = 2/2 = 1.0
        - 3 visited = 3/2 = 1.0 (capped)
        """
        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(simple_wizard_config)
        reasoning = WizardReasoning(wizard_fsm=wizard_fsm)

        # At start - 1 stage visited out of 3 (minus 1 for end)
        state = WizardState(current_stage="welcome", history=["welcome"])
        progress = reasoning._calculate_progress(state)
        assert progress == 0.5  # 1 / (3-1) = 0.5

        # Middle of wizard - 2 unique stages visited
        state = WizardState(
            current_stage="configure", history=["welcome", "configure"]
        )
        progress = reasoning._calculate_progress(state)
        assert progress == 1.0  # 2 / (3-1) = 1.0, capped at 1.0

        # At end - all stages visited
        state = WizardState(
            current_stage="complete",
            history=["welcome", "configure", "complete"],
        )
        progress = reasoning._calculate_progress(state)
        assert progress == 1.0

    def test_get_last_user_message(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Test extracting last user message."""
        manager = WizardTestManager()

        # String content
        manager.messages = [
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "Response"},
            {"role": "user", "content": "Last message"},
        ]
        message = wizard_reasoning._get_last_user_message(manager)
        assert message == "Last message"

        # Structured content
        manager.messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Structured message"}],
            }
        ]
        message = wizard_reasoning._get_last_user_message(manager)
        assert message == "Structured message"

        # No messages
        manager.messages = []
        message = wizard_reasoning._get_last_user_message(manager)
        assert message == ""


class TestWizardReasoningFromConfig:
    """Tests for creating WizardReasoning from config."""

    def test_from_config_basic(self, simple_wizard_config: dict) -> None:
        """Test creating WizardReasoning from config dict."""
        import tempfile

        import yaml

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(simple_wizard_config, f)
            config_path = f.name

        config = {"wizard_config": config_path, "strict_validation": True}

        reasoning = WizardReasoning.from_config(config)
        assert reasoning is not None
        assert reasoning._strict_validation is True

    def test_from_config_missing_path(self) -> None:
        """Test error when wizard_config path is missing."""
        with pytest.raises(ValueError, match="wizard_config path is required"):
            WizardReasoning.from_config({})
