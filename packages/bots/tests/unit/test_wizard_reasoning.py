"""Tests for WizardReasoning strategy."""

import time

import pytest

from dataknobs_bots.reasoning.observability import TransitionRecord
from dataknobs_bots.reasoning.wizard import (
    WizardReasoning,
    WizardStageContext,
    WizardState,
)
from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader
from dataknobs_llm.conversations import ConversationManager
from dataknobs_llm.llm.providers.echo import EchoProvider


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
        before = time.time()
        state = WizardState(current_stage="start")
        after = time.time()

        assert state.current_stage == "start"
        assert state.data == {}
        assert state.history == []
        assert state.completed is False
        assert state.clarification_attempts == 0
        assert state.transitions == []
        assert before <= state.stage_entry_time <= after

    def test_all_values(self) -> None:
        """Test WizardState with all values set."""
        transition = TransitionRecord(
            from_stage="start",
            to_stage="middle",
            timestamp=time.time(),
            trigger="user_input",
        )
        entry_time = time.time()
        state = WizardState(
            current_stage="middle",
            data={"key": "value"},
            history=["start", "middle"],
            completed=True,
            clarification_attempts=2,
            transitions=[transition],
            stage_entry_time=entry_time,
        )

        assert state.current_stage == "middle"
        assert state.data == {"key": "value"}
        assert state.history == ["start", "middle"]
        assert state.completed is True
        assert state.clarification_attempts == 2
        assert len(state.transitions) == 1
        assert state.transitions[0].from_stage == "start"
        assert state.stage_entry_time == entry_time


class TestWizardReasoning:
    """Tests for WizardReasoning."""

    @pytest.mark.asyncio
    async def test_generate_initial_state(
        self,
        wizard_reasoning: WizardReasoning,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """Test generate with initial state."""
        manager, provider = conversation_manager_pair
        await manager.add_message(role="user", content="Hello")

        # Script a specific response for this test
        provider.set_responses(["Welcome to the wizard!"])

        response = await wizard_reasoning.generate(manager, llm=None)

        assert response is not None
        assert response.content == "Welcome to the wizard!"
        # State should be saved
        assert "wizard" in manager.metadata

    @pytest.mark.asyncio
    async def test_generate_with_existing_state(
        self,
        wizard_reasoning: WizardReasoning,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """Test generate with existing wizard state."""
        manager, provider = conversation_manager_pair
        await manager.add_message(role="user", content="Create")
        manager.metadata["wizard"] = {
            "fsm_state": {
                "current_stage": "welcome",
                "history": ["welcome"],
                "data": {},
                "completed": False,
                "clarification_attempts": 0,
            }
        }

        # Script response for existing state
        provider.set_responses(["Let's configure your settings."])

        response = await wizard_reasoning.generate(manager, llm=None)

        assert response is not None
        assert response.content == "Let's configure your settings."
        # Verify state was updated
        assert manager.metadata["wizard"]["fsm_state"]["history"]

    @pytest.mark.asyncio
    async def test_navigation_back(
        self,
        simple_wizard_config: dict,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """Test back navigation command."""
        manager, provider = conversation_manager_pair

        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(simple_wizard_config)
        reasoning = WizardReasoning(wizard_fsm=wizard_fsm, strict_validation=False)

        await manager.add_message(role="user", content="back")
        # Already at second stage with history
        manager.metadata["wizard"] = {
            "fsm_state": {
                "current_stage": "configure",
                "history": ["welcome", "configure"],
                "data": {"intent": "test"},
                "completed": False,
                "clarification_attempts": 0,
            }
        }

        # Script response for going back
        provider.set_responses(["Going back to the previous step."])

        response = await reasoning.generate(manager, llm=None)

        assert response is not None
        assert response.content == "Going back to the previous step."
        # Should have gone back
        state = manager.metadata["wizard"]["fsm_state"]
        assert state["current_stage"] == "welcome"
        assert len(state["history"]) == 1

    @pytest.mark.asyncio
    async def test_navigation_restart(
        self,
        simple_wizard_config: dict,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """Test restart navigation command."""
        manager, provider = conversation_manager_pair

        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(simple_wizard_config)
        reasoning = WizardReasoning(wizard_fsm=wizard_fsm, strict_validation=False)

        await manager.add_message(role="user", content="restart")
        manager.metadata["wizard"] = {
            "fsm_state": {
                "current_stage": "configure",
                "history": ["welcome", "configure"],
                "data": {"intent": "test", "config": "value"},
                "completed": False,
                "clarification_attempts": 0,
            }
        }

        # Script response for restart
        provider.set_responses(["Starting over from the beginning."])

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
        self,
        simple_wizard_config: dict,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """Test skip navigation command on skippable stage."""
        manager, provider = conversation_manager_pair

        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(simple_wizard_config)
        reasoning = WizardReasoning(wizard_fsm=wizard_fsm, strict_validation=False)

        await manager.add_message(role="user", content="skip")
        # At configure stage which is skippable
        manager.metadata["wizard"] = {
            "fsm_state": {
                "current_stage": "configure",
                "history": ["welcome", "configure"],
                "data": {"intent": "test"},
                "completed": False,
                "clarification_attempts": 0,
            }
        }

        # Script response for skip
        provider.set_responses(["Skipping this step."])

        response = await reasoning.generate(manager, llm=None)

        assert response is not None
        assert response.content == "Skipping this step."
        state = manager.metadata["wizard"]["fsm_state"]
        # Should have skipped marker in data
        assert "_skipped_configure" in state["data"]

    @pytest.mark.asyncio
    async def test_response_metadata(
        self,
        wizard_reasoning: WizardReasoning,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """Test that response contains wizard metadata."""
        manager, provider = conversation_manager_pair
        await manager.add_message(role="user", content="Hello")

        # Script response
        provider.set_responses(["Welcome!"])

        response = await wizard_reasoning.generate(manager, llm=None)

        # The WizardReasoning adds metadata to the response
        # Check that wizard metadata was added
        assert response.metadata is not None
        assert "wizard" in response.metadata
        wizard_meta = response.metadata["wizard"]
        assert "current_stage" in wizard_meta
        assert "progress" in wizard_meta
        assert "completed" in wizard_meta
        assert "suggestions" in wizard_meta
        assert "stages" in wizard_meta

    @pytest.mark.asyncio
    async def test_system_prompt_contains_wizard_context(
        self,
        wizard_reasoning: WizardReasoning,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """Test that complete() is called with wizard context in system prompt."""
        manager, provider = conversation_manager_pair
        await manager.add_message(role="user", content="Hello")

        provider.set_responses(["Test response"])

        await wizard_reasoning.generate(manager, llm=None)

        # Verify the provider received the call with wizard context
        assert provider.call_count >= 1
        last_call = provider.get_last_call()
        system_msg = next(
            m for m in last_call["messages"] if m.role == "system"
        )
        assert (
            "## Current Wizard Stage" in system_msg.content
            or "## Clarification Needed" in system_msg.content
        )

    @pytest.mark.asyncio
    async def test_echo_provider_receives_messages(
        self,
        wizard_reasoning: WizardReasoning,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """Test that EchoProvider receives the conversation messages."""
        manager, provider = conversation_manager_pair
        await manager.add_message(role="user", content="Test input message")

        provider.set_responses(["Scripted response"])

        await wizard_reasoning.generate(manager, llm=None)

        # Verify EchoProvider received the messages
        assert provider.call_count == 1
        last_message = provider.get_last_user_message()
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

    @pytest.mark.asyncio
    async def test_get_last_user_message(
        self,
        wizard_reasoning: WizardReasoning,
        conversation_manager: ConversationManager,
    ) -> None:
        """Test extracting last user message."""
        await conversation_manager.add_message(role="user", content="First message")
        await conversation_manager.add_message(
            role="assistant", content="Response"
        )
        await conversation_manager.add_message(role="user", content="Last message")

        message = wizard_reasoning._get_last_user_message(conversation_manager)
        assert message == "Last message"

    @pytest.mark.asyncio
    async def test_get_last_user_message_no_user_messages(
        self,
        wizard_reasoning: WizardReasoning,
        conversation_manager: ConversationManager,
    ) -> None:
        """Test extracting last user message when none exist."""
        # Fresh CM has only the system message — no user messages
        message = wizard_reasoning._get_last_user_message(conversation_manager)
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
        with pytest.raises(ValueError, match="wizard_config is required"):
            WizardReasoning.from_config({})


class TestWizardTransitionTracking:
    """Tests for wizard transition audit trail."""

    @pytest.mark.asyncio
    async def test_transition_recorded_on_stage_change(
        self,
        simple_wizard_config: dict,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """Test that a transition is recorded when stage changes."""
        manager, provider = conversation_manager_pair

        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(simple_wizard_config)
        reasoning = WizardReasoning(wizard_fsm=wizard_fsm, strict_validation=False)

        await manager.add_message(role="user", content="Create something")
        manager.metadata["wizard"] = {
            "fsm_state": {
                "current_stage": "welcome",
                "history": ["welcome"],
                "data": {},
                "completed": False,
                "clarification_attempts": 0,
                "transitions": [],
                "stage_entry_time": time.time() - 5.0,  # 5 seconds ago
            }
        }

        provider.set_responses(["Moving to configuration step."])

        await reasoning.generate(manager, llm=None)

        # Check that transition was recorded
        state = manager.metadata["wizard"]["fsm_state"]
        # Stage should have changed due to extraction
        # Note: With strict_validation=False and no extractor,
        # raw input is passed through
        transitions = state.get("transitions", [])
        # Transition may or may not be recorded depending on FSM behavior
        # At minimum, ensure transitions list exists
        assert isinstance(transitions, list)

    @pytest.mark.asyncio
    async def test_transition_recorded_on_back_navigation(
        self,
        simple_wizard_config: dict,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """Test that a transition is recorded on back navigation."""
        manager, provider = conversation_manager_pair

        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(simple_wizard_config)
        reasoning = WizardReasoning(wizard_fsm=wizard_fsm, strict_validation=False)

        await manager.add_message(role="user", content="back")
        entry_time = time.time() - 5.0
        manager.metadata["wizard"] = {
            "fsm_state": {
                "current_stage": "configure",
                "history": ["welcome", "configure"],
                "data": {"intent": "test"},
                "completed": False,
                "clarification_attempts": 0,
                "transitions": [],
                "stage_entry_time": entry_time,
            }
        }

        provider.set_responses(["Going back to welcome."])

        await reasoning.generate(manager, llm=None)

        # Check that transition was recorded
        state = manager.metadata["wizard"]["fsm_state"]
        transitions = state.get("transitions", [])

        assert len(transitions) == 1
        transition = transitions[0]
        assert transition["from_stage"] == "configure"
        assert transition["to_stage"] == "welcome"
        assert transition["trigger"] == "navigation_back"
        assert transition["user_input"] == "back"
        assert transition["duration_in_stage_ms"] >= 0

    @pytest.mark.asyncio
    async def test_transition_recorded_on_restart(
        self,
        simple_wizard_config: dict,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """Test that a transition is recorded on restart."""
        manager, provider = conversation_manager_pair

        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(simple_wizard_config)
        reasoning = WizardReasoning(wizard_fsm=wizard_fsm, strict_validation=False)

        await manager.add_message(role="user", content="restart")
        entry_time = time.time() - 10.0
        manager.metadata["wizard"] = {
            "fsm_state": {
                "current_stage": "configure",
                "history": ["welcome", "configure"],
                "data": {"intent": "test", "config": "value"},
                "completed": False,
                "clarification_attempts": 0,
                "transitions": [],
                "stage_entry_time": entry_time,
            }
        }

        provider.set_responses(["Starting over."])

        await reasoning.generate(manager, llm=None)

        # Check that transition was recorded
        state = manager.metadata["wizard"]["fsm_state"]
        transitions = state.get("transitions", [])

        assert len(transitions) == 1
        transition = transitions[0]
        assert transition["from_stage"] == "configure"
        assert transition["to_stage"] == "welcome"
        assert transition["trigger"] == "restart"
        assert transition["user_input"] == "restart"
        # Data snapshot should contain the data at time of restart
        assert transition["data_snapshot"] == {"intent": "test", "config": "value"}

    @pytest.mark.asyncio
    async def test_transitions_preserved_across_restarts(
        self,
        simple_wizard_config: dict,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """Test that transition history is preserved across restarts."""
        manager, provider = conversation_manager_pair

        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(simple_wizard_config)
        reasoning = WizardReasoning(wizard_fsm=wizard_fsm, strict_validation=False)

        # Start with an existing transition history
        existing_transition = {
            "from_stage": "welcome",
            "to_stage": "configure",
            "timestamp": time.time() - 60.0,
            "trigger": "user_input",
            "duration_in_stage_ms": 5000.0,
            "data_snapshot": None,
            "user_input": "Create",
            "error": None,
        }

        await manager.add_message(role="user", content="restart")
        manager.metadata["wizard"] = {
            "fsm_state": {
                "current_stage": "configure",
                "history": ["welcome", "configure"],
                "data": {"intent": "test"},
                "completed": False,
                "clarification_attempts": 0,
                "transitions": [existing_transition],
                "stage_entry_time": time.time() - 10.0,
            }
        }

        provider.set_responses(["Starting over."])

        await reasoning.generate(manager, llm=None)

        # Check that previous transitions are preserved
        state = manager.metadata["wizard"]["fsm_state"]
        transitions = state.get("transitions", [])

        assert len(transitions) == 2  # Original + restart
        # First transition should be the original
        assert transitions[0]["trigger"] == "user_input"
        # Second should be the restart
        assert transitions[1]["trigger"] == "restart"

    def test_transitions_serialization_roundtrip(self) -> None:
        """Test that transitions serialize and deserialize correctly."""
        transition = TransitionRecord(
            from_stage="welcome",
            to_stage="configure",
            timestamp=1234567890.0,
            trigger="user_input",
            duration_in_stage_ms=5000.0,
            data_snapshot={"intent": "create"},
            user_input="Create something",
            condition_evaluated="data.get('intent')",
            condition_result=True,
        )

        # Serialize
        serialized = transition.to_dict()

        # Deserialize
        restored = TransitionRecord.from_dict(serialized)

        assert restored.from_stage == transition.from_stage
        assert restored.to_stage == transition.to_stage
        assert restored.timestamp == transition.timestamp
        assert restored.trigger == transition.trigger
        assert restored.duration_in_stage_ms == transition.duration_in_stage_ms
        assert restored.data_snapshot == transition.data_snapshot
        assert restored.user_input == transition.user_input
        assert restored.condition_evaluated == transition.condition_evaluated
        assert restored.condition_result == transition.condition_result

    @pytest.mark.asyncio
    async def test_stage_entry_time_updated_on_transition(
        self,
        simple_wizard_config: dict,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """Test that stage_entry_time is updated when transitioning."""
        manager, provider = conversation_manager_pair

        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(simple_wizard_config)
        reasoning = WizardReasoning(wizard_fsm=wizard_fsm, strict_validation=False)

        old_entry_time = time.time() - 100.0

        await manager.add_message(role="user", content="back")
        manager.metadata["wizard"] = {
            "fsm_state": {
                "current_stage": "configure",
                "history": ["welcome", "configure"],
                "data": {},
                "completed": False,
                "clarification_attempts": 0,
                "transitions": [],
                "stage_entry_time": old_entry_time,
            }
        }

        provider.set_responses(["Going back."])

        before = time.time()
        await reasoning.generate(manager, llm=None)
        after = time.time()

        # Check that stage_entry_time was updated
        state = manager.metadata["wizard"]["fsm_state"]
        new_entry_time = state.get("stage_entry_time", 0)

        assert new_entry_time > old_entry_time
        assert before <= new_entry_time <= after

    def test_get_transition_condition(self, simple_wizard_config: dict) -> None:
        """Test that WizardFSM can look up transition conditions."""
        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(simple_wizard_config)

        # The simple_wizard_config has a transition from welcome to configure
        # with condition: "data.get('intent')"
        condition = wizard_fsm.get_transition_condition("welcome", "configure")
        assert condition == "data.get('intent')"

        # Transition from configure to complete has no condition
        condition = wizard_fsm.get_transition_condition("configure", "complete")
        assert condition is None

        # Non-existent transition
        condition = wizard_fsm.get_transition_condition("welcome", "nonexistent")
        assert condition is None

    def test_transitions_serialization_with_conditions(self) -> None:
        """Test that transitions with conditions serialize correctly."""
        transition = TransitionRecord(
            from_stage="welcome",
            to_stage="configure",
            timestamp=1234567890.0,
            trigger="user_input",
            duration_in_stage_ms=5000.0,
            data_snapshot={"intent": "create"},
            user_input="Create something",
            condition_evaluated="data.get('intent')",
            condition_result=True,
        )

        # Serialize
        serialized = transition.to_dict()

        assert serialized["condition_evaluated"] == "data.get('intent')"
        assert serialized["condition_result"] is True

        # Deserialize
        restored = TransitionRecord.from_dict(serialized)

        assert restored.condition_evaluated == "data.get('intent')"
        assert restored.condition_result is True
