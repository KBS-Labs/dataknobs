"""Tests for wizard auto-advance functionality (enhancement 2b).

Auto-advance allows the wizard to skip stages where all required schema
fields already have values in the wizard state.
"""

import pytest

from dataknobs_bots.reasoning.wizard import WizardReasoning, WizardState
from dataknobs_bots.reasoning.wizard_fsm import WizardFSM
from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader


class TestCanAutoAdvance:
    """Tests for _can_auto_advance logic."""

    @pytest.fixture
    def simple_fsm(self) -> WizardFSM:
        """Create a simple wizard FSM for testing."""
        config = {
            "name": "test-wizard",
            "settings": {"auto_advance_filled_stages": True},
            "stages": [
                {
                    "name": "identity",
                    "is_start": True,
                    "prompt": "What is your bot called?",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "domain_id": {"type": "string"},
                            "domain_name": {"type": "string"},
                        },
                        "required": ["domain_id", "domain_name"],
                    },
                    "transitions": [
                        {"target": "done", "condition": "data.get('domain_id')"}
                    ],
                },
                {"name": "done", "is_end": True, "prompt": "Complete!"},
            ],
        }
        loader = WizardConfigLoader()
        return loader.load_from_dict(config)

    def test_can_auto_advance_with_all_required_fields(
        self, simple_fsm: WizardFSM
    ) -> None:
        """Stage with all required fields filled can auto-advance."""
        reasoning = WizardReasoning(
            wizard_fsm=simple_fsm, auto_advance_filled_stages=True
        )

        state = WizardState(
            current_stage="identity",
            data={"domain_id": "math-tutor", "domain_name": "Math Tutor"},
        )

        stage = simple_fsm.current_metadata
        assert reasoning._can_auto_advance(state, stage) is True

    def test_cannot_auto_advance_missing_required_field(
        self, simple_fsm: WizardFSM
    ) -> None:
        """Stage missing required field cannot auto-advance."""
        reasoning = WizardReasoning(
            wizard_fsm=simple_fsm, auto_advance_filled_stages=True
        )

        state = WizardState(
            current_stage="identity",
            data={"domain_id": "math-tutor"},  # Missing domain_name
        )

        stage = simple_fsm.current_metadata
        assert reasoning._can_auto_advance(state, stage) is False

    def test_cannot_auto_advance_end_stage(self) -> None:
        """End stages cannot auto-advance."""
        config = {
            "name": "test-wizard",
            "settings": {"auto_advance_filled_stages": True},
            "stages": [
                {
                    "name": "start",
                    "is_start": True,
                    "prompt": "Start",
                    "transitions": [{"target": "end"}],
                },
                {"name": "end", "is_end": True, "prompt": "Done"},
            ],
        }
        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(config)

        reasoning = WizardReasoning(wizard_fsm=fsm, auto_advance_filled_stages=True)

        # Advance to end stage
        fsm.step({})
        state = WizardState(current_stage="end", data={"_saved": True})
        stage = fsm.current_metadata

        assert reasoning._can_auto_advance(state, stage) is False

    def test_cannot_auto_advance_when_disabled(self, simple_fsm: WizardFSM) -> None:
        """Auto-advance doesn't happen when disabled."""
        reasoning = WizardReasoning(
            wizard_fsm=simple_fsm, auto_advance_filled_stages=False  # Disabled globally
        )

        state = WizardState(
            current_stage="identity",
            data={"domain_id": "test", "domain_name": "Test"},
        )

        stage = simple_fsm.current_metadata
        assert reasoning._can_auto_advance(state, stage) is False

    def test_auto_advance_with_empty_string_value(self, simple_fsm: WizardFSM) -> None:
        """Empty string values don't count as filled."""
        reasoning = WizardReasoning(
            wizard_fsm=simple_fsm, auto_advance_filled_stages=True
        )

        state = WizardState(
            current_stage="identity",
            data={"domain_id": "valid", "domain_name": "   "},  # Whitespace-only
        )

        stage = simple_fsm.current_metadata
        assert reasoning._can_auto_advance(state, stage) is False

    def test_auto_advance_with_none_value(self, simple_fsm: WizardFSM) -> None:
        """None values don't count as filled."""
        reasoning = WizardReasoning(
            wizard_fsm=simple_fsm, auto_advance_filled_stages=True
        )

        state = WizardState(
            current_stage="identity",
            data={"domain_id": "valid", "domain_name": None},
        )

        stage = simple_fsm.current_metadata
        assert reasoning._can_auto_advance(state, stage) is False

    def test_auto_advance_treats_all_properties_as_required(self) -> None:
        """When no 'required' list, all properties are treated as required."""
        config = {
            "name": "test-wizard",
            "stages": [
                {
                    "name": "config",
                    "is_start": True,
                    "prompt": "Configure",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "field_a": {"type": "string"},
                            "field_b": {"type": "string"},
                        },
                        # No 'required' list
                    },
                    "transitions": [{"target": "done"}],
                },
                {"name": "done", "is_end": True},
            ],
        }
        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(config)

        reasoning = WizardReasoning(wizard_fsm=fsm, auto_advance_filled_stages=True)

        # Missing field_b
        state = WizardState(current_stage="config", data={"field_a": "value"})
        stage = fsm.current_metadata
        assert reasoning._can_auto_advance(state, stage) is False

        # Both fields present
        state.data["field_b"] = "value"
        assert reasoning._can_auto_advance(state, stage) is True

    def test_cannot_auto_advance_no_schema(self) -> None:
        """Stages without schema cannot auto-advance based on data."""
        config = {
            "name": "test-wizard",
            "stages": [
                {
                    "name": "welcome",
                    "is_start": True,
                    "prompt": "Welcome!",
                    # No schema
                    "transitions": [{"target": "done"}],
                },
                {"name": "done", "is_end": True},
            ],
        }
        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(config)

        reasoning = WizardReasoning(wizard_fsm=fsm, auto_advance_filled_stages=True)

        state = WizardState(current_stage="welcome", data={})
        stage = fsm.current_metadata
        assert reasoning._can_auto_advance(state, stage) is False

    def test_stage_auto_advance_false_overrides_global_true(self) -> None:
        """Stage auto_advance: false must prevent auto-advance when global is true."""
        config = {
            "name": "test-wizard",
            "settings": {"auto_advance_filled_stages": True},
            "stages": [
                {
                    "name": "review",
                    "is_start": True,
                    "auto_advance": False,  # Explicitly opt out
                    "prompt": "Review",
                    "schema": {
                        "type": "object",
                        "properties": {"confirmed": {"type": "boolean"}},
                        "required": ["confirmed"],
                    },
                    "transitions": [
                        {"target": "done", "condition": "data.get('confirmed')"},
                    ],
                },
                {"name": "done", "is_end": True},
            ],
        }
        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(config)

        reasoning = WizardReasoning(wizard_fsm=fsm, auto_advance_filled_stages=True)

        state = WizardState(current_stage="review", data={"confirmed": True})
        stage = fsm.current_metadata
        # Stage-level auto_advance: false must win over global true
        assert stage.get("auto_advance") is False
        assert reasoning._can_auto_advance(state, stage) is False

    def test_global_true_advances_when_stage_not_set(self) -> None:
        """Global auto_advance_filled_stages: true works for stages without explicit setting."""
        config = {
            "name": "test-wizard",
            "settings": {"auto_advance_filled_stages": True},
            "stages": [
                {
                    "name": "gather",
                    "is_start": True,
                    "prompt": "Gather",
                    # No auto_advance key — should defer to global
                    "schema": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"],
                    },
                    "transitions": [{"target": "done"}],
                },
                {"name": "done", "is_end": True},
            ],
        }
        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(config)

        reasoning = WizardReasoning(wizard_fsm=fsm, auto_advance_filled_stages=True)

        state = WizardState(current_stage="gather", data={"name": "Alice"})
        stage = fsm.current_metadata
        # Stage has no explicit auto_advance — loader should leave it as None
        # so the global setting applies
        assert stage.get("auto_advance") is None
        assert reasoning._can_auto_advance(state, stage) is True

    def test_stage_level_auto_advance_override(self) -> None:
        """Per-stage auto_advance setting overrides global setting."""
        config = {
            "name": "test-wizard",
            "settings": {"auto_advance_filled_stages": False},  # Disabled globally
            "stages": [
                {
                    "name": "config",
                    "is_start": True,
                    "auto_advance": True,  # Enabled for this stage
                    "prompt": "Configure",
                    "schema": {
                        "type": "object",
                        "properties": {"field": {"type": "string"}},
                        "required": ["field"],
                    },
                    "transitions": [{"target": "done"}],
                },
                {"name": "done", "is_end": True},
            ],
        }
        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(config)

        # Globally disabled but stage has auto_advance: true
        reasoning = WizardReasoning(wizard_fsm=fsm, auto_advance_filled_stages=False)

        state = WizardState(current_stage="config", data={"field": "value"})
        stage = fsm.current_metadata
        # Stage-level auto_advance is loaded by WizardConfigLoader
        assert stage.get("auto_advance") is True
        assert reasoning._can_auto_advance(state, stage) is True


class TestEvaluateCondition:
    """Tests for _evaluate_condition helper."""

    @pytest.fixture
    def reasoning(self) -> WizardReasoning:
        """Create a minimal WizardReasoning instance."""
        config = {
            "name": "test",
            "stages": [{"name": "start", "is_start": True, "is_end": True}],
        }
        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(config)
        return WizardReasoning(wizard_fsm=fsm)

    def test_evaluate_simple_condition(self, reasoning: WizardReasoning) -> None:
        """Test evaluating simple data.get() condition."""
        assert reasoning._evaluate_condition("data.get('name')", {"name": "Test"})
        assert not reasoning._evaluate_condition("data.get('name')", {})

    def test_evaluate_condition_with_comparison(
        self, reasoning: WizardReasoning
    ) -> None:
        """Test evaluating condition with comparison."""
        assert reasoning._evaluate_condition(
            "data.get('count', 0) > 5", {"count": 10}
        )
        assert not reasoning._evaluate_condition(
            "data.get('count', 0) > 5", {"count": 3}
        )

    def test_evaluate_condition_with_return(self, reasoning: WizardReasoning) -> None:
        """Test condition that already has 'return'."""
        assert reasoning._evaluate_condition(
            "return data.get('name')", {"name": "Test"}
        )

    def test_evaluate_invalid_condition(self, reasoning: WizardReasoning) -> None:
        """Invalid conditions return False without raising."""
        # Syntax error in condition
        assert not reasoning._evaluate_condition(
            "data.get('name'",  # Missing closing paren
            {"name": "Test"},
        )

    def test_evaluate_condition_falsy_values(
        self, reasoning: WizardReasoning
    ) -> None:
        """Falsy values correctly evaluate to False."""
        assert not reasoning._evaluate_condition("data.get('flag')", {"flag": False})
        assert not reasoning._evaluate_condition("data.get('flag')", {"flag": 0})
        assert not reasoning._evaluate_condition("data.get('flag')", {"flag": ""})
        assert not reasoning._evaluate_condition("data.get('flag')", {"flag": None})


class TestAutoAdvanceIntegration:
    """Integration tests for auto-advance flow."""

    @pytest.fixture
    def multi_stage_config(self) -> dict:
        """Multi-stage wizard config for testing auto-advance."""
        return {
            "name": "multi-stage-wizard",
            "settings": {"auto_advance_filled_stages": True},
            "stages": [
                {
                    "name": "identity",
                    "is_start": True,
                    "prompt": "What's your bot's name?",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "domain_id": {"type": "string"},
                            "domain_name": {"type": "string"},
                        },
                        "required": ["domain_id", "domain_name"],
                    },
                    "transitions": [
                        {"target": "llm", "condition": "data.get('domain_id')"}
                    ],
                },
                {
                    "name": "llm",
                    "prompt": "Which LLM provider?",
                    "schema": {
                        "type": "object",
                        "properties": {"llm_provider": {"type": "string"}},
                        "required": ["llm_provider"],
                    },
                    "transitions": [
                        {"target": "done", "condition": "data.get('llm_provider')"}
                    ],
                },
                {"name": "done", "is_end": True, "prompt": "Complete!"},
            ],
        }

    def test_auto_advance_skips_multiple_stages(
        self, multi_stage_config: dict
    ) -> None:
        """Wizard auto-advances through multiple stages with pre-filled data."""
        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(multi_stage_config)

        reasoning = WizardReasoning(wizard_fsm=fsm, auto_advance_filled_stages=True)

        # State with data for both identity and llm stages
        state = WizardState(
            current_stage="identity",
            data={
                "domain_id": "math-tutor",
                "domain_name": "Math Tutor",
                "llm_provider": "anthropic",
            },
            history=["identity"],
        )

        # First check: identity stage can auto-advance
        stage = fsm.current_metadata
        assert reasoning._can_auto_advance(state, stage) is True

    def test_from_config_loads_auto_advance_setting(self) -> None:
        """from_config correctly reads auto_advance_filled_stages from settings."""
        # Create a temp YAML file
        import tempfile
        from pathlib import Path

        config_content = """
name: test-wizard
settings:
  auto_advance_filled_stages: true
stages:
  - name: start
    is_start: true
    prompt: Hello
    transitions:
      - target: done
  - name: done
    is_end: true
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(config_content)
            config_path = Path(f.name)

        try:
            reasoning = WizardReasoning.from_config({"wizard_config": str(config_path)})
            assert reasoning._auto_advance_filled_stages is True
        finally:
            config_path.unlink()

    def test_settings_accessible_on_wizard_fsm(self) -> None:
        """Settings are accessible via WizardFSM.settings property."""
        config = {
            "name": "test",
            "settings": {
                "auto_advance_filled_stages": True,
                "custom_setting": "value",
            },
            "stages": [{"name": "start", "is_start": True, "is_end": True}],
        }
        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(config)

        assert fsm.settings["auto_advance_filled_stages"] is True
        assert fsm.settings["custom_setting"] == "value"


class TestSkipExtraction:
    """Tests for skip_extraction flag on WizardState (item 19).

    When auto-advance lands on a new stage, the next generate() call
    should skip extraction because the user hasn't responded to the
    landing stage yet — their message was directed at the previous stage.
    """

    def test_skip_extraction_persisted_in_state(self) -> None:
        """skip_extraction flag is included in to_dict/from_dict round-trip."""
        state = WizardState(current_stage="review", skip_extraction=True)
        serialized = state.to_dict()
        assert serialized["skip_extraction"] is True

        restored = WizardState.from_dict(serialized)
        assert restored.skip_extraction is True

    def test_skip_extraction_defaults_false(self) -> None:
        """skip_extraction defaults to False."""
        state = WizardState(current_stage="start")
        assert state.skip_extraction is False

    @pytest.mark.asyncio
    async def test_auto_advance_sets_skip_extraction(self) -> None:
        """Auto-advance loop sets skip_extraction on the landing stage."""
        config = {
            "name": "test-wizard",
            "settings": {"auto_advance_filled_stages": True},
            "stages": [
                {
                    "name": "gather",
                    "is_start": True,
                    "prompt": "Gather info",
                    "schema": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"],
                    },
                    "transitions": [{"target": "review"}],
                },
                {
                    "name": "review",
                    "prompt": "Review: {{ name }}",
                    "schema": {
                        "type": "object",
                        "properties": {"confirmed": {"type": "boolean"}},
                        "required": ["confirmed"],
                    },
                    "transitions": [
                        {"target": "done", "condition": "data.get('confirmed')"},
                    ],
                },
                {"name": "done", "is_end": True},
            ],
        }
        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(config)

        reasoning = WizardReasoning(
            wizard_fsm=fsm, auto_advance_filled_stages=True
        )

        state = WizardState(
            current_stage="gather",
            data={"name": "Alice"},
        )

        # Auto-advance should fire on gather (all fields filled),
        # land on review, and set skip_extraction
        stage = fsm.current_metadata
        await reasoning._run_auto_advance_loop(state, fsm, stage)

        assert state.current_stage == "review"
        assert state.skip_extraction is True

    @pytest.mark.asyncio
    async def test_no_skip_extraction_when_no_advance(self) -> None:
        """skip_extraction stays False when auto-advance doesn't fire."""
        config = {
            "name": "test-wizard",
            "settings": {"auto_advance_filled_stages": True},
            "stages": [
                {
                    "name": "gather",
                    "is_start": True,
                    "prompt": "Gather info",
                    "schema": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"],
                    },
                    "transitions": [{"target": "done"}],
                },
                {"name": "done", "is_end": True},
            ],
        }
        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(config)

        reasoning = WizardReasoning(
            wizard_fsm=fsm, auto_advance_filled_stages=True
        )

        # Missing required field — auto-advance won't fire
        state = WizardState(current_stage="gather", data={})

        stage = fsm.current_metadata
        await reasoning._run_auto_advance_loop(state, fsm, stage)

        assert state.current_stage == "gather"
        assert state.skip_extraction is False


class TestSkipExtractionLifecycle:
    """Tests for skip_extraction flag lifecycle across navigation paths."""

    @pytest.mark.asyncio
    async def test_navigate_back_clears_skip_extraction(self) -> None:
        """Back navigation clears skip_extraction flag."""
        config = {
            "name": "test-wizard",
            "stages": [
                {
                    "name": "start",
                    "is_start": True,
                    "prompt": "Start",
                    "schema": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"],
                    },
                    "transitions": [{"target": "review"}],
                },
                {
                    "name": "review",
                    "prompt": "Review",
                    "transitions": [{"target": "done"}],
                },
                {"name": "done", "is_end": True},
            ],
        }
        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(config)
        reasoning = WizardReasoning(wizard_fsm=fsm)

        state = WizardState(
            current_stage="review",
            data={"name": "Alice"},
            history=["start", "review"],
            skip_extraction=True,
        )
        # Sync FSM to review stage
        fsm.restore({"current_stage": "review", "data": state.data})

        result = await reasoning._navigate_back(state)
        assert result is True
        assert state.current_stage == "start"
        assert state.skip_extraction is False

    @pytest.mark.asyncio
    async def test_restart_clears_skip_extraction(self) -> None:
        """Restart clears skip_extraction flag."""
        config = {
            "name": "test-wizard",
            "stages": [
                {
                    "name": "start",
                    "is_start": True,
                    "prompt": "Start",
                    "transitions": [{"target": "done"}],
                },
                {"name": "done", "is_end": True},
            ],
        }
        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(config)
        reasoning = WizardReasoning(wizard_fsm=fsm)

        state = WizardState(
            current_stage="done",
            data={"some_field": "value"},
            history=["start", "done"],
            skip_extraction=True,
            completed=True,
        )

        await reasoning._restart_cleanup(state, "restart please")
        assert state.current_stage == "start"
        assert state.skip_extraction is False

    @pytest.mark.asyncio
    async def test_greet_auto_advance_clears_skip_extraction(self) -> None:
        """After greet() auto-advances, skip_extraction should be False.

        The user's first message after greet IS directed at the landing
        stage, so extraction should run normally.
        """
        from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader
        from dataknobs_data.backends.memory import AsyncMemoryDatabase
        from dataknobs_llm.conversations import (
            ConversationManager,
            DataknobsConversationStorage,
        )
        from dataknobs_llm.llm import LLMConfig
        from dataknobs_llm.llm.providers.echo import EchoProvider
        from dataknobs_llm.prompts import AsyncPromptBuilder, ConfigPromptLibrary

        config = {
            "name": "test-wizard",
            "stages": [
                {
                    "name": "welcome",
                    "is_start": True,
                    "auto_advance": True,
                    "prompt": "Welcome!",
                    "response_template": "Welcome to the wizard!",
                    "transitions": [{"target": "gather"}],
                },
                {
                    "name": "gather",
                    "prompt": "What is your name?",
                    "schema": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"],
                    },
                    "transitions": [{"target": "done"}],
                },
                {"name": "done", "is_end": True},
            ],
        }
        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(config)
        reasoning = WizardReasoning(wizard_fsm=fsm)

        llm_config = LLMConfig(
            provider="echo", model="echo-test", options={"echo_prefix": ""}
        )
        provider = EchoProvider(llm_config)
        library = ConfigPromptLibrary({
            "system": {"assistant": {"template": "You are a helper."}},
        })
        builder = AsyncPromptBuilder(library=library)
        storage = DataknobsConversationStorage(AsyncMemoryDatabase())
        manager = await ConversationManager.create(
            llm=provider,
            prompt_builder=builder,
            storage=storage,
            system_prompt_name="assistant",
        )

        # Script responses: one for welcome stage, one for landing stage
        provider.set_responses(["Welcome!", "What is your name?"])

        response = await reasoning.greet(manager, llm=None)
        assert response is not None

        # Verify: auto-advance landed on gather
        wizard_meta = manager.metadata.get("wizard", {})
        fsm_state = wizard_meta.get("fsm_state", {})
        assert fsm_state["current_stage"] == "gather"

        # Key assertion: skip_extraction must be False after greet
        # so the user's first message gets extracted normally
        assert fsm_state.get("skip_extraction", False) is False

        await provider.close()

    @pytest.mark.asyncio
    async def test_navigate_skip_clears_skip_extraction(self) -> None:
        """Skip navigation clears skip_extraction flag.

        Bug: if a user says "skip" right after being auto-advanced to an
        optional stage, the stale skip_extraction=True flag survives to the
        next stage, suppressing extraction on the user's first real message.
        """
        config = {
            "name": "test-wizard",
            "stages": [
                {
                    "name": "start",
                    "is_start": True,
                    "prompt": "Start",
                    "can_skip": True,
                    "schema": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                    },
                    "transitions": [{"target": "next"}],
                },
                {
                    "name": "next",
                    "prompt": "Next",
                    "transitions": [{"target": "done"}],
                },
                {"name": "done", "is_end": True},
            ],
        }
        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(config)
        reasoning = WizardReasoning(wizard_fsm=fsm)

        state = WizardState(
            current_stage="start",
            data={},
            history=["start"],
            skip_extraction=True,
        )
        fsm.restore({"current_stage": "start", "data": state.data})

        success, _msgs = await reasoning._navigate_skip(state)
        assert success is True
        assert state.current_stage == "next"
        assert state.skip_extraction is False

    @pytest.mark.asyncio
    async def test_skip_extraction_does_not_inject_stale_message(self) -> None:
        """When skip_extraction is active, the prior stage's message must not
        be injected as _message for FSM condition evaluation.

        Bug: _execute_fsm_step injects user_message as data["_message"] for
        condition evaluation.  When skip_extraction is True the user_message
        was directed at the previous stage and could spuriously trigger a
        message-conditional transition on the landing stage.
        """
        config = {
            "name": "test-wizard",
            "stages": [
                {
                    "name": "gather",
                    "is_start": True,
                    "prompt": "Enter name",
                    "schema": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"],
                    },
                    "transitions": [{"target": "review"}],
                },
                {
                    "name": "review",
                    "prompt": "Looks good?",
                    "transitions": [
                        {
                            "target": "done",
                            "condition": "data.get('_message', '') == 'magic'",
                        },
                    ],
                },
                {"name": "done", "is_end": True},
            ],
        }
        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(config)
        reasoning = WizardReasoning(wizard_fsm=fsm)

        state = WizardState(
            current_stage="review",
            data={"name": "Alice"},
            history=["gather", "review"],
            skip_extraction=True,
        )
        fsm.restore({"current_stage": "review", "data": state.data})

        # Simulate generate()'s _execute_fsm_step call with the prior
        # stage's message.  If user_message leaks as _message, the
        # condition "data.get('_message') == 'magic'" would fire and
        # transition to "done" — which is wrong.
        _skip_extraction = state.skip_extraction
        if _skip_extraction:
            state.skip_extraction = False

        from_stage, _ = await reasoning._execute_fsm_step(
            state,
            user_message=None if _skip_extraction else "magic",
        )

        # Should NOT have transitioned — the "magic" message was from the
        # prior stage and should not have been injected
        assert state.current_stage == "review"


class TestAutoAdvanceTransitionRecording:
    """Tests for auto-advance transition recording."""

    def test_auto_advance_creates_transition_record(self) -> None:
        """Auto-advance creates proper transition records."""
        from dataknobs_bots.reasoning.observability import create_transition_record

        # Verify the function works as expected for auto-advance trigger
        record = create_transition_record(
            from_stage="identity",
            to_stage="llm",
            trigger="auto_advance",
            data_snapshot={"domain_id": "test"},
            condition_evaluated="data.get('domain_id')",
            condition_result=True,
        )

        assert record.from_stage == "identity"
        assert record.to_stage == "llm"
        assert record.trigger == "auto_advance"
        assert record.condition_evaluated == "data.get('domain_id')"
        assert record.condition_result is True
