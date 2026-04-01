"""Tests for the non-conversational wizard advance API.

Tests WizardReasoning.advance(), initial_stage, and get_wizard_metadata()
— the public API for advancing a wizard without DynaBot/LLM infrastructure.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from dataknobs_bots.reasoning.observability import TransitionRecord
from dataknobs_bots.reasoning.wizard import (
    ExtractionPipelineResult,
    SubflowContext,
    WizardAdvanceResult,
    WizardReasoning,
    WizardState,
)
from dataknobs_bots.reasoning.wizard_hooks import WizardHooks
from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader
from dataknobs_llm import EchoProvider
from dataknobs_llm.testing import ConfigurableExtractor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_reasoning(
    config: dict[str, Any],
    *,
    hooks: WizardHooks | None = None,
    consistent_navigation_lifecycle: bool = True,
    custom_functions: dict[str, Any] | None = None,
) -> WizardReasoning:
    """Create a WizardReasoning from a config dict."""
    loader = WizardConfigLoader()
    wizard_fsm = loader.load_from_dict(config, custom_functions=custom_functions)
    return WizardReasoning(
        wizard_fsm=wizard_fsm,
        strict_validation=False,
        hooks=hooks,
        consistent_navigation_lifecycle=consistent_navigation_lifecycle,
    )


def _make_state(
    reasoning: WizardReasoning,
    *,
    current_stage: str | None = None,
    data: dict[str, Any] | None = None,
    history: list[str] | None = None,
) -> WizardState:
    """Create a WizardState at the given stage."""
    stage = current_stage or reasoning.initial_stage
    hist = history or [stage]
    return WizardState(
        current_stage=stage,
        data=data or {},
        history=hist,
        stage_entry_time=time.time(),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAdvanceTransitions:
    """Test normal FSM transitions through advance()."""

    @pytest.mark.asyncio
    async def test_advance_transitions_to_next_stage(
        self, simple_wizard_config: dict[str, Any],
    ) -> None:
        """advance() transitions when condition is met."""
        reasoning = _make_reasoning(simple_wizard_config)
        state = _make_state(reasoning)

        result = await reasoning.advance({"intent": "create"}, state)

        assert result.transitioned is True
        assert result.stage_name == "configure"
        assert result.from_stage == "welcome"
        assert state.current_stage == "configure"

    @pytest.mark.asyncio
    async def test_advance_stays_when_condition_not_met(
        self, simple_wizard_config: dict[str, Any],
    ) -> None:
        """advance() stays at current stage when condition is not met."""
        reasoning = _make_reasoning(simple_wizard_config)
        state = _make_state(reasoning)

        result = await reasoning.advance({}, state)

        assert result.transitioned is False
        assert result.stage_name == "welcome"
        assert result.from_stage is None
        assert state.current_stage == "welcome"

    @pytest.mark.asyncio
    async def test_advance_reaches_completion(
        self, simple_wizard_config: dict[str, Any],
    ) -> None:
        """advance() sets completed when reaching end stage."""
        reasoning = _make_reasoning(simple_wizard_config)
        state = _make_state(
            reasoning,
            current_stage="configure",
            history=["welcome", "configure"],
        )

        result = await reasoning.advance({}, state)

        assert result.completed is True
        assert result.stage_name == "complete"

    @pytest.mark.asyncio
    async def test_advance_merges_user_input(
        self, simple_wizard_config: dict[str, Any],
    ) -> None:
        """advance() merges user_input into state.data."""
        reasoning = _make_reasoning(simple_wizard_config)
        state = _make_state(reasoning)

        await reasoning.advance({"intent": "create", "name": "Alice"}, state)

        assert state.data["intent"] == "create"
        assert state.data["name"] == "Alice"


class TestAdvanceNavigation:
    """Test navigation commands through advance()."""

    @pytest.mark.asyncio
    async def test_advance_navigation_back(
        self, simple_wizard_config: dict[str, Any],
    ) -> None:
        """advance() with navigation='back' goes to previous stage."""
        reasoning = _make_reasoning(simple_wizard_config)
        state = _make_state(
            reasoning,
            current_stage="configure",
            data={"intent": "test"},
            history=["welcome", "configure"],
        )

        result = await reasoning.advance({}, state, navigation="back")

        assert result.stage_name == "welcome"
        assert result.transitioned is True

    @pytest.mark.asyncio
    async def test_advance_navigation_skip(
        self, simple_wizard_config: dict[str, Any],
    ) -> None:
        """advance() with navigation='skip' skips current stage."""
        reasoning = _make_reasoning(simple_wizard_config)
        state = _make_state(
            reasoning,
            current_stage="configure",
            history=["welcome", "configure"],
        )

        result = await reasoning.advance({}, state, navigation="skip")

        assert result.stage_name == "complete"
        assert result.transitioned is True

    @pytest.mark.asyncio
    async def test_advance_navigation_restart(
        self, simple_wizard_config: dict[str, Any],
    ) -> None:
        """advance() with navigation='restart' resets to initial state."""
        reasoning = _make_reasoning(simple_wizard_config)
        state = _make_state(
            reasoning,
            current_stage="configure",
            data={"intent": "test", "name": "Alice"},
            history=["welcome", "configure"],
        )

        result = await reasoning.advance({}, state, navigation="restart")

        assert result.stage_name == "welcome"
        assert state.data == {}
        assert result.completed is False
        assert result.transitioned is True

    @pytest.mark.asyncio
    async def test_advance_restart_from_initial_stage_not_transitioned(
        self, simple_wizard_config: dict[str, Any],
    ) -> None:
        """Restart from initial stage sets transitioned=False."""
        reasoning = _make_reasoning(simple_wizard_config)
        state = _make_state(reasoning)  # already at initial stage

        result = await reasoning.advance({}, state, navigation="restart")

        assert result.stage_name == "welcome"
        assert result.transitioned is False
        assert result.from_stage is None


class TestAdvanceHooks:
    """Test hook lifecycle through advance()."""

    @pytest.mark.asyncio
    async def test_advance_fires_hooks(
        self, simple_wizard_config: dict[str, Any],
    ) -> None:
        """advance() fires enter/exit/complete hooks."""
        enter_calls: list[tuple[str, dict[str, Any]]] = []
        exit_calls: list[tuple[str, dict[str, Any]]] = []
        complete_calls: list[dict[str, Any]] = []

        hooks = WizardHooks()
        hooks.on_enter(lambda s, d: enter_calls.append((s, dict(d))))
        hooks.on_exit(lambda s, d: exit_calls.append((s, dict(d))))
        hooks.on_complete(lambda d: complete_calls.append(dict(d)))

        reasoning = _make_reasoning(simple_wizard_config, hooks=hooks)
        state = _make_state(reasoning)

        # Advance welcome -> configure (fires exit on welcome, enter on configure)
        await reasoning.advance({"intent": "create"}, state)

        assert len(exit_calls) == 1
        assert exit_calls[0][0] == "welcome"
        assert len(enter_calls) == 1
        assert enter_calls[0][0] == "configure"

        # Advance configure -> complete (fires exit, enter, complete)
        exit_calls.clear()
        enter_calls.clear()
        await reasoning.advance({}, state)

        assert len(exit_calls) == 1
        assert exit_calls[0][0] == "configure"
        assert len(enter_calls) == 1
        assert enter_calls[0][0] == "complete"
        assert len(complete_calls) == 1

    @pytest.mark.asyncio
    async def test_advance_no_transition_does_not_fire_enter_hook(
        self, simple_wizard_config: dict[str, Any],
    ) -> None:
        """advance() must NOT fire enter hook when FSM stays at same stage."""
        enter_calls: list[tuple[str, dict[str, Any]]] = []
        exit_calls: list[tuple[str, dict[str, Any]]] = []

        hooks = WizardHooks()
        hooks.on_enter(lambda s, d: enter_calls.append((s, dict(d))))
        hooks.on_exit(lambda s, d: exit_calls.append((s, dict(d))))

        reasoning = _make_reasoning(simple_wizard_config, hooks=hooks)
        state = _make_state(reasoning)

        # Advance with no data — condition not met, no transition
        result = await reasoning.advance({}, state)

        assert result.transitioned is False
        # Exit hook fires before attempting transition (expected)
        assert len(exit_calls) == 1
        # Enter hook must NOT fire — we never entered a new stage
        assert len(enter_calls) == 0


class TestAdvanceTransitionRecords:
    """Test transition audit trail through advance()."""

    @pytest.mark.asyncio
    async def test_advance_records_transitions(
        self, simple_wizard_config: dict[str, Any],
    ) -> None:
        """advance() records transitions in state.transitions."""
        reasoning = _make_reasoning(simple_wizard_config)
        state = _make_state(reasoning)

        # Advance through all stages
        await reasoning.advance({"intent": "create"}, state)
        await reasoning.advance({}, state)

        assert len(state.transitions) >= 2
        assert state.transitions[0].from_stage == "welcome"
        assert state.transitions[0].to_stage == "configure"
        assert state.transitions[0].trigger == "user_input"
        assert state.transitions[1].from_stage == "configure"
        assert state.transitions[1].to_stage == "complete"


class TestAdvanceResult:
    """Test WizardAdvanceResult content."""

    @pytest.mark.asyncio
    async def test_advance_result_contains_stage_info(
        self, simple_wizard_config: dict[str, Any],
    ) -> None:
        """WizardAdvanceResult includes prompt, schema, suggestions, nav flags."""
        reasoning = _make_reasoning(simple_wizard_config)
        state = _make_state(reasoning)

        result = await reasoning.advance({}, state)

        # Welcome stage has prompt, schema, and suggestions
        assert result.stage_prompt == "What would you like to do?"
        assert result.stage_schema is not None
        assert "properties" in result.stage_schema
        assert result.suggestions == ["Create something", "Edit something"]
        assert result.can_go_back is False  # At first stage
        assert isinstance(result.metadata, dict)

    @pytest.mark.asyncio
    async def test_advance_result_type(
        self, simple_wizard_config: dict[str, Any],
    ) -> None:
        """advance() returns a WizardAdvanceResult instance."""
        reasoning = _make_reasoning(simple_wizard_config)
        state = _make_state(reasoning)

        result = await reasoning.advance({}, state)

        assert isinstance(result, WizardAdvanceResult)


class TestAdvanceDecoupling:
    """Test that advance() is decoupled from DynaBot infrastructure."""

    @pytest.mark.asyncio
    async def test_advance_does_not_require_llm_or_manager(
        self, simple_wizard_config: dict[str, Any],
    ) -> None:
        """advance() works without LLM, ConversationManager, or DynaBot."""
        reasoning = _make_reasoning(simple_wizard_config)
        state = _make_state(reasoning)

        # Should complete without any LLM or manager setup
        result = await reasoning.advance({"intent": "create"}, state)

        assert result.transitioned is True
        assert result.stage_name == "configure"

        result = await reasoning.advance({}, state)
        assert result.completed is True


class TestInitialStageProperty:
    """Test the initial_stage property."""

    def test_initial_stage_property(
        self, simple_wizard_config: dict[str, Any],
    ) -> None:
        """initial_stage returns the start stage name."""
        reasoning = _make_reasoning(simple_wizard_config)
        assert reasoning.initial_stage == "welcome"

    def test_initial_stage_respects_is_start_marker(self) -> None:
        """initial_stage returns the is_start stage, not the first-defined."""
        config: dict[str, Any] = {
            "name": "non-first-start",
            "version": "1.0",
            "stages": [
                {
                    "name": "setup",
                    "prompt": "Setup step",
                    "transitions": [{"target": "done"}],
                },
                {
                    "name": "greeting",
                    "is_start": True,
                    "prompt": "Hello!",
                    "transitions": [{"target": "setup"}],
                },
                {
                    "name": "done",
                    "is_end": True,
                    "prompt": "Done",
                },
            ],
        }
        reasoning = _make_reasoning(config)
        assert reasoning.initial_stage == "greeting"


class TestWizardStateHistoryInit:
    """Test WizardState auto-initializes history from current_stage."""

    def test_empty_history_seeded_from_current_stage(self) -> None:
        """WizardState() with no history gets history=[current_stage]."""
        state = WizardState(current_stage="welcome")
        assert state.history == ["welcome"]

    def test_explicit_history_preserved(self) -> None:
        """Explicit history is not overwritten by __post_init__."""
        state = WizardState(
            current_stage="configure",
            history=["welcome", "configure"],
        )
        assert state.history == ["welcome", "configure"]

    @pytest.mark.asyncio
    async def test_back_navigation_works_with_simple_constructor(
        self, simple_wizard_config: dict[str, Any],
    ) -> None:
        """Back navigation works when state is created with simple constructor."""
        reasoning = _make_reasoning(simple_wizard_config)
        # Simple constructor — no explicit history
        state = WizardState(current_stage=reasoning.initial_stage)

        # Advance to second stage
        result = await reasoning.advance({"intent": "create"}, state)
        assert result.stage_name == "configure"

        # Back should work — history was auto-seeded with initial stage
        result = await reasoning.advance({}, state, navigation="back")
        assert result.transitioned is True
        assert result.stage_name == "welcome"


class TestWizardStateSerialization:
    """Test WizardState.to_dict() / from_dict() round-trip."""

    def test_round_trip_with_nested_types(self) -> None:
        """to_dict/from_dict preserves nested dataclass fields."""
        original = WizardState(
            current_stage="configure",
            data={"name": "Alice"},
            history=["welcome", "configure"],
            completed=False,
            clarification_attempts=1,
            transitions=[
                TransitionRecord(
                    from_stage="welcome",
                    to_stage="configure",
                    timestamp=1000.0,
                    trigger="user_input",
                    duration_in_stage_ms=500.0,
                    condition_evaluated="data.get('name')",
                    condition_result=True,
                ),
            ],
            stage_entry_time=1000.0,
        )

        restored = WizardState.from_dict(original.to_dict())

        assert restored.current_stage == "configure"
        assert restored.data == {"name": "Alice"}
        assert restored.history == ["welcome", "configure"]
        assert restored.clarification_attempts == 1
        assert len(restored.transitions) == 1
        assert isinstance(restored.transitions[0], TransitionRecord)
        assert restored.transitions[0].from_stage == "welcome"
        assert restored.transitions[0].condition_result is True
        assert restored.stage_entry_time == 1000.0

    def test_round_trip_json_compatible(self) -> None:
        """to_dict output survives json.dumps/json.loads."""
        import json

        state = WizardState(
            current_stage="start",
            data={"key": "value"},
            history=["start"],
            stage_entry_time=2000.0,
        )

        json_str = json.dumps(state.to_dict())
        restored = WizardState.from_dict(json.loads(json_str))

        assert restored.current_stage == "start"
        assert restored.data == {"key": "value"}

    def test_from_dict_defaults(self) -> None:
        """from_dict fills defaults for missing optional fields."""
        minimal = {"current_stage": "welcome"}
        restored = WizardState.from_dict(minimal)

        assert restored.current_stage == "welcome"
        assert restored.data == {}
        assert restored.history == ["welcome"]  # __post_init__ seeds from current_stage
        assert restored.completed is False
        assert restored.transitions == []
        assert restored.subflow_stack == []


class TestGetWizardMetadata:
    """Test get_wizard_metadata()."""

    def test_get_wizard_metadata(
        self, simple_wizard_config: dict[str, Any],
    ) -> None:
        """get_wizard_metadata() returns metadata with expected keys."""
        reasoning = _make_reasoning(simple_wizard_config)
        state = _make_state(
            reasoning,
            current_stage="configure",
            history=["welcome", "configure"],
        )

        metadata = reasoning.get_wizard_metadata(state)

        assert metadata["current_stage"] == "configure"
        assert metadata["completed"] is False
        assert metadata["total_stages"] == 3
        assert "progress" in metadata
        assert "progress_percent" in metadata
        assert "stage_prompt" in metadata
        assert "can_skip" in metadata
        assert "can_go_back" in metadata


class TestAdvanceSubflowRestore:
    """Test that advance() correctly restores subflow FSM from state."""

    @pytest.fixture
    def subflow_wizard_config(self) -> dict[str, Any]:
        """Wizard config with a multi-stage subflow network."""
        return {
            "name": "subflow-wizard",
            "version": "1.0",
            "stages": [
                {
                    "name": "welcome",
                    "is_start": True,
                    "prompt": "Welcome!",
                    "transitions": [{"target": "configure"}],
                },
                {
                    "name": "configure",
                    "prompt": "Configure",
                    "transitions": [
                        {
                            "target": "_subflow",
                            "condition": "data.get('use_subflow')",
                            "subflow": {
                                "network": "detail_flow",
                                "return_stage": "complete",
                                "data_mapping": {},
                                "result_mapping": {"detail": "detail"},
                            },
                        },
                        {"target": "complete"},
                    ],
                },
                {
                    "name": "complete",
                    "is_end": True,
                    "prompt": "Done!",
                },
            ],
            "subflows": {
                "detail_flow": {
                    "stages": [
                        {
                            "name": "detail_start",
                            "is_start": True,
                            "prompt": "Enter details",
                            "schema": {
                                "type": "object",
                                "properties": {"detail": {"type": "string"}},
                                "required": ["detail"],
                            },
                            "transitions": [
                                {
                                    "target": "detail_confirm",
                                    "condition": "data.get('detail')",
                                },
                            ],
                        },
                        {
                            "name": "detail_confirm",
                            "prompt": "Confirm details",
                            "transitions": [{"target": "detail_end"}],
                        },
                        {
                            "name": "detail_end",
                            "is_end": True,
                            "prompt": "Details complete",
                        },
                    ],
                },
            },
        }

    @pytest.mark.asyncio
    async def test_advance_restores_subflow_fsm(
        self, subflow_wizard_config: dict[str, Any],
    ) -> None:
        """advance() sets up _active_subflow_fsm when state has subflow_stack.

        Simulates a stateless server: fresh WizardReasoning receives a
        WizardState that is mid-subflow.  The advance call must operate
        on the subflow FSM, not the main FSM.

        Uses a 3-stage subflow so the first advance stays within the
        subflow (detail_start -> detail_confirm) rather than popping
        back to the parent.
        """
        reasoning = _make_reasoning(subflow_wizard_config)

        # Build a state as if we're mid-subflow at detail_start
        state = WizardState(
            current_stage="detail_start",
            data={"use_subflow": True},
            history=["detail_start"],
            stage_entry_time=1000.0,
            subflow_stack=[
                SubflowContext(
                    parent_stage="configure",
                    parent_data={"use_subflow": True},
                    parent_history=["welcome", "configure"],
                    return_stage="complete",
                    result_mapping={"detail": "detail"},
                    subflow_network="detail_flow",
                ),
            ],
        )

        # Advance within the subflow — should transition to detail_confirm
        # (not to a main-flow stage like "complete")
        result = await reasoning.advance({"detail": "some value"}, state)

        assert result.stage_name == "detail_confirm"
        # Still in the subflow
        assert result.state.is_in_subflow is True

    @pytest.mark.asyncio
    async def test_get_wizard_metadata_restores_subflow_fsm(
        self, subflow_wizard_config: dict[str, Any],
    ) -> None:
        """get_wizard_metadata() returns correct info for subflow states."""
        reasoning = _make_reasoning(subflow_wizard_config)

        state = WizardState(
            current_stage="detail_start",
            data={"use_subflow": True},
            history=["detail_start"],
            stage_entry_time=1000.0,
            subflow_stack=[
                SubflowContext(
                    parent_stage="configure",
                    parent_data={"use_subflow": True},
                    parent_history=["welcome", "configure"],
                    return_stage="complete",
                    result_mapping={"detail": "detail"},
                    subflow_network="detail_flow",
                ),
            ],
        )

        metadata = reasoning.get_wizard_metadata(state)

        # Should reflect the subflow stage, not the main flow
        assert metadata["current_stage"] == "detail_start"

    @pytest.mark.asyncio
    async def test_back_navigation_within_subflow(
        self, subflow_wizard_config: dict[str, Any],
    ) -> None:
        """Back navigation within a subflow operates on the subflow FSM.

        Start at detail_confirm (mid-subflow), navigate back — should
        return to detail_start within the subflow, not a main-flow stage.
        """
        reasoning = _make_reasoning(subflow_wizard_config)

        # Build state as if we're at detail_confirm in the subflow
        state = WizardState(
            current_stage="detail_confirm",
            data={"use_subflow": True, "detail": "some value"},
            history=["detail_start", "detail_confirm"],
            stage_entry_time=1000.0,
            subflow_stack=[
                SubflowContext(
                    parent_stage="configure",
                    parent_data={"use_subflow": True},
                    parent_history=["welcome", "configure"],
                    return_stage="complete",
                    result_mapping={"detail": "detail"},
                    subflow_network="detail_flow",
                ),
            ],
        )

        result = await reasoning.advance({}, state, navigation="back")

        assert result.transitioned is True
        assert result.stage_name == "detail_start"
        assert result.state.is_in_subflow is True


class TestGenerateBackwardCompatibility:
    """Verify generate() still works after refactoring."""

    @pytest.mark.asyncio
    async def test_generate_still_works(
        self,
        simple_wizard_config: dict[str, Any],
        conversation_manager_pair: tuple,
    ) -> None:
        """generate() produces a response after refactoring."""
        from dataknobs_llm.conversations import ConversationManager
        from dataknobs_llm.llm.providers.echo import EchoProvider

        manager: ConversationManager = conversation_manager_pair[0]
        provider: EchoProvider = conversation_manager_pair[1]

        reasoning = _make_reasoning(simple_wizard_config)
        provider.set_responses(["Welcome to the wizard!"])
        await manager.add_message(role="user", content="hello")

        response = await reasoning.generate(manager, llm=None)

        assert response is not None


class TestAutoAdvanceMessages:
    """Test that auto_advance_messages are captured in WizardAdvanceResult."""

    @pytest.fixture
    def auto_advance_wizard_config(self) -> dict[str, Any]:
        """Wizard where skipping stage 1 lands on an auto-advance stage.

        Flow: ask_name (skippable) -> greeting (auto_advance) -> done (end)

        The greeting stage has auto_advance: true, a response_template, and
        its required field ('name') is pre-filled via skip_default.  So
        skipping ask_name lands on greeting, which auto-advances through to
        done, producing a rendered template message.
        """
        return {
            "name": "auto-advance-wizard",
            "version": "1.0",
            "stages": [
                {
                    "name": "ask_name",
                    "is_start": True,
                    "prompt": "What is your name?",
                    "can_skip": True,
                    "skip_default": {"name": "Anonymous"},
                    "schema": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"],
                    },
                    "transitions": [
                        {
                            "target": "greeting",
                            "condition": "data.get('name')",
                        },
                    ],
                },
                {
                    "name": "greeting",
                    "prompt": "Hello!",
                    "auto_advance": True,
                    "response_template": "Welcome, {{ name }}!",
                    "schema": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"],
                    },
                    "transitions": [{"target": "done"}],
                },
                {
                    "name": "done",
                    "is_end": True,
                    "prompt": "All done.",
                },
            ],
        }

    @pytest.mark.asyncio
    async def test_skip_captures_auto_advance_messages(
        self, auto_advance_wizard_config: dict[str, Any],
    ) -> None:
        """Skip through auto-advance stages populates auto_advance_messages."""
        reasoning = _make_reasoning(auto_advance_wizard_config)
        state = _make_state(reasoning)

        result = await reasoning.advance({}, state, navigation="skip")

        assert result.transitioned is True
        assert result.completed is True
        assert len(result.auto_advance_messages) > 0
        assert "Welcome, Anonymous!" in result.auto_advance_messages[0]

    @pytest.mark.asyncio
    async def test_normal_advance_captures_auto_advance_messages(
        self, auto_advance_wizard_config: dict[str, Any],
    ) -> None:
        """Normal advance through auto-advance stages populates auto_advance_messages."""
        reasoning = _make_reasoning(auto_advance_wizard_config)
        state = _make_state(reasoning)

        # Provide name to satisfy condition — transitions to greeting,
        # which auto-advances through to done
        result = await reasoning.advance({"name": "Alice"}, state)

        assert result.completed is True
        assert len(result.auto_advance_messages) > 0
        assert "Welcome, Alice!" in result.auto_advance_messages[0]

    @pytest.mark.asyncio
    async def test_no_auto_advance_gives_empty_messages(
        self, simple_wizard_config: dict[str, Any],
    ) -> None:
        """auto_advance_messages is empty when no auto-advance occurs."""
        reasoning = _make_reasoning(simple_wizard_config)
        state = _make_state(reasoning)

        result = await reasoning.advance({"intent": "create"}, state)

        assert result.transitioned is True
        assert result.auto_advance_messages == []


class TestNavigationHookCoverage:
    """Assert documented hook coverage for each navigation type.

    See the 'Hooks by navigation type' table in WIZARD_ADVANCE_API.md.
    """

    @pytest.mark.asyncio
    async def test_forward_fires_exit_and_enter(
        self, simple_wizard_config: dict[str, Any],
    ) -> None:
        """Forward advance fires exit before transition, enter after."""
        exit_calls: list[str] = []
        enter_calls: list[str] = []
        hooks = WizardHooks()
        hooks.on_exit(lambda s, d: exit_calls.append(s))
        hooks.on_enter(lambda s, d: enter_calls.append(s))

        reasoning = _make_reasoning(simple_wizard_config, hooks=hooks)
        state = _make_state(reasoning)

        await reasoning.advance({"intent": "create"}, state)

        assert exit_calls == ["welcome"]
        assert enter_calls == ["configure"]

    @pytest.mark.asyncio
    async def test_back_fires_enter_but_not_exit(
        self, simple_wizard_config: dict[str, Any],
    ) -> None:
        """Back fires enter on destination but NOT exit on source."""
        reasoning = _make_reasoning(simple_wizard_config)
        state = _make_state(reasoning)

        # Advance to configure first
        await reasoning.advance({"intent": "create"}, state)
        assert state.current_stage == "configure"

        # Now track hooks for back
        exit_calls: list[str] = []
        enter_calls: list[str] = []
        hooks = WizardHooks()
        hooks.on_exit(lambda s, d: exit_calls.append(s))
        hooks.on_enter(lambda s, d: enter_calls.append(s))
        reasoning._hooks = hooks

        await reasoning.advance({}, state, navigation="back")

        assert exit_calls == []  # No exit hook on back
        assert enter_calls == ["welcome"]

    @pytest.mark.asyncio
    async def test_skip_fires_enter_but_not_exit(
        self, auto_advance_wizard_config: dict[str, Any],
    ) -> None:
        """Skip fires post-transition lifecycle (enter) but NOT exit."""
        exit_calls: list[str] = []
        enter_calls: list[str] = []
        hooks = WizardHooks()
        hooks.on_exit(lambda s, d: exit_calls.append(s))
        hooks.on_enter(lambda s, d: enter_calls.append(s))

        reasoning = _make_reasoning(auto_advance_wizard_config, hooks=hooks)
        state = _make_state(reasoning)

        await reasoning.advance({}, state, navigation="skip")

        assert exit_calls == []  # No exit hook on skip
        assert len(enter_calls) > 0  # Enter hook fires

    @pytest.mark.asyncio
    async def test_restart_fires_restart_hook_only(
        self, simple_wizard_config: dict[str, Any],
    ) -> None:
        """Restart fires restart hook but NOT enter or exit."""
        exit_calls: list[str] = []
        enter_calls: list[str] = []
        restart_calls: list[bool] = []
        hooks = WizardHooks()
        hooks.on_exit(lambda s, d: exit_calls.append(s))
        hooks.on_enter(lambda s, d: enter_calls.append(s))
        hooks.on_restart(lambda: restart_calls.append(True))

        reasoning = _make_reasoning(simple_wizard_config, hooks=hooks)
        state = _make_state(reasoning)

        # Advance first so restart has somewhere to go back from
        await reasoning.advance({"intent": "create"}, state)
        exit_calls.clear()
        enter_calls.clear()

        await reasoning.advance({}, state, navigation="restart")

        assert exit_calls == []  # No exit hook on restart
        assert enter_calls == []  # No enter hook on restart
        assert len(restart_calls) == 1

    @pytest.fixture
    def auto_advance_wizard_config(self) -> dict[str, Any]:
        """Wizard with a skippable stage leading to auto-advance."""
        return {
            "name": "auto-advance-wizard",
            "version": "1.0",
            "stages": [
                {
                    "name": "ask_name",
                    "is_start": True,
                    "prompt": "What is your name?",
                    "can_skip": True,
                    "skip_default": {"name": "Anonymous"},
                    "schema": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"],
                    },
                    "transitions": [
                        {
                            "target": "greeting",
                            "condition": "data.get('name')",
                        },
                    ],
                },
                {
                    "name": "greeting",
                    "prompt": "Hello!",
                    "auto_advance": True,
                    "response_template": "Welcome, {{ name }}!",
                    "schema": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"],
                    },
                    "transitions": [{"target": "done"}],
                },
                {
                    "name": "done",
                    "is_end": True,
                    "prompt": "All done.",
                },
            ],
        }


class TestConsistentNavigationLifecycleFlag:
    """Test the consistent_navigation_lifecycle flag behavior."""

    @pytest.mark.asyncio
    async def test_back_fires_enter_hook_when_enabled(
        self, simple_wizard_config: dict[str, Any],
    ) -> None:
        """Back navigation fires enter hook when flag is True (default)."""
        enter_calls: list[str] = []
        hooks = WizardHooks()
        hooks.on_enter(lambda s, d: enter_calls.append(s))

        reasoning = _make_reasoning(
            simple_wizard_config, hooks=hooks,
            consistent_navigation_lifecycle=True,
        )
        state = _make_state(
            reasoning,
            current_stage="configure",
            data={"intent": "test"},
            history=["welcome", "configure"],
        )

        await reasoning.advance({}, state, navigation="back")

        assert "welcome" in enter_calls

    @pytest.mark.asyncio
    async def test_back_skips_enter_hook_when_disabled(
        self, simple_wizard_config: dict[str, Any],
    ) -> None:
        """Back navigation does NOT fire enter hook when flag is False."""
        enter_calls: list[str] = []
        hooks = WizardHooks()
        hooks.on_enter(lambda s, d: enter_calls.append(s))

        reasoning = _make_reasoning(
            simple_wizard_config, hooks=hooks,
            consistent_navigation_lifecycle=False,
        )
        state = _make_state(
            reasoning,
            current_stage="configure",
            data={"intent": "test"},
            history=["welcome", "configure"],
        )

        await reasoning.advance({}, state, navigation="back")

        assert len(enter_calls) == 0

    @pytest.mark.asyncio
    async def test_skip_fires_lifecycle_when_enabled(
        self, simple_wizard_config: dict[str, Any],
    ) -> None:
        """Skip fires post-transition lifecycle (hooks, auto-advance) when True."""
        enter_calls: list[str] = []
        complete_calls: list[dict[str, Any]] = []
        hooks = WizardHooks()
        hooks.on_enter(lambda s, d: enter_calls.append(s))
        hooks.on_complete(lambda d: complete_calls.append(dict(d)))

        reasoning = _make_reasoning(
            simple_wizard_config, hooks=hooks,
            consistent_navigation_lifecycle=True,
        )
        state = _make_state(
            reasoning,
            current_stage="configure",
            history=["welcome", "configure"],
        )

        result = await reasoning.advance({}, state, navigation="skip")

        # Skip configure -> complete fires enter + complete hooks
        assert result.stage_name == "complete"
        assert "complete" in enter_calls
        assert len(complete_calls) == 1

    @pytest.mark.asyncio
    async def test_skip_skips_lifecycle_when_disabled(
        self, simple_wizard_config: dict[str, Any],
    ) -> None:
        """Skip does NOT fire post-transition lifecycle when flag is False."""
        enter_calls: list[str] = []
        complete_calls: list[dict[str, Any]] = []
        hooks = WizardHooks()
        hooks.on_enter(lambda s, d: enter_calls.append(s))
        hooks.on_complete(lambda d: complete_calls.append(dict(d)))

        reasoning = _make_reasoning(
            simple_wizard_config, hooks=hooks,
            consistent_navigation_lifecycle=False,
        )
        state = _make_state(
            reasoning,
            current_stage="configure",
            history=["welcome", "configure"],
        )

        result = await reasoning.advance({}, state, navigation="skip")

        # Skip still transitions but no lifecycle hooks
        assert result.stage_name == "complete"
        assert len(enter_calls) == 0
        assert len(complete_calls) == 0


class TestSkipMessageInjection:
    """Test that _message is injected into state.data during skip.

    The original _execute_skip() always injected state.data["_message"].
    After refactoring, _navigate_skip() must pass user_message through to
    _execute_fsm_step() so that transition conditions referencing _message
    still work.
    """

    @pytest.mark.asyncio
    async def test_skip_injects_message_in_transition_record(
        self,
    ) -> None:
        """Skip via _execute_skip passes user message to transition record."""
        # Config where skip is allowed and transition records user_input
        config = {
            "name": "test-wizard",
            "stages": [
                {
                    "name": "start",
                    "is_start": True,
                    "prompt": "Start",
                    "can_skip": True,
                    "transitions": [{"target": "end"}],
                },
                {"name": "end", "is_end": True, "prompt": "Done"},
            ],
        }
        reasoning = _make_reasoning(config)
        state = _make_state(reasoning, current_stage="start")

        # Use advance (non-conversational path) with skip
        result = await reasoning.advance({}, state, navigation="skip")

        assert result.stage_name == "end"
        assert result.transitioned is True
        # Transition was recorded
        skip_transitions = [
            t for t in state.transitions if t.trigger == "navigation_skip"
        ]
        assert len(skip_transitions) == 1

    @pytest.mark.asyncio
    async def test_skip_with_message_condition(self) -> None:
        """Skip correctly injects _message for condition evaluation.

        Verifies that _message is available in state.data during the FSM
        step when skip is called with a user_message.  Uses a condition
        that reads _message to prove it was injected.
        """
        config = {
            "name": "test-wizard",
            "stages": [
                {
                    "name": "start",
                    "is_start": True,
                    "prompt": "Start",
                    "can_skip": True,
                    "transitions": [
                        {
                            "target": "special",
                            "condition": "data.get('_message') == 'skip please'",
                            "priority": 0,
                        },
                        {"target": "default", "priority": 1},
                    ],
                },
                {"name": "special", "is_end": True, "prompt": "Special"},
                {"name": "default", "is_end": True, "prompt": "Default"},
            ],
        }
        reasoning = _make_reasoning(config)

        # Call _navigate_skip with user_message — should match the condition
        state = _make_state(reasoning, current_stage="start")
        await reasoning._navigate_skip(state, user_message="skip please")

        assert state.current_stage == "special"

        # Without message — should fall through to default
        reasoning2 = _make_reasoning(config)
        state2 = _make_state(reasoning2, current_stage="start")
        await reasoning2._navigate_skip(state2)

        assert state2.current_stage == "default"


class TestSkipTurnContext:
    """Test that TurnContext is set during skip FSM step.

    The original _execute_skip() did NOT set self._current_turn.
    After refactoring, _execute_fsm_step() always sets it, which is
    an improvement for transforms that access TurnContext during skip.
    """

    @pytest.mark.asyncio
    async def test_skip_sets_turn_context_during_step(self) -> None:
        """_current_turn is set during skip FSM step and cleared after."""
        config = {
            "name": "test-wizard",
            "stages": [
                {
                    "name": "start",
                    "is_start": True,
                    "prompt": "Start",
                    "can_skip": True,
                    "transitions": [{"target": "end"}],
                },
                {"name": "end", "is_end": True, "prompt": "Done"},
            ],
        }
        reasoning = _make_reasoning(config)
        state = _make_state(reasoning, current_stage="start")

        # Before skip, _current_turn should be None
        assert reasoning._current_turn is None

        await reasoning._navigate_skip(state, user_message="skip it")

        # After skip, _current_turn should be cleared
        assert reasoning._current_turn is None
        # But the skip did transition
        assert state.current_stage == "end"


class TestFromConfigFlag:
    """Test that consistent_navigation_lifecycle is wired through from_config."""

    def test_from_config_default_is_true(self) -> None:
        """from_config defaults consistent_navigation_lifecycle to True."""
        config = {
            "name": "test-wizard",
            "stages": [
                {"name": "start", "is_start": True, "prompt": "S",
                 "transitions": [{"target": "end"}]},
                {"name": "end", "is_end": True, "prompt": "E"},
            ],
        }
        reasoning = _make_reasoning(config)
        assert reasoning._consistent_navigation_lifecycle is True

    def test_from_config_respects_false(self) -> None:
        """from_config passes consistent_navigation_lifecycle=False."""
        config = {
            "name": "test-wizard",
            "stages": [
                {"name": "start", "is_start": True, "prompt": "S",
                 "transitions": [{"target": "end"}]},
                {"name": "end", "is_end": True, "prompt": "E"},
            ],
        }
        reasoning = _make_reasoning(
            config, consistent_navigation_lifecycle=False,
        )
        assert reasoning._consistent_navigation_lifecycle is False

    def test_from_config_class_method_reads_flag(self) -> None:
        """WizardReasoning.from_config() reads the flag from config dict."""
        config = {
            "wizard_config": {
                "name": "test-wizard",
                "stages": [
                    {"name": "start", "is_start": True, "prompt": "S",
                     "transitions": [{"target": "end"}]},
                    {"name": "end", "is_end": True, "prompt": "E"},
                ],
            },
            "strict_validation": False,
            "consistent_navigation_lifecycle": False,
        }
        reasoning = WizardReasoning.from_config(config)
        assert reasoning._consistent_navigation_lifecycle is False

    def test_from_config_class_method_defaults_true(self) -> None:
        """WizardReasoning.from_config() defaults to True when key absent."""
        config = {
            "wizard_config": {
                "name": "test-wizard",
                "stages": [
                    {"name": "start", "is_start": True, "prompt": "S",
                     "transitions": [{"target": "end"}]},
                    {"name": "end", "is_end": True, "prompt": "E"},
                ],
            },
            "strict_validation": False,
        }
        reasoning = WizardReasoning.from_config(config)
        assert reasoning._consistent_navigation_lifecycle is True


# ---------------------------------------------------------------------------
# Extraction mode tests
# ---------------------------------------------------------------------------


class TestAdvanceExtractionMode:
    """Test advance() with string input (extraction mode)."""

    @pytest.mark.asyncio
    async def test_str_input_requires_llm(
        self, simple_wizard_config: dict[str, Any],
    ) -> None:
        """advance() raises ValueError when str input without llm."""
        reasoning = _make_reasoning(simple_wizard_config)
        state = _make_state(reasoning)

        with pytest.raises(ValueError, match="llm parameter is required"):
            await reasoning.advance("hello", state)

    @pytest.mark.asyncio
    async def test_str_input_runs_extraction(self) -> None:
        """advance() with str input runs extraction pipeline."""
        config = {
            "name": "test",
            "version": "1.0",
            "stages": [
                {
                    "name": "gather",
                    "is_start": True,
                    "prompt": "Provide details",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "age": {"type": "integer"},
                        },
                        "required": ["name"],
                    },
                    "transitions": [
                        {
                            "target": "done",
                            "condition": "data.get('name')",
                        },
                    ],
                },
                {"name": "done", "is_end": True, "prompt": "Done!"},
            ],
        }
        reasoning = _make_reasoning(config)
        extractor = ConfigurableExtractor(
            result_data={"name": "Alice", "age": 30}, confidence=0.95,
        )
        reasoning.set_extractor(extractor)
        state = _make_state(reasoning)
        provider = EchoProvider({"provider": "echo", "model": "test"})

        result = await reasoning.advance(
            "My name is Alice and I am 30", state, llm=provider,
        )

        # Extraction result is populated
        assert result.extraction is not None
        assert result.extraction.confidence == 0.95
        # Extracted data was merged into state
        assert state.data.get("name") == "Alice"
        # changed_fields tracks what was extracted
        assert result.changed_fields is not None
        assert "name" in result.changed_fields
        # Transition happened (gather → done via name condition)
        assert result.transitioned is True
        assert result.stage_name == "done"

    @pytest.mark.asyncio
    async def test_str_input_reports_missing_fields(self) -> None:
        """advance() with str input reports missing required fields."""
        config = {
            "name": "test",
            "version": "1.0",
            "stages": [
                {
                    "name": "gather",
                    "is_start": True,
                    "prompt": "Provide your details",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "email": {"type": "string"},
                        },
                        "required": ["name", "email"],
                    },
                    "transitions": [
                        {
                            "target": "done",
                            "condition": (
                                "data.get('name') and data.get('email')"
                            ),
                        },
                    ],
                },
                {"name": "done", "is_end": True, "prompt": "Done!"},
            ],
        }
        reasoning = _make_reasoning(config)
        # Extractor only returns name, not email
        extractor = ConfigurableExtractor(
            result_data={"name": "Alice"}, confidence=0.5,
        )
        reasoning.set_extractor(extractor)
        state = _make_state(reasoning)
        provider = EchoProvider({"provider": "echo", "model": "test"})

        result = await reasoning.advance(
            "My name is Alice", state, llm=provider,
        )

        assert result.missing_fields is not None
        assert "email" in result.missing_fields
        # Name was still merged
        assert state.data.get("name") == "Alice"
        # Did not transition (missing email)
        assert result.transitioned is False

    @pytest.mark.asyncio
    async def test_dict_input_no_extraction_fields(
        self, simple_wizard_config: dict[str, Any],
    ) -> None:
        """advance() with dict input has None extraction/missing_fields."""
        reasoning = _make_reasoning(simple_wizard_config)
        state = _make_state(reasoning)

        result = await reasoning.advance({"intent": "create"}, state)

        assert result.extraction is None
        assert result.missing_fields is None
        # But the dict was merged and transition happened
        assert result.transitioned is True
        assert state.data["intent"] == "create"

    @pytest.mark.asyncio
    async def test_str_input_on_schema_less_stage(self) -> None:
        """advance() with str input on a stage without schema."""
        config = {
            "name": "test",
            "version": "1.0",
            "stages": [
                {
                    "name": "chat",
                    "is_start": True,
                    "prompt": "Tell me anything",
                    "transitions": [{"target": "done"}],
                },
                {"name": "done", "is_end": True, "prompt": "Done!"},
            ],
        }
        reasoning = _make_reasoning(config)
        state = _make_state(reasoning)
        provider = EchoProvider({"provider": "echo", "model": "test"})

        result = await reasoning.advance(
            "hello world", state, llm=provider,
        )

        # Raw input captured, no missing fields
        assert result.extraction is not None
        assert result.missing_fields == set()
        # Unconditional transition fires
        assert result.transitioned is True

    @pytest.mark.asyncio
    async def test_str_input_with_navigation_skips_extraction(
        self, simple_wizard_config: dict[str, Any],
    ) -> None:
        """advance() with str input and navigation skips extraction."""
        reasoning = _make_reasoning(simple_wizard_config)
        state = _make_state(
            reasoning,
            current_stage="configure",
            history=["welcome", "configure"],
        )
        provider = EchoProvider({"provider": "echo", "model": "test"})

        result = await reasoning.advance(
            "go back", state, llm=provider, navigation="back",
        )

        # Navigation happened, extraction was skipped
        assert result.extraction is None
        assert result.missing_fields is None
        assert result.stage_name == "welcome"


# ---------------------------------------------------------------------------
# Routing transforms tests
# ---------------------------------------------------------------------------


class TestRoutingTransforms:
    """Test routing transforms in wizard config."""

    @pytest.mark.asyncio
    async def test_routing_transform_sets_signal(self) -> None:
        """Routing transform sets a signal used by transition conditions."""

        def set_route_signal(data: dict[str, Any]) -> dict[str, Any]:
            data["_route"] = "path_a"
            return data

        config = {
            "name": "routing-test",
            "version": "1.0",
            "stages": [
                {
                    "name": "start",
                    "is_start": True,
                    "prompt": "Start",
                    "routing_transforms": ["set_route_signal"],
                    "transitions": [
                        {
                            "target": "path_a",
                            "condition": "data.get('_route') == 'path_a'",
                        },
                        {"target": "path_b"},
                    ],
                },
                {"name": "path_a", "is_end": True, "prompt": "A"},
                {"name": "path_b", "is_end": True, "prompt": "B"},
            ],
        }
        reasoning = _make_reasoning(
            config,
            custom_functions={"set_route_signal": set_route_signal},
        )
        state = _make_state(reasoning)

        result = await reasoning.advance({"input": "test"}, state)

        # The routing transform set _route='path_a', so path_a wins
        assert result.transitioned is True
        assert result.stage_name == "path_a"

    @pytest.mark.asyncio
    async def test_async_routing_transform(self) -> None:
        """Async routing transforms are awaited correctly."""

        async def async_classify(data: dict[str, Any]) -> dict[str, Any]:
            data["_route"] = "async_path"
            return data

        config = {
            "name": "async-routing-test",
            "version": "1.0",
            "stages": [
                {
                    "name": "start",
                    "is_start": True,
                    "prompt": "Start",
                    "routing_transforms": ["async_classify"],
                    "transitions": [
                        {
                            "target": "async_path",
                            "condition": (
                                "data.get('_route') == 'async_path'"
                            ),
                        },
                        {"target": "fallback"},
                    ],
                },
                {"name": "async_path", "is_end": True, "prompt": "Async"},
                {"name": "fallback", "is_end": True, "prompt": "Fallback"},
            ],
        }
        reasoning = _make_reasoning(
            config,
            custom_functions={"async_classify": async_classify},
        )
        state = _make_state(reasoning)

        result = await reasoning.advance({"input": "test"}, state)

        assert result.transitioned is True
        assert result.stage_name == "async_path"

    @pytest.mark.asyncio
    async def test_no_routing_transforms_is_noop(
        self, simple_wizard_config: dict[str, Any],
    ) -> None:
        """Stages without routing_transforms work normally."""
        reasoning = _make_reasoning(simple_wizard_config)
        state = _make_state(reasoning)

        result = await reasoning.advance({"intent": "create"}, state)

        assert result.transitioned is True
        assert result.stage_name == "configure"

    @pytest.mark.asyncio
    async def test_missing_routing_transform_logs_warning(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Unknown routing transform name logs a warning."""
        config = {
            "name": "test",
            "version": "1.0",
            "stages": [
                {
                    "name": "start",
                    "is_start": True,
                    "prompt": "S",
                    "routing_transforms": ["nonexistent_func"],
                    "transitions": [{"target": "end"}],
                },
                {"name": "end", "is_end": True, "prompt": "E"},
            ],
        }
        reasoning = _make_reasoning(config)
        state = _make_state(reasoning)

        result = await reasoning.advance({"x": 1}, state)

        # Should still work (transition fires unconditionally)
        assert result.transitioned is True
        # Warning was logged
        assert any(
            "nonexistent_func" in r.message and "not found" in r.message
            for r in caplog.records
        )

    @pytest.mark.asyncio
    async def test_routing_transforms_in_metadata(self) -> None:
        """routing_transforms are present in stage metadata."""
        config = {
            "name": "test",
            "version": "1.0",
            "stages": [
                {
                    "name": "start",
                    "is_start": True,
                    "prompt": "S",
                    "routing_transforms": ["classify_intent"],
                    "transitions": [{"target": "end"}],
                },
                {"name": "end", "is_end": True, "prompt": "E"},
            ],
        }
        reasoning = _make_reasoning(config)
        stages = reasoning._fsm.stages

        assert stages["start"]["routing_transforms"] == ["classify_intent"]
        assert stages["end"]["routing_transforms"] == []
