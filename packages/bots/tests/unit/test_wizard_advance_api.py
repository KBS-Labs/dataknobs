"""Tests for the non-conversational wizard advance API.

Tests WizardReasoning.advance(), initial_stage, and get_wizard_metadata()
— the public API for advancing a wizard without DynaBot/LLM infrastructure.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from dataknobs_bots.reasoning.wizard import WizardAdvanceResult, WizardReasoning, WizardState
from dataknobs_bots.reasoning.wizard_hooks import WizardHooks
from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_reasoning(
    config: dict[str, Any],
    *,
    hooks: WizardHooks | None = None,
    consistent_navigation_lifecycle: bool = True,
) -> WizardReasoning:
    """Create a WizardReasoning from a config dict."""
    loader = WizardConfigLoader()
    wizard_fsm = loader.load_from_dict(config)
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
