"""Unit tests for WizardNavigator.

Tests static helpers (``_is_ancestor_of``), config resolution
(``_resolve_navigation_config``, ``map_section_to_stage``), and
behavioral navigation methods (``handle_navigation``, ``navigate_back``,
``navigate_skip``, ``restart_cleanup``, ``branch_for_revisited_stage``)
using real ``WizardFSM`` and ``SubflowManager`` instances.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import pytest

from dataknobs_bots.reasoning.wizard_hooks import WizardHooks
from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader
from dataknobs_bots.reasoning.wizard_navigation import WizardNavigator
from dataknobs_bots.reasoning.wizard_subflows import SubflowManager
from dataknobs_bots.reasoning.wizard_types import (
    NavigationConfig,
    WizardState,
)


# ---------------------------------------------------------------------------
# Shared config and helpers
# ---------------------------------------------------------------------------


def _two_stage_config(
    *,
    settings: dict[str, Any] | None = None,
    start_nav: dict[str, Any] | None = None,
    can_skip: bool = True,
) -> dict[str, Any]:
    """Build a minimal two-stage wizard config."""
    start: dict[str, Any] = {
        "name": "start",
        "is_start": True,
        "prompt": "Hello",
        "can_skip": can_skip,
        "transitions": [{"target": "done"}],
    }
    if start_nav is not None:
        start["navigation"] = start_nav
    config: dict[str, Any] = {
        "name": "nav-test",
        "stages": [
            start,
            {"name": "done", "is_end": True, "prompt": "Done!"},
        ],
    }
    if settings:
        config["settings"] = settings
    return config


def _three_stage_config() -> dict[str, Any]:
    """Three stages: start → middle → done."""
    return {
        "name": "nav-test",
        "stages": [
            {
                "name": "start",
                "is_start": True,
                "prompt": "Start",
                "can_skip": True,
                "transitions": [{"target": "middle"}],
            },
            {
                "name": "middle",
                "prompt": "Middle",
                "can_skip": True,
                "transitions": [{"target": "done"}],
            },
            {"name": "done", "is_end": True, "prompt": "Done"},
        ],
    }


@dataclass
class _FakeResponse:
    """Minimal response object for testing."""

    content: str = "test response"


def _build_navigator(
    config: dict[str, Any],
    *,
    hooks: WizardHooks | None = None,
    consistent_lifecycle: bool = True,
    allow_amendments: bool = False,
    section_to_stage_mapping: dict[str, str] | None = None,
) -> WizardNavigator:
    """Build a WizardNavigator from a config dict using real FSM/subflows.

    Callbacks are simple stubs — ``execute_fsm_step`` advances the FSM
    to the next stage, ``generate_stage_response`` returns a
    ``_FakeResponse``, and ``run_post_transition_lifecycle`` is a no-op.
    These are sufficient for testing the navigator's own logic without
    pulling in the full WizardReasoning orchestrator.
    """
    loader = WizardConfigLoader()
    fsm = loader.load_from_dict(config)
    subflows = SubflowManager(
        fsm=fsm,
        evaluate_condition=lambda cond, data: True,
    )
    nav_settings = config.get("settings", {}).get("navigation", {})

    async def _fsm_step(
        state: WizardState,
        *,
        user_message: str | None = None,
        trigger: str = "user_input",
    ) -> tuple[str, Any]:
        """Stub: step the FSM forward and update state."""
        from_stage = state.current_stage
        active = subflows.get_active_fsm()
        result = active.step(state.data)
        new_stage = active.current_stage
        if new_stage != from_stage:
            state.current_stage = new_stage
            if new_stage not in state.history:
                state.history.append(new_stage)
        return from_stage, result

    async def _post_lifecycle(state: WizardState) -> list[str]:
        return []

    async def _gen_response(*args: Any, **kwargs: Any) -> _FakeResponse:
        return _FakeResponse()

    return WizardNavigator(
        fsm=fsm,
        subflows=subflows,
        hooks=hooks,
        navigation_config=NavigationConfig.from_dict(nav_settings),
        consistent_lifecycle=consistent_lifecycle,
        allow_amendments=allow_amendments,
        section_to_stage_mapping=section_to_stage_mapping or {},
        extractor=None,
        banks={},
        artifact=None,
        catalog=None,
        execute_fsm_step=_fsm_step,
        run_post_transition_lifecycle=_post_lifecycle,
        generate_stage_response=_gen_response,
        prepend_messages_to_response=lambda r, m: None,
    )


def _make_state(
    nav: WizardNavigator,
    *,
    current_stage: str | None = None,
) -> WizardState:
    """Create a WizardState at the navigator's initial or given stage."""
    stage = current_stage or nav._fsm.current_stage
    return WizardState(
        current_stage=stage,
        data={},
        history=[stage],
        stage_entry_time=time.time(),
    )


# ---------------------------------------------------------------------------
# _is_ancestor_of — pure static, no dependencies
# ---------------------------------------------------------------------------


class TestIsAncestorOf:
    """Tests for the static _is_ancestor_of helper."""

    def test_root_is_ancestor_of_everything(self) -> None:
        assert WizardNavigator._is_ancestor_of("", "0") is True
        assert WizardNavigator._is_ancestor_of("", "0.1.2") is True

    def test_direct_parent(self) -> None:
        assert WizardNavigator._is_ancestor_of("0", "0.1") is True

    def test_deep_ancestor(self) -> None:
        assert WizardNavigator._is_ancestor_of("0", "0.1.2.3") is True

    def test_not_ancestor_different_branch(self) -> None:
        assert WizardNavigator._is_ancestor_of("0.1", "0.2") is False

    def test_node_is_not_own_ancestor(self) -> None:
        assert WizardNavigator._is_ancestor_of("0.1", "0.1") is False

    def test_prefix_match_requires_dot_separator(self) -> None:
        # "0.1" should NOT be ancestor of "0.10" — they share a prefix
        # but "0.1" + "." does not start "0.10".
        assert WizardNavigator._is_ancestor_of("0.1", "0.10") is False

    def test_longer_id_not_ancestor_of_shorter(self) -> None:
        assert WizardNavigator._is_ancestor_of("0.1.2", "0.1") is False


# ---------------------------------------------------------------------------
# _resolve_navigation_config — uses real FSM stage metadata
# ---------------------------------------------------------------------------


class TestResolveNavigationConfig:
    """Tests for _resolve_navigation_config."""

    def test_returns_wizard_config_when_stage_has_no_override(self) -> None:
        nav = _build_navigator(_two_stage_config())
        result = nav._resolve_navigation_config("start")
        assert result is nav._navigation_config

    def test_stage_override_replaces_keywords(self) -> None:
        nav = _build_navigator(_two_stage_config(
            start_nav={"back": {"keywords": ["undo", "previous"]}},
        ))
        result = nav._resolve_navigation_config("start")
        assert result.back.keywords == ("undo", "previous")
        # Skip and restart inherit from wizard-level defaults
        assert result.skip.keywords == nav._navigation_config.skip.keywords
        assert result.restart.keywords == nav._navigation_config.restart.keywords

    def test_stage_override_can_disable_command(self) -> None:
        nav = _build_navigator(_two_stage_config(
            start_nav={"skip": {"enabled": False}},
        ))
        result = nav._resolve_navigation_config("start")
        assert result.skip.enabled is False
        assert result.skip.keywords == nav._navigation_config.skip.keywords

    def test_unknown_stage_returns_wizard_config(self) -> None:
        nav = _build_navigator(_two_stage_config())
        result = nav._resolve_navigation_config("nonexistent")
        assert result is nav._navigation_config


# ---------------------------------------------------------------------------
# map_section_to_stage — uses real FSM stage metadata
# ---------------------------------------------------------------------------


class TestMapSectionToStage:
    """Tests for map_section_to_stage."""

    def test_empty_section_returns_none(self) -> None:
        nav = _build_navigator(_two_stage_config())
        assert nav.map_section_to_stage("") is None

    def test_custom_mapping_takes_precedence(self) -> None:
        nav = _build_navigator(
            _two_stage_config(),
            section_to_stage_mapping={"mykey": "start"},
        )
        assert nav.map_section_to_stage("mykey") == "start"

    def test_default_mapping_returns_none_if_stage_missing(self) -> None:
        # Default maps "llm" to "configure_llm" which doesn't exist here
        nav = _build_navigator(_two_stage_config())
        assert nav.map_section_to_stage("llm") is None

    def test_case_insensitive(self) -> None:
        nav = _build_navigator(
            _two_stage_config(),
            section_to_stage_mapping={"llm": "start"},
        )
        assert nav.map_section_to_stage("LLM") == "start"
        assert nav.map_section_to_stage("  Llm  ") == "start"


# ---------------------------------------------------------------------------
# handle_navigation — behavioral: keyword dispatch
# ---------------------------------------------------------------------------


class TestHandleNavigation:
    """Behavioral tests for handle_navigation dispatch."""

    @pytest.mark.asyncio
    async def test_back_keyword_triggers_back(self) -> None:
        nav = _build_navigator(_three_stage_config())
        state = _make_state(nav)

        # Advance to middle first so back has somewhere to go
        state.current_stage = "middle"
        state.history = ["start", "middle"]
        nav._fsm.restore({"current_stage": "middle", "data": {}})

        result = await nav.handle_navigation("back", state, None, None)
        assert result is not None  # Got a response
        assert state.current_stage == "start"

    @pytest.mark.asyncio
    async def test_skip_keyword_triggers_skip(self) -> None:
        nav = _build_navigator(_three_stage_config())
        state = _make_state(nav)

        result = await nav.handle_navigation("skip", state, None, None)
        assert result is not None
        assert state.current_stage == "middle"

    @pytest.mark.asyncio
    async def test_restart_keyword_triggers_restart(self) -> None:
        nav = _build_navigator(_three_stage_config())
        state = _make_state(nav)

        # Advance to middle first
        state.current_stage = "middle"
        state.history = ["start", "middle"]
        nav._fsm.restore({"current_stage": "middle", "data": {}})

        result = await nav.handle_navigation("restart", state, None, None)
        assert result is not None
        assert state.current_stage == "start"
        assert state.data == {}

    @pytest.mark.asyncio
    async def test_non_navigation_returns_none(self) -> None:
        nav = _build_navigator(_two_stage_config())
        state = _make_state(nav)
        result = await nav.handle_navigation("hello there", state, None, None)
        assert result is None

    @pytest.mark.asyncio
    async def test_disabled_command_returns_none(self) -> None:
        nav = _build_navigator(_two_stage_config(
            start_nav={"skip": {"enabled": False}},
        ))
        state = _make_state(nav)
        result = await nav.handle_navigation("skip", state, None, None)
        assert result is None


# ---------------------------------------------------------------------------
# navigate_back — behavioral: FSM-level back
# ---------------------------------------------------------------------------


class TestNavigateBack:
    """Behavioral tests for navigate_back."""

    @pytest.mark.asyncio
    async def test_back_pops_history_and_restores_stage(self) -> None:
        nav = _build_navigator(_three_stage_config())
        state = _make_state(nav)
        state.current_stage = "middle"
        state.history = ["start", "middle"]
        nav._fsm.restore({"current_stage": "middle", "data": {}})

        result = await nav.navigate_back(state)
        assert result is True
        assert state.current_stage == "start"
        assert state.history == ["start"]

    @pytest.mark.asyncio
    async def test_back_at_start_returns_false(self) -> None:
        nav = _build_navigator(_two_stage_config())
        state = _make_state(nav)

        result = await nav.navigate_back(state)
        assert result is False
        assert state.current_stage == "start"

    @pytest.mark.asyncio
    async def test_back_records_transition(self) -> None:
        nav = _build_navigator(_three_stage_config())
        state = _make_state(nav)
        state.current_stage = "middle"
        state.history = ["start", "middle"]
        nav._fsm.restore({"current_stage": "middle", "data": {}})

        await nav.navigate_back(state, user_message="back")
        assert len(state.transitions) == 1
        assert state.transitions[0].trigger == "navigation_back"
        assert state.transitions[0].from_stage == "middle"
        assert state.transitions[0].to_stage == "start"

    @pytest.mark.asyncio
    async def test_back_fires_enter_hook_when_consistent(self) -> None:
        enter_calls: list[str] = []
        hooks = WizardHooks()
        hooks.on_enter(lambda s, d: enter_calls.append(s))

        nav = _build_navigator(
            _three_stage_config(), hooks=hooks, consistent_lifecycle=True,
        )
        state = _make_state(nav)
        state.current_stage = "middle"
        state.history = ["start", "middle"]
        nav._fsm.restore({"current_stage": "middle", "data": {}})

        await nav.navigate_back(state)
        assert enter_calls == ["start"]

    @pytest.mark.asyncio
    async def test_back_no_hooks_when_not_consistent(self) -> None:
        enter_calls: list[str] = []
        hooks = WizardHooks()
        hooks.on_enter(lambda s, d: enter_calls.append(s))

        nav = _build_navigator(
            _three_stage_config(), hooks=hooks, consistent_lifecycle=False,
        )
        state = _make_state(nav)
        state.current_stage = "middle"
        state.history = ["start", "middle"]
        nav._fsm.restore({"current_stage": "middle", "data": {}})

        await nav.navigate_back(state)
        assert enter_calls == []


# ---------------------------------------------------------------------------
# navigate_skip — behavioral: FSM-level skip
# ---------------------------------------------------------------------------


class TestNavigateSkip:
    """Behavioral tests for navigate_skip."""

    @pytest.mark.asyncio
    async def test_skip_advances_and_marks_skipped(self) -> None:
        nav = _build_navigator(_three_stage_config())
        state = _make_state(nav)

        success, _ = await nav.navigate_skip(state)
        assert success is True
        assert state.data.get("_skipped_start") is True

    @pytest.mark.asyncio
    async def test_skip_not_allowed_returns_false(self) -> None:
        nav = _build_navigator(_two_stage_config(can_skip=False))
        state = _make_state(nav)

        success, msgs = await nav.navigate_skip(state)
        assert success is False
        assert msgs == []

    @pytest.mark.asyncio
    async def test_skip_applies_defaults(self) -> None:
        config = _two_stage_config()
        config["stages"][0]["skip_default"] = {"color": "blue"}
        nav = _build_navigator(config)
        state = _make_state(nav)

        await nav.navigate_skip(state)
        assert state.data.get("color") == "blue"


# ---------------------------------------------------------------------------
# restart_cleanup — behavioral: full state reset
# ---------------------------------------------------------------------------


class TestRestartCleanup:
    """Behavioral tests for restart_cleanup."""

    @pytest.mark.asyncio
    async def test_restart_clears_data_and_history(self) -> None:
        nav = _build_navigator(_three_stage_config())
        state = _make_state(nav)
        state.current_stage = "middle"
        state.history = ["start", "middle"]
        state.data = {"name": "test", "age": 25}
        nav._fsm.restore({"current_stage": "middle", "data": state.data})

        await nav.restart_cleanup(state, "restart")
        assert state.data == {}
        assert state.current_stage == "start"
        assert state.history == ["start"]
        assert state.completed is False

    @pytest.mark.asyncio
    async def test_restart_preserves_transition_history(self) -> None:
        nav = _build_navigator(_two_stage_config())
        state = _make_state(nav)
        from dataknobs_bots.reasoning.observability import create_transition_record

        prev = create_transition_record(
            from_stage="x", to_stage="y", trigger="prev",
        )
        state.transitions = [prev]

        await nav.restart_cleanup(state, "restart")
        assert len(state.transitions) == 2  # prev + restart
        assert state.transitions[-1].trigger == "restart"

    @pytest.mark.asyncio
    async def test_restart_fires_restart_hook(self) -> None:
        restart_called = []
        hooks = WizardHooks()
        hooks.on_restart(lambda: restart_called.append(True))

        nav = _build_navigator(_two_stage_config(), hooks=hooks)
        state = _make_state(nav)

        await nav.restart_cleanup(state, "restart")
        assert restart_called == [True]

    @pytest.mark.asyncio
    async def test_restart_clears_banks(self) -> None:
        nav = _build_navigator(_two_stage_config())
        # Simulate a bank with a clear() method
        cleared = []

        class FakeBank:
            def clear(self) -> None:
                cleared.append(True)

        nav._banks = {"test_bank": FakeBank()}
        state = _make_state(nav)

        await nav.restart_cleanup(state, "restart")
        assert cleared == [True]


# ---------------------------------------------------------------------------
# branch_for_revisited_stage — behavioral: graceful degradation
# ---------------------------------------------------------------------------


class TestBranchForRevisitedStage:
    """Behavioral tests for branch_for_revisited_stage."""

    @pytest.mark.asyncio
    async def test_no_crash_when_manager_has_no_state(self) -> None:
        nav = _build_navigator(_two_stage_config())
        # Plain object with no .state — should degrade gracefully
        await nav.branch_for_revisited_stage(object(), "start")

    @pytest.mark.asyncio
    async def test_no_crash_when_manager_is_none(self) -> None:
        nav = _build_navigator(_two_stage_config())
        # None manager — _find_stage_node_id handles getattr(None, "state")
        await nav.branch_for_revisited_stage(None, "start")
