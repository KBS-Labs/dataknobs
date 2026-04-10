"""Unit tests for WizardNavigator pure/near-pure methods.

Tests the static helpers and config resolution logic directly on
``WizardNavigator``, independent of the full wizard stack.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from dataknobs_bots.reasoning.wizard_navigation import WizardNavigator
from dataknobs_bots.reasoning.wizard_types import (
    NavigationCommandConfig,
    NavigationConfig,
)


# ---------------------------------------------------------------------------
# _is_ancestor_of
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
# _find_stage_node_id
# ---------------------------------------------------------------------------


class TestFindStageNodeId:
    """Tests for the static _find_stage_node_id helper."""

    def test_returns_none_when_manager_has_no_state(self) -> None:
        manager = object()  # No .state attribute
        assert WizardNavigator._find_stage_node_id(manager, "start") is None

    def test_returns_none_when_state_is_none(self) -> None:
        manager = MagicMock()
        manager.state = None
        assert WizardNavigator._find_stage_node_id(manager, "start") is None

    def test_returns_none_when_no_matches(self) -> None:
        manager = MagicMock()
        manager.state.current_node_id = "0.1"
        manager.state.message_tree.find_nodes.return_value = []
        assert WizardNavigator._find_stage_node_id(manager, "start") is None


# ---------------------------------------------------------------------------
# _resolve_navigation_config
# ---------------------------------------------------------------------------


def _make_navigator(
    *,
    stage_metadata: dict[str, Any] | None = None,
    nav_config: NavigationConfig | None = None,
) -> WizardNavigator:
    """Build a minimal WizardNavigator for config resolution tests.

    Only wires the FSM (for stage metadata) and navigation config — the
    callbacks are stubs since these tests don't exercise navigation flow.
    """
    fsm = MagicMock()
    fsm._stage_metadata = stage_metadata or {}

    async def _noop_fsm_step(
        state: Any, *, user_message: Any = None, trigger: str = "",
    ) -> tuple[str, Any]:
        return ("", None)

    async def _noop_lifecycle(state: Any) -> list[str]:
        return []

    async def _noop_response(*args: Any, **kwargs: Any) -> Any:
        return None

    return WizardNavigator(
        fsm=fsm,
        subflows=MagicMock(),
        hooks=None,
        navigation_config=nav_config or NavigationConfig.from_dict({}),
        consistent_lifecycle=True,
        allow_amendments=False,
        section_to_stage_mapping={},
        extractor=None,
        banks={},
        artifact=None,
        catalog=None,
        execute_fsm_step=_noop_fsm_step,
        run_post_transition_lifecycle=_noop_lifecycle,
        generate_stage_response=_noop_response,
        prepend_messages_to_response=lambda r, m: None,
    )


class TestResolveNavigationConfig:
    """Tests for _resolve_navigation_config."""

    def test_returns_wizard_config_when_stage_has_no_override(self) -> None:
        nav = _make_navigator()
        result = nav._resolve_navigation_config("start")
        assert result is nav._navigation_config

    def test_stage_override_replaces_keywords(self) -> None:
        nav = _make_navigator(
            stage_metadata={
                "start": {
                    "navigation": {
                        "back": {"keywords": ["undo", "previous"]},
                    },
                },
            },
        )
        result = nav._resolve_navigation_config("start")
        assert result.back.keywords == ("undo", "previous")
        # Skip and restart inherit from wizard-level defaults
        assert result.skip.keywords == nav._navigation_config.skip.keywords
        assert result.restart.keywords == nav._navigation_config.restart.keywords

    def test_stage_override_can_disable_command(self) -> None:
        nav = _make_navigator(
            stage_metadata={
                "start": {
                    "navigation": {
                        "skip": {"enabled": False},
                    },
                },
            },
        )
        result = nav._resolve_navigation_config("start")
        assert result.skip.enabled is False
        # Keywords still inherited when only enabled is overridden
        assert result.skip.keywords == nav._navigation_config.skip.keywords

    def test_unknown_stage_returns_wizard_config(self) -> None:
        nav = _make_navigator(
            stage_metadata={"start": {"navigation": {"back": {"keywords": ["x"]}}}},
        )
        result = nav._resolve_navigation_config("nonexistent")
        assert result is nav._navigation_config


# ---------------------------------------------------------------------------
# map_section_to_stage
# ---------------------------------------------------------------------------


class TestMapSectionToStage:
    """Tests for map_section_to_stage."""

    def test_empty_section_returns_none(self) -> None:
        nav = _make_navigator()
        assert nav.map_section_to_stage("") is None

    def test_custom_mapping_takes_precedence(self) -> None:
        nav = _make_navigator(
            stage_metadata={"my_llm_stage": {}},
        )
        nav._section_to_stage_mapping = {"llm": "my_llm_stage"}
        assert nav.map_section_to_stage("llm") == "my_llm_stage"

    def test_default_mapping_verified_against_fsm(self) -> None:
        # Default maps "llm" to "configure_llm" but only if that stage exists
        nav = _make_navigator(
            stage_metadata={"configure_llm": {}},
        )
        assert nav.map_section_to_stage("llm") == "configure_llm"

    def test_default_mapping_returns_none_if_stage_missing(self) -> None:
        nav = _make_navigator(stage_metadata={})
        assert nav.map_section_to_stage("llm") is None

    def test_case_insensitive(self) -> None:
        nav = _make_navigator(
            stage_metadata={"configure_llm": {}},
        )
        assert nav.map_section_to_stage("LLM") == "configure_llm"
        assert nav.map_section_to_stage("  Llm  ") == "configure_llm"
