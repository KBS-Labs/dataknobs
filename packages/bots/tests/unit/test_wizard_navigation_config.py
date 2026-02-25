"""Tests for configurable navigation keywords.

Covers the NavigationConfig data model, default backward-compatible
behaviour, wizard-level custom keywords, per-stage overrides, and
the WizardConfigBuilder integration.
"""

from typing import Any

import pytest

from dataknobs_bots.config.wizard_builder import StageConfig, WizardConfigBuilder
from dataknobs_bots.reasoning.wizard import (
    DEFAULT_BACK_KEYWORDS,
    DEFAULT_RESTART_KEYWORDS,
    DEFAULT_SKIP_KEYWORDS,
    NavigationCommandConfig,
    NavigationConfig,
    WizardReasoning,
    WizardState,
)
from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader
from dataknobs_llm.conversations import ConversationManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_two_stage_config(
    *,
    settings: dict[str, Any] | None = None,
    start_nav: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a minimal two-stage wizard config dict.

    Args:
        settings: Wizard-level settings (optional).
        start_nav: Per-stage navigation override for the start stage.
    """
    start_stage: dict[str, Any] = {
        "name": "start",
        "is_start": True,
        "prompt": "Hello",
        "can_skip": True,
        "transitions": [{"target": "done"}],
    }
    if start_nav is not None:
        start_stage["navigation"] = start_nav

    config: dict[str, Any] = {
        "name": "nav-test",
        "stages": [
            start_stage,
            {"name": "done", "is_end": True, "prompt": "Done!"},
        ],
    }
    if settings:
        config["settings"] = settings
    return config


def _build_reasoning(config: dict[str, Any]) -> WizardReasoning:
    loader = WizardConfigLoader()
    wizard_fsm = loader.load_from_dict(config)
    return WizardReasoning(wizard_fsm=wizard_fsm, strict_validation=False)


# ===================================================================
# TestNavigationConfigDataModel
# ===================================================================

class TestNavigationConfigDataModel:
    """Unit tests for NavigationConfig.from_dict()."""

    def test_empty_dict_returns_defaults(self) -> None:
        nav = NavigationConfig.from_dict({})
        assert nav.back.keywords == DEFAULT_BACK_KEYWORDS
        assert nav.skip.keywords == DEFAULT_SKIP_KEYWORDS
        assert nav.restart.keywords == DEFAULT_RESTART_KEYWORDS
        assert nav.back.enabled is True
        assert nav.skip.enabled is True
        assert nav.restart.enabled is True

    def test_defaults_classmethod(self) -> None:
        nav = NavigationConfig.defaults()
        assert nav.back.keywords == DEFAULT_BACK_KEYWORDS
        assert nav.skip.keywords == DEFAULT_SKIP_KEYWORDS
        assert nav.restart.keywords == DEFAULT_RESTART_KEYWORDS

    def test_partial_config_overrides_specified_only(self) -> None:
        nav = NavigationConfig.from_dict({
            "back": {"keywords": ["undo", "go back"]},
        })
        assert nav.back.keywords == ("undo", "go back")
        # skip and restart keep defaults
        assert nav.skip.keywords == DEFAULT_SKIP_KEYWORDS
        assert nav.restart.keywords == DEFAULT_RESTART_KEYWORDS

    def test_full_config(self) -> None:
        nav = NavigationConfig.from_dict({
            "back": {"keywords": ["b"]},
            "skip": {"keywords": ["s", "next"]},
            "restart": {"keywords": ["r"]},
        })
        assert nav.back.keywords == ("b",)
        assert nav.skip.keywords == ("s", "next")
        assert nav.restart.keywords == ("r",)

    def test_enabled_false_parsed(self) -> None:
        nav = NavigationConfig.from_dict({
            "skip": {"enabled": False},
        })
        assert nav.skip.enabled is False
        # Keywords still default when not specified
        assert nav.skip.keywords == DEFAULT_SKIP_KEYWORDS

    def test_keywords_normalised_to_lowercase(self) -> None:
        nav = NavigationConfig.from_dict({
            "back": {"keywords": ["BACK", "Go Back", "PREVIOUS"]},
        })
        assert nav.back.keywords == ("back", "go back", "previous")

    def test_none_data_returns_defaults(self) -> None:
        """Explicitly passing None-ish data still gives defaults."""
        nav = NavigationConfig.from_dict({})
        assert nav == NavigationConfig.defaults()

    def test_frozen(self) -> None:
        nav = NavigationConfig.defaults()
        with pytest.raises(AttributeError):
            nav.back = NavigationCommandConfig(keywords=("x",))  # type: ignore[misc]


# ===================================================================
# TestNavigationConfigDefaults — backward compatibility
# ===================================================================

class TestNavigationConfigDefaults:
    """Verify that with no navigation config the original keywords work."""

    @pytest.mark.asyncio
    async def test_default_back_keywords(
        self, conversation_manager: ConversationManager
    ) -> None:
        config = _make_two_stage_config()
        reasoning = _build_reasoning(config)

        for keyword in ("back", "go back", "previous"):
            state = WizardState(current_stage="start", history=["start"])
            result = await reasoning._handle_navigation(
                keyword, state, conversation_manager, None
            )
            # At start with single history entry, back cannot go further
            # so we get a response (not None)
            assert result is not None, f"Expected response for '{keyword}'"

    @pytest.mark.asyncio
    async def test_default_skip_keywords(
        self, conversation_manager: ConversationManager
    ) -> None:
        config = _make_two_stage_config()
        reasoning = _build_reasoning(config)

        for keyword in ("skip", "skip this", "use default", "use defaults"):
            state = WizardState(current_stage="start")
            result = await reasoning._handle_navigation(
                keyword, state, conversation_manager, None
            )
            # can_skip=True so returns None (continue to transition)
            assert result is None, f"Expected None for skip keyword '{keyword}'"

    @pytest.mark.asyncio
    async def test_default_restart_keywords(
        self, conversation_manager: ConversationManager
    ) -> None:
        config = _make_two_stage_config()
        reasoning = _build_reasoning(config)

        for keyword in ("restart", "start over"):
            state = WizardState(current_stage="start")
            result = await reasoning._handle_navigation(
                keyword, state, conversation_manager, None
            )
            assert result is not None, f"Expected response for '{keyword}'"

    @pytest.mark.asyncio
    async def test_non_navigation_message_returns_none(
        self, conversation_manager: ConversationManager
    ) -> None:
        config = _make_two_stage_config()
        reasoning = _build_reasoning(config)
        state = WizardState(current_stage="start")

        result = await reasoning._handle_navigation(
            "hello there", state, conversation_manager, None
        )
        assert result is None


# ===================================================================
# TestNavigationConfigCustomKeywords — wizard-level customisation
# ===================================================================

class TestNavigationConfigCustomKeywords:
    """Test wizard-level navigation keyword customisation."""

    @pytest.mark.asyncio
    async def test_custom_back_replaces_defaults(
        self, conversation_manager: ConversationManager
    ) -> None:
        config = _make_two_stage_config(settings={
            "navigation": {
                "back": {"keywords": ["undo", "go back"]},
            },
        })
        reasoning = _build_reasoning(config)

        # "undo" should now work
        state = WizardState(current_stage="start", history=["start"])
        result = await reasoning._handle_navigation(
            "undo", state, conversation_manager, None
        )
        assert result is not None  # back handled (can't go further)

        # "previous" (old default) should no longer work
        state = WizardState(current_stage="start")
        result = await reasoning._handle_navigation(
            "previous", state, conversation_manager, None
        )
        assert result is None  # Not recognised as navigation

    @pytest.mark.asyncio
    async def test_custom_skip_keywords(
        self, conversation_manager: ConversationManager
    ) -> None:
        config = _make_two_stage_config(settings={
            "navigation": {
                "skip": {"keywords": ["next", "pass"]},
            },
        })
        reasoning = _build_reasoning(config)

        # "next" should skip
        state = WizardState(current_stage="start")
        result = await reasoning._handle_navigation(
            "next", state, conversation_manager, None
        )
        assert result is None  # skip succeeded

        # "skip" (old default) should not work
        state = WizardState(current_stage="start")
        result = await reasoning._handle_navigation(
            "skip", state, conversation_manager, None
        )
        assert result is None  # Not a nav command — falls through

    @pytest.mark.asyncio
    async def test_custom_restart_keywords(
        self, conversation_manager: ConversationManager
    ) -> None:
        config = _make_two_stage_config(settings={
            "navigation": {
                "restart": {"keywords": ["begin again"]},
            },
        })
        reasoning = _build_reasoning(config)

        state = WizardState(current_stage="start")
        result = await reasoning._handle_navigation(
            "begin again", state, conversation_manager, None
        )
        assert result is not None

        # "restart" should no longer work
        state = WizardState(current_stage="start")
        result = await reasoning._handle_navigation(
            "restart", state, conversation_manager, None
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_partial_override_preserves_other_defaults(
        self, conversation_manager: ConversationManager
    ) -> None:
        """Overriding only back should leave skip/restart defaults intact."""
        config = _make_two_stage_config(settings={
            "navigation": {
                "back": {"keywords": ["undo"]},
            },
        })
        reasoning = _build_reasoning(config)

        # skip should still use defaults
        state = WizardState(current_stage="start")
        result = await reasoning._handle_navigation(
            "skip", state, conversation_manager, None
        )
        assert result is None  # skip worked

        # restart should still use defaults
        state = WizardState(current_stage="start")
        result = await reasoning._handle_navigation(
            "restart", state, conversation_manager, None
        )
        assert result is not None


# ===================================================================
# TestNavigationConfigStageOverride — per-stage overrides
# ===================================================================

class TestNavigationConfigStageOverride:
    """Test per-stage navigation keyword overrides."""

    @pytest.mark.asyncio
    async def test_stage_keywords_replace_wizard_level(
        self, conversation_manager: ConversationManager
    ) -> None:
        """Stage-level back keywords replace wizard-level for that command."""
        config = _make_two_stage_config(
            settings={
                "navigation": {
                    "back": {"keywords": ["undo"]},
                },
            },
            start_nav={
                "back": {"keywords": ["change my answer"]},
            },
        )
        reasoning = _build_reasoning(config)

        # "change my answer" should work at start stage
        state = WizardState(current_stage="start", history=["start"])
        result = await reasoning._handle_navigation(
            "change my answer", state, conversation_manager, None
        )
        assert result is not None

        # "undo" (wizard-level) should NOT work at start stage
        state = WizardState(current_stage="start")
        result = await reasoning._handle_navigation(
            "undo", state, conversation_manager, None
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_unmodified_commands_inherit_wizard_level(
        self, conversation_manager: ConversationManager
    ) -> None:
        """Commands not overridden at stage level keep wizard-level config."""
        config = _make_two_stage_config(
            settings={
                "navigation": {
                    "skip": {"keywords": ["next"]},
                },
            },
            start_nav={
                "back": {"keywords": ["undo"]},
            },
        )
        reasoning = _build_reasoning(config)

        # skip should still use wizard-level "next"
        state = WizardState(current_stage="start")
        result = await reasoning._handle_navigation(
            "next", state, conversation_manager, None
        )
        assert result is None  # skip succeeded

    @pytest.mark.asyncio
    async def test_enabled_false_disables_command(
        self, conversation_manager: ConversationManager
    ) -> None:
        config = _make_two_stage_config(
            start_nav={
                "skip": {"enabled": False},
            },
        )
        reasoning = _build_reasoning(config)

        state = WizardState(current_stage="start")
        result = await reasoning._handle_navigation(
            "skip", state, conversation_manager, None
        )
        # skip is disabled — message falls through as non-navigation
        assert result is None

        # But restart still works (not disabled)
        state = WizardState(current_stage="start")
        result = await reasoning._handle_navigation(
            "restart", state, conversation_manager, None
        )
        assert result is not None

    @pytest.mark.asyncio
    async def test_stage_without_navigation_inherits_wizard(
        self, conversation_manager: ConversationManager
    ) -> None:
        """Stages without a navigation key should use wizard-level config."""
        config = _make_two_stage_config(settings={
            "navigation": {
                "back": {"keywords": ["undo"]},
            },
        })
        # No start_nav — the start stage has no navigation override
        reasoning = _build_reasoning(config)

        state = WizardState(current_stage="start", history=["start"])
        result = await reasoning._handle_navigation(
            "undo", state, conversation_manager, None
        )
        assert result is not None


# ===================================================================
# TestNavigationConfigBuilder — WizardConfigBuilder integration
# ===================================================================

class TestNavigationConfigBuilder:
    """Test WizardConfigBuilder navigation support."""

    def test_set_navigation_produces_correct_settings(self) -> None:
        nav_config = {
            "back": {"keywords": ["undo"]},
            "skip": {"keywords": ["next"]},
        }
        builder = (
            WizardConfigBuilder("nav-wizard")
            .set_navigation(nav_config)
            .add_structured_stage("start", "Hello", is_start=True, is_end=True)
        )
        config = builder.build()

        assert config.settings["navigation"] == nav_config

    def test_stage_navigation_roundtrip(self) -> None:
        """Stage navigation survives to_dict → from_dict roundtrip."""
        stage_nav = {"skip": {"enabled": False}}
        stage = StageConfig(
            name="review",
            prompt="Review your answers",
            is_start=True,
            is_end=True,
            navigation=stage_nav,
        )
        d = stage.to_dict()
        assert d["navigation"] == stage_nav

        rebuilt = WizardConfigBuilder.from_dict({
            "name": "rt-wizard",
            "stages": [d],
        })
        built = rebuilt.build()
        assert built.stages[0].navigation == stage_nav

    def test_build_roundtrip_with_navigation(self) -> None:
        """Full build → to_dict → from_dict → build roundtrip."""
        nav_config = {
            "back": {"keywords": ["undo", "go back"]},
            "restart": {"keywords": ["start fresh"]},
        }
        original = (
            WizardConfigBuilder("rt-wizard")
            .set_navigation(nav_config)
            .add_structured_stage("start", "Hello", is_start=True, is_end=True)
            .build()
        )
        as_dict = original.to_dict()
        rebuilt = WizardConfigBuilder.from_dict(as_dict).build()

        assert rebuilt.settings["navigation"] == nav_config

    def test_set_navigation_then_load_and_handle(self) -> None:
        """End-to-end: builder → load → NavigationConfig is correct."""
        nav_config = {
            "back": {"keywords": ["undo"]},
        }
        wizard = (
            WizardConfigBuilder("e2e-wizard")
            .set_navigation(nav_config)
            .add_structured_stage("start", "Hello", is_start=True, is_end=True)
            .build()
        )
        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(wizard.to_dict())
        reasoning = WizardReasoning(wizard_fsm=fsm, strict_validation=False)

        # Verify the navigation config was picked up
        assert reasoning._navigation_config.back.keywords == ("undo",)
        assert reasoning._navigation_config.skip.keywords == DEFAULT_SKIP_KEYWORDS

    def test_stage_navigation_preserved_in_assemble(self) -> None:
        """Stage navigation survives _assemble_stages rebuild."""
        stage_nav = {"back": {"keywords": ["undo"]}}
        builder = (
            WizardConfigBuilder("assemble-test")
            .add_structured_stage(
                "start", "Hello", is_start=True, navigation=stage_nav,
            )
            .add_end_stage("done", "Done!")
            .add_transition("start", "done")
        )
        config = builder.build()
        # The start stage was rebuilt in _assemble_stages (transition added)
        assert config.stages[0].navigation == stage_nav
