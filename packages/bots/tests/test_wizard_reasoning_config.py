"""Tests for the wizard reasoning-strategy ``StructuredConfig`` family.

Covers the wizard config migration:

- ``WizardReasoningConfig`` — the new thin strategy-section envelope that
  ``WizardReasoning.CONFIG_CLS`` points at.
- The leaf wizard configs flipped to ``StructuredConfig``:
  ``NavigationConfig``, ``NavigationCommandConfig``, and
  ``ToolResultMappingEntry``.

Verifies round-trip serialization, frozen immutability, that each is a
real ``StructuredConfig`` subclass, the ``NavigationConfig`` dict-loading
shape quirks preserved via ``_normalize_dict`` (per-command defaults +
lowercasing), ``ToolResultMappingEntry``'s ``on_error`` validation, and
that ``WizardReasoning`` declares ``CONFIG_CLS``.

No validation binding is exercised here — that lands with the resolver in
a later PR.  These configs are typed/redacted/round-trippable on their
own.  ``NavigationConfig``'s end-to-end navigation behaviour is covered by
``tests/unit/test_wizard_navigation_config.py``; this file is the typed-
config layer.
"""

from __future__ import annotations

import dataclasses

import pytest

from dataknobs_common.structured_config import StructuredConfig
from dataknobs_common.testing import assert_structured_config_roundtrip

from dataknobs_bots.reasoning.wizard import WizardReasoning
from dataknobs_bots.reasoning.wizard_config import WizardReasoningConfig
from dataknobs_bots.reasoning.wizard_types import (
    DEFAULT_BACK_KEYWORDS,
    DEFAULT_RESTART_KEYWORDS,
    DEFAULT_SKIP_KEYWORDS,
    NavigationCommandConfig,
    NavigationConfig,
    ToolResultMappingEntry,
)


# ===================================================================
# TestConfigClsPointer
# ===================================================================


class TestConfigClsPointer:
    """WizardReasoning declares CONFIG_CLS pointing at its config."""

    def test_wizard_points_at_config(self) -> None:
        assert WizardReasoning.CONFIG_CLS is WizardReasoningConfig

    def test_all_configs_are_structured_config(self) -> None:
        for cls in (
            WizardReasoningConfig,
            NavigationConfig,
            NavigationCommandConfig,
            ToolResultMappingEntry,
        ):
            assert issubclass(cls, StructuredConfig), cls


# ===================================================================
# TestWizardReasoningConfig
# ===================================================================


class TestWizardReasoningConfig:
    """The thin reasoning-section envelope."""

    def test_defaults(self) -> None:
        cfg = WizardReasoningConfig(wizard_config="w.yaml")
        assert cfg.wizard_config == "w.yaml"
        assert cfg.config_base_path is None
        assert cfg.custom_functions is None
        assert cfg.extraction_config is None
        assert cfg.strict_validation is True
        assert cfg.hooks is None
        assert cfg.artifacts is None
        assert cfg.review_protocols is None
        assert cfg.initial_data is None
        assert cfg.consistent_navigation_lifecycle is True

    def test_roundtrip_with_path(self) -> None:
        cfg = WizardReasoningConfig(
            wizard_config="wizards/onboarding.yaml",
            config_base_path="/cfg",
            extraction_config={"provider": "ollama", "model": "x"},
            strict_validation=False,
            consistent_navigation_lifecycle=False,
        )
        assert_structured_config_roundtrip(cfg)

    def test_roundtrip_with_inline_dict(self) -> None:
        cfg = WizardReasoningConfig(
            wizard_config={
                "name": "w",
                "stages": [
                    {"name": "s", "prompt": "p", "is_start": True, "is_end": True}
                ],
            },
            hooks={"on_complete": ["myapp.hooks:save"]},
            initial_data={"seed": 1},
        )
        assert_structured_config_roundtrip(cfg)

    def test_from_dict_ignores_discriminator_and_unknown_keys(self) -> None:
        cfg = WizardReasoningConfig.from_dict(
            {
                "strategy": "wizard",  # registry discriminator — ignored
                "wizard_config": "w.yaml",
                "unknown_future_key": 123,  # tolerated
            }
        )
        assert cfg.wizard_config == "w.yaml"
        assert cfg.strict_validation is True

    def test_frozen(self) -> None:
        cfg = WizardReasoningConfig(wizard_config="w.yaml")
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.strict_validation = False  # type: ignore[misc]

    def test_wizard_config_required(self) -> None:
        with pytest.raises(TypeError):
            WizardReasoningConfig()  # type: ignore[call-arg]

    def test_extraction_api_key_redacted_in_repr(self) -> None:
        """Credentials nested in extraction_config are masked by the base repr."""
        cfg = WizardReasoningConfig(
            wizard_config="w.yaml",
            extraction_config={"provider": "openai", "api_key": "sk-secret"},
        )
        rendered = repr(cfg)
        assert "sk-secret" not in rendered
        assert "openai" in rendered


# ===================================================================
# TestNavigationCommandConfig
# ===================================================================


class TestNavigationCommandConfig:
    """The single-command leaf config."""

    def test_roundtrip(self) -> None:
        assert_structured_config_roundtrip(
            NavigationCommandConfig(keywords=("a", "b"))
        )

    def test_keywords_coerced_to_tuple(self) -> None:
        cfg = NavigationCommandConfig(keywords=["a", "b"])  # type: ignore[arg-type]
        assert cfg.keywords == ("a", "b")

    def test_from_dict_coerces_list(self) -> None:
        cfg = NavigationCommandConfig.from_dict(
            {"keywords": ["x", "y"], "enabled": False}
        )
        assert cfg.keywords == ("x", "y")
        assert cfg.enabled is False

    def test_frozen(self) -> None:
        cfg = NavigationCommandConfig(keywords=("a",))
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.enabled = False  # type: ignore[misc]


# ===================================================================
# TestNavigationConfig
# ===================================================================


class TestNavigationConfig:
    """The back/skip/restart container — _normalize_dict shape quirks."""

    def test_empty_dict_returns_defaults(self) -> None:
        nav = NavigationConfig.from_dict({})
        assert nav == NavigationConfig.defaults()
        assert nav.back.keywords == DEFAULT_BACK_KEYWORDS
        assert nav.skip.keywords == DEFAULT_SKIP_KEYWORDS
        assert nav.restart.keywords == DEFAULT_RESTART_KEYWORDS

    def test_partial_override_keeps_other_defaults(self) -> None:
        nav = NavigationConfig.from_dict({"back": {"keywords": ["undo"]}})
        assert nav.back.keywords == ("undo",)
        assert nav.skip.keywords == DEFAULT_SKIP_KEYWORDS
        assert nav.restart.keywords == DEFAULT_RESTART_KEYWORDS

    def test_keywords_lowercased(self) -> None:
        nav = NavigationConfig.from_dict(
            {"back": {"keywords": ["BACK", "Go Back"]}}
        )
        assert nav.back.keywords == ("back", "go back")

    def test_enabled_false_keeps_default_keywords(self) -> None:
        nav = NavigationConfig.from_dict({"skip": {"enabled": False}})
        assert nav.skip.enabled is False
        assert nav.skip.keywords == DEFAULT_SKIP_KEYWORDS

    def test_nested_children_are_structured_config(self) -> None:
        nav = NavigationConfig.from_dict({"back": {"keywords": ["b"]}})
        assert isinstance(nav.back, NavigationCommandConfig)
        assert isinstance(nav.skip, NavigationCommandConfig)

    def test_roundtrip(self) -> None:
        assert_structured_config_roundtrip(NavigationConfig.defaults())
        assert_structured_config_roundtrip(
            NavigationConfig.from_dict(
                {
                    "back": {"keywords": ["b"]},
                    "skip": {"keywords": ["s", "next"], "enabled": False},
                    "restart": {"keywords": ["r"]},
                }
            )
        )

    def test_frozen(self) -> None:
        nav = NavigationConfig.defaults()
        with pytest.raises(dataclasses.FrozenInstanceError):
            nav.back = NavigationCommandConfig(keywords=("x",))  # type: ignore[misc]


# ===================================================================
# TestToolResultMappingEntry
# ===================================================================


class TestToolResultMappingEntry:
    """The post-extraction tool-to-state mapping leaf config."""

    def test_roundtrip(self) -> None:
        assert_structured_config_roundtrip(
            ToolResultMappingEntry(
                tool_name="lookup",
                params={"q": "query"},
                mapping={"result": "answer"},
            )
        )

    def test_default_on_error(self) -> None:
        entry = ToolResultMappingEntry(tool_name="t", params={}, mapping={})
        assert entry.on_error == "skip"

    def test_invalid_on_error_rejected(self) -> None:
        with pytest.raises(ValueError, match="on_error"):
            ToolResultMappingEntry(
                tool_name="t", params={}, mapping={}, on_error="bogus"
            )

    def test_from_dict(self) -> None:
        entry = ToolResultMappingEntry.from_dict(
            {
                "tool_name": "lookup",
                "params": {"q": "query"},
                "mapping": {"result": "answer"},
                "on_error": "fail",
            }
        )
        assert entry.tool_name == "lookup"
        assert entry.on_error == "fail"

    def test_frozen(self) -> None:
        entry = ToolResultMappingEntry(tool_name="t", params={}, mapping={})
        with pytest.raises(dataclasses.FrozenInstanceError):
            entry.on_error = "fail"  # type: ignore[misc]
