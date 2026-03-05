"""Tests for bot capability validation with per-role requirements.

Verifies that:
- Wizard strategy assigns json_mode to extraction role, not main
- React strategy assigns function_calling to main role
- validate_bot_capabilities checks each role against the correct resource
- Backward-compat infer_main_capability_requirements works
"""

import pytest

from dataknobs_bots.bot.validation import (
    infer_capability_requirements,
    infer_main_capability_requirements,
    validate_bot_capabilities,
)


# ---------------------------------------------------------------------------
# infer_capability_requirements — per-role
# ---------------------------------------------------------------------------


class TestInferCapabilityRequirements:
    def test_wizard_assigns_json_mode_to_extraction(self):
        config = {"reasoning": {"strategy": "wizard"}}
        reqs = infer_capability_requirements(config)
        assert "extraction" in reqs
        assert "json_mode" in reqs["extraction"]
        assert "main" not in reqs  # wizard doesn't require main capabilities

    def test_react_with_tools_assigns_function_calling_to_main(self):
        config = {
            "reasoning": {"strategy": "react"},
            "tools": [{"name": "search"}],
        }
        reqs = infer_capability_requirements(config)
        assert "main" in reqs
        assert "function_calling" in reqs["main"]
        assert "extraction" not in reqs

    def test_react_without_tools_no_requirements(self):
        config = {"reasoning": {"strategy": "react"}, "tools": []}
        reqs = infer_capability_requirements(config)
        assert reqs == {}

    def test_simple_strategy_no_requirements(self):
        config = {"reasoning": {"strategy": "simple"}}
        reqs = infer_capability_requirements(config)
        assert reqs == {}

    def test_explicit_requires_goes_to_main(self):
        config = {
            "reasoning": {"strategy": "simple"},
            "llm": {"$requires": ["streaming", "vision"]},
        }
        reqs = infer_capability_requirements(config)
        assert "main" in reqs
        assert "streaming" in reqs["main"]
        assert "vision" in reqs["main"]

    def test_wizard_with_explicit_requires(self):
        """Wizard + explicit $requires: json_mode to extraction, explicit to main."""
        config = {
            "reasoning": {"strategy": "wizard"},
            "llm": {"$requires": ["streaming"]},
        }
        reqs = infer_capability_requirements(config)
        assert reqs["extraction"] == ["json_mode"]
        assert reqs["main"] == ["streaming"]

    def test_empty_config(self):
        reqs = infer_capability_requirements({})
        assert reqs == {}


# ---------------------------------------------------------------------------
# infer_main_capability_requirements — backward compat
# ---------------------------------------------------------------------------


class TestInferMainCapabilityRequirements:
    def test_wizard_returns_empty_for_main(self):
        """Wizard's json_mode is extraction-only, main has no requirements."""
        config = {"reasoning": {"strategy": "wizard"}}
        main_reqs = infer_main_capability_requirements(config)
        assert main_reqs == []

    def test_react_returns_function_calling(self):
        config = {
            "reasoning": {"strategy": "react"},
            "tools": [{"name": "search"}],
        }
        main_reqs = infer_main_capability_requirements(config)
        assert "function_calling" in main_reqs

    def test_explicit_requires(self):
        config = {"llm": {"$requires": ["vision"]}}
        main_reqs = infer_main_capability_requirements(config)
        assert "vision" in main_reqs


# ---------------------------------------------------------------------------
# validate_bot_capabilities — per-role resource resolution
# ---------------------------------------------------------------------------


from typing import Any


class _FakeEnvironment:
    """Minimal EnvironmentConfig stand-in for validation tests."""

    def __init__(self, resources: dict[str, dict[str, Any]] | None = None):
        self.name = "test"
        self._resources = resources or {}

    def get_resource(self, resource_type: str, name: str) -> dict[str, Any]:
        key = f"{resource_type}/{name}"
        if key not in self._resources:
            raise KeyError(name)
        return self._resources[key]


class TestValidateBotCapabilities:
    def test_wizard_validates_extraction_resource(self):
        """json_mode requirement is checked against the extraction LLM resource."""
        config = {
            "reasoning": {
                "strategy": "wizard",
                "config": {
                    "wizard_config": {},
                    "extraction_config": {
                        "$resource": "extraction-llm",
                        "type": "llm_providers",
                    },
                },
            },
            "llm": {"$resource": "main-llm", "type": "llm_providers"},
        }
        env = _FakeEnvironment({
            "llm_providers/main-llm": {"model": "llama3.2"},
            "llm_providers/extraction-llm": {
                "model": "qwen3",
                "capabilities": ["text_generation", "json_mode"],
            },
        })
        warnings = validate_bot_capabilities(config, env)
        assert warnings == []

    def test_wizard_warns_when_extraction_lacks_json_mode(self):
        config = {
            "reasoning": {
                "strategy": "wizard",
                "config": {
                    "extraction_config": {
                        "$resource": "extraction-llm",
                        "type": "llm_providers",
                    },
                },
            },
            "llm": {"$resource": "main-llm"},
        }
        env = _FakeEnvironment({
            "llm_providers/main-llm": {"model": "llama3.2"},
            "llm_providers/extraction-llm": {
                "model": "tinyllama",
                "capabilities": ["text_generation"],
            },
        })
        warnings = validate_bot_capabilities(config, env)
        assert any("json_mode" in w and "Extraction" in w for w in warnings)

    def test_wizard_does_not_warn_main_about_json_mode(self):
        """Main LLM should NOT be checked for json_mode in wizard bots."""
        config = {
            "reasoning": {
                "strategy": "wizard",
                "config": {
                    "extraction_config": {
                        "model": "qwen3",
                        "capabilities": ["text_generation", "json_mode"],
                    },
                },
            },
            "llm": {
                "$resource": "main-llm",
                "type": "llm_providers",
            },
        }
        env = _FakeEnvironment({
            "llm_providers/main-llm": {
                "model": "llama3.2",
                "capabilities": ["text_generation", "chat"],  # no json_mode
            },
        })
        warnings = validate_bot_capabilities(config, env)
        # No warnings about main LLM needing json_mode
        assert not any("Main" in w and "json_mode" in w for w in warnings)

    def test_wizard_inline_extraction_config(self):
        """Inline extraction_config (no $resource) validated via model name."""
        config = {
            "reasoning": {
                "strategy": "wizard",
                "config": {
                    "extraction_config": {
                        "provider": "ollama",
                        "model": "qwen3",
                        "capabilities": ["text_generation", "json_mode"],
                    },
                },
            },
            "llm": {"$resource": "main-llm"},
        }
        env = _FakeEnvironment({
            "llm_providers/main-llm": {"model": "llama3.2"},
        })
        warnings = validate_bot_capabilities(config, env)
        assert warnings == []

    def test_react_validates_main_resource(self):
        """function_calling is checked against the main LLM resource."""
        config = {
            "reasoning": {"strategy": "react"},
            "tools": [{"name": "search"}],
            "llm": {"$resource": "main-llm", "type": "llm_providers"},
        }
        env = _FakeEnvironment({
            "llm_providers/main-llm": {
                "model": "llama3.2",
                "capabilities": ["text_generation", "function_calling"],
            },
        })
        warnings = validate_bot_capabilities(config, env)
        assert warnings == []

    def test_react_warns_when_main_lacks_function_calling(self):
        config = {
            "reasoning": {"strategy": "react"},
            "tools": [{"name": "search"}],
            "llm": {"$resource": "main-llm", "type": "llm_providers"},
        }
        env = _FakeEnvironment({
            "llm_providers/main-llm": {
                "model": "tinyllama",
                "capabilities": ["text_generation"],
            },
        })
        warnings = validate_bot_capabilities(config, env)
        assert any("function_calling" in w and "Main" in w for w in warnings)

    def test_no_resource_reference_skips_validation(self):
        config = {
            "reasoning": {"strategy": "react"},
            "tools": [{"name": "search"}],
            "llm": {"model": "llama3.2"},  # no $resource
        }
        env = _FakeEnvironment()
        warnings = validate_bot_capabilities(config, env)
        assert warnings == []

    def test_missing_resource_returns_warning(self):
        config = {
            "reasoning": {"strategy": "react"},
            "tools": [{"name": "search"}],
            "llm": {"$resource": "nonexistent"},
        }
        env = _FakeEnvironment()
        warnings = validate_bot_capabilities(config, env)
        assert any("not found" in w for w in warnings)
