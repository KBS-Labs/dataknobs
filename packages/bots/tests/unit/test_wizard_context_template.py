"""Tests for wizard configurable context template (enhancement 2i).

The context_template setting allows customizing how stage context is
formatted in the system prompt using Jinja2 templates.
"""

import pytest

from dataknobs_bots.reasoning.wizard import WizardReasoning, WizardState
from dataknobs_bots.reasoning.wizard_fsm import WizardFSM
from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader


@pytest.fixture
def simple_fsm() -> WizardFSM:
    """Create a simple wizard FSM for testing."""
    config = {
        "name": "test-wizard",
        "stages": [
            {
                "name": "welcome",
                "is_start": True,
                "prompt": "Tell me about your bot",
                "can_skip": True,
                "help_text": "Just describe what you want",
                "suggestions": ["math tutor", "quiz bot"],
                "transitions": [{"target": "done"}],
            },
            {"name": "done", "is_end": True, "prompt": "Complete!"},
        ],
    }
    loader = WizardConfigLoader()
    return loader.load_from_dict(config)


class TestRenderCustomContext:
    """Tests for _render_custom_context method."""

    def test_render_custom_context_basic(self, simple_fsm: WizardFSM) -> None:
        """Custom template renders with basic variables."""
        reasoning = WizardReasoning(
            wizard_fsm=simple_fsm,
            context_template="Stage: {{stage_name}}, Goal: {{stage_prompt}}",
        )

        stage = simple_fsm.current_metadata
        state = WizardState(current_stage="welcome", data={})

        result = reasoning._render_custom_context(stage, state)

        assert "Stage: welcome" in result
        assert "Goal: Tell me about your bot" in result

    def test_render_custom_context_collected_data(self, simple_fsm: WizardFSM) -> None:
        """Custom template renders collected data with Jinja2 loop."""
        reasoning = WizardReasoning(
            wizard_fsm=simple_fsm,
            context_template="""
{% if collected_data %}
Collected:
{% for k, v in collected_data.items() %}
- {{k}}: {{v}}
{% endfor %}
{% endif %}
""",
        )

        stage = simple_fsm.current_metadata
        state = WizardState(
            current_stage="welcome",
            data={"subject": "math", "level": "advanced", "_internal": "hidden"},
        )

        result = reasoning._render_custom_context(stage, state)

        assert "subject: math" in result
        assert "level: advanced" in result
        assert "_internal" not in result  # Internal keys filtered

    def test_render_custom_context_conditional(self, simple_fsm: WizardFSM) -> None:
        """Conditional ((sections)) work correctly."""
        reasoning = WizardReasoning(
            wizard_fsm=simple_fsm,
            context_template="Main((, with help: {{help_text}}))",
        )

        # With help text
        stage = {"name": "test", "help_text": "Some help"}
        state = WizardState(current_stage="test", data={})
        result = reasoning._render_custom_context(stage, state)
        assert ", with help: Some help" in result

        # Without help text - conditional section removed
        stage = {"name": "test", "help_text": None}
        result = reasoning._render_custom_context(stage, state)
        assert ", with help:" not in result

    def test_render_suggestions_list(self, simple_fsm: WizardFSM) -> None:
        """Suggestions list is available in template with Jinja2 filters."""
        reasoning = WizardReasoning(
            wizard_fsm=simple_fsm,
            context_template="Suggestions: {{ suggestions | join(', ') }}",
        )

        stage = simple_fsm.current_metadata
        state = WizardState(current_stage="welcome", data={})

        result = reasoning._render_custom_context(stage, state)

        assert "math tutor" in result
        assert "quiz bot" in result

    def test_render_completed_status(self, simple_fsm: WizardFSM) -> None:
        """Completed status is available in template."""
        reasoning = WizardReasoning(
            wizard_fsm=simple_fsm,
            context_template="{% if completed %}DONE{% else %}IN PROGRESS{% endif %}",
        )

        stage = simple_fsm.current_metadata

        # Not completed
        state = WizardState(current_stage="welcome", data={}, completed=False)
        result = reasoning._render_custom_context(stage, state)
        assert "IN PROGRESS" in result
        assert "DONE" not in result

        # Completed
        state = WizardState(current_stage="done", data={}, completed=True)
        result = reasoning._render_custom_context(stage, state)
        assert "DONE" in result

    def test_render_navigation_variables(self, simple_fsm: WizardFSM) -> None:
        """can_skip and can_go_back are available in template."""
        reasoning = WizardReasoning(
            wizard_fsm=simple_fsm,
            context_template="Skip: {{can_skip}}, Back: {{can_go_back}}",
        )

        stage = simple_fsm.current_metadata
        state = WizardState(
            current_stage="welcome",
            data={},
            history=["welcome"],
        )

        result = reasoning._render_custom_context(stage, state)

        # can_skip is True for this stage (configured in fixture)
        assert "Skip: True" in result

    def test_render_history(self, simple_fsm: WizardFSM) -> None:
        """History is available in template with Jinja2 filters."""
        reasoning = WizardReasoning(
            wizard_fsm=simple_fsm,
            context_template="History: {{ history | join(' -> ') }}",
        )

        stage = simple_fsm.current_metadata
        state = WizardState(
            current_stage="done",
            data={},
            history=["welcome", "config", "done"],
        )

        result = reasoning._render_custom_context(stage, state)

        assert "welcome -> config -> done" in result


class TestBuildStageContext:
    """Tests for _build_stage_context dispatch logic."""

    def test_build_stage_context_uses_template(self, simple_fsm: WizardFSM) -> None:
        """_build_stage_context dispatches to custom template when set."""
        reasoning = WizardReasoning(
            wizard_fsm=simple_fsm, context_template="CUSTOM: {{stage_name}}"
        )

        stage = {"name": "test"}
        state = WizardState(current_stage="test", data={})

        result = reasoning._build_stage_context(stage, state)

        assert result == "CUSTOM: test"

    def test_build_stage_context_default_without_template(
        self, simple_fsm: WizardFSM
    ) -> None:
        """_build_stage_context uses default when no template."""
        reasoning = WizardReasoning(wizard_fsm=simple_fsm, context_template=None)

        stage = {"name": "test", "prompt": "Do something"}
        state = WizardState(current_stage="test", data={})

        result = reasoning._build_stage_context(stage, state)

        # Should use default format
        assert "## Current Wizard Stage" in result
        assert "Stage: test" in result
        assert "Goal: Do something" in result

    def test_default_context_includes_collected_data(
        self, simple_fsm: WizardFSM
    ) -> None:
        """Default context format includes collected data section."""
        reasoning = WizardReasoning(wizard_fsm=simple_fsm, context_template=None)

        stage = {"name": "test", "prompt": "Configure"}
        state = WizardState(
            current_stage="test",
            data={"field1": "value1", "_internal": "hidden"},
        )

        result = reasoning._build_default_context(stage, state)

        assert "ALREADY COLLECTED" in result
        assert "field1: value1" in result
        assert "_internal" not in result


class TestContextTemplateIntegration:
    """Integration tests for context template feature."""

    def test_from_config_loads_context_template(self) -> None:
        """from_config correctly reads context_template from settings."""
        import tempfile
        from pathlib import Path

        config_content = """
name: test-wizard
settings:
  context_template: |
    ## Stage: {{stage_name}}
    Goal: {{stage_prompt}}
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
            assert reasoning._context_template is not None
            assert "{{stage_name}}" in reasoning._context_template
            assert "{{stage_prompt}}" in reasoning._context_template
        finally:
            config_path.unlink()

    def test_settings_context_template_accessible(self) -> None:
        """Context template is accessible via WizardFSM.settings."""
        config = {
            "name": "test",
            "settings": {"context_template": "Custom template here"},
            "stages": [{"name": "start", "is_start": True, "is_end": True}],
        }
        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(config)

        assert fsm.settings.get("context_template") == "Custom template here"

    def test_complex_template_with_all_features(self) -> None:
        """Complex template using all available variables and features."""
        config = {
            "name": "test-wizard",
            "settings": {
                "context_template": """
## Wizard Stage: {{stage_name}}

**Goal**: {{stage_prompt}}

((Additional help: {{help_text}}))

{% if collected_data %}
### Already Collected (DO NOT ASK AGAIN)
{% for key, value in collected_data.items() %}
- **{{key}}**: {{value}}
{% endfor %}
{% endif %}

{% if not completed %}
Navigation: {% if can_skip %}Can skip{% endif %}{% if can_go_back %}, Can go back{% endif %}
{% endif %}

{% if suggestions %}
Suggestions: {{ suggestions | join(', ') }}
{% endif %}
"""
            },
            "stages": [
                {
                    "name": "config",
                    "is_start": True,
                    "prompt": "Configure your bot",
                    "help_text": "Provide details about your bot",
                    "can_skip": True,
                    "suggestions": ["option1", "option2"],
                    "transitions": [{"target": "done"}],
                },
                {"name": "done", "is_end": True},
            ],
        }
        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(config)

        reasoning = WizardReasoning(
            wizard_fsm=fsm,
            context_template=config["settings"]["context_template"],
        )

        stage = fsm.current_metadata
        state = WizardState(
            current_stage="config",
            data={"bot_name": "TestBot", "_internal": "hidden"},
            history=["config"],
        )

        result = reasoning._render_custom_context(stage, state)

        # Verify all components rendered correctly
        assert "## Wizard Stage: config" in result
        assert "**Goal**: Configure your bot" in result
        assert "Additional help: Provide details about your bot" in result
        assert "### Already Collected" in result
        assert "**bot_name**: TestBot" in result
        assert "_internal" not in result
        assert "Can skip" in result
        assert "option1, option2" in result


class TestBackwardCompatibility:
    """Tests ensuring backward compatibility when no template is set."""

    def test_no_template_uses_default_format(self, simple_fsm: WizardFSM) -> None:
        """Wizards without context_template use original default format."""
        reasoning = WizardReasoning(wizard_fsm=simple_fsm)

        assert reasoning._context_template is None

        stage = {"name": "test", "prompt": "Test prompt", "suggestions": ["a", "b"]}
        state = WizardState(current_stage="test", data={"field": "value"})

        result = reasoning._build_stage_context(stage, state)

        # Should have default format markers
        assert "## Current Wizard Stage" in result
        assert "Stage: test" in result
        assert "Goal: Test prompt" in result
        assert "field: value" in result

    def test_empty_template_still_uses_default(self, simple_fsm: WizardFSM) -> None:
        """Empty string template is treated as no template (uses default)."""
        # Empty string is falsy, so should use default
        reasoning = WizardReasoning(wizard_fsm=simple_fsm, context_template="")

        stage = {"name": "test", "prompt": "Test"}
        state = WizardState(current_stage="test", data={})

        result = reasoning._build_stage_context(stage, state)

        # Empty string is falsy, dispatches to default
        assert "## Current Wizard Stage" in result
