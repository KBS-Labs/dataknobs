"""Tests for wizard template response rendering."""

from dataknobs_bots.reasoning.wizard import WizardReasoning, WizardState


class TestRenderResponseTemplate:
    """Tests for _render_response_template."""

    def test_simple_variable_substitution(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """{{ var }} renders from state data."""
        state = WizardState(
            current_stage="welcome",
            data={"name": "Alice"},
        )
        stage = {"name": "welcome", "label": "Welcome"}

        result = wizard_reasoning._render_response_template(
            "Hello {{ name }}!", stage, state
        )
        assert result == "Hello Alice!"

    def test_collected_data_accessible(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Template can access collected_data dict."""
        state = WizardState(
            current_stage="welcome",
            data={"color": "blue", "size": "large"},
        )
        stage = {"name": "welcome"}

        result = wizard_reasoning._render_response_template(
            "Items: {{ collected_data.color }}, {{ collected_data.size }}",
            stage,
            state,
        )
        assert result == "Items: blue, large"

    def test_internal_keys_filtered(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Keys starting with _ are excluded from top-level template context."""
        state = WizardState(
            current_stage="welcome",
            data={"visible": "yes", "_internal": "hidden"},
        )
        stage = {"name": "welcome"}

        result = wizard_reasoning._render_response_template(
            "visible={{ visible }}", stage, state
        )
        assert result == "visible=yes"

        # _internal should not be in collected_data
        result2 = wizard_reasoning._render_response_template(
            "count={{ collected_data | length }}", stage, state
        )
        assert result2 == "count=1"

    def test_stage_metadata_in_context(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """stage_name and stage_label are available in template."""
        state = WizardState(current_stage="welcome", data={})
        stage = {"name": "welcome", "label": "Getting Started"}

        result = wizard_reasoning._render_response_template(
            "Stage: {{ stage_name }} ({{ stage_label }})", stage, state
        )
        assert result == "Stage: welcome (Getting Started)"

    def test_stage_label_falls_back_to_name(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """stage_label falls back to name when no label set."""
        state = WizardState(current_stage="welcome", data={})
        stage = {"name": "welcome"}

        result = wizard_reasoning._render_response_template(
            "{{ stage_label }}", stage, state
        )
        assert result == "welcome"

    def test_undefined_variables_render_empty(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """No error on undefined vars; they render as empty string."""
        state = WizardState(current_stage="welcome", data={})
        stage = {"name": "welcome"}

        result = wizard_reasoning._render_response_template(
            "Hello {{ nonexistent }}!", stage, state
        )
        assert result == "Hello !"

    def test_history_and_completed_in_context(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """history and completed are accessible in template."""
        state = WizardState(
            current_stage="configure",
            data={},
            history=["welcome", "configure"],
            completed=False,
        )
        stage = {"name": "configure"}

        result = wizard_reasoning._render_response_template(
            "steps={{ history | length }}, done={{ completed }}", stage, state
        )
        assert result == "steps=2, done=False"


class TestCreateTemplateResponse:
    """Tests for _create_template_response."""

    def test_response_has_content(self) -> None:
        """Response carries the rendered content."""
        resp = WizardReasoning._create_template_response("Hello world")
        assert resp.content == "Hello world"

    def test_response_model_is_template(self) -> None:
        """Model field is 'template'."""
        resp = WizardReasoning._create_template_response("test")
        assert resp.model == "template"

    def test_response_has_finish_reason(self) -> None:
        """finish_reason is 'stop'."""
        resp = WizardReasoning._create_template_response("test")
        assert resp.finish_reason == "stop"

    def test_response_has_metadata_dict(self) -> None:
        """Response metadata is an empty dict by default."""
        resp = WizardReasoning._create_template_response("test")
        assert isinstance(resp.metadata, dict)

    def test_response_tool_calls_is_none(self) -> None:
        """tool_calls defaults to None."""
        resp = WizardReasoning._create_template_response("test")
        assert resp.tool_calls is None

    def test_response_has_created_at(self) -> None:
        """Response has a created_at timestamp."""
        resp = WizardReasoning._create_template_response("test")
        assert resp.created_at is not None
