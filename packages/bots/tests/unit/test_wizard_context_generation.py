"""Tests for wizard context generation and transition derivation."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from dataknobs_bots.reasoning.wizard import WizardReasoning, WizardState
from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader


# =========================================================================
# Tests for _render_response_template with extra_context
# =========================================================================


class TestRenderResponseTemplateExtraContext:
    """Tests for _render_response_template with extra_context parameter."""

    def test_extra_context_available_in_template(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Extra context variables are accessible in the template."""
        state = WizardState(current_stage="welcome", data={})
        stage = {"name": "welcome"}

        result = wizard_reasoning._render_response_template(
            "Names: {{ suggested_names }}",
            stage,
            state,
            extra_context={"suggested_names": "Alpha, Beta, Gamma"},
        )
        assert result == "Names: Alpha, Beta, Gamma"

    def test_extra_context_merged_with_state_data(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Both state data and extra context are available."""
        state = WizardState(
            current_stage="welcome",
            data={"subject": "Chemistry"},
        )
        stage = {"name": "welcome"}

        result = wizard_reasoning._render_response_template(
            "{{ subject }}: {{ suggestions }}",
            stage,
            state,
            extra_context={"suggestions": "Chem Coach, Mol Master"},
        )
        assert result == "Chemistry: Chem Coach, Mol Master"

    def test_none_extra_context_handled(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """None extra_context works (backwards compatible)."""
        state = WizardState(
            current_stage="welcome",
            data={"name": "Alice"},
        )
        stage = {"name": "welcome"}

        result = wizard_reasoning._render_response_template(
            "Hello {{ name }}!", stage, state, extra_context=None
        )
        assert result == "Hello Alice!"

    def test_empty_extra_context_handled(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Empty dict extra_context works."""
        state = WizardState(
            current_stage="welcome",
            data={"name": "Bob"},
        )
        stage = {"name": "welcome"}

        result = wizard_reasoning._render_response_template(
            "Hello {{ name }}!", stage, state, extra_context={}
        )
        assert result == "Hello Bob!"


# =========================================================================
# Tests for _generate_context_variables
# =========================================================================


class TestGenerateContextVariables:
    """Tests for _generate_context_variables."""

    @pytest.mark.asyncio
    async def test_no_context_generation_returns_empty(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Stage without context_generation returns empty dict."""
        state = WizardState(current_stage="welcome", data={})
        stage = {"name": "welcome"}
        llm = MagicMock()

        result = await wizard_reasoning._generate_context_variables(
            stage, state, llm
        )
        assert result == {}

    @pytest.mark.asyncio
    async def test_successful_llm_generation(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """LLM response is stored under the configured variable name."""
        state = WizardState(
            current_stage="configure_identity",
            data={"subject": "Physics", "intent": "tutor"},
        )
        stage = {
            "name": "configure_identity",
            "context_generation": {
                "prompt": "Suggest names for a {{ intent }} bot about {{ subject }}.",
                "variable": "suggested_names",
                "fallback": "Study Buddy",
            },
        }

        llm = AsyncMock()
        llm.complete.return_value = MagicMock(
            content="- **Physics Pro** (`physics-pro`)\n- **Force Field** (`force-field`)"
        )

        result = await wizard_reasoning._generate_context_variables(
            stage, state, llm
        )
        assert "suggested_names" in result
        assert "Physics Pro" in result["suggested_names"]

    @pytest.mark.asyncio
    async def test_llm_prompt_rendered_with_state_data(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """The prompt template is rendered with wizard state data before LLM call."""
        state = WizardState(
            current_stage="configure_identity",
            data={"subject": "Biology", "intent": "quiz"},
        )
        stage = {
            "name": "configure_identity",
            "context_generation": {
                "prompt": "Names for a {{ intent }} bot about {{ subject }}.",
                "variable": "names",
                "fallback": "default",
            },
        }

        llm = AsyncMock()
        llm.complete.return_value = MagicMock(content="Bio Quiz")

        await wizard_reasoning._generate_context_variables(stage, state, llm)

        # Verify the prompt was rendered before being sent to LLM
        call_args = llm.complete.call_args[0][0]  # First positional arg (messages)
        sent_content = call_args[0].content
        assert "quiz" in sent_content
        assert "Biology" in sent_content
        assert "{{" not in sent_content  # No unrendered templates

    @pytest.mark.asyncio
    async def test_fallback_on_llm_failure(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Fallback value used when LLM call raises an exception."""
        state = WizardState(current_stage="welcome", data={})
        stage = {
            "name": "welcome",
            "context_generation": {
                "prompt": "Suggest names.",
                "variable": "names",
                "fallback": "Default Bot",
            },
        }

        llm = AsyncMock()
        llm.complete.side_effect = RuntimeError("Connection timeout")

        result = await wizard_reasoning._generate_context_variables(
            stage, state, llm
        )
        assert result == {"names": "Default Bot"}

    @pytest.mark.asyncio
    async def test_fallback_on_empty_response(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Fallback used when LLM returns empty content."""
        state = WizardState(current_stage="welcome", data={})
        stage = {
            "name": "welcome",
            "context_generation": {
                "prompt": "Suggest names.",
                "variable": "names",
                "fallback": "Fallback Name",
            },
        }

        llm = AsyncMock()
        llm.complete.return_value = MagicMock(content="")

        result = await wizard_reasoning._generate_context_variables(
            stage, state, llm
        )
        assert result == {"names": "Fallback Name"}

    @pytest.mark.asyncio
    async def test_missing_prompt_returns_empty(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Missing prompt in context_generation returns empty dict."""
        state = WizardState(current_stage="welcome", data={})
        stage = {
            "name": "welcome",
            "context_generation": {
                "variable": "names",
            },
        }
        llm = MagicMock()

        result = await wizard_reasoning._generate_context_variables(
            stage, state, llm
        )
        assert result == {}

    @pytest.mark.asyncio
    async def test_missing_variable_returns_empty(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Missing variable name in context_generation returns empty dict."""
        state = WizardState(current_stage="welcome", data={})
        stage = {
            "name": "welcome",
            "context_generation": {
                "prompt": "Suggest names.",
            },
        }
        llm = MagicMock()

        result = await wizard_reasoning._generate_context_variables(
            stage, state, llm
        )
        assert result == {}


# =========================================================================
# Tests for _render_suggestions
# =========================================================================


class TestRenderSuggestions:
    """Tests for _render_suggestions."""

    def test_plain_suggestions_unchanged(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Suggestions without {{ }} pass through unchanged."""
        state = WizardState(current_stage="welcome", data={"subject": "Math"})
        suggestions = ["Create a bot", "Skip this step"]

        result = wizard_reasoning._render_suggestions(suggestions, state)
        assert result == ["Create a bot", "Skip this step"]

    def test_template_suggestions_rendered(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Suggestions with {{ }} are rendered with state data."""
        state = WizardState(
            current_stage="welcome",
            data={"subject": "Chemistry"},
        )
        suggestions = [
            "Call it '{{ subject }} Ace'",
            "Name it '{{ subject }} Helper'",
        ]

        result = wizard_reasoning._render_suggestions(suggestions, state)
        assert result == [
            "Call it 'Chemistry Ace'",
            "Name it 'Chemistry Helper'",
        ]

    def test_mixed_plain_and_template(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Mix of plain and templated suggestions works."""
        state = WizardState(
            current_stage="welcome",
            data={"subject": "Physics"},
        )
        suggestions = [
            "I have my own name",
            "Call it '{{ subject }} Pro'",
        ]

        result = wizard_reasoning._render_suggestions(suggestions, state)
        assert result == [
            "I have my own name",
            "Call it 'Physics Pro'",
        ]

    def test_empty_suggestions_returned(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Empty list returns empty list."""
        state = WizardState(current_stage="welcome", data={})
        assert wizard_reasoning._render_suggestions([], state) == []

    def test_undefined_variable_renders_empty(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Undefined variables in suggestions render as empty string."""
        state = WizardState(current_stage="welcome", data={})
        suggestions = ["Try '{{ subject }} Bot'"]

        result = wizard_reasoning._render_suggestions(suggestions, state)
        assert result == ["Try ' Bot'"]

    def test_internal_keys_excluded(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Internal keys (starting with _) are not available in suggestions."""
        state = WizardState(
            current_stage="welcome",
            data={"subject": "Math", "_internal": "hidden"},
        )
        suggestions = ["{{ subject }} ({{ _internal }})"]

        result = wizard_reasoning._render_suggestions(suggestions, state)
        # _internal should render as empty
        assert result == ["Math ()"]


# =========================================================================
# Tests for _apply_transition_derivations
# =========================================================================


class TestApplyTransitionDerivations:
    """Tests for _apply_transition_derivations."""

    def test_jinja2_derivation(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Derive a value from Jinja2 template referencing state data."""
        state = WizardState(
            current_stage="welcome",
            data={"intent": "quiz"},
        )
        stage = {
            "name": "welcome",
            "transitions": [
                {
                    "target": "configure_identity",
                    "derive": {
                        "template_name": "{{ intent }}",
                    },
                    "condition": "data.get('intent')",
                },
            ],
        }

        wizard_reasoning._apply_transition_derivations(stage, state)
        assert state.data["template_name"] == "quiz"

    def test_literal_derivation(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Derive a literal (non-template) value."""
        state = WizardState(
            current_stage="welcome",
            data={"intent": "quiz"},
        )
        stage = {
            "name": "welcome",
            "transitions": [
                {
                    "target": "configure_identity",
                    "derive": {
                        "use_template": True,
                    },
                },
            ],
        }

        wizard_reasoning._apply_transition_derivations(stage, state)
        assert state.data["use_template"] is True

    def test_does_not_overwrite_existing_data(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Derivation does not overwrite user-provided data."""
        state = WizardState(
            current_stage="welcome",
            data={"intent": "quiz", "template_name": "tutor"},
        )
        stage = {
            "name": "welcome",
            "transitions": [
                {
                    "target": "configure_identity",
                    "derive": {
                        "template_name": "{{ intent }}",
                    },
                },
            ],
        }

        wizard_reasoning._apply_transition_derivations(stage, state)
        # Should keep user's value, not overwrite with derived
        assert state.data["template_name"] == "tutor"

    def test_empty_render_skipped(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Template that renders to empty string is not set."""
        state = WizardState(
            current_stage="welcome",
            data={},  # No intent → {{ intent }} renders empty
        )
        stage = {
            "name": "welcome",
            "transitions": [
                {
                    "target": "configure_identity",
                    "derive": {
                        "template_name": "{{ intent }}",
                    },
                },
            ],
        }

        wizard_reasoning._apply_transition_derivations(stage, state)
        assert "template_name" not in state.data

    def test_no_transitions_is_noop(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Stage with no transitions does nothing."""
        state = WizardState(current_stage="welcome", data={"x": 1})
        stage = {"name": "welcome", "transitions": []}

        wizard_reasoning._apply_transition_derivations(stage, state)
        assert state.data == {"x": 1}

    def test_no_derive_in_transition_is_noop(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Transition without derive block is skipped."""
        state = WizardState(current_stage="welcome", data={"x": 1})
        stage = {
            "name": "welcome",
            "transitions": [
                {"target": "next", "condition": "data.get('x')"},
            ],
        }

        wizard_reasoning._apply_transition_derivations(stage, state)
        assert state.data == {"x": 1}

    def test_multiple_transitions_derive_multiple_keys(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Derivations from multiple transitions are all applied."""
        state = WizardState(
            current_stage="welcome",
            data={"intent": "tutor", "subject": "Math"},
        )
        stage = {
            "name": "welcome",
            "transitions": [
                {
                    "target": "fast_path",
                    "derive": {
                        "template_name": "{{ intent }}",
                        "use_template": True,
                    },
                    "condition": "data.get('intent') in ('tutor', 'quiz')",
                },
                {
                    "target": "slow_path",
                    "derive": {
                        "needs_review": True,
                    },
                },
            ],
        }

        wizard_reasoning._apply_transition_derivations(stage, state)
        assert state.data["template_name"] == "tutor"
        assert state.data["use_template"] is True
        assert state.data["needs_review"] is True


# =========================================================================
# Tests for loader metadata extraction of new fields
# =========================================================================


class TestLoaderContextGeneration:
    """Tests for WizardConfigLoader extracting context_generation."""

    def test_context_generation_in_metadata(self) -> None:
        """context_generation block is extracted into stage metadata."""
        config: dict[str, Any] = {
            "name": "test",
            "stages": [
                {
                    "name": "start",
                    "is_start": True,
                    "is_end": True,
                    "prompt": "Hello",
                    "context_generation": {
                        "prompt": "Suggest names for {{ subject }}",
                        "variable": "suggested_names",
                        "fallback": "Default",
                    },
                },
            ],
        }
        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(config)

        meta = wizard_fsm.stages["start"]
        assert meta["context_generation"] is not None
        assert meta["context_generation"]["variable"] == "suggested_names"

    def test_no_context_generation_is_none(self) -> None:
        """Stage without context_generation has None in metadata."""
        config: dict[str, Any] = {
            "name": "test",
            "stages": [
                {
                    "name": "start",
                    "is_start": True,
                    "is_end": True,
                    "prompt": "Hello",
                },
            ],
        }
        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(config)

        meta = wizard_fsm.stages["start"]
        assert meta["context_generation"] is None


class TestLoaderTransitionDerive:
    """Tests for WizardConfigLoader extracting derive from transitions."""

    def test_derive_in_transition_metadata(self) -> None:
        """derive block is extracted into transition metadata."""
        config: dict[str, Any] = {
            "name": "test",
            "stages": [
                {
                    "name": "start",
                    "is_start": True,
                    "prompt": "Hello",
                    "transitions": [
                        {
                            "target": "end",
                            "derive": {
                                "template_name": "{{ intent }}",
                                "use_template": True,
                            },
                            "condition": "data.get('intent')",
                        },
                    ],
                },
                {
                    "name": "end",
                    "is_end": True,
                    "prompt": "Done",
                },
            ],
        }
        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(config)

        transitions = wizard_fsm.stages["start"]["transitions"]
        assert len(transitions) == 1
        assert transitions[0]["derive"] == {
            "template_name": "{{ intent }}",
            "use_template": True,
        }

    def test_no_derive_is_none(self) -> None:
        """Transition without derive has None in metadata."""
        config: dict[str, Any] = {
            "name": "test",
            "stages": [
                {
                    "name": "start",
                    "is_start": True,
                    "prompt": "Hello",
                    "transitions": [{"target": "end"}],
                },
                {
                    "name": "end",
                    "is_end": True,
                    "prompt": "Done",
                },
            ],
        }
        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(config)

        transitions = wizard_fsm.stages["start"]["transitions"]
        assert transitions[0]["derive"] is None
