"""Tests for review personas."""

import pytest

from dataknobs_bots.review.personas import (
    BUILT_IN_PERSONAS,
    ReviewPersona,
    get_persona,
    list_personas,
)


class TestReviewPersona:
    """Tests for ReviewPersona dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic persona creation."""
        persona = ReviewPersona(
            id="test",
            name="Test Reviewer",
            focus="testing things",
            prompt_template="Review this: {artifact_content}",
        )

        assert persona.id == "test"
        assert persona.name == "Test Reviewer"
        assert persona.focus == "testing things"
        assert persona.default_score_threshold == 0.7

    def test_serialization(self) -> None:
        """Test to_dict and from_dict."""
        original = ReviewPersona(
            id="test",
            name="Test Reviewer",
            focus="testing",
            prompt_template="Test template",
            scoring_criteria="Test criteria",
            default_score_threshold=0.85,
            metadata={"domain": "test"},
        )

        data = original.to_dict()
        restored = ReviewPersona.from_dict(data)

        assert restored.id == original.id
        assert restored.name == original.name
        assert restored.focus == original.focus
        assert restored.prompt_template == original.prompt_template
        assert restored.scoring_criteria == original.scoring_criteria
        assert restored.default_score_threshold == original.default_score_threshold
        assert restored.metadata == original.metadata

    def test_from_config(self) -> None:
        """Test creation from config dict."""
        config = {
            "name": "Custom Reviewer",
            "focus": "custom focus",
            "prompt_template": "Custom template",
            "scoring_criteria": "Custom criteria",
            "score_threshold": 0.9,
        }
        persona = ReviewPersona.from_config("custom", config)

        assert persona.id == "custom"
        assert persona.name == "Custom Reviewer"
        assert persona.focus == "custom focus"
        assert persona.default_score_threshold == 0.9


class TestBuiltInPersonas:
    """Tests for built-in personas."""

    def test_all_personas_exist(self) -> None:
        """Test all expected personas exist."""
        expected_personas = [
            "adversarial",
            "skeptical",
            "insightful",
            "minimalist",
            "downstream",
        ]
        for persona_id in expected_personas:
            assert persona_id in BUILT_IN_PERSONAS

    def test_adversarial_persona(self) -> None:
        """Test adversarial persona properties."""
        persona = BUILT_IN_PERSONAS["adversarial"]
        assert persona.id == "adversarial"
        assert "edge case" in persona.focus.lower()
        assert "{artifact_content}" in persona.prompt_template
        assert persona.default_score_threshold == 0.7

    def test_skeptical_persona(self) -> None:
        """Test skeptical persona properties."""
        persona = BUILT_IN_PERSONAS["skeptical"]
        assert persona.id == "skeptical"
        assert "accuracy" in persona.focus.lower()
        assert persona.default_score_threshold == 0.8

    def test_insightful_persona(self) -> None:
        """Test insightful persona properties."""
        persona = BUILT_IN_PERSONAS["insightful"]
        assert persona.id == "insightful"
        assert "context" in persona.focus.lower()

    def test_minimalist_persona(self) -> None:
        """Test minimalist persona properties."""
        persona = BUILT_IN_PERSONAS["minimalist"]
        assert persona.id == "minimalist"
        assert "simplicity" in persona.focus.lower()

    def test_downstream_persona(self) -> None:
        """Test downstream persona properties."""
        persona = BUILT_IN_PERSONAS["downstream"]
        assert persona.id == "downstream"
        assert "usability" in persona.focus.lower()
        assert persona.default_score_threshold == 0.8

    def test_all_personas_have_json_response_format(self) -> None:
        """Test all personas request JSON response format."""
        for persona_id, persona in BUILT_IN_PERSONAS.items():
            assert "JSON" in persona.prompt_template or "json" in persona.prompt_template, (
                f"{persona_id} missing JSON format instruction"
            )

    def test_all_personas_have_placeholders(self) -> None:
        """Test all personas have required placeholders."""
        required_placeholders = [
            "{artifact_type}",
            "{artifact_name}",
            "{artifact_purpose}",
            "{artifact_content}",
        ]
        for persona_id, persona in BUILT_IN_PERSONAS.items():
            for placeholder in required_placeholders:
                assert placeholder in persona.prompt_template, (
                    f"{persona_id} missing placeholder: {placeholder}"
                )


class TestPersonaHelpers:
    """Tests for persona helper functions."""

    def test_get_persona_found(self) -> None:
        """Test getting existing persona."""
        persona = get_persona("adversarial")
        assert persona is not None
        assert persona.id == "adversarial"

    def test_get_persona_not_found(self) -> None:
        """Test getting non-existent persona."""
        persona = get_persona("nonexistent")
        assert persona is None

    def test_list_personas(self) -> None:
        """Test listing all personas."""
        personas = list_personas()
        assert "adversarial" in personas
        assert "skeptical" in personas
        assert "insightful" in personas
        assert "minimalist" in personas
        assert "downstream" in personas
        assert len(personas) >= 5
