"""Tests for stripping schema defaults from extraction.

Tests for the _strip_schema_defaults method in WizardReasoning,
ensuring defaults are removed before passing schemas to the extraction LLM.
"""

from dataclasses import dataclass, field
from typing import Any

import pytest

from dataknobs_bots.reasoning.wizard import WizardReasoning
from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader


@pytest.fixture
def minimal_wizard_config() -> dict[str, Any]:
    """Create minimal wizard config for testing."""
    return {
        "name": "test-wizard",
        "stages": [
            {
                "name": "start",
                "is_start": True,
                "is_end": True,
                "prompt": "Test",
            }
        ],
    }


@pytest.fixture
def wizard_reasoning(minimal_wizard_config: dict[str, Any]) -> WizardReasoning:
    """Create WizardReasoning instance for testing."""
    loader = WizardConfigLoader()
    wizard_fsm = loader.load_from_dict(minimal_wizard_config)
    return WizardReasoning(wizard_fsm=wizard_fsm, strict_validation=False)


class TestStripSchemaDefaults:
    """Tests for _strip_schema_defaults method."""

    def test_strip_simple_defaults(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Test stripping defaults from simple properties."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "provider": {"type": "string", "default": "anthropic"},
                "model": {"type": "string"},
            },
        }

        result = wizard_reasoning._strip_schema_defaults(schema)

        assert "default" not in result["properties"]["provider"]
        assert result["properties"]["model"]["type"] == "string"

    def test_strip_nested_defaults(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Test stripping defaults from nested object properties."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "config": {
                    "type": "object",
                    "properties": {
                        "temperature": {"type": "number", "default": 0.7}
                    },
                }
            },
        }

        result = wizard_reasoning._strip_schema_defaults(schema)

        nested_temp = result["properties"]["config"]["properties"]["temperature"]
        assert "default" not in nested_temp

    def test_strip_array_item_defaults(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Test stripping defaults from array item schemas."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "enabled": {"type": "boolean", "default": True}
                        },
                    },
                }
            },
        }

        result = wizard_reasoning._strip_schema_defaults(schema)

        item_enabled = result["properties"]["items"]["items"]["properties"]["enabled"]
        assert "default" not in item_enabled

    def test_original_schema_unchanged(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Test that original schema is not modified."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {"value": {"type": "string", "default": "test"}},
        }

        wizard_reasoning._strip_schema_defaults(schema)

        # Original should be unchanged
        assert schema["properties"]["value"]["default"] == "test"

    def test_strip_allof_defaults(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Test stripping defaults from allOf schemas."""
        schema: dict[str, Any] = {
            "allOf": [
                {"properties": {"field": {"type": "string", "default": "x"}}}
            ]
        }

        result = wizard_reasoning._strip_schema_defaults(schema)

        assert "default" not in result["allOf"][0]["properties"]["field"]

    def test_strip_anyof_defaults(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Test stripping defaults from anyOf schemas."""
        schema: dict[str, Any] = {
            "anyOf": [
                {"properties": {"a": {"type": "string", "default": "x"}}},
                {"properties": {"b": {"type": "number", "default": 5}}},
            ]
        }

        result = wizard_reasoning._strip_schema_defaults(schema)

        assert "default" not in result["anyOf"][0]["properties"]["a"]
        assert "default" not in result["anyOf"][1]["properties"]["b"]

    def test_strip_oneof_defaults(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Test stripping defaults from oneOf schemas."""
        schema: dict[str, Any] = {
            "oneOf": [
                {"properties": {"x": {"type": "string", "default": "hello"}}},
            ]
        }

        result = wizard_reasoning._strip_schema_defaults(schema)

        assert "default" not in result["oneOf"][0]["properties"]["x"]

    def test_preserves_other_schema_properties(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Test that non-default properties are preserved."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "default": "test",
                    "description": "User's name",
                    "minLength": 1,
                    "maxLength": 100,
                }
            },
            "required": ["name"],
        }

        result = wizard_reasoning._strip_schema_defaults(schema)

        prop = result["properties"]["name"]
        assert "default" not in prop
        assert prop["description"] == "User's name"
        assert prop["minLength"] == 1
        assert prop["maxLength"] == 100
        assert result["required"] == ["name"]

    def test_schema_without_defaults(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Test schema with no defaults returns equivalent copy."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
        }

        result = wizard_reasoning._strip_schema_defaults(schema)

        assert result["properties"]["name"]["type"] == "string"
        assert result["properties"]["age"]["type"] == "integer"
        # Should be a copy, not the same object
        assert result is not schema

    def test_empty_properties(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Test schema with empty properties dict."""
        schema: dict[str, Any] = {"type": "object", "properties": {}}

        result = wizard_reasoning._strip_schema_defaults(schema)

        assert result["properties"] == {}

    def test_deeply_nested_defaults(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Test stripping defaults from deeply nested structures."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "level1": {
                    "type": "object",
                    "properties": {
                        "level2": {
                            "type": "object",
                            "properties": {
                                "level3": {
                                    "type": "string",
                                    "default": "deep_value",
                                }
                            },
                        }
                    },
                }
            },
        }

        result = wizard_reasoning._strip_schema_defaults(schema)

        level3 = result["properties"]["level1"]["properties"]["level2"][
            "properties"
        ]["level3"]
        assert "default" not in level3
        assert level3["type"] == "string"

    def test_multiple_defaults_same_level(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Test stripping multiple defaults at the same nesting level."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "a": {"type": "string", "default": "val_a"},
                "b": {"type": "number", "default": 42},
                "c": {"type": "boolean", "default": True},
            },
        }

        result = wizard_reasoning._strip_schema_defaults(schema)

        assert "default" not in result["properties"]["a"]
        assert "default" not in result["properties"]["b"]
        assert "default" not in result["properties"]["c"]


@dataclass
class SimpleExtractionResult:
    """Simple extraction result for testing."""

    data: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def is_confident(self) -> bool:
        return self.confidence >= 0.8 and not self.errors


class RecordingExtractor:
    """Extractor that records the schema it receives.

    This is not a mock - it's a real extractor implementation that
    captures inputs for verification. It returns a valid extraction
    result structure.
    """

    def __init__(self) -> None:
        self.received_schema: dict[str, Any] | None = None
        self.received_text: str | None = None
        self.received_context: dict[str, Any] | None = None

    async def extract(
        self,
        text: str,
        schema: dict[str, Any],
        context: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> SimpleExtractionResult:
        """Record inputs and return empty extraction."""
        self.received_schema = schema
        self.received_text = text
        self.received_context = context
        # Return valid result with no extracted data
        return SimpleExtractionResult(data={}, confidence=0.5)


class TestSchemaDefaultsIntegration:
    """Integration tests for schema defaults stripping."""

    @pytest.mark.asyncio
    async def test_extractor_receives_schema_without_defaults(self) -> None:
        """Extractor should receive schema with defaults stripped."""
        extractor = RecordingExtractor()

        wizard_config: dict[str, Any] = {
            "name": "test-wizard",
            "stages": [
                {
                    "name": "configure_llm",
                    "is_start": True,
                    "is_end": True,
                    "prompt": "Choose your LLM",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "provider": {"type": "string", "default": "anthropic"},
                            "model": {"type": "string"},
                        },
                    },
                }
            ],
        }

        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(wizard_config)
        reasoning = WizardReasoning(
            wizard_fsm=wizard_fsm, extractor=extractor, strict_validation=False
        )

        stage = wizard_fsm.current_metadata
        await reasoning._extract_data("some message", stage, llm=None)

        # Verify extractor received schema without default
        assert extractor.received_schema is not None
        assert "default" not in extractor.received_schema["properties"]["provider"]
        assert extractor.received_schema["properties"]["model"]["type"] == "string"

    @pytest.mark.asyncio
    async def test_extraction_result_has_no_autofilled_defaults(self) -> None:
        """Verify that extraction doesn't produce values from schema defaults."""
        extractor = RecordingExtractor()

        wizard_config: dict[str, Any] = {
            "name": "test-wizard",
            "stages": [
                {
                    "name": "llm_config",
                    "is_start": True,
                    "is_end": True,
                    "prompt": "Configure LLM",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "provider": {"type": "string", "default": "anthropic"},
                            "model": {"type": "string", "default": "claude-3-sonnet"},
                        },
                    },
                }
            ],
        }

        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(wizard_config)
        reasoning = WizardReasoning(
            wizard_fsm=wizard_fsm, extractor=extractor, strict_validation=False
        )

        stage = wizard_fsm.current_metadata
        result = await reasoning._extract_data(
            "I want to build a math tutor",  # No LLM info mentioned
            stage,
            llm=None,
        )

        # RecordingExtractor returns empty data
        # In real usage, even a real extractor shouldn't fill defaults
        assert "provider" not in result.data
        assert "model" not in result.data

    @pytest.mark.asyncio
    async def test_no_extractor_still_works(self) -> None:
        """When no extractor is configured, extraction still works."""
        wizard_config: dict[str, Any] = {
            "name": "test-wizard",
            "stages": [
                {
                    "name": "configure",
                    "is_start": True,
                    "is_end": True,
                    "prompt": "Configure",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "value": {"type": "string", "default": "test"},
                        },
                    },
                }
            ],
        }

        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(wizard_config)
        # No extractor provided
        reasoning = WizardReasoning(wizard_fsm=wizard_fsm, strict_validation=False)

        stage = wizard_fsm.current_metadata
        result = await reasoning._extract_data("user message", stage, llm=None)

        # Should return fallback result with raw input
        assert result.data == {"_raw_input": "user message"}
        assert result.confidence == 0.5

    @pytest.mark.asyncio
    async def test_no_schema_stage(self) -> None:
        """Stage without schema should pass through raw input."""
        wizard_config: dict[str, Any] = {
            "name": "test-wizard",
            "stages": [
                {
                    "name": "welcome",
                    "is_start": True,
                    "is_end": True,
                    "prompt": "Welcome!",
                    # No schema
                }
            ],
        }

        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(wizard_config)
        reasoning = WizardReasoning(wizard_fsm=wizard_fsm, strict_validation=False)

        stage = wizard_fsm.current_metadata
        result = await reasoning._extract_data("hello", stage, llm=None)

        assert result.data == {"_raw_input": "hello"}
        assert result.confidence == 1.0
