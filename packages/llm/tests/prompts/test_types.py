"""Unit tests for core types and validation."""

import pytest
from dataknobs_llm.prompts import (
    ValidationLevel,
    ValidationConfig,
    RenderResult,
)


class TestValidationLevel:
    """Test suite for ValidationLevel enum."""

    def test_validation_levels_exist(self):
        """Test that all validation levels are defined."""
        assert ValidationLevel.ERROR
        assert ValidationLevel.WARN
        assert ValidationLevel.IGNORE

    def test_validation_level_values(self):
        """Test validation level string values."""
        assert ValidationLevel.ERROR.value == "error"
        assert ValidationLevel.WARN.value == "warn"
        assert ValidationLevel.IGNORE.value == "ignore"


class TestValidationConfig:
    """Test suite for ValidationConfig."""

    def test_default_initialization(self):
        """Test ValidationConfig with defaults."""
        config = ValidationConfig()
        assert config.level is None  # None means inherit from context
        assert config.required_params == set()
        assert config.optional_params == set()

    def test_initialization_with_level(self):
        """Test ValidationConfig with custom level."""
        config = ValidationConfig(level=ValidationLevel.ERROR)
        assert config.level == ValidationLevel.ERROR

    def test_initialization_with_params(self):
        """Test ValidationConfig with parameter lists."""
        config = ValidationConfig(
            level=ValidationLevel.ERROR,
            required_params=["name", "age"],
            optional_params=["city", "country"]
        )
        assert config.required_params == {"name", "age"}
        assert config.optional_params == {"city", "country"}

    def test_params_converted_to_sets(self):
        """Test that parameter lists are converted to sets."""
        config = ValidationConfig(
            required_params=["name", "name", "age"]  # Duplicate "name"
        )
        assert config.required_params == {"name", "age"}
        assert len(config.required_params) == 2


class TestRenderResult:
    """Test suite for RenderResult dataclass."""

    def test_default_initialization(self):
        """Test RenderResult with minimal data."""
        result = RenderResult(content="Hello World")
        assert result.content == "Hello World"
        assert result.params_used == {}
        assert result.params_missing == []
        assert result.validation_warnings == []
        assert result.metadata == {}

    def test_full_initialization(self):
        """Test RenderResult with all fields."""
        result = RenderResult(
            content="Hello Alice",
            params_used={"name": "Alice"},
            params_missing=["age"],
            validation_warnings=["Missing parameter: age"],
            metadata={"template_name": "greeting"}
        )
        assert result.content == "Hello Alice"
        assert result.params_used == {"name": "Alice"}
        assert result.params_missing == ["age"]
        assert len(result.validation_warnings) == 1
        assert result.metadata["template_name"] == "greeting"

    def test_render_result_is_mutable(self):
        """Test that RenderResult fields can be modified."""
        result = RenderResult(content="Hello")
        result.params_used["name"] = "Alice"
        result.params_missing.append("age")
        result.validation_warnings.append("warning")
        result.metadata["key"] = "value"

        assert result.params_used == {"name": "Alice"}
        assert result.params_missing == ["age"]
        assert result.validation_warnings == ["warning"]
        assert result.metadata == {"key": "value"}
