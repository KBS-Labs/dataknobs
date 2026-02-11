"""Tests for config/validation.py."""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_bots.config.schema import DynaBotConfigSchema
from dataknobs_bots.config.validation import ConfigValidator, ValidationResult


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_ok(self) -> None:
        result = ValidationResult.ok()
        assert result.valid is True
        assert result.errors == []
        assert result.warnings == []

    def test_error(self) -> None:
        result = ValidationResult.error("something broke")
        assert result.valid is False
        assert result.errors == ["something broke"]
        assert result.warnings == []

    def test_warning(self) -> None:
        result = ValidationResult.warning("heads up")
        assert result.valid is True
        assert result.errors == []
        assert result.warnings == ["heads up"]

    def test_merge_both_valid(self) -> None:
        a = ValidationResult.ok()
        b = ValidationResult.warning("note")
        merged = a.merge(b)
        assert merged.valid is True
        assert merged.warnings == ["note"]

    def test_merge_one_invalid(self) -> None:
        a = ValidationResult.error("bad")
        b = ValidationResult.ok()
        merged = a.merge(b)
        assert merged.valid is False
        assert merged.errors == ["bad"]

    def test_merge_both_invalid(self) -> None:
        a = ValidationResult.error("err1")
        b = ValidationResult.error("err2")
        merged = a.merge(b)
        assert merged.valid is False
        assert merged.errors == ["err1", "err2"]

    def test_merge_accumulates(self) -> None:
        a = ValidationResult(valid=True, warnings=["w1"])
        b = ValidationResult(valid=False, errors=["e1"], warnings=["w2"])
        merged = a.merge(b)
        assert merged.valid is False
        assert merged.errors == ["e1"]
        assert merged.warnings == ["w1", "w2"]

    def test_to_dict(self) -> None:
        result = ValidationResult(
            valid=False, errors=["err"], warnings=["warn"]
        )
        d = result.to_dict()
        assert d == {"valid": False, "errors": ["err"], "warnings": ["warn"]}


class TestConfigValidator:
    """Tests for ConfigValidator."""

    def test_validate_completeness_valid(self) -> None:
        validator = ConfigValidator()
        config: dict[str, Any] = {
            "llm": {"provider": "ollama"},
            "conversation_storage": {"backend": "memory"},
        }
        result = validator.validate_completeness(config)
        assert result.valid is True

    def test_validate_completeness_missing_llm(self) -> None:
        validator = ConfigValidator()
        config: dict[str, Any] = {
            "conversation_storage": {"backend": "memory"},
        }
        result = validator.validate_completeness(config)
        assert result.valid is False
        assert any("llm" in e for e in result.errors)

    def test_validate_completeness_missing_storage(self) -> None:
        validator = ConfigValidator()
        config: dict[str, Any] = {
            "llm": {"provider": "ollama"},
        }
        result = validator.validate_completeness(config)
        assert result.valid is False
        assert any("conversation_storage" in e for e in result.errors)

    def test_validate_completeness_missing_both(self) -> None:
        validator = ConfigValidator()
        result = validator.validate_completeness({})
        assert result.valid is False
        assert len(result.errors) == 2

    def test_validate_completeness_portable_format(self) -> None:
        validator = ConfigValidator()
        config: dict[str, Any] = {
            "bot": {
                "llm": {"$resource": "default"},
                "conversation_storage": {"$resource": "db"},
            },
        }
        result = validator.validate_completeness(config)
        assert result.valid is True

    def test_validate_portability_clean(self) -> None:
        validator = ConfigValidator()
        config: dict[str, Any] = {
            "llm": {"provider": "ollama", "model": "llama3.2"},
        }
        result = validator.validate_portability(config)
        assert result.valid is True

    def test_validate_portability_with_local_path(self) -> None:
        validator = ConfigValidator()
        config: dict[str, Any] = {
            "storage": {"path": "/Users/dev/data"},
        }
        result = validator.validate_portability(config)
        assert result.valid is True  # warnings, not errors
        assert len(result.warnings) > 0

    def test_register_custom_validator(self) -> None:
        validator = ConfigValidator()

        def check_name(config: dict[str, Any]) -> ValidationResult:
            if "name" not in config:
                return ValidationResult.warning("Config has no name")
            return ValidationResult.ok()

        validator.register_validator("name_check", check_name)
        config: dict[str, Any] = {
            "llm": {"provider": "ollama"},
            "conversation_storage": {"backend": "memory"},
        }
        result = validator.validate(config)
        assert result.valid is True
        assert any("name" in w for w in result.warnings)

    def test_validate_with_schema(self) -> None:
        schema = DynaBotConfigSchema()
        validator = ConfigValidator(schema=schema)
        config: dict[str, Any] = {
            "llm": {"provider": "invalid_provider"},
            "conversation_storage": {"backend": "memory"},
        }
        result = validator.validate(config)
        assert result.valid is False
        assert any("invalid_provider" in e for e in result.errors)

    def test_validate_component(self) -> None:
        schema = DynaBotConfigSchema()
        validator = ConfigValidator(schema=schema)
        result = validator.validate_component(
            "llm", {"provider": "ollama", "model": "llama3.2"}
        )
        assert result.valid is True

    def test_validate_component_invalid(self) -> None:
        schema = DynaBotConfigSchema()
        validator = ConfigValidator(schema=schema)
        result = validator.validate_component(
            "llm", {"provider": "not_a_provider"}
        )
        assert result.valid is False

    def test_validator_exception_handling(self) -> None:
        validator = ConfigValidator()

        def bad_validator(config: dict[str, Any]) -> ValidationResult:
            raise RuntimeError("boom")

        validator.register_validator("bad", bad_validator)
        config: dict[str, Any] = {
            "llm": {"provider": "ollama"},
            "conversation_storage": {"backend": "memory"},
        }
        result = validator.validate(config)
        assert result.valid is False
        assert any("bad" in e for e in result.errors)
