"""Tests for wizard schema-aware data normalization."""

from typing import Any

import pytest

from dataknobs_bots.reasoning.wizard import StageSchema, WizardReasoning


class TestNormalizeExtractedData:
    """Tests for _normalize_extracted_data."""

    # --- Boolean coercion ---

    @pytest.mark.parametrize(
        "input_val",
        ["yes", "true", "1", "y", "on", "enable", "enabled", "YES", "True", " yes "],
    )
    def test_boolean_coercion_true(
        self, wizard_reasoning: WizardReasoning, input_val: str
    ) -> None:
        """String truthy values are coerced to True."""
        schema: dict[str, Any] = {
            "properties": {"flag": {"type": "boolean"}},
        }
        result = wizard_reasoning._extraction._normalize_extracted_data(
            {"flag": input_val}, StageSchema.from_dict(schema)
        )
        assert result["flag"] is True

    @pytest.mark.parametrize(
        "input_val",
        ["no", "false", "0", "n", "off", "disable", "disabled", "NO", "False", " no "],
    )
    def test_boolean_coercion_false(
        self, wizard_reasoning: WizardReasoning, input_val: str
    ) -> None:
        """String falsy values are coerced to False."""
        schema: dict[str, Any] = {
            "properties": {"flag": {"type": "boolean"}},
        }
        result = wizard_reasoning._extraction._normalize_extracted_data(
            {"flag": input_val}, StageSchema.from_dict(schema)
        )
        assert result["flag"] is False

    def test_boolean_already_bool_unchanged(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Actual bools pass through unchanged."""
        schema: dict[str, Any] = {
            "properties": {"flag": {"type": "boolean"}},
        }
        for val in (True, False):
            result = wizard_reasoning._extraction._normalize_extracted_data(
                {"flag": val}, StageSchema.from_dict(schema)
            )
            assert result["flag"] is val

    # --- Array handling ---

    def test_array_wrapping_bare_string(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Bare string for array field is wrapped in list."""
        schema: dict[str, Any] = {
            "properties": {"tags": {"type": "array", "items": {"type": "string"}}},
        }
        result = wizard_reasoning._extraction._normalize_extracted_data(
            {"tags": "python"}, StageSchema.from_dict(schema)
        )
        assert result["tags"] == ["python"]

    def test_array_all_shortcut(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """["all"] expands to full enum list."""
        schema: dict[str, Any] = {
            "properties": {
                "tools": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["hammer", "saw", "drill"]},
                }
            },
        }
        result = wizard_reasoning._extraction._normalize_extracted_data(
            {"tools": ["all"]}, StageSchema.from_dict(schema)
        )
        assert result["tools"] == ["hammer", "saw", "drill"]

    def test_array_none_shortcut(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """["none"] normalizes to []."""
        schema: dict[str, Any] = {
            "properties": {
                "tools": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["hammer", "saw"]},
                }
            },
        }
        result = wizard_reasoning._extraction._normalize_extracted_data(
            {"tools": ["none"]}, StageSchema.from_dict(schema)
        )
        assert result["tools"] == []

    # --- Number coercion ---

    def test_integer_coercion(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """String "42" is coerced to int 42 for integer fields."""
        schema: dict[str, Any] = {
            "properties": {"count": {"type": "integer"}},
        }
        result = wizard_reasoning._extraction._normalize_extracted_data(
            {"count": "42"}, StageSchema.from_dict(schema)
        )
        assert result["count"] == 42
        assert isinstance(result["count"], int)

    def test_integer_negative(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Negative integer string is coerced correctly."""
        schema: dict[str, Any] = {
            "properties": {"offset": {"type": "integer"}},
        }
        result = wizard_reasoning._extraction._normalize_extracted_data(
            {"offset": "-5"}, StageSchema.from_dict(schema)
        )
        assert result["offset"] == -5

    def test_number_coercion(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """String "3.14" is coerced to float for number fields."""
        schema: dict[str, Any] = {
            "properties": {"price": {"type": "number"}},
        }
        result = wizard_reasoning._extraction._normalize_extracted_data(
            {"price": "3.14"}, StageSchema.from_dict(schema)
        )
        assert result["price"] == pytest.approx(3.14)
        assert isinstance(result["price"], float)

    def test_number_invalid_rejected(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Non-numeric string for number field is rejected (type mismatch)."""
        schema: dict[str, Any] = {
            "properties": {"price": {"type": "number"}},
        }
        result = wizard_reasoning._extraction._normalize_extracted_data(
            {"price": "not-a-number"}, StageSchema.from_dict(schema)
        )
        assert result["price"] is None

    # --- Type mismatch rejection ---

    def test_bool_for_integer_rejected(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Bool True for integer field is rejected (bool is subclass of int)."""
        schema: dict[str, Any] = {
            "properties": {"count": {"type": "integer"}},
        }
        result = wizard_reasoning._extraction._normalize_extracted_data(
            {"count": True}, StageSchema.from_dict(schema)
        )
        assert result["count"] is None

    def test_bool_for_number_rejected(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Bool False for number field is rejected."""
        schema: dict[str, Any] = {
            "properties": {"price": {"type": "number"}},
        }
        result = wizard_reasoning._extraction._normalize_extracted_data(
            {"price": False}, StageSchema.from_dict(schema)
        )
        assert result["price"] is None

    def test_bool_for_string_rejected(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Bool for string field is rejected, not coerced."""
        schema: dict[str, Any] = {
            "properties": {"tone": {"type": "string"}},
        }
        result = wizard_reasoning._extraction._normalize_extracted_data(
            {"tone": True}, StageSchema.from_dict(schema)
        )
        assert result["tone"] is None

    def test_int_for_string_rejected(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Int for string field is rejected, not coerced."""
        schema: dict[str, Any] = {
            "properties": {"name": {"type": "string"}},
        }
        result = wizard_reasoning._extraction._normalize_extracted_data(
            {"name": 42}, StageSchema.from_dict(schema)
        )
        assert result["name"] is None

    def test_list_for_string_rejected(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """List for string field is rejected."""
        schema: dict[str, Any] = {
            "properties": {"name": {"type": "string"}},
        }
        result = wizard_reasoning._extraction._normalize_extracted_data(
            {"name": ["a", "b"]}, StageSchema.from_dict(schema)
        )
        assert result["name"] is None

    # --- Coercion + type-mismatch interaction ---

    def test_coercion_then_type_check_passes(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """String 'yes' for boolean field is coerced to True before type check."""
        schema: dict[str, Any] = {
            "properties": {"enabled": {"type": "boolean"}},
        }
        result = wizard_reasoning._extraction._normalize_extracted_data(
            {"enabled": "yes"}, StageSchema.from_dict(schema)
        )
        assert result["enabled"] is True

    def test_failed_coercion_then_type_check_rejects(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """String 'abc' for integer field — coercion fails, type check rejects."""
        schema: dict[str, Any] = {
            "properties": {"count": {"type": "integer"}},
        }
        result = wizard_reasoning._extraction._normalize_extracted_data(
            {"count": "abc"}, StageSchema.from_dict(schema)
        )
        assert result["count"] is None

    # --- Skip behavior ---

    def test_internal_keys_skipped(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Keys starting with _ are not normalized."""
        schema: dict[str, Any] = {
            "properties": {"_internal": {"type": "boolean"}},
        }
        result = wizard_reasoning._extraction._normalize_extracted_data(
            {"_internal": "yes"}, StageSchema.from_dict(schema)
        )
        # Should remain as string, not coerced to True
        assert result["_internal"] == "yes"

    def test_unknown_fields_skipped(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Fields not in schema properties are left unchanged."""
        schema: dict[str, Any] = {
            "properties": {"known": {"type": "boolean"}},
        }
        result = wizard_reasoning._extraction._normalize_extracted_data(
            {"known": "yes", "unknown": "yes"}, StageSchema.from_dict(schema)
        )
        assert result["known"] is True
        assert result["unknown"] == "yes"

    def test_schema_without_properties(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Schema without 'properties' returns data unchanged."""
        schema: dict[str, Any] = {"type": "object"}
        data = {"flag": "yes", "count": "42"}
        result = wizard_reasoning._extraction._normalize_extracted_data(data, StageSchema.from_dict(schema))
        assert result == data
