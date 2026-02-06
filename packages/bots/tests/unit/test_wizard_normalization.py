"""Tests for wizard schema-aware data normalization."""

from typing import Any

import pytest

from dataknobs_bots.reasoning.wizard import WizardReasoning


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
        result = wizard_reasoning._normalize_extracted_data(
            {"flag": input_val}, schema
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
        result = wizard_reasoning._normalize_extracted_data(
            {"flag": input_val}, schema
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
            result = wizard_reasoning._normalize_extracted_data(
                {"flag": val}, schema
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
        result = wizard_reasoning._normalize_extracted_data(
            {"tags": "python"}, schema
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
        result = wizard_reasoning._normalize_extracted_data(
            {"tools": ["all"]}, schema
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
        result = wizard_reasoning._normalize_extracted_data(
            {"tools": ["none"]}, schema
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
        result = wizard_reasoning._normalize_extracted_data(
            {"count": "42"}, schema
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
        result = wizard_reasoning._normalize_extracted_data(
            {"offset": "-5"}, schema
        )
        assert result["offset"] == -5

    def test_number_coercion(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """String "3.14" is coerced to float for number fields."""
        schema: dict[str, Any] = {
            "properties": {"price": {"type": "number"}},
        }
        result = wizard_reasoning._normalize_extracted_data(
            {"price": "3.14"}, schema
        )
        assert result["price"] == pytest.approx(3.14)
        assert isinstance(result["price"], float)

    def test_number_invalid_leaves_unchanged(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Non-numeric string for number field is left as-is."""
        schema: dict[str, Any] = {
            "properties": {"price": {"type": "number"}},
        }
        result = wizard_reasoning._normalize_extracted_data(
            {"price": "not-a-number"}, schema
        )
        assert result["price"] == "not-a-number"

    # --- Skip behavior ---

    def test_internal_keys_skipped(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Keys starting with _ are not normalized."""
        schema: dict[str, Any] = {
            "properties": {"_internal": {"type": "boolean"}},
        }
        result = wizard_reasoning._normalize_extracted_data(
            {"_internal": "yes"}, schema
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
        result = wizard_reasoning._normalize_extracted_data(
            {"known": "yes", "unknown": "yes"}, schema
        )
        assert result["known"] is True
        assert result["unknown"] == "yes"

    def test_schema_without_properties(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Schema without 'properties' returns data unchanged."""
        schema: dict[str, Any] = {"type": "object"}
        data = {"flag": "yes", "count": "42"}
        result = wizard_reasoning._normalize_extracted_data(data, schema)
        assert result == data
