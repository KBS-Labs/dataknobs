"""Tests for intent classification and verbatim capture in wizard collection mode.

Covers Phase 3 of 03b:
- _needs_llm_extraction: schema-based auto-detection and capture_mode config
- _classify_collection_intent: rule-based help detection
- Verbatim capture path in _extract_data: single string field → no LLM call
"""

from __future__ import annotations

from typing import Any

import pytest
import pytest_asyncio

from dataknobs_bots.reasoning.wizard import WizardReasoning, WizardState
from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader


# =====================================================================
# Helpers
# =====================================================================

def _make_wizard(
    schema: dict[str, Any] | None = None,
    collection_config: dict[str, Any] | None = None,
) -> WizardReasoning:
    """Create a minimal WizardReasoning for unit tests."""
    stage: dict[str, Any] = {
        "name": "collect",
        "is_start": True,
        "is_end": True,
        "prompt": "Provide data",
    }
    if schema:
        stage["schema"] = schema
    if collection_config:
        stage["collection_mode"] = "collection"
        stage["collection_config"] = collection_config

    config: dict[str, Any] = {
        "name": "test-wizard",
        "version": "1.0",
        "stages": [stage],
    }
    loader = WizardConfigLoader()
    fsm = loader.load_from_dict(config)
    return WizardReasoning(wizard_fsm=fsm, strict_validation=False)


# =====================================================================
# _needs_llm_extraction tests
# =====================================================================

class TestNeedsLlmExtraction:
    """Tests for _needs_llm_extraction auto-detection and config override."""

    def test_single_string_field_auto_detect(self) -> None:
        """Single required string field → verbatim (no LLM)."""
        wizard = _make_wizard()
        schema = {
            "type": "object",
            "properties": {"instruction": {"type": "string"}},
            "required": ["instruction"],
        }
        stage: dict[str, Any] = {"name": "test"}
        assert wizard._needs_llm_extraction(schema, stage) is False

    def test_multi_field_requires_llm(self) -> None:
        """Multiple fields → LLM extraction needed."""
        wizard = _make_wizard()
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "amount": {"type": "string"},
            },
            "required": ["name", "amount"],
        }
        stage: dict[str, Any] = {"name": "test"}
        assert wizard._needs_llm_extraction(schema, stage) is True

    def test_single_field_with_enum_requires_llm(self) -> None:
        """Single string field with enum constraint → LLM extraction."""
        wizard = _make_wizard()
        schema = {
            "type": "object",
            "properties": {
                "difficulty": {
                    "type": "string",
                    "enum": ["easy", "medium", "hard"],
                },
            },
            "required": ["difficulty"],
        }
        stage: dict[str, Any] = {"name": "test"}
        assert wizard._needs_llm_extraction(schema, stage) is True

    def test_single_field_with_pattern_requires_llm(self) -> None:
        """Single string field with pattern constraint → LLM extraction."""
        wizard = _make_wizard()
        schema = {
            "type": "object",
            "properties": {
                "email": {"type": "string", "pattern": r"^\S+@\S+$"},
            },
            "required": ["email"],
        }
        stage: dict[str, Any] = {"name": "test"}
        assert wizard._needs_llm_extraction(schema, stage) is True

    def test_single_field_with_format_requires_llm(self) -> None:
        """Single string field with format constraint → LLM extraction."""
        wizard = _make_wizard()
        schema = {
            "type": "object",
            "properties": {
                "date": {"type": "string", "format": "date"},
            },
            "required": ["date"],
        }
        stage: dict[str, Any] = {"name": "test"}
        assert wizard._needs_llm_extraction(schema, stage) is True

    def test_single_integer_field_requires_llm(self) -> None:
        """Single non-string field → LLM extraction."""
        wizard = _make_wizard()
        schema = {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
        }
        stage: dict[str, Any] = {"name": "test"}
        assert wizard._needs_llm_extraction(schema, stage) is True

    def test_capture_mode_verbatim_overrides_auto(self) -> None:
        """capture_mode='verbatim' forces verbatim even for complex schemas."""
        wizard = _make_wizard()
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "amount": {"type": "string"},
            },
            "required": ["name", "amount"],
        }
        stage: dict[str, Any] = {
            "name": "test",
            "collection_config": {"capture_mode": "verbatim"},
        }
        assert wizard._needs_llm_extraction(schema, stage) is False

    def test_capture_mode_extract_overrides_auto(self) -> None:
        """capture_mode='extract' forces LLM even for trivial schemas."""
        wizard = _make_wizard()
        schema = {
            "type": "object",
            "properties": {"instruction": {"type": "string"}},
            "required": ["instruction"],
        }
        stage: dict[str, Any] = {
            "name": "test",
            "collection_config": {"capture_mode": "extract"},
        }
        assert wizard._needs_llm_extraction(schema, stage) is True

    def test_capture_mode_auto_is_default(self) -> None:
        """Without capture_mode config, auto-detection is used."""
        wizard = _make_wizard()
        schema = {
            "type": "object",
            "properties": {"instruction": {"type": "string"}},
            "required": ["instruction"],
        }
        # No collection_config at all
        stage: dict[str, Any] = {"name": "test"}
        assert wizard._needs_llm_extraction(schema, stage) is False

    def test_optional_field_only_requires_llm(self) -> None:
        """Single property with no required fields → LLM extraction."""
        wizard = _make_wizard()
        schema = {
            "type": "object",
            "properties": {"note": {"type": "string"}},
            "required": [],
        }
        stage: dict[str, Any] = {"name": "test"}
        assert wizard._needs_llm_extraction(schema, stage) is True

    def test_extra_optional_property_requires_llm(self) -> None:
        """One required string + one optional → LLM extraction."""
        wizard = _make_wizard()
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "note": {"type": "string"},
            },
            "required": ["name"],
        }
        stage: dict[str, Any] = {"name": "test"}
        # 2 properties, even though only 1 required → needs LLM
        assert wizard._needs_llm_extraction(schema, stage) is True


# =====================================================================
# _classify_collection_intent tests
# =====================================================================

class TestClassifyCollectionIntent:
    """Tests for _classify_collection_intent rule-based classification."""

    def test_data_input_default(self) -> None:
        """Regular data input returns 'data_input'."""
        stage: dict[str, Any] = {"name": "collect"}
        assert WizardReasoning._classify_collection_intent(
            "2 cups flour", stage
        ) == "data_input"

    def test_question_mark_is_help(self) -> None:
        """Messages ending with ? are classified as help."""
        stage: dict[str, Any] = {"name": "collect"}
        assert WizardReasoning._classify_collection_intent(
            "What goes here?", stage
        ) == "help"

    def test_help_keyword(self) -> None:
        """'help' alone is classified as help."""
        stage: dict[str, Any] = {"name": "collect"}
        assert WizardReasoning._classify_collection_intent(
            "help", stage
        ) == "help"

    def test_what_should_i(self) -> None:
        """'what should i...' is classified as help."""
        stage: dict[str, Any] = {"name": "collect"}
        assert WizardReasoning._classify_collection_intent(
            "what should I put here", stage
        ) == "help"

    def test_i_dont_understand(self) -> None:
        """'i don't understand' is classified as help."""
        stage: dict[str, Any] = {"name": "collect"}
        assert WizardReasoning._classify_collection_intent(
            "I don't understand what you need", stage
        ) == "help"

    def test_explain(self) -> None:
        """'explain...' is classified as help."""
        stage: dict[str, Any] = {"name": "collect"}
        assert WizardReasoning._classify_collection_intent(
            "explain what this step is about", stage
        ) == "help"

    def test_what_format(self) -> None:
        """'what format...' is classified as help."""
        stage: dict[str, Any] = {"name": "collect"}
        assert WizardReasoning._classify_collection_intent(
            "what format should I use", stage
        ) == "help"

    def test_custom_help_keywords(self) -> None:
        """Stage-configured help_keywords are respected."""
        stage: dict[str, Any] = {
            "name": "collect",
            "collection_config": {
                "help_keywords": ["info", "details"],
            },
        }
        assert WizardReasoning._classify_collection_intent(
            "info", stage
        ) == "help"
        assert WizardReasoning._classify_collection_intent(
            "details", stage
        ) == "help"

    def test_custom_help_keywords_exact_match(self) -> None:
        """Custom help keywords require exact match."""
        stage: dict[str, Any] = {
            "name": "collect",
            "collection_config": {
                "help_keywords": ["info"],
            },
        }
        # "information" should NOT match "info" (exact match)
        assert WizardReasoning._classify_collection_intent(
            "information about cooking", stage
        ) == "data_input"

    def test_data_with_question_word_no_question_mark(self) -> None:
        """Data starting with 'what' but no ? is data_input."""
        stage: dict[str, Any] = {"name": "collect"}
        # "what should i" starts a help phrase, so this IS help
        result = WizardReasoning._classify_collection_intent(
            "what should I", stage
        )
        assert result == "help"

    def test_regular_sentence_not_help(self) -> None:
        """Normal sentences are data_input."""
        stage: dict[str, Any] = {"name": "collect"}
        assert WizardReasoning._classify_collection_intent(
            "Mix the flour and sugar together", stage
        ) == "data_input"

    def test_case_insensitive(self) -> None:
        """Help detection is case-insensitive."""
        stage: dict[str, Any] = {"name": "collect"}
        assert WizardReasoning._classify_collection_intent(
            "HELP", stage
        ) == "help"
        assert WizardReasoning._classify_collection_intent(
            "What Should I Do?", stage
        ) == "help"


# =====================================================================
# Verbatim capture in _extract_data
# =====================================================================

class TestVerbatimCaptureInExtraction:
    """Tests for verbatim capture path in _extract_data."""

    @pytest.mark.asyncio
    async def test_trivial_schema_verbatim_capture(self) -> None:
        """Single string field → verbatim capture, no LLM call."""
        wizard = _make_wizard(
            schema={
                "type": "object",
                "properties": {"instruction": {"type": "string"}},
                "required": ["instruction"],
            },
        )
        stage = {
            "name": "collect",
            "schema": {
                "type": "object",
                "properties": {"instruction": {"type": "string"}},
                "required": ["instruction"],
            },
        }

        result = await wizard._extract_data(
            "Mix, bake, and enjoy!", stage, llm=None,
        )

        assert result.data == {"instruction": "Mix, bake, and enjoy!"}
        assert result.confidence == 1.0
        assert result.metadata.get("capture_mode") == "verbatim"

    @pytest.mark.asyncio
    async def test_multi_field_schema_not_verbatim(self) -> None:
        """Multi-field schema does NOT use verbatim — falls through to extractor."""
        wizard = _make_wizard()
        stage = {
            "name": "collect",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "amount": {"type": "string"},
                },
                "required": ["name", "amount"],
            },
        }

        # No extractor configured → falls back to SimpleExtractionResult
        result = await wizard._extract_data(
            "2 cups flour", stage, llm=None,
        )

        # Should NOT be verbatim — should fall through to fallback
        assert result.data == {"_raw_input": "2 cups flour"}
        assert result.confidence == 0.5

    @pytest.mark.asyncio
    async def test_verbatim_config_override(self) -> None:
        """capture_mode='verbatim' forces verbatim for multi-field schema."""
        wizard = _make_wizard()
        stage = {
            "name": "collect",
            "collection_config": {"capture_mode": "verbatim"},
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "amount": {"type": "string"},
                },
                "required": ["name"],
            },
        }

        result = await wizard._extract_data(
            "flour", stage, llm=None,
        )

        # Verbatim captures into the first property key
        assert result.data == {"name": "flour"}
        assert result.confidence == 1.0
        assert result.metadata.get("capture_mode") == "verbatim"

    @pytest.mark.asyncio
    async def test_extract_config_override(self) -> None:
        """capture_mode='extract' forces LLM even for trivial schema."""
        wizard = _make_wizard()
        stage = {
            "name": "collect",
            "collection_config": {"capture_mode": "extract"},
            "schema": {
                "type": "object",
                "properties": {"instruction": {"type": "string"}},
                "required": ["instruction"],
            },
        }

        # No extractor → falls back to SimpleExtractionResult (not verbatim)
        result = await wizard._extract_data(
            "Mix and bake", stage, llm=None,
        )

        assert result.data == {"_raw_input": "Mix and bake"}
        assert result.confidence == 0.5

    @pytest.mark.asyncio
    async def test_no_schema_still_returns_raw_input(self) -> None:
        """No schema defined → raw input passthrough (existing behavior)."""
        wizard = _make_wizard()
        stage: dict[str, Any] = {"name": "collect"}

        result = await wizard._extract_data("hello", stage, llm=None)

        assert result.data == {"_raw_input": "hello"}
        assert result.confidence == 1.0
