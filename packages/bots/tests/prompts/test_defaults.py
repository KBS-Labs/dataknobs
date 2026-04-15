"""Tests for the aggregated default prompt library.

Verifies:
- All prompt keys from all modules are collected without conflicts
- get_default_prompt_library() returns a working ConfigPromptLibrary
- get_full_prompt_library() returns a CompositePromptLibrary with both
  bots and extraction prompt keys
- No duplicate keys exist across modules
- Key count matches expected totals
"""

import pytest

from dataknobs_llm.prompts import ConfigPromptLibrary, CompositePromptLibrary

from dataknobs_bots.prompts.defaults import (
    ALL_BOTS_PROMPT_KEYS,
    get_default_prompt_library,
    get_full_prompt_library,
)
from dataknobs_bots.prompts.wizard import WIZARD_PROMPT_KEYS
from dataknobs_bots.prompts.memory import MEMORY_PROMPT_KEYS
from dataknobs_bots.prompts.rubric import RUBRIC_PROMPT_KEYS
from dataknobs_bots.prompts.review import REVIEW_PROMPT_KEYS
from dataknobs_bots.prompts.grounded import GROUNDED_PROMPT_KEYS
from dataknobs_bots.prompts.focus import FOCUS_PROMPT_KEYS


class TestAllBotsPromptKeys:

    def test_total_key_count(self) -> None:
        """Total keys equals the sum of all module key counts."""
        expected = (
            len(WIZARD_PROMPT_KEYS)
            + len(MEMORY_PROMPT_KEYS)
            + len(RUBRIC_PROMPT_KEYS)
            + len(REVIEW_PROMPT_KEYS)
            + len(GROUNDED_PROMPT_KEYS)
            + len(FOCUS_PROMPT_KEYS)
        )
        assert len(ALL_BOTS_PROMPT_KEYS) == expected

    def test_no_namespace_collisions(self) -> None:
        """All keys use distinct namespace prefixes per module."""
        wizard_keys = {k for k in ALL_BOTS_PROMPT_KEYS if k.startswith("wizard.")}
        memory_keys = {k for k in ALL_BOTS_PROMPT_KEYS if k.startswith("memory.")}
        rubric_keys = {k for k in ALL_BOTS_PROMPT_KEYS if k.startswith("rubric.")}
        review_keys = {k for k in ALL_BOTS_PROMPT_KEYS if k.startswith("review.")}
        grounded_keys = {k for k in ALL_BOTS_PROMPT_KEYS if k.startswith("grounded.")}
        focus_keys = {k for k in ALL_BOTS_PROMPT_KEYS if k.startswith("focus.")}

        assert len(wizard_keys) == len(WIZARD_PROMPT_KEYS)
        assert len(memory_keys) == len(MEMORY_PROMPT_KEYS)
        assert len(rubric_keys) == len(RUBRIC_PROMPT_KEYS)
        assert len(review_keys) == len(REVIEW_PROMPT_KEYS)
        assert len(grounded_keys) == len(GROUNDED_PROMPT_KEYS)
        assert len(focus_keys) == len(FOCUS_PROMPT_KEYS)

    def test_all_values_are_prompt_template_dicts(self) -> None:
        """Every value has the required 'template' and 'template_syntax' keys."""
        for key, value in ALL_BOTS_PROMPT_KEYS.items():
            assert "template" in value, f"Key {key!r} missing 'template'"
            assert "template_syntax" in value, f"Key {key!r} missing 'template_syntax'"


class TestGetDefaultPromptLibrary:

    def test_returns_config_prompt_library(self) -> None:
        library = get_default_prompt_library()
        assert isinstance(library, ConfigPromptLibrary)

    def test_wizard_keys_accessible(self) -> None:
        library = get_default_prompt_library()
        for key in WIZARD_PROMPT_KEYS:
            result = library.get_system_prompt(key)
            assert result is not None, f"Key {key!r} not found in library"

    def test_review_keys_accessible(self) -> None:
        library = get_default_prompt_library()
        for key in REVIEW_PROMPT_KEYS:
            result = library.get_system_prompt(key)
            assert result is not None, f"Key {key!r} not found in library"

    def test_grounded_keys_accessible(self) -> None:
        library = get_default_prompt_library()
        for key in GROUNDED_PROMPT_KEYS:
            result = library.get_system_prompt(key)
            assert result is not None, f"Key {key!r} not found in library"

    def test_focus_keys_accessible(self) -> None:
        library = get_default_prompt_library()
        for key in FOCUS_PROMPT_KEYS:
            result = library.get_system_prompt(key)
            assert result is not None, f"Key {key!r} not found in library"

    def test_memory_keys_accessible(self) -> None:
        library = get_default_prompt_library()
        for key in MEMORY_PROMPT_KEYS:
            result = library.get_system_prompt(key)
            assert result is not None, f"Key {key!r} not found in library"

    def test_rubric_keys_accessible(self) -> None:
        library = get_default_prompt_library()
        for key in RUBRIC_PROMPT_KEYS:
            result = library.get_system_prompt(key)
            assert result is not None, f"Key {key!r} not found in library"

    def test_nonexistent_key_returns_none(self) -> None:
        library = get_default_prompt_library()
        assert library.get_system_prompt("nonexistent.key") is None


class TestGetFullPromptLibrary:

    def test_returns_composite_prompt_library(self) -> None:
        library = get_full_prompt_library()
        assert isinstance(library, CompositePromptLibrary)

    def test_bots_keys_accessible(self) -> None:
        library = get_full_prompt_library()
        result = library.get_system_prompt("wizard.clarification")
        assert result is not None

    def test_extraction_keys_accessible(self) -> None:
        library = get_full_prompt_library()
        result = library.get_system_prompt("extraction.default")
        assert result is not None

    def test_extraction_fragment_accessible(self) -> None:
        library = get_full_prompt_library()
        result = library.get_system_prompt("extraction.default.schema_section")
        assert result is not None

    def test_nonexistent_key_returns_none(self) -> None:
        library = get_full_prompt_library()
        assert library.get_system_prompt("nonexistent.key") is None
