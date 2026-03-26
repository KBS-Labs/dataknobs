"""Tests for extraction grounding utility.

Validates that :func:`is_field_grounded` and :func:`ground_extraction`
correctly identify whether extracted values are grounded in the user's
message, using type-dispatched heuristics and ``x-extraction`` overrides.
"""

from __future__ import annotations

import pytest

from dataknobs_llm.extraction.grounding import (
    DEFAULT_NEGATION_KEYWORDS,
    DEFAULT_STOPWORDS,
    FieldGroundingResult,
    GroundingConfig,
    detect_boolean_signal,
    field_keywords,
    ground_extraction,
    has_negation,
    is_field_grounded,
    significant_words,
)


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

def _string_prop(description: str = "", **extra: object) -> dict:
    """Build a string schema property."""
    p: dict = {"type": "string"}
    if description:
        p["description"] = description
    p.update(extra)
    return p


def _enum_prop(values: list[str], description: str = "", **extra: object) -> dict:
    p: dict = {"type": "string", "enum": values}
    if description:
        p["description"] = description
    p.update(extra)
    return p


def _bool_prop(description: str = "", **extra: object) -> dict:
    p: dict = {"type": "boolean"}
    if description:
        p["description"] = description
    p.update(extra)
    return p


def _number_prop(num_type: str = "integer", **extra: object) -> dict:
    p: dict = {"type": num_type}
    p.update(extra)
    return p


def _array_prop(description: str = "", **extra: object) -> dict:
    p: dict = {"type": "array", "items": {"type": "string"}}
    if description:
        p["description"] = description
    p.update(extra)
    return p


# ──────────────────────────────────────────────────────────────────
# TestIsFieldGrounded — type-dispatched checks
# ──────────────────────────────────────────────────────────────────

class TestIsFieldGrounded:
    """Core type-dispatched grounding checks."""

    def test_enum_grounded(self) -> None:
        prop = _enum_prop(["red", "blue", "green"])
        result = is_field_grounded("color", "red", "I want the red one", prop)
        assert result.grounded is True
        assert result.strategy == "enum"

    def test_enum_ungrounded(self) -> None:
        prop = _enum_prop(["red", "blue", "green"])
        result = is_field_grounded("color", "blue", "I want the red one", prop)
        assert result.grounded is False
        assert result.strategy == "enum"

    def test_string_overlap_above_threshold(self) -> None:
        prop = _string_prop()
        result = is_field_grounded(
            "topic", "machine learning", "tell me about machine learning", prop,
        )
        assert result.grounded is True
        assert result.strategy == "string_overlap"

    def test_string_overlap_below_threshold(self) -> None:
        prop = _string_prop()
        result = is_field_grounded(
            "topic", "quantum computing research",
            "tell me about the weather today", prop,
        )
        assert result.grounded is False

    def test_string_all_stopwords_trusted(self) -> None:
        """Value consisting only of stopwords is trusted."""
        prop = _string_prop()
        result = is_field_grounded(
            "filler", "the is a", "some random message", prop,
        )
        assert result.grounded is True

    def test_number_literal_match(self) -> None:
        prop = _number_prop()
        result = is_field_grounded("count", 42, "I need 42 items", prop)
        assert result.grounded is True

    def test_number_no_match(self) -> None:
        prop = _number_prop()
        result = is_field_grounded("count", 5, "I need 15 items", prop)
        assert result.grounded is False

    def test_number_no_match_prefix(self) -> None:
        """5 should not match inside 50."""
        prop = _number_prop()
        result = is_field_grounded("count", 5, "I need 50 items", prop)
        assert result.grounded is False

    def test_boolean_field_keywords_present(self) -> None:
        prop = _bool_prop(description="Enable notifications")
        result = is_field_grounded(
            "notifications", True, "yes enable notifications", prop,
        )
        assert result.grounded is True
        assert result.strategy == "boolean"

    def test_boolean_field_keywords_absent(self) -> None:
        prop = _bool_prop(description="Enable notifications")
        result = is_field_grounded(
            "notifications", True, "tell me about the weather", prop,
        )
        assert result.grounded is False

    def test_boolean_direction_false_with_negation(self) -> None:
        """False value grounded when negation keyword is near field keyword."""
        prop = _bool_prop(description="Enable notifications")
        result = is_field_grounded(
            "notifications", False, "no notifications please", prop,
        )
        assert result.grounded is True

    def test_boolean_direction_false_without_negation(self) -> None:
        """False value ungrounded when no negation keyword."""
        prop = _bool_prop(description="Enable notifications")
        result = is_field_grounded(
            "notifications", False, "yes notifications please", prop,
        )
        assert result.grounded is False

    def test_array_element_in_message(self) -> None:
        prop = _array_prop(description="Topics to search")
        result = is_field_grounded(
            "topics", ["python", "rust"], "I want to learn python", prop,
        )
        assert result.grounded is True
        assert result.strategy == "array"

    def test_array_no_elements(self) -> None:
        prop = _array_prop(description="Topics to search")
        result = is_field_grounded(
            "topics", ["quantum", "biology"], "tell me about python", prop,
        )
        assert result.grounded is False

    def test_empty_string_with_negation(self) -> None:
        prop = _string_prop(description="Subject area")
        result = is_field_grounded(
            "subject", "", "clear the subject please", prop,
        )
        assert result.grounded is True

    def test_empty_string_without_negation(self) -> None:
        prop = _string_prop(description="Subject area")
        result = is_field_grounded(
            "subject", "", "tell me about history", prop,
        )
        assert result.grounded is False

    def test_unknown_type_trusted(self) -> None:
        prop = {"type": "object"}
        result = is_field_grounded("data", {"a": 1}, "anything", prop)
        assert result.grounded is True
        assert result.strategy == "unknown"


# ──────────────────────────────────────────────────────────────────
# TestXExtractionOverrides
# ──────────────────────────────────────────────────────────────────

class TestXExtractionOverrides:
    """Per-field x-extraction annotation overrides."""

    def test_grounding_skip(self) -> None:
        prop = _string_prop(**{"x-extraction": {"grounding": "skip"}})
        result = is_field_grounded("field", "anything", "no match", prop)
        assert result.grounded is True
        assert result.strategy == "skip"

    def test_grounding_exact(self) -> None:
        prop = _string_prop(**{"x-extraction": {"grounding": "exact"}})
        result = is_field_grounded("field", "hello", "say hello world", prop)
        assert result.grounded is True

    def test_grounding_exact_no_match(self) -> None:
        prop = _string_prop(**{"x-extraction": {"grounding": "exact"}})
        result = is_field_grounded("field", "hello", "say hi world", prop)
        assert result.grounded is False
        assert result.strategy == "exact"

    def test_grounding_fuzzy(self) -> None:
        prop = _string_prop(**{"x-extraction": {"grounding": "fuzzy"}})
        result = is_field_grounded("field", "anything", "no match at all", prop)
        assert result.grounded is True
        assert result.strategy == "fuzzy"

    def test_custom_overlap_threshold(self) -> None:
        """Per-field threshold overrides config threshold."""
        prop = _string_prop(**{"x-extraction": {"overlap_threshold": 0.1}})
        # "research" overlaps, that's 1/3 words = 0.33 > 0.1
        result = is_field_grounded(
            "topic", "quantum computing research",
            "I want research papers", prop,
        )
        assert result.grounded is True

    def test_empty_allowed(self) -> None:
        prop = _string_prop(**{"x-extraction": {"empty_allowed": True}})
        result = is_field_grounded("field", "", "any message", prop)
        assert result.grounded is True

    def test_custom_negation_keywords(self) -> None:
        prop = _bool_prop(
            description="Enable alerts",
            **{"x-extraction": {"negation_keywords": ["deactivate"]}},
        )
        result = is_field_grounded(
            "alerts", False, "deactivate alerts", prop,
        )
        assert result.grounded is True

    def test_negation_proximity(self) -> None:
        """Negation within proximity window is detected."""
        prop = _bool_prop(
            description="Enable alerts",
            **{"x-extraction": {"negation_proximity": 2}},
        )
        # "no" is 1 word from "alerts" — within proximity 2
        result = is_field_grounded(
            "alerts", False, "no alerts please", prop,
        )
        assert result.grounded is True

    def test_check_direction_false(self) -> None:
        """When check_direction is false, just field mention is enough."""
        prop = _bool_prop(
            description="Enable alerts",
            **{"x-extraction": {"check_direction": False}},
        )
        # False value but no negation — normally ungrounded, but
        # check_direction=false means field mention is sufficient
        result = is_field_grounded(
            "alerts", False, "yes I want alerts", prop,
        )
        assert result.grounded is True


# ──────────────────────────────────────────────────────────────────
# TestGroundExtraction — bulk check
# ──────────────────────────────────────────────────────────────────

class TestGroundExtraction:
    """Bulk extraction grounding via ground_extraction()."""

    def test_bulk_check_mixed(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "color": _enum_prop(["red", "blue"]),
                "size": _number_prop(),
            },
        }
        extracted = {"color": "red", "size": 99}
        results = ground_extraction(extracted, "I want the red widget", schema)

        assert results["color"].grounded is True
        assert results["size"].grounded is False

    def test_skips_none_values(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "name": _string_prop(),
                "age": _number_prop(),
            },
        }
        extracted = {"name": "Alice", "age": None}
        results = ground_extraction(extracted, "my name is Alice", schema)

        assert "name" in results
        assert "age" not in results  # None skipped

    def test_config_threshold_applied(self) -> None:
        """GroundingConfig.overlap_threshold flows through to string checks."""
        schema = {
            "type": "object",
            "properties": {
                "topic": _string_prop(),
            },
        }
        # "research" overlaps (1/3 value words)
        extracted = {"topic": "quantum computing research"}
        msg = "I want research papers"

        # Default threshold (0.5) — 0.33 < 0.5 → ungrounded
        results = ground_extraction(extracted, msg, schema)
        assert results["topic"].grounded is False

        # Low threshold (0.1) — 0.33 > 0.1 → grounded
        cfg = GroundingConfig(overlap_threshold=0.1)
        results = ground_extraction(extracted, msg, schema, config=cfg)
        assert results["topic"].grounded is True

    def test_missing_schema_property_skipped(self) -> None:
        """Fields in data but not in schema are skipped."""
        schema = {
            "type": "object",
            "properties": {
                "name": _string_prop(),
            },
        }
        extracted = {"name": "Alice", "extra_field": "value"}
        results = ground_extraction(extracted, "Alice says hello", schema)

        assert "name" in results
        assert "extra_field" not in results

    def test_config_stopwords_applied(self) -> None:
        """Custom stopwords on GroundingConfig flow through."""
        schema = {
            "type": "object",
            "properties": {
                "topic": _string_prop(),
            },
        }
        # "python" is the only significant word. With custom stopwords
        # that include "python", value_words becomes empty → trusted.
        custom_sw = DEFAULT_STOPWORDS | {"python"}
        cfg = GroundingConfig(stopwords=custom_sw)
        extracted = {"topic": "python"}
        results = ground_extraction(
            extracted, "tell me about something", schema, config=cfg,
        )
        # "python" is now a stopword → value_words is empty → trusted
        assert results["topic"].grounded is True

    def test_config_negation_keywords_applied(self) -> None:
        """Custom negation_keywords on GroundingConfig flow through."""
        schema = {
            "type": "object",
            "properties": {
                "subject": _string_prop(description="Subject area"),
            },
        }
        # "purge" is not a default negation keyword
        cfg = GroundingConfig(
            negation_keywords=DEFAULT_NEGATION_KEYWORDS | {"purge"},
        )
        extracted = {"subject": ""}
        results = ground_extraction(
            extracted, "purge the subject", schema, config=cfg,
        )
        assert results["subject"].grounded is True


# ──────────────────────────────────────────────────────────────────
# TestUtilities
# ──────────────────────────────────────────────────────────────────

class TestUtilities:
    """Text utility functions."""

    def test_significant_words_filters_stopwords(self) -> None:
        words = significant_words("the quick brown fox is very fast")
        assert "quick" in words
        assert "brown" in words
        assert "fox" in words
        assert "fast" in words
        assert "the" not in words
        assert "is" not in words
        assert "very" not in words

    def test_significant_words_strips_punctuation(self) -> None:
        words = significant_words("hello, world! (test)")
        assert "hello" in words
        assert "world" in words
        assert "test" in words

    def test_significant_words_custom_stopwords(self) -> None:
        words = significant_words(
            "python is great",
            stopwords=frozenset({"python", "great"}),
        )
        assert words == set()  # "is" is short (< 3 chars)

    def test_field_keywords_from_description(self) -> None:
        prop = {"description": "Enable email notifications for alerts"}
        kw = field_keywords("email_alerts", prop)
        assert "enable" in kw
        assert "email" in kw
        assert "notifications" in kw
        assert "alerts" in kw

    def test_field_keywords_from_field_name(self) -> None:
        prop: dict = {}
        kw = field_keywords("save_confirmed", prop)
        assert "save" in kw
        assert "confirmed" in kw

    def test_has_negation_basic(self) -> None:
        assert has_negation("no I don't want that", DEFAULT_NEGATION_KEYWORDS)
        assert not has_negation("yes I want that", DEFAULT_NEGATION_KEYWORDS)

    def test_has_negation_proximity(self) -> None:
        kw = {"alerts"}
        # "no" is 1 word from "alerts"
        assert has_negation(
            "no alerts please", DEFAULT_NEGATION_KEYWORDS,
            field_keywords=kw, proximity=2,
        )
        # "no" is 5 words from "alerts" — outside proximity 2
        assert not has_negation(
            "no I really do want alerts", DEFAULT_NEGATION_KEYWORDS,
            field_keywords=kw, proximity=2,
        )

    def test_detect_boolean_signal_affirmative(self) -> None:
        result = detect_boolean_signal(
            "yes please",
            affirmative_signals=frozenset({"yes", "confirm"}),
            affirmative_phrases=("sounds good",),
            negative_signals=frozenset({"no", "cancel"}),
            negative_phrases=("not yet",),
        )
        assert result is True

    def test_detect_boolean_signal_negative(self) -> None:
        result = detect_boolean_signal(
            "no thanks",
            affirmative_signals=frozenset({"yes", "confirm"}),
            affirmative_phrases=("sounds good",),
            negative_signals=frozenset({"no", "cancel"}),
            negative_phrases=("not yet",),
        )
        assert result is False

    def test_detect_boolean_signal_none(self) -> None:
        result = detect_boolean_signal(
            "tell me more about it",
            affirmative_signals=frozenset({"yes", "confirm"}),
            affirmative_phrases=("sounds good",),
            negative_signals=frozenset({"no", "cancel"}),
            negative_phrases=("not yet",),
        )
        assert result is None


# ──────────────────────────────────────────────────────────────────
# TestGroundingConfig
# ──────────────────────────────────────────────────────────────────

class TestGroundingConfig:
    """GroundingConfig defaults and customization."""

    def test_defaults(self) -> None:
        cfg = GroundingConfig()
        assert cfg.overlap_threshold == 0.5
        assert cfg.stopwords is DEFAULT_STOPWORDS
        assert cfg.negation_keywords is DEFAULT_NEGATION_KEYWORDS

    def test_custom_values(self) -> None:
        custom_sw = frozenset({"custom"})
        custom_neg = frozenset({"nope"})
        cfg = GroundingConfig(
            overlap_threshold=0.3,
            stopwords=custom_sw,
            negation_keywords=custom_neg,
        )
        assert cfg.overlap_threshold == 0.3
        assert cfg.stopwords == custom_sw
        assert cfg.negation_keywords == custom_neg

    def test_frozen(self) -> None:
        cfg = GroundingConfig()
        with pytest.raises(AttributeError):
            cfg.overlap_threshold = 0.9  # type: ignore[misc]


class TestFieldGroundingResult:
    """FieldGroundingResult structure."""

    def test_attributes(self) -> None:
        r = FieldGroundingResult(
            field="color", grounded=True,
            reason="enum match", strategy="enum",
        )
        assert r.field == "color"
        assert r.grounded is True
        assert r.reason == "enum match"
        assert r.strategy == "enum"

    def test_frozen(self) -> None:
        r = FieldGroundingResult(
            field="x", grounded=False, reason="no", strategy="skip",
        )
        with pytest.raises(AttributeError):
            r.grounded = True  # type: ignore[misc]
