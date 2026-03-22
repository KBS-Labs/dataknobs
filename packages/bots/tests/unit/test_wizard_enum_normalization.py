"""Tests for wizard enum normalization.

When an extraction model returns a value that is semantically correct but
syntactically wrong for an enum constraint (e.g. ``"Tutor"`` instead of
``"tutor"``), enum normalization matches it to the canonical entry.

Unit tests verify the normalization function directly.
Integration tests exercise the full DynaBot.from_config() → bot.chat()
path via ``BotTestHarness``.
"""

from typing import Any

import pytest

from dataknobs_bots.reasoning.wizard import _normalize_enum_value
from dataknobs_bots.testing import BotTestHarness, WizardConfigBuilder


# ---------------------------------------------------------------------------
# Unit tests: _normalize_enum_value()
# ---------------------------------------------------------------------------

ENUM = ["tutor", "quiz", "study_companion", "custom"]


class TestNormalizeEnumValue:
    """Verify the tiered matching algorithm."""

    # -- Tier 1: exact match --

    def test_exact_match(self) -> None:
        assert _normalize_enum_value("tutor", ENUM) == "tutor"

    def test_exact_match_preserves_case(self) -> None:
        """Exact match returns the value as-is (it IS the canonical form)."""
        mixed = ["Tutor", "Quiz"]
        assert _normalize_enum_value("Tutor", mixed) == "Tutor"

    # -- Tier 2: case-insensitive --

    def test_case_insensitive_upper(self) -> None:
        assert _normalize_enum_value("TUTOR", ENUM) == "tutor"

    def test_case_insensitive_title(self) -> None:
        assert _normalize_enum_value("Tutor", ENUM) == "tutor"

    def test_case_insensitive_mixed(self) -> None:
        assert _normalize_enum_value("Study_Companion", ENUM) == "study_companion"

    # -- Tier 3: substring match --

    def test_substring_value_contains_enum(self) -> None:
        """'tutor bot' contains 'tutor'."""
        assert _normalize_enum_value("tutor bot", ENUM) == "tutor"

    def test_substring_enum_contains_value(self) -> None:
        """'study' is contained in 'study_companion' (after normalisation)."""
        assert _normalize_enum_value("study", ENUM) == "study_companion"

    def test_substring_underscore_to_space(self) -> None:
        """'study companion' matches 'study_companion'."""
        assert _normalize_enum_value("study companion", ENUM) == "study_companion"

    def test_substring_hyphen_to_space(self) -> None:
        enum = ["lower-hyphen", "upper-case"]
        assert _normalize_enum_value("lower hyphen", enum) == "lower-hyphen"

    # -- Tier 4: token overlap --

    def test_token_overlap_above_threshold(self) -> None:
        """'interactive quiz mode' shares 'quiz' with enum 'quiz'."""
        assert _normalize_enum_value(
            "interactive quiz", ENUM, threshold=0.5,
        ) == "quiz"

    def test_token_overlap_below_threshold(self) -> None:
        """Low overlap does not match at high threshold."""
        # Use an enum where no tier-1/2/3 match exists, only partial
        # token overlap: "study" appears in both but score = 1/3 < 0.9
        enum = ["study_companion_advanced"]
        result = _normalize_enum_value(
            "study helper mode", enum, threshold=0.9,
        )
        assert result is None

    # -- Edge cases --

    def test_empty_value_returns_none(self) -> None:
        assert _normalize_enum_value("", ENUM) is None

    def test_empty_enum_returns_none(self) -> None:
        assert _normalize_enum_value("tutor", []) is None

    def test_no_match_returns_none(self) -> None:
        assert _normalize_enum_value("completely unrelated", ENUM) is None

    def test_exact_wins_over_substring(self) -> None:
        """Exact match (tier 1) takes priority over substring (tier 3)."""
        enum = ["ab", "abc"]
        assert _normalize_enum_value("ab", enum) == "ab"

    def test_non_string_enum_values_skipped(self) -> None:
        """Integer/mixed enum values don't crash — non-strings are filtered."""
        enum: list[Any] = [1, 2, "tutor", 3]
        assert _normalize_enum_value("Tutor", enum) == "tutor"

    def test_all_non_string_enum_returns_none(self) -> None:
        enum: list[Any] = [1, 2, 3]
        assert _normalize_enum_value("two", enum) is None

    def test_short_enum_values_skip_substring(self) -> None:
        """Single-char enum values don't false-positive via substring."""
        enum = ["y", "n"]
        # "y" is a substring of "yesterday" but tier 3 skips len < 2
        assert _normalize_enum_value("yesterday", enum) is None

    def test_word_boundary_prevents_partial_match(self) -> None:
        """Enum 'no' must not match 'nobody' via character containment."""
        enum = ["no", "yes"]
        assert _normalize_enum_value("nobody", enum) is None
        assert _normalize_enum_value("notable", enum) is None

    def test_word_boundary_prevents_suffix_match(self) -> None:
        """Enum 'tutor' must not match 'tutored' or 'tutoring'."""
        enum = ["tutor", "quiz"]
        assert _normalize_enum_value("tutored", enum) is None
        assert _normalize_enum_value("tutoring", enum) is None

    def test_word_boundary_allows_whole_word(self) -> None:
        """Enum 'no' should still match when it appears as a whole word."""
        enum = ["no", "yes"]
        assert _normalize_enum_value("no thanks", enum) == "no"

    def test_longest_substring_wins(self) -> None:
        """When multiple enums substring-match, longest match wins."""
        enum = ["quiz", "study_companion"]
        # "study companion quiz" contains both "quiz" and "study companion"
        # study_companion is longer → preferred
        result = _normalize_enum_value("study companion quiz", enum)
        assert result == "study_companion"

    def test_threshold_zero_matches_any_overlap(self) -> None:
        result = _normalize_enum_value(
            "quiz helper tool", ENUM, threshold=0.0,
        )
        assert result == "quiz"

    def test_threshold_one_requires_exact_token_match(self) -> None:
        """threshold=1.0 requires all tokens to overlap."""
        # "quiz" has 1 token, "quiz" has 1 token → score=1.0 → match
        assert _normalize_enum_value("quiz", ENUM, threshold=1.0) == "quiz"
        # "interactive quiz" has 2 tokens, "quiz" has 1 → score=0.5 → no match
        # But tier 3 substring catches "quiz" ⊆ "interactive quiz"
        # So this tests that substring takes priority over threshold
        assert _normalize_enum_value(
            "interactive quiz", ENUM, threshold=1.0,
        ) == "quiz"


# ---------------------------------------------------------------------------
# Integration tests: BotTestHarness
# ---------------------------------------------------------------------------


class TestEnumNormalizationIntegration:
    """Full pipeline tests via BotTestHarness."""

    @pytest.mark.asyncio
    async def test_case_mismatch_normalized(self) -> None:
        """Extracted 'Tutor' normalized to 'tutor', transition succeeds."""
        config = (
            WizardConfigBuilder("test")
            .settings(extraction_hints={"enum_normalize": True})
            .stage("gather", is_start=True, prompt="What type?")
                .field(
                    "intent", field_type="string",
                    enum=["tutor", "quiz", "study_companion", "custom"],
                    required=True,
                )
                .transition("done", "data.get('intent') == 'tutor'")
            .stage("done", is_end=True, prompt="Done!")
            .build()
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Got it!"],
            extraction_results=[[{"intent": "Tutor"}]],
        ) as harness:
            await harness.chat("I want a tutor")
            assert harness.wizard_data["intent"] == "tutor"
            assert harness.wizard_stage == "done"

    @pytest.mark.asyncio
    async def test_per_field_disable(self) -> None:
        """Per-field normalize=false prevents normalization."""
        config = (
            WizardConfigBuilder("test")
            .settings(extraction_hints={"enum_normalize": True})
            .stage("gather", is_start=True, prompt="Which provider?")
                .field(
                    "provider", field_type="string",
                    enum=["ollama", "openai", "anthropic"],
                    required=True,
                    x_extraction={
                        "normalize": False,
                        "reject_unmatched": False,
                    },
                )
                .transition("done", "has('provider')")
            .stage("done", is_end=True, prompt="Done!")
            .build()
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Got it!"],
            extraction_results=[[{"provider": "Ollama"}]],
        ) as harness:
            await harness.chat("Use Ollama")
            # Normalization disabled per-field, reject disabled: value as-is
            assert harness.wizard_data["provider"] == "Ollama"

    @pytest.mark.asyncio
    async def test_per_field_enable_overrides_class_disable(self) -> None:
        """Per-field normalize=true works even when class-level is false."""
        config = (
            WizardConfigBuilder("test")
            .settings(extraction_hints={"enum_normalize": False})
            .stage("gather", is_start=True, prompt="What type?")
                .field(
                    "intent", field_type="string",
                    enum=["tutor", "quiz"],
                    required=True,
                    x_extraction={"normalize": True},
                )
                .transition("done", "has('intent')")
            .stage("done", is_end=True, prompt="Done!")
            .build()
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Got it!"],
            extraction_results=[[{"intent": "TUTOR"}]],
        ) as harness:
            await harness.chat("I want a tutor")
            assert harness.wizard_data["intent"] == "tutor"

    @pytest.mark.asyncio
    async def test_default_enabled(self) -> None:
        """Default behavior: enum_normalize=true without explicit settings."""
        config = (
            WizardConfigBuilder("test")
            .stage("gather", is_start=True, prompt="What type?")
                .field(
                    "intent", field_type="string",
                    enum=["tutor", "quiz", "study_companion"],
                    required=True,
                )
                .transition("done", "has('intent')")
            .stage("done", is_end=True, prompt="Done!")
            .build()
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Got it!"],
            extraction_results=[[{"intent": "Study_Companion"}]],
        ) as harness:
            await harness.chat("study companion")
            assert harness.wizard_data["intent"] == "study_companion"

    @pytest.mark.asyncio
    async def test_exact_match_passthrough(self) -> None:
        """Values that already match exactly are not changed."""
        config = (
            WizardConfigBuilder("test")
            .stage("gather", is_start=True, prompt="What type?")
                .field(
                    "intent", field_type="string",
                    enum=["tutor", "quiz"],
                    required=True,
                )
                .transition("done", "has('intent')")
            .stage("done", is_end=True, prompt="Done!")
            .build()
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Got it!"],
            extraction_results=[[{"intent": "quiz"}]],
        ) as harness:
            await harness.chat("quiz please")
            assert harness.wizard_data["intent"] == "quiz"

    @pytest.mark.asyncio
    async def test_custom_threshold(self) -> None:
        """Per-field normalize_threshold controls fuzzy sensitivity."""
        config = (
            WizardConfigBuilder("test")
            .stage("gather", is_start=True, prompt="What type?")
                .field(
                    "intent", field_type="string",
                    enum=["tutor", "quiz"],
                    required=True,
                    x_extraction={
                        "normalize": True,
                        "normalize_threshold": 1.0,
                    },
                )
                .transition("done", "has('intent')")
            .stage("done", is_end=True, prompt="Done!")
            .build()
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Got it!", "Got it!"],
            extraction_results=[
                # Tier 4 overlap too low for threshold=1.0 but
                # "quiz" substring-matches tier 3, so it still works
                [{"intent": "quiz helper"}],
            ],
        ) as harness:
            await harness.chat("I want a quiz helper")
            assert harness.wizard_data["intent"] == "quiz"

    @pytest.mark.asyncio
    async def test_no_match_rejected_by_default(self) -> None:
        """Default: non-matching enum values are rejected (not merged)."""
        config = (
            WizardConfigBuilder("test")
            .stage("gather", is_start=True, prompt="What type?")
                .field(
                    "intent", field_type="string",
                    enum=["tutor", "quiz"],
                    required=True,
                )
                .transition("done", "has('intent')")
            .stage("done", is_end=True, prompt="Done!")
            .build()
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Got it!"],
            extraction_results=[[{"intent": "completely_unrelated"}]],
        ) as harness:
            await harness.chat("something random")
            # reject_unmatched=True by default — field not merged
            assert "intent" not in harness.wizard_data
            assert harness.wizard_stage == "gather"

    @pytest.mark.asyncio
    async def test_no_match_preserved_when_reject_disabled(self) -> None:
        """With reject_unmatched=false, non-matching values pass through."""
        config = (
            WizardConfigBuilder("test")
            .settings(extraction_hints={
                "enum_normalize": True,
                "reject_unmatched": False,
            })
            .stage("gather", is_start=True, prompt="What type?")
                .field(
                    "intent", field_type="string",
                    enum=["tutor", "quiz"],
                    required=True,
                )
                .transition("done", "has('intent')")
            .stage("done", is_end=True, prompt="Done!")
            .build()
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Got it!"],
            extraction_results=[[{"intent": "completely_unrelated"}]],
        ) as harness:
            await harness.chat("something random")
            # reject_unmatched=False — value stored as-is
            assert harness.wizard_data.get("intent") == "completely_unrelated"

    @pytest.mark.asyncio
    async def test_per_field_reject_override(self) -> None:
        """Per-field reject_unmatched overrides class-level setting."""
        config = (
            WizardConfigBuilder("test")
            .settings(extraction_hints={
                "enum_normalize": True,
                "reject_unmatched": False,
            })
            .stage("gather", is_start=True, prompt="Pick a provider.")
                .field(
                    "provider", field_type="string",
                    enum=["ollama", "openai", "anthropic"],
                    required=True,
                    x_extraction={"reject_unmatched": True},
                )
                .transition("done", "has('provider')")
            .stage("done", is_end=True, prompt="Done!")
            .build()
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Got it!"],
            extraction_results=[[{"provider": "magic"}]],
        ) as harness:
            await harness.chat("use magic")
            # Per-field reject=True overrides class-level reject=False
            assert "provider" not in harness.wizard_data
            assert harness.wizard_stage == "gather"

    @pytest.mark.asyncio
    async def test_reject_does_not_affect_valid_match(self) -> None:
        """reject_unmatched only applies when normalization finds no match."""
        config = (
            WizardConfigBuilder("test")
            .stage("gather", is_start=True, prompt="Pick a provider.")
                .field(
                    "provider", field_type="string",
                    enum=["ollama", "openai", "anthropic"],
                    required=True,
                )
                .transition("done", "data.get('provider')")
            .stage("done", is_end=True, prompt="Done!")
            .build()
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Got it!"],
            extraction_results=[[{"provider": "Ollama"}]],
        ) as harness:
            await harness.chat("use ollama")
            # Case-insensitive match succeeds — value normalized and accepted
            assert harness.wizard_data["provider"] == "ollama"
            assert harness.wizard_stage == "done"

    @pytest.mark.asyncio
    async def test_reject_keeps_wizard_at_stage(self) -> None:
        """Rejected enum value doesn't satisfy required field — stays at stage."""
        config = (
            WizardConfigBuilder("test")
            .stage("gather", is_start=True, prompt="What type?")
                .field(
                    "intent", field_type="string",
                    enum=["tutor", "quiz"],
                    required=True,
                )
                .transition("done", "data.get('intent')")
            .stage("done", is_end=True, prompt="Done!")
            .build()
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Got it!", "Got it!"],
            extraction_results=[
                [{"intent": "magic"}],      # Rejected — no match
                [{"intent": "tutor"}],       # Accepted — exact match
            ],
        ) as harness:
            await harness.chat("use magic")
            assert "intent" not in harness.wizard_data
            assert harness.wizard_stage == "gather"

            await harness.chat("tutor please")
            assert harness.wizard_data["intent"] == "tutor"
            assert harness.wizard_stage == "done"

    @pytest.mark.asyncio
    async def test_reject_preserves_existing_valid_data(self) -> None:
        """Rejected value on a subsequent turn does not clobber valid data."""
        config = (
            WizardConfigBuilder("test")
            .stage("gather", is_start=True, prompt="Pick provider and model.")
                .field(
                    "provider", field_type="string",
                    enum=["ollama", "openai", "anthropic"],
                    required=True,
                )
                .field(
                    "model", field_type="string",
                    required=True,
                )
                .transition(
                    "done",
                    "data.get('provider') and data.get('model')",
                )
            .stage("done", is_end=True, prompt="Done!")
            .build()
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Got it!", "Got it!"],
            extraction_results=[
                [{"provider": "ollama"}],                # Valid — stored
                [{"provider": "magic", "model": "llama"}],  # provider rejected, model accepted
            ],
        ) as harness:
            await harness.chat("use ollama")
            assert harness.wizard_data["provider"] == "ollama"
            # Still at gather — model not yet provided
            assert harness.wizard_stage == "gather"

            await harness.chat("use magic model llama")
            # "magic" rejected — existing "ollama" preserved
            assert harness.wizard_data["provider"] == "ollama"
            # "model" accepted — no enum constraint
            assert harness.wizard_data["model"] == "llama"
            assert harness.wizard_stage == "done"

    @pytest.mark.asyncio
    async def test_reject_without_normalization(self) -> None:
        """reject_unmatched works as strict enum check when normalize=false."""
        config = (
            WizardConfigBuilder("test")
            .settings(extraction_hints={
                "enum_normalize": False,
                "reject_unmatched": True,
            })
            .stage("gather", is_start=True, prompt="Pick a provider.")
                .field(
                    "provider", field_type="string",
                    enum=["ollama", "openai", "anthropic"],
                    required=True,
                )
                .transition("done", "data.get('provider')")
            .stage("done", is_end=True, prompt="Done!")
            .build()
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Got it!", "Got it!"],
            extraction_results=[
                [{"provider": "Ollama"}],    # Case mismatch — rejected (no normalization)
                [{"provider": "ollama"}],    # Exact match — accepted
            ],
        ) as harness:
            await harness.chat("use Ollama")
            # normalize=False, so "Ollama" is not normalized to "ollama"
            # reject_unmatched=True, so "Ollama" is rejected (not in enum)
            assert "provider" not in harness.wizard_data
            assert harness.wizard_stage == "gather"

            await harness.chat("use ollama")
            assert harness.wizard_data["provider"] == "ollama"
            assert harness.wizard_stage == "done"
