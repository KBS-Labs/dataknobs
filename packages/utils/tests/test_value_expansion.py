"""Tests for conjunction-bounded value expansion."""

from dataknobs_utils.value_expansion import (
    ValueExpansionConfig,
    ValueExpansionPlugin,
    expand_value_in_message,
)

import pytest

# Representative stopword set for testing — the canonical set lives in
# dataknobs_llm.extraction.grounding and is passed at the integration layer.
_STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "is", "of", "for", "to", "or", "and", "whether",
    "it", "in", "on", "at", "by", "with", "this", "that", "be", "as",
    "can", "do", "does", "did", "has", "have", "had", "was", "were",
    "are", "been", "being", "will", "would", "could", "should", "may",
    "might", "shall", "must", "not", "but", "if", "then", "than",
    "so", "just", "also", "very", "too", "really", "quite",
})


def _expand(value: str, message: str, **kwargs) -> str | None:
    """Shorthand for calling expand_value_in_message with test stopwords."""
    return expand_value_in_message(
        value, message, stopwords=_STOPWORDS, **kwargs,
    )


# ──────────────────────────────────────────────────────────────────
# Cases where expansion SHOULD find more words
# ──────────────────────────────────────────────────────────────────


class TestExpansionCases:
    """Values that should expand across conjunctions."""

    def test_formal_expands_to_formal_and_academic(self):
        result = _expand("formal", "Let's set the tone to formal and academic")
        assert result == "formal and academic"

    def test_clear_expands_to_clear_and_concise(self):
        result = _expand("clear", "Make the writing clear and concise")
        assert result == "clear and concise"

    def test_warm_expands_to_warm_and_inviting(self):
        result = _expand("warm", "Make it warm and inviting")
        assert result == "warm and inviting"

    def test_or_conjunction(self):
        result = _expand("formal", "Set the tone to formal or academic")
        assert result == "formal or academic"

    def test_multiple_occurrences_picks_best(self):
        result = _expand(
            "formal",
            "A formal review. Set the tone to formal and academic.",
        )
        assert result == "formal and academic"

    def test_professional_and_well_structured(self):
        result = _expand(
            "professional",
            "Make it professional and well-structured",
        )
        assert result == "professional and well-structured"

    def test_preserves_original_casing(self):
        """Expansion should preserve the user's original casing."""
        result = _expand(
            "formal",
            "Set the tone to Formal and Academic",
        )
        assert result == "Formal and Academic"

    def test_multi_conjunction_chain(self):
        """Two conjunctions: 'formal and academic and technical'."""
        result = _expand(
            "formal",
            "Set the tone to formal and academic and technical",
        )
        assert result == "formal and academic and technical"

    def test_multi_conjunction_stops_at_boundary(self):
        """Two conjunctions but second hits a boundary pattern."""
        result = _expand(
            "formal",
            "Set the tone to formal and academic and the style to narrative",
        )
        assert result == "formal and academic"

    def test_stopword_between_conjunction_and_content(self):
        """Stopwords between conjunction and significant word are included."""
        result = _expand(
            "formal",
            "Set the tone to formal and very academic",
        )
        assert result == "formal and very academic"


# ──────────────────────────────────────────────────────────────────
# Cases where expansion should NOT happen
# ──────────────────────────────────────────────────────────────────


class TestNoExpansionCases:
    """Values that should NOT expand — boundary conditions."""

    def test_already_complete(self):
        result = _expand(
            "formal and academic",
            "Let's set the tone to formal and academic",
        )
        assert result is None

    def test_end_of_sentence(self):
        result = _expand("formal", "The tone should be formal.")
        assert result is None

    def test_comma_boundary(self):
        result = _expand(
            "formal",
            "Set the tone to formal, and make it concise.",
        )
        assert result is None

    def test_and_the_boundary(self):
        result = _expand(
            "formal",
            "Set the tone to formal and the style to academic",
        )
        assert result is None

    def test_and_set_boundary(self):
        result = _expand(
            "formal",
            "Set the tone to formal and set the audience to experts",
        )
        assert result is None

    def test_but_boundary(self):
        result = _expand(
            "formal",
            "Set the tone to formal but keep it accessible",
        )
        assert result is None

    def test_and_make_boundary(self):
        result = _expand(
            "formal",
            "Set the tone to formal and make it engaging",
        )
        assert result is None

    def test_and_add_boundary(self):
        result = _expand(
            "warm",
            "Make it warm and add some citations",
        )
        assert result is None

    def test_and_use_boundary(self):
        result = _expand(
            "formal",
            "Set the tone to formal and use APA style",
        )
        assert result is None

    def test_value_not_in_message(self):
        result = _expand("casual", "Set the tone to formal and academic")
        assert result is None

    def test_empty_value(self):
        result = _expand("", "Set the tone to formal and academic")
        assert result is None

    def test_substring_match_inside_word(self):
        """Value found as substring of a longer word — should not match.

        'form' appears inside 'formal' but is not a word boundary match.
        Without word-boundary checking, this would produce garbage like
        'form al and academic'.
        """
        result = _expand("form", "Set the tone to formal and academic")
        assert result is None

    def test_short_value_substring_of_word(self):
        """Short value 'or' inside 'laboratory' — should not match."""
        result = _expand("or", "The laboratory and research center")
        assert result is None

    def test_value_at_word_boundary_still_expands(self):
        """Value at a proper word boundary should still expand normally."""
        result = _expand("formal", "Set the tone to formal and academic")
        assert result == "formal and academic"

    def test_conjunction_with_attached_punctuation(self):
        """Conjunction with punctuation (e.g., 'and,') should stop expansion.

        The comma is a phrase-break character; expansion should not
        continue through a conjunction that has punctuation attached.
        """
        result = _expand(
            "formal",
            "Set the tone to formal and, academic",
        )
        assert result is None


# ──────────────────────────────────────────────────────────────────
# Config overrides
# ──────────────────────────────────────────────────────────────────


class TestConfigOverrides:
    """Custom configuration changes expansion behaviour."""

    def test_custom_conjunctions(self):
        """Only 'plus' bridges — 'and' does not."""
        config = ValueExpansionConfig(conjunctions=frozenset({"plus"}))
        result = _expand(
            "formal",
            "Set the tone to formal and academic",
            config=config,
        )
        assert result is None

        result = _expand(
            "formal",
            "Set the tone to formal plus academic",
            config=config,
        )
        assert result == "formal plus academic"

    def test_custom_break_chars_no_comma(self):
        """Removing comma from break chars allows expansion across commas.

        Also requires ``require_conjunction=False`` because in a
        comma-separated list like ``"formal, concise, and academic"``
        there is no conjunction between ``"formal"`` and ``"concise"``.
        """
        config = ValueExpansionConfig(
            phrase_break_chars=frozenset(".;:!?"),  # no comma
            require_conjunction=False,
        )
        result = _expand(
            "formal",
            "The tone should be formal, concise, and academic",
            config=config,
        )
        assert result is not None
        assert "concise" in result
        assert "academic" in result

    def test_require_conjunction_false(self):
        """Without requiring conjunction, adjacent significant words expand."""
        config = ValueExpansionConfig(require_conjunction=False)
        result = _expand(
            "formal",
            "Set the tone to formal academic",
            config=config,
        )
        assert result == "formal academic"


# ──────────────────────────────────────────────────────────────────
# Direction parameter
# ──────────────────────────────────────────────────────────────────


class TestDirectionParameter:
    """RTL readiness — only 'right' is implemented."""

    def test_right_works(self):
        config = ValueExpansionConfig(expand_direction="right")
        result = _expand(
            "formal",
            "Set the tone to formal and academic",
            config=config,
        )
        assert result == "formal and academic"

    def test_left_raises_not_implemented(self):
        config = ValueExpansionConfig(expand_direction="left")
        with pytest.raises(NotImplementedError):
            _expand(
                "formal",
                "Set the tone to formal and academic",
                config=config,
            )

    def test_both_raises_not_implemented(self):
        config = ValueExpansionConfig(expand_direction="both")
        with pytest.raises(NotImplementedError):
            _expand(
                "formal",
                "Set the tone to formal and academic",
                config=config,
            )


# ──────────────────────────────────────────────────────────────────
# Plugin protocol
# ──────────────────────────────────────────────────────────────────


class TestPluginProtocol:
    """Extension point for augmented expansion."""

    def test_plugin_overrides_result(self):
        class OverridePlugin:
            def expand(self, value, message, deterministic_result, config):
                return "plugin-override"

        result = _expand(
            "formal",
            "Set the tone to formal and academic",
            plugin=OverridePlugin(),
        )
        assert result == "plugin-override"

    def test_plugin_returns_none_falls_back(self):
        class PassthroughPlugin:
            def expand(self, value, message, deterministic_result, config):
                return None

        result = _expand(
            "formal",
            "Set the tone to formal and academic",
            plugin=PassthroughPlugin(),
        )
        assert result == "formal and academic"

    def test_no_plugin_uses_deterministic(self):
        result = _expand(
            "formal",
            "Set the tone to formal and academic",
            plugin=None,
        )
        assert result == "formal and academic"

    def test_plugin_receives_deterministic_result(self):
        received = {}

        class InspectPlugin:
            def expand(self, value, message, deterministic_result, config):
                received["value"] = value
                received["deterministic"] = deterministic_result
                return None

        _expand(
            "formal",
            "Set the tone to formal and academic",
            plugin=InspectPlugin(),
        )
        assert received["value"] == "formal"
        assert received["deterministic"] == "formal and academic"

    def test_plugin_satisfies_protocol(self):
        class MyPlugin:
            def expand(self, value, message, deterministic_result, config):
                return None

        assert isinstance(MyPlugin(), ValueExpansionPlugin)
