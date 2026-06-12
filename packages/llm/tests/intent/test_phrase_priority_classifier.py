"""Tests for ``KeywordIntentClassifier``'s ``phrase_priority`` mode.

The mode lets the caller surface ambiguity (two intents tying at the
same tier) instead of resolving by iteration order, and lets
multi-word phrases beat single-word matches. The
:func:`dataknobs_llm.extraction.grounding.detect_boolean_signal`
helper opts in for the affirmative/negative two-intent shape.

Default behaviour (``phrase_priority=False``) is byte-identical
first-match-wins iteration; the intent-confirm primitive's call sites
do not opt in. See the parity-equivalence test in
``packages/bots/tests/unit/test_wizard_boolean_recovery.py`` for the
cross-path drift guard against ``detect_boolean_signal``.
"""
from __future__ import annotations

import pytest

from dataknobs_llm.intent import (
    IntentSpec,
    KeywordIntentClassifier,
)


_AFF_NEG_SPECS: list[IntentSpec] = [
    IntentSpec(name="affirmative", target="accept"),
    IntentSpec(name="negative", target="decline"),
]


# ---------------------------------------------------------------------------
# phrase_priority — phrases beat single words
# ---------------------------------------------------------------------------


class TestPhrasePriorityBeatsWord:
    @pytest.mark.asyncio
    async def test_phrase_beats_word_tie_break(self) -> None:
        """``"no, looks good"`` — negative single-word + affirmative phrase.

        Without ``phrase_priority``, iteration order would resolve to
        ``affirmative`` (first spec wins on the single-word match);
        with ``phrase_priority``, the affirmative PHRASE wins
        regardless of iteration order.
        """
        classifier = KeywordIntentClassifier(
            vocabulary={
                "affirmative": frozenset({"yes"}),
                "negative": frozenset({"no"}),
            },
            phrases={
                "affirmative": frozenset({"looks good"}),
                "negative": frozenset({"not yet"}),
            },
            phrase_priority=True,
        )
        result = await classifier.classify(
            "no, looks good", _AFF_NEG_SPECS,
        )
        assert result.intent is not None
        assert result.intent.name == "affirmative"

    @pytest.mark.asyncio
    async def test_phrase_beats_word_reversed_iteration(self) -> None:
        """Reversed spec order — affirmative phrase still wins."""
        classifier = KeywordIntentClassifier(
            vocabulary={
                "affirmative": frozenset({"yes"}),
                "negative": frozenset({"no"}),
            },
            phrases={
                "affirmative": frozenset({"looks good"}),
                "negative": frozenset({"not yet"}),
            },
            phrase_priority=True,
        )
        specs = list(reversed(_AFF_NEG_SPECS))
        result = await classifier.classify("no, looks good", specs)
        assert result.intent is not None
        assert result.intent.name == "affirmative"


# ---------------------------------------------------------------------------
# phrase_priority — same-priority ambiguity returns None
# ---------------------------------------------------------------------------


class TestPhrasePriorityAmbiguity:
    @pytest.mark.asyncio
    async def test_same_priority_word_ambiguity_returns_none(self) -> None:
        """``"yes and no"`` — both word-level matches → ambiguous."""
        classifier = KeywordIntentClassifier(
            vocabulary={
                "affirmative": frozenset({"yes"}),
                "negative": frozenset({"no"}),
            },
            phrases={
                "affirmative": frozenset(),
                "negative": frozenset(),
            },
            phrase_priority=True,
        )
        result = await classifier.classify("yes and no", _AFF_NEG_SPECS)
        assert result.intent is None

    @pytest.mark.asyncio
    async def test_same_priority_phrase_ambiguity_returns_none(self) -> None:
        """Two phrase matches across intents → ambiguous."""
        classifier = KeywordIntentClassifier(
            vocabulary={
                "affirmative": frozenset(),
                "negative": frozenset(),
            },
            phrases={
                "affirmative": frozenset({"looks good"}),
                "negative": frozenset({"not yet"}),
            },
            phrase_priority=True,
        )
        result = await classifier.classify(
            "looks good but not yet", _AFF_NEG_SPECS,
        )
        assert result.intent is None


# ---------------------------------------------------------------------------
# Default behaviour preserved when phrase_priority is off
# ---------------------------------------------------------------------------


class TestDefaultBehaviourPreserved:
    @pytest.mark.asyncio
    async def test_default_no_phrase_priority_preserves_first_match_wins(
        self,
    ) -> None:
        """Without ``phrase_priority=True``, iteration order wins."""
        classifier = KeywordIntentClassifier(
            vocabulary={
                "affirmative": frozenset({"yes"}),
                "negative": frozenset({"no"}),
            },
        )
        result = await classifier.classify("yes and no", _AFF_NEG_SPECS)
        assert result.intent is not None
        assert result.intent.name == "affirmative"

    @pytest.mark.asyncio
    async def test_default_phrases_kwarg_omitted_word_tier_only(
        self,
    ) -> None:
        """``phrases`` defaults to empty — phrase_priority falls
        straight through to word-tier handling.
        """
        classifier = KeywordIntentClassifier(
            vocabulary={
                "affirmative": frozenset({"confirm"}),
                "negative": frozenset({"cancel"}),
            },
            phrase_priority=True,
        )
        result = await classifier.classify("please confirm", _AFF_NEG_SPECS)
        assert result.intent is not None
        assert result.intent.name == "affirmative"
