"""Keyword-based intent classifier.

Consumes :class:`IntentSpec` sequences and dispatches through an
injectable ``tokenizer`` (default: word-boundary regex). The
``vocabulary`` mapping supplies per-intent default keywords when an
:class:`IntentSpec` has no ``keywords`` override.

This is the single keyword-classification primitive registered as
the ``"keyword"`` backend in :data:`intent_classifier_backends`. It
also backs the two-intent affirmative/negative shape exposed via
:func:`dataknobs_llm.extraction.grounding.detect_boolean_signal`
(through the ``phrase_priority`` opt-in below), so a single
keyword-iteration loop covers both call paths.

The classification work is pure-CPU; :meth:`classify` is a thin async
wrapper over the private :meth:`_classify_sync` core. Same-package
synchronous callers can dispatch through the sync core directly
without an event-loop bridge.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from dataknobs_llm.intent.defaults import (
    DEFAULT_VOCABULARY,
    default_word_boundary_tokenizer,
)
from dataknobs_llm.intent.protocol import (
    IntentClassifier,
    IntentMatchResult,
    IntentSpec,
)


class KeywordIntentClassifier(IntentClassifier):
    """Rule-based keyword classifier.

    Args:
        vocabulary: Per-intent default keyword sets, keyed by intent
            name. Used as fallback when an :class:`IntentSpec` has no
            per-intent ``keywords`` override. Defaults to
            :data:`DEFAULT_VOCABULARY`.
        tokenizer: Callable ``(keyword, message) -> bool`` determining
            whether a keyword matches the message. Both arguments are
            passed pre-lowercased. Defaults to
            :func:`default_word_boundary_tokenizer` (word-boundary
            regex). Inject custom tokenizers for I18N / fuzzy / N-gram
            / morphological matching without forking dataknobs.
        phrase_priority: When ``True``, intents whose ``phrases`` entry
            matches take precedence over intents whose only match is
            in ``vocabulary``. Two intents tying at the same tier
            (both phrase-matched or both word-matched only) resolve to
            ``IntentMatchResult(intent=None, ...)`` rather than
            picking by iteration order — the caller decides what
            "ambiguous" means in its domain.
            :func:`dataknobs_llm.extraction.grounding.detect_boolean_signal`
            opts in so multi-word affirmative/negative phrases beat
            single-word matches. Default ``False`` keeps the
            first-match-wins iteration semantic.
        phrases: Per-intent PHRASE sets (multi-word strings),
            consulted ONLY when ``phrase_priority=True``. ``None`` is
            treated as "no phrases known" (every spec falls through to
            the word tier). Keep ``vocabulary`` (single words) and
            ``phrases`` (multi-word strings) disjoint per intent; an
            entry that appears in both is treated as a phrase match
            for tie-break purposes (the phrase tier wins).
    """

    def __init__(
        self,
        *,
        vocabulary: Mapping[str, frozenset[str]] | None = None,
        tokenizer: Callable[[str, str], bool] | None = None,
        phrase_priority: bool = False,
        phrases: Mapping[str, frozenset[str]] | None = None,
    ) -> None:
        self._vocabulary = (
            vocabulary if vocabulary is not None else DEFAULT_VOCABULARY
        )
        self._tokenizer = tokenizer or default_word_boundary_tokenizer
        self._phrase_priority = phrase_priority
        self._phrases: Mapping[str, frozenset[str]] = phrases or {}

    async def classify(
        self,
        message: str,
        intents: Sequence[IntentSpec],
        **_: Any,
    ) -> IntentMatchResult:
        return self._classify_sync(message, intents)

    # ------------------------------------------------------------------
    # Synchronous core — callable from both async and sync contexts.
    # Keyword classification is pure-CPU; the protocol's async outer
    # skin exists only because LLM-backed classifiers need it.
    # ------------------------------------------------------------------

    def _classify_sync(
        self,
        message: str,
        intents: Sequence[IntentSpec],
    ) -> IntentMatchResult:
        lower = message.lower()
        if self._phrase_priority:
            return self._classify_phrase_priority_sync(
                lower, intents, message,
            )
        return self._classify_first_match_sync(lower, intents, message)

    def _classify_first_match_sync(
        self,
        lower: str,
        intents: Sequence[IntentSpec],
        raw_message: str,
    ) -> IntentMatchResult:
        for spec in intents:
            keywords = self._keywords_for(spec)
            if any(self._tokenizer(kw.lower(), lower) for kw in keywords):
                return IntentMatchResult(
                    intent=spec, extracted=None,
                    rule_based=True, raw_reply=raw_message,
                )
        return IntentMatchResult(
            intent=None, extracted=None,
            rule_based=False, raw_reply=raw_message,
        )

    def _classify_phrase_priority_sync(
        self,
        lower: str,
        intents: Sequence[IntentSpec],
        raw_message: str,
    ) -> IntentMatchResult:
        phrase_matches: list[IntentSpec] = []
        word_matches: list[IntentSpec] = []
        for spec in intents:
            spec_phrases = self._phrases.get(spec.name, frozenset())
            if any(self._tokenizer(p.lower(), lower) for p in spec_phrases):
                phrase_matches.append(spec)
            spec_words = self._keywords_for(spec)
            if any(self._tokenizer(kw.lower(), lower) for kw in spec_words):
                word_matches.append(spec)

        # Phrase tier wins outright when unique.
        if len(phrase_matches) == 1:
            return IntentMatchResult(
                intent=phrase_matches[0], extracted=None,
                rule_based=True, raw_reply=raw_message,
            )
        if len(phrase_matches) > 1:
            # Same-priority ambiguity at the phrase tier.
            return IntentMatchResult(
                intent=None, extracted=None,
                rule_based=False, raw_reply=raw_message,
            )

        # No phrase match — fall back to the word tier, but exclude
        # specs that ONLY matched at the phrase tier (impossible here
        # because we already returned above; kept explicit so the
        # word-tier branch never sees a phrase-match-only spec).
        only_word_matches = [
            spec for spec in word_matches if spec not in phrase_matches
        ]
        if len(only_word_matches) == 1:
            return IntentMatchResult(
                intent=only_word_matches[0], extracted=None,
                rule_based=True, raw_reply=raw_message,
            )
        if len(only_word_matches) > 1:
            # Same-priority ambiguity at the word tier.
            return IntentMatchResult(
                intent=None, extracted=None,
                rule_based=False, raw_reply=raw_message,
            )

        return IntentMatchResult(
            intent=None, extracted=None,
            rule_based=False, raw_reply=raw_message,
        )

    def _keywords_for(self, spec: IntentSpec) -> list[str]:
        return (
            list(spec.keywords) if spec.keywords
            else list(self._vocabulary.get(spec.name, ()))
        )
