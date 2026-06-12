"""Keyword-based intent classifier.

Consumes :class:`IntentSpec` sequences and dispatches through an
injectable ``tokenizer`` (default: word-boundary regex). The
``vocabulary`` mapping supplies per-intent default keywords when an
:class:`IntentSpec` has no ``keywords`` override.

This is the single keyword-classification primitive consumed by:

* The new public :class:`IntentClassifier` API.
* The wizard's ``intent_detection: {classifier: keyword, ...}`` block
  (after the v3 re-flooring).
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from dataknobs_bots.intent.defaults import (
    DEFAULT_VOCABULARY,
    default_word_boundary_tokenizer,
)
from dataknobs_bots.intent.protocol import (
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
    """

    def __init__(
        self,
        *,
        vocabulary: Mapping[str, frozenset[str]] | None = None,
        tokenizer: Callable[[str, str], bool] | None = None,
    ) -> None:
        self._vocabulary = (
            vocabulary if vocabulary is not None else DEFAULT_VOCABULARY
        )
        self._tokenizer = tokenizer or default_word_boundary_tokenizer

    async def classify(
        self,
        message: str,
        intents: Sequence[IntentSpec],
        **_: Any,
    ) -> IntentMatchResult:
        lower = message.lower()
        for spec in intents:
            keywords = (
                list(spec.keywords) if spec.keywords
                else list(self._vocabulary.get(spec.name, ()))
            )
            if any(self._tokenizer(kw.lower(), lower) for kw in keywords):
                return IntentMatchResult(
                    intent=spec, extracted=None,
                    rule_based=True, raw_reply=message,
                )
        return IntentMatchResult(
            intent=None, extracted=None,
            rule_based=False, raw_reply=message,
        )
