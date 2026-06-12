"""Shared defaults: vocabulary, tokenizer, LLM prompt template.

This module is the canonical home for the small text-classification
primitives used by:

* :data:`DEFAULT_AFFIRMATIVE_SIGNALS` / :data:`DEFAULT_NEGATIVE_SIGNALS`
  — single-token English yes/no vocabulary, used as the
  :class:`KeywordIntentClassifier` default vocabulary and importable
  by any consumer that needs the same primitives for boolean
  recovery or analogous text-classification tasks.
* :func:`word_in_text` — case-insensitive word-boundary regex helper.
* :data:`DEFAULT_NEGATION_KEYWORDS` — re-export of the canonical
  negation set used by :class:`NegationFilter`.

This module is a leaf within :mod:`dataknobs_llm.intent`: it has no
upward dependencies on any consumer layer. LLM-layer reasoning
strategies, tool routers, and downstream packages (the wizard in
``dataknobs-bots``, for example) consume these from here.
"""
from __future__ import annotations

import re

from dataknobs_llm.extraction.grounding import (
    DEFAULT_NEGATION_KEYWORDS as _DEFAULT_NEGATION_KEYWORDS,
)

# Re-export so consumers can import from one place.
DEFAULT_NEGATION_KEYWORDS = _DEFAULT_NEGATION_KEYWORDS


# ── Single-token English yes/no signals ──────────────────────────────
#
# Canonical home for the public vocab. The keyword intent classifier
# consumes these by name; consumers needing the same primitives for
# boolean recovery (or analogous text-classification tasks) import
# them from here directly.

DEFAULT_AFFIRMATIVE_SIGNALS: frozenset[str] = frozenset({
    "yes", "confirm", "save", "approve", "correct", "sure",
    "ok", "okay", "agreed", "accept", "absolutely", "definitely",
    "yep", "yeah",
})

DEFAULT_NEGATIVE_SIGNALS: frozenset[str] = frozenset({
    "no", "wait", "stop", "cancel", "wrong", "redo", "nope",
    "nah", "incorrect",
})


# ── Word-boundary tokenizer ───────────────────────────────────────────


def word_in_text(word: str, text: str) -> bool:
    r"""Case-insensitive word-boundary membership check.

    Uses ``\b`` regex anchors to avoid substring false positives
    (e.g. ``"base"`` matching ``"database"``). Supports multi-word
    phrases (``word_in_text("study companion", "I want a study
    companion bot")`` returns ``True``).

    Both arguments may be in any case. Used as the default tokenizer
    for :class:`KeywordIntentClassifier` and re-exported from
    ``wizard_utils`` for the wizard's extraction layer.
    """
    return bool(re.search(r"\b" + re.escape(word) + r"\b", text))


# ── Intent vocabulary ────────────────────────────────────────────────


_INTENT_ACCEPT_PHRASES: frozenset[str] = frozenset({
    "do it", "go ahead", "sounds good", "that works",
    "please proceed", "let's go", "alright",
})
_INTENT_DECLINE_PHRASES: frozenset[str] = frozenset({
    "not really", "no thanks", "don't", "skip that",
})
_INTENT_UNCLEAR_PHRASES: frozenset[str] = frozenset({
    "not sure", "maybe", "i don't know",
})


DEFAULT_VOCABULARY: dict[str, frozenset[str]] = {
    "accept": DEFAULT_AFFIRMATIVE_SIGNALS | _INTENT_ACCEPT_PHRASES,
    "decline": DEFAULT_NEGATIVE_SIGNALS | _INTENT_DECLINE_PHRASES,
    "unclear": _INTENT_UNCLEAR_PHRASES,
}


# ── LLM prompt template ──────────────────────────────────────────────


DEFAULT_LLM_PROMPT_TEMPLATE = (
    "Classify the user's reply into exactly one of these intents: "
    "{intent_list}, or null if none apply.\n\n"
    'Reply: "{message}"\n\n'
    'Output JSON only: '
    '{{"intent": <name|null>, "extracted": <string|null>}}. '
    "Set 'extracted' to the user-named alternative only for intents "
    "that capture one ({extract_intents}); otherwise null."
)


def default_word_boundary_tokenizer(keyword: str, message: str) -> bool:
    """Default tokenizer for :class:`KeywordIntentClassifier`.

    Case-insensitive word-boundary match via :func:`word_in_text`.
    Bare-token keywords (``"yes"``) match standalone tokens but NOT
    substrings of other words (``"yesterday"``).

    Consumer-supplied tokenizers (I18N / fuzzy / N-gram / morphological)
    follow this signature: ``(keyword: str, message: str) -> bool``.
    Both arguments are passed pre-lowercased.
    """
    return word_in_text(keyword, message)
