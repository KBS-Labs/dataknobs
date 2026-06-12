"""Shared defaults: vocabulary, tokenizer, LLM prompt template.

The defaults compose existing dataknobs primitives:

* :func:`dataknobs_bots.reasoning.wizard_utils.word_in_text` — the
  word-boundary regex helper used as the default tokenizer.
* :data:`dataknobs_bots.reasoning.wizard_types._DEFAULT_AFFIRMATIVE_SIGNALS` /
  ``_DEFAULT_NEGATIVE_SIGNALS`` — canonical English yes/no vocab,
  shared with the extraction layer's boolean-recovery code.
* :data:`dataknobs_llm.extraction.grounding.DEFAULT_NEGATION_KEYWORDS` —
  the negation set used by ``has_negation`` and reused as the
  default for :class:`NegationFilter`.
"""
from __future__ import annotations

from dataknobs_bots.reasoning.wizard_types import (
    _DEFAULT_AFFIRMATIVE_SIGNALS,
    _DEFAULT_NEGATIVE_SIGNALS,
)
from dataknobs_bots.reasoning.wizard_utils import word_in_text
from dataknobs_llm.extraction.grounding import (
    DEFAULT_NEGATION_KEYWORDS as _DEFAULT_NEGATION_KEYWORDS,
)

# Re-export so consumers can import from one place.
DEFAULT_NEGATION_KEYWORDS = _DEFAULT_NEGATION_KEYWORDS


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
    "accept": _DEFAULT_AFFIRMATIVE_SIGNALS | _INTENT_ACCEPT_PHRASES,
    "decline": _DEFAULT_NEGATIVE_SIGNALS | _INTENT_DECLINE_PHRASES,
    "unclear": _INTENT_UNCLEAR_PHRASES,
}


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

    Case-insensitive word-boundary match via the existing
    :func:`wizard_utils.word_in_text` helper. Bare-token keywords
    (``"yes"``) match standalone tokens but NOT substrings of other
    words (``"yesterday"``).

    Consumer-supplied tokenizers (I18N / fuzzy / N-gram / morphological)
    follow this signature: ``(keyword: str, message: str) -> bool``.
    Both arguments are passed pre-lowercased.
    """
    return word_in_text(keyword, message)
