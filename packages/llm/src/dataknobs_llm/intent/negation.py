"""NegationFilter decorator.

Wraps any :class:`IntentClassifier` and composes the existing
:func:`dataknobs_llm.extraction.grounding.has_negation` helper to
suppress matches when a negation pattern is detected.

Closes the foot-gun where ``"no, I don't want to accept that"``
substring-matches the ``accept`` vocab without negation context.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from dataknobs_llm.extraction.grounding import has_negation
from dataknobs_llm.intent.defaults import DEFAULT_NEGATION_KEYWORDS
from dataknobs_llm.intent.protocol import (
    IntentClassifier,
    IntentMatchResult,
    IntentSpec,
)

logger = logging.getLogger(__name__)


class NegationFilter(IntentClassifier):
    """Decorator: wrap any classifier; suppress matches when negation
    is detected in the user message.

    Args:
        inner: Wrapped classifier.
        negation_keywords: Override the negation keyword set (defaults
            to :data:`DEFAULT_NEGATION_KEYWORDS` from
            ``dataknobs_llm.extraction.grounding``).
        suppress_intents: Optional whitelist of intent names to filter.
            ``None`` (default) suppresses ALL matches when negation is
            detected; consumers can scope it to e.g. only the
            ``"accept"`` intent if they want the negation filter to
            stay out of the way for ``"decline"`` matches.
    """

    def __init__(
        self,
        inner: IntentClassifier,
        *,
        negation_keywords: frozenset[str] | None = None,
        suppress_intents: frozenset[str] | None = None,
    ) -> None:
        self._inner = inner
        self._negation_keywords = (
            negation_keywords if negation_keywords is not None
            else DEFAULT_NEGATION_KEYWORDS
        )
        self._suppress_intents = suppress_intents

    async def classify(
        self,
        message: str,
        intents: Sequence[IntentSpec],
        **kwargs: Any,
    ) -> IntentMatchResult:
        result = await self._inner.classify(message, intents, **kwargs)
        if result.intent is None:
            return result
        if (
            self._suppress_intents is not None
            and result.intent.name not in self._suppress_intents
        ):
            return result
        if has_negation(message.lower(), self._negation_keywords):
            logger.debug(
                "NegationFilter suppressed intent '%s' for message '%s'",
                result.intent.name, message,
            )
            # ``rule_based`` describes "how did we MATCH this intent" —
            # when the result is no-match, the field is semantically
            # vacuous. Set False rather than inheriting the inner
            # result's flag so a downstream observer doesn't see
            # ``intent=None, rule_based=True`` (a contradictory pair).
            return IntentMatchResult(
                intent=None, extracted=None,
                rule_based=False, raw_reply=message,
            )
        return result
