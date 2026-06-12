"""Composite intent classifier — chains multiple backends.

Two strategies:

* ``"first_match"`` — try each classifier in order; return the first
  non-None match. Reproduces the v2 default "rule-first, optional
  LLM fallback" behaviour:
  ``CompositeIntentClassifier([KeywordIntentClassifier(),
  LLMIntentClassifier()], strategy="first_match")``.
* ``"vote"`` — query all classifiers, return the intent with the
  most votes (ties broken by classifier order). Useful for
  ensemble classification (consumer use cases: critical decisions
  needing redundancy).

Consumer-supplied combine strategies are out of scope here —
follow-up 163-FU8 captures the registry shape for that extension.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any, Literal

from dataknobs_bots.intent.protocol import (
    IntentClassifier,
    IntentMatchResult,
    IntentSpec,
)


class CompositeIntentClassifier(IntentClassifier):
    def __init__(
        self,
        classifiers: list[IntentClassifier],
        *,
        strategy: Literal["first_match", "vote"] = "first_match",
    ) -> None:
        if not classifiers:
            raise ValueError(
                "CompositeIntentClassifier requires at least one "
                "inner classifier",
            )
        self._classifiers = list(classifiers)
        self._strategy = strategy

    async def classify(
        self,
        message: str,
        intents: Sequence[IntentSpec],
        **kwargs: Any,
    ) -> IntentMatchResult:
        if self._strategy == "first_match":
            for clf in self._classifiers:
                result = await clf.classify(message, intents, **kwargs)
                if result.intent is not None:
                    return result
            return IntentMatchResult(
                intent=None, extracted=None,
                rule_based=False, raw_reply=message,
            )

        # "vote" strategy
        votes: Counter[str] = Counter()
        first_match: dict[str, IntentMatchResult] = {}
        for clf in self._classifiers:
            result = await clf.classify(message, intents, **kwargs)
            if result.intent is not None:
                votes[result.intent.name] += 1
                first_match.setdefault(result.intent.name, result)
        if not votes:
            return IntentMatchResult(
                intent=None, extracted=None,
                rule_based=False, raw_reply=message,
            )
        winner, _ = votes.most_common(1)[0]
        return first_match[winner]
