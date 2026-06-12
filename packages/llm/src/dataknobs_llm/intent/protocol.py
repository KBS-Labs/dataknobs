"""IntentClassifier protocol and shared data types."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class IntentSpec:
    """One intent declaration.

    Attributes:
        name: Intent identifier. For wizard ``intent_confirm:``
            usage, written into ``state.data[name] = True`` on match.
        target: Routing destination (interpretation depends on the
            consumer — wizard uses stage name; a tool router might use
            tool name; etc.).
        keywords: Optional override vocabulary. If absent, the
            classifier falls back to its configured default
            vocabulary keyed by ``name``.
        extract: Optional field name; when set, the matched payload
            (LLM / embedding-classifier output) is written into the
            consumer's target state under this key.
        llm_fallback: Legacy compatibility flag — preserved for the
            v2 → v3 transition. New code should compose
            classifiers explicitly via
            :class:`CompositeIntentClassifier`.
    """
    name: str
    target: str
    keywords: tuple[str, ...] | None = None
    extract: str | None = None
    llm_fallback: bool = False


@dataclass(frozen=True)
class IntentMatchResult:
    """Result of classifying a user reply.

    Attributes:
        intent: Matched IntentSpec, or ``None``.
        extracted: For an extract-bearing intent, the captured string
            (LLM-fallback only today); ``None`` otherwise.
        rule_based: ``True`` if matched via a rule-based classifier
            (keyword, embedding-similarity above threshold);
            ``False`` if matched via LLM classification.
        raw_reply: Preserved user reply for audit/provenance.
        confidence: Calibrated ``[0.0, 1.0]`` confidence in the match,
            or ``None`` when the underlying classifier doesn't expose
            calibrated confidence. The built-in classifiers (keyword,
            JSON-output LLM, composite, negation-filter) all return
            ``None``; calibrated-confidence backends such as
            embedding-similarity, sklearn, or structured-output LLM
            adapters populate it. The field is part of the dataclass
            shape so that consumers reading ``result.confidence`` see
            a uniform schema across backend kinds.
    """
    intent: IntentSpec | None
    extracted: str | None
    rule_based: bool
    raw_reply: str
    confidence: float | None = None


@runtime_checkable
class IntentClassifier(Protocol):
    """Protocol for an intent classifier.

    A classifier inspects a user message against a sequence of
    :class:`IntentSpec` declarations and returns an
    :class:`IntentMatchResult`.

    Implementations may be deterministic (keyword matching,
    embedding-similarity above threshold), LLM-based, or compositions
    of either. Built-in implementations live in this package
    (:class:`KeywordIntentClassifier`, :class:`LLMIntentClassifier`,
    :class:`CompositeIntentClassifier`, :class:`NegationFilter`);
    consumers register their own via
    :data:`intent_classifier_backends`.

    The protocol is intentionally minimal: one async ``classify``
    method. Per-classifier configuration (vocabulary, LLM provider,
    tokenizer, etc.) lives on the implementation, injected at
    construction time.
    """

    async def classify(
        self,
        message: str,
        intents: Sequence[IntentSpec],
        **kwargs: Any,
    ) -> IntentMatchResult:
        """Classify ``message`` against ``intents``.

        Args:
            message: Raw user message.
            intents: Sequence of IntentSpec declarations.
            **kwargs: Implementation-specific keyword arguments (e.g.
                LLM-based implementations may accept ``llm`` if not
                already injected at construction time).

        Returns:
            IntentMatchResult with ``intent=None`` on no match.
        """
        ...


# Factory signature for registry registration.
IntentClassifierFactory = Callable[[dict[str, Any]], IntentClassifier]
