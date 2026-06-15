"""Intent classification — pluggable IntentClassifier protocol.

Forward-looking intent classification for any LLM-layer consumer
that needs to route user input by intent — tool routers, reasoning
strategies, RAG query classifiers, downstream packages such as
``dataknobs-bots`` (wizard advisor flows), etc.

The package ships a small protocol (:class:`IntentClassifier`) plus
four built-in implementations covering the common cases:

* :class:`KeywordIntentClassifier` — vocabulary + tokenizer
  injectable (default: word-boundary regex). Replace ``tokenizer``
  for I18N / fuzzy / N-gram variants.
* :class:`LLMIntentClassifier` — LLM provider + prompt template
  injectable. Use standalone for LLM-only classification.
* :class:`CompositeIntentClassifier` — chain classifiers
  (``first_match`` short-circuit, ``vote`` ensemble). The default
  "rule-first, optional LLM fallback" behaviour is just
  ``CompositeIntentClassifier([KeywordIntentClassifier(),
  LLMIntentClassifier()])`` — no special class needed.
* :class:`NegationFilter` — decorator over any classifier that
  composes the existing ``has_negation`` helper from
  ``dataknobs_llm.extraction.grounding``. Wrap any classifier to
  suppress matches when a negation pattern is detected (closes the
  ``"no, I don't want to accept"`` foot-gun).

:data:`intent_classifier_backends` is a
:class:`~dataknobs_common.registry.PluginRegistry` of
:data:`IntentClassifierFactory` callables that consumers register their
own backends into (``embedding``, ``fuzzy_match``, locale-specific
keyword variants). Consumers dispatch through it via
:func:`create_intent_classifier` (or
:func:`create_intent_classifier_async` for backends whose construction
is asynchronous).

Consumer-agnostic: the protocol, defaults, and registry have no
dependencies on any consumer layer. Any LLM-layer reasoning
strategy, tool router, memory query classifier — or downstream
packages — can use this module directly.

The protocol is intentionally narrow (one async ``classify`` method
returning :class:`IntentMatchResult`). The ``confidence`` field on
the result is populated by calibrated-confidence backends
(embedding-similarity, structured-output LLM, sklearn adapters) and
left ``None`` by the built-in keyword / JSON-output LLM / composite /
negation-filter classifiers.
"""
from __future__ import annotations

from dataknobs_llm.intent.composite import CompositeIntentClassifier
from dataknobs_llm.intent.defaults import (
    DEFAULT_AFFIRMATIVE_SIGNALS,
    DEFAULT_LLM_PROMPT_TEMPLATE,
    DEFAULT_NEGATION_KEYWORDS,
    DEFAULT_NEGATIVE_SIGNALS,
    DEFAULT_VOCABULARY,
    default_word_boundary_tokenizer,
    word_in_text,
)
from dataknobs_llm.intent.keyword import KeywordIntentClassifier
from dataknobs_llm.intent.llm import LLMIntentClassifier
from dataknobs_llm.intent.negation import NegationFilter
from dataknobs_llm.intent.protocol import (
    IntentClassifier,
    IntentClassifierFactory,
    IntentMatchResult,
    IntentSpec,
)
from dataknobs_llm.intent.registry import (
    create_intent_classifier,
    create_intent_classifier_async,
    intent_classifier_backends,
)

__all__ = [
    "DEFAULT_AFFIRMATIVE_SIGNALS",
    "DEFAULT_LLM_PROMPT_TEMPLATE",
    "DEFAULT_NEGATION_KEYWORDS",
    "DEFAULT_NEGATIVE_SIGNALS",
    "DEFAULT_VOCABULARY",
    "CompositeIntentClassifier",
    "IntentClassifier",
    "IntentClassifierFactory",
    "IntentMatchResult",
    "IntentSpec",
    "KeywordIntentClassifier",
    "LLMIntentClassifier",
    "NegationFilter",
    "create_intent_classifier",
    "create_intent_classifier_async",
    "default_word_boundary_tokenizer",
    "intent_classifier_backends",
    "word_in_text",
]
