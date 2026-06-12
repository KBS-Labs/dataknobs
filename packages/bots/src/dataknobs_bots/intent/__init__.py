"""Intent classification — pluggable IntentClassifier protocol.

Forward-looking intent classification for any consumer that needs
to route user input by intent — wizard advisor flows, tool routers,
memory query classifiers, etc.

The package ships a small protocol (:class:`IntentClassifier`) plus
four built-in implementations covering the common cases:

* :class:`KeywordIntentClassifier` — vocabulary + tokenizer
  injectable (default: word-boundary regex). Replace ``tokenizer``
  for I18N / fuzzy / N-gram variants.
* :class:`LLMIntentClassifier` — LLM provider + prompt template
  injectable. Use standalone for LLM-only classification.
* :class:`CompositeIntentClassifier` — chain classifiers
  (``first_match`` short-circuit, ``vote`` ensemble). The v3 default
  "rule-first, optional LLM fallback" behaviour is just
  ``CompositeIntentClassifier([KeywordIntentClassifier(),
  LLMIntentClassifier()])`` — no special class needed.
* :class:`NegationFilter` — decorator over any classifier that
  composes the existing ``has_negation`` helper from
  ``dataknobs_llm.extraction.grounding``. Wrap any classifier to
  suppress matches when a negation pattern is detected (closes the
  ``"no, I don't want to accept"`` foot-gun).

:data:`intent_classifier_backends` is a
:class:`Registry[IntentClassifierFactory]` consumers register their
own backends into (``embedding``, ``fuzzy_match``, locale-specific
keyword variants). The wizard's ``intent_detection: {classifier:
<name>, ...}`` block dispatches through it.

Wizard-agnostic: the protocol, defaults, and registry have no
dependencies on the :mod:`dataknobs_bots.reasoning` (wizard) layer.
The wizard consumes intent classification via the synthesized
``intent_detection:`` block produced by the ``intent_confirm:``
primitive, but any other reasoning strategy — ReAct router, tool
router, memory query classifier — can use this module directly.

The protocol is intentionally narrow (one async ``classify`` method
returning :class:`IntentMatchResult`). The ``confidence`` field on
the result is populated by calibrated-confidence backends
(embedding-similarity, structured-output LLM, sklearn adapters) and
left ``None`` by the built-in keyword / JSON-output LLM / composite /
negation-filter classifiers.
"""
from __future__ import annotations

from dataknobs_bots.intent.composite import CompositeIntentClassifier
from dataknobs_bots.intent.defaults import (
    DEFAULT_LLM_PROMPT_TEMPLATE,
    DEFAULT_NEGATION_KEYWORDS,
    DEFAULT_VOCABULARY,
    default_word_boundary_tokenizer,
)
from dataknobs_bots.intent.keyword import KeywordIntentClassifier
from dataknobs_bots.intent.llm import LLMIntentClassifier
from dataknobs_bots.intent.negation import NegationFilter
from dataknobs_bots.intent.protocol import (
    IntentClassifier,
    IntentClassifierFactory,
    IntentMatchResult,
    IntentSpec,
)
from dataknobs_bots.intent.registry import (
    create_intent_classifier,
    intent_classifier_backends,
)

__all__ = [
    "DEFAULT_LLM_PROMPT_TEMPLATE",
    "DEFAULT_NEGATION_KEYWORDS",
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
    "default_word_boundary_tokenizer",
    "intent_classifier_backends",
]
