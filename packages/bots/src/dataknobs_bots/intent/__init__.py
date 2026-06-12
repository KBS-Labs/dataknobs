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

Wizard-agnostic — no wizard dependencies in the protocol. The
:mod:`dataknobs_bots.reasoning` package consumes the protocol via
the synthesized ``intent_detection:`` block produced by the
``intent_confirm:`` primitive.

**Lift trajectory.** This protocol is the wizard-domain narrowing
of a broader ``Classifier[InputT, LabelT]`` concept. When that
generic classification surface lands, this module becomes a thin
intent-domain adapter layer (``IntentDispatcher`` wraps a
``Classifier[str, str]``) over ``dataknobs_common.classification``
primitives (with LLM and embedding backends in
``dataknobs_llm.classification`` and external adapters — sklearn,
HuggingFace, spaCy — in a future ``dataknobs_ml.classification``
package). Forward-compat seeds (a ``confidence`` field on
:class:`IntentMatchResult`, this docstring, and the registry
docstring) make the lift mechanical: rename + adapter pass, no
behavior changes.
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
from dataknobs_bots.intent.registry import intent_classifier_backends

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
    "default_word_boundary_tokenizer",
    "intent_classifier_backends",
]
