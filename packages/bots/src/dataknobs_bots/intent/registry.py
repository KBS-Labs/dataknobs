"""Classifier-backend registry.

Mirrors the shape of
:data:`dataknobs_common.events.event_bus_backends` and
:data:`dataknobs_common.locks.lock_backends`. Consumers register
their own classifier backends (embedding-similarity, fuzzy-match,
locale-specific keyword variants) under a name; the wizard's
``intent_detection: {classifier: <name>, ...}`` block dispatches
through this registry.

**Lift trajectory (163-FU12).** This registry is the wizard-domain
projection of a planned ``classifier_backends`` registry in
``dataknobs_common.classification`` (generic
``Classifier[InputT, LabelT]`` factories — sklearn, HuggingFace,
spaCy adapters, plus the lifted LLM / embedding / keyword
backends). Consumer-registered intent backends here migrate via a
one-line rename + an ``IntentDispatcher`` wrap once FU12 lands.
"""
from __future__ import annotations

from typing import Any

from dataknobs_bots.intent.composite import CompositeIntentClassifier
from dataknobs_bots.intent.keyword import KeywordIntentClassifier
from dataknobs_bots.intent.llm import LLMIntentClassifier
from dataknobs_bots.intent.protocol import (
    IntentClassifier,
    IntentClassifierFactory,
)
from dataknobs_common.registry import Registry

intent_classifier_backends: Registry[IntentClassifierFactory] = Registry(
    name="intent_classifier_backends",
)


# ── Built-in factories ───────────────────────────────────────────────


def _keyword_factory(config: dict[str, Any]) -> IntentClassifier:
    return KeywordIntentClassifier(
        vocabulary=config.get("vocabulary"),
        tokenizer=config.get("tokenizer"),
    )


def _llm_factory(config: dict[str, Any]) -> IntentClassifier:
    return LLMIntentClassifier(
        llm=config.get("llm"),
        prompt_template=config.get("prompt_template"),
    )


def _composite_factory(config: dict[str, Any]) -> IntentClassifier:
    """Build a composite from a list of ``{classifier: <name>,
    config: {...}}`` entries.
    """
    child_specs = config.get("classifiers", [])
    children: list[IntentClassifier] = []
    for spec in child_specs:
        name = spec.get("classifier") or spec.get("name")
        if name is None:
            continue
        sub_factory = intent_classifier_backends.get_optional(name)
        if sub_factory is None:
            raise ValueError(
                f"Unknown intent_classifier_backend '{name}' "
                f"(registered: "
                f"{sorted(intent_classifier_backends.list_keys())})",
            )
        children.append(sub_factory(spec.get("config", {})))
    return CompositeIntentClassifier(
        classifiers=children,
        strategy=config.get("strategy", "first_match"),
    )


intent_classifier_backends.register("keyword", _keyword_factory)
intent_classifier_backends.register("llm", _llm_factory)
intent_classifier_backends.register("composite", _composite_factory)
