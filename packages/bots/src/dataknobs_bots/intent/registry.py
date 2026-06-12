"""Classifier-backend registry.

Mirrors the shape of
:data:`dataknobs_common.events.event_bus_backends` and
:data:`dataknobs_common.locks.lock_backends`. Consumers register
their own classifier backends (embedding-similarity, fuzzy-match,
locale-specific keyword variants) under a name; the wizard's
``intent_detection: {classifier: <name>, ...}`` block dispatches
through this registry.
"""
from __future__ import annotations

from collections.abc import Mapping
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


def create_intent_classifier(
    name: str,
    config: Mapping[str, Any] | None = None,
) -> IntentClassifier:
    """Build an :class:`IntentClassifier` from a registered backend.

    Looks up ``name`` in :data:`intent_classifier_backends` and calls
    the registered factory with ``config`` (defaults to empty dict).

    Args:
        name: Registered backend name (built-in: ``"keyword"``,
            ``"llm"``, ``"composite"``; consumer-registered names are
            also resolved here).
        config: Backend-specific config dict forwarded to the factory.

    Returns:
        The constructed :class:`IntentClassifier` instance.

    Raises:
        ValueError: If ``name`` is not registered. The error message
            lists every currently registered backend so the typo is
            self-diagnosing.
    """
    factory = intent_classifier_backends.get_optional(name)
    if factory is None:
        raise ValueError(
            f"Unknown intent_classifier '{name}' (registered: "
            f"{sorted(intent_classifier_backends.list_keys())})",
        )
    return factory(dict(config) if config else {})


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

    Each child spec MUST declare ``classifier:`` (or its legacy
    alias ``name:``). Missing the discriminator raises ``ValueError``
    with the offending spec — silently dropping malformed entries
    would surface later as a misleading "requires at least one inner
    classifier" error pointing at the wrong root cause (a typo in
    ``classifier:`` would have rejected the entry without saying so).
    """
    child_specs = config.get("classifiers", [])
    children: list[IntentClassifier] = []
    for spec in child_specs:
        if not isinstance(spec, Mapping):
            raise ValueError(
                f"composite classifier child spec must be a mapping, "
                f"got {type(spec).__name__}: {spec!r}",
            )
        name = spec.get("classifier") or spec.get("name")
        if name is None:
            raise ValueError(
                f"composite classifier child spec missing required "
                f"'classifier' field (or legacy 'name'): {dict(spec)!r}",
            )
        children.append(create_intent_classifier(name, spec.get("config", {})))
    return CompositeIntentClassifier(
        classifiers=children,
        strategy=config.get("strategy", "first_match"),
    )


intent_classifier_backends.register("keyword", _keyword_factory)
intent_classifier_backends.register("llm", _llm_factory)
intent_classifier_backends.register("composite", _composite_factory)
