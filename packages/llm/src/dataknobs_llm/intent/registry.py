# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Classifier-backend registry.

Mirrors the shape of
:data:`dataknobs_common.events.event_bus_backends`,
:data:`dataknobs_common.locks.lock_backends`, and
:data:`dataknobs_common.ratelimit.rate_limiter_backends` — all four are
:class:`~dataknobs_common.registry.PluginRegistry` instances with
domain-shaped not-found error text (``not_found_kind`` /
``not_found_exception``) so the consolidating shims keep their
historical error contracts. Consumers register their own classifier
backends (embedding-similarity, fuzzy-match, locale-specific keyword
variants) under a name and resolve them via
:func:`create_intent_classifier`.

Unlike the other three (which read the discriminator from a
``config["backend"]`` field), this registry uses the explicit-key mode
of :meth:`PluginRegistry.create` — :func:`create_intent_classifier`
passes ``key=name`` directly. No ``config_key`` is configured.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dataknobs_common.exceptions import _written_by_dataknobs
from dataknobs_common.registry import (
    PluginRegistry,
    create_surfacing,
    create_surfacing_async,
)
from dataknobs_llm.intent.composite import CompositeIntentClassifier
from dataknobs_llm.intent.keyword import KeywordIntentClassifier
from dataknobs_llm.intent.llm import LLMIntentClassifier
from dataknobs_llm.intent.protocol import (
    IntentClassifier,
    IntentClassifierFactory,
)

# Discriminator is the explicit ``name`` first arg to
# :func:`create_intent_classifier`; NOT a field in config. With no
# ``config_key`` configured, :meth:`PluginRegistry.create` routes via
# the explicit ``key=`` kwarg.
intent_classifier_backends: PluginRegistry[IntentClassifier] = PluginRegistry(
    name="intent_classifier_backends",
    validate_type=IntentClassifier,
    not_found_kind="intent_classifier",
    not_found_exception=ValueError,
)
"""Registry of named :data:`IntentClassifierFactory` callables.

Register a custom backend with
``intent_classifier_backends.register("name", factory)`` and select it
via ``create_intent_classifier("name", {...})``. The registry conforms
to :class:`~dataknobs_common.registry.BackendRegistry` for ``isinstance``
checks.
"""


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
        ValueError: If ``name`` is not registered (the message lists
            every currently registered backend so the typo is
            self-diagnosing), or if a backend factory refuses its config.
            The historical consumer contract is that a configuration
            problem arrives as ``ValueError``, and it still does: dataknobs'
            own refusals (a composite child spec that is not a mapping or
            names no ``classifier:``, a child name not registered, a
            composite with no children) arrive as themselves, with their
            words. A ``ValueError`` a consumer's factory raised with its own
            words arrives as a bounded
            :class:`~dataknobs_common.exceptions.FactoryValueError` --
            still a ``ValueError`` -- naming the classifier and the error's
            type, because a factory's text can carry what it was handed.
            The factory's error is on ``__cause__``.
        OperationError: If a factory fails some other way, bounded the same
            way. A factory's ``ResourceError``, ``TimeoutError``,
            ``ConcurrencyError``, ``SerializationError`` or
            ``ConfigurationError`` arrives instead as a bounded error of that
            family. The originating exception is on ``__cause__``.
    """
    return create_surfacing(
        intent_classifier_backends,
        subject="intent classifier",
        key=name,
        config=dict(config) if config else {},
    )


async def create_intent_classifier_async(
    name: str,
    config: Mapping[str, Any] | None = None,
) -> IntentClassifier:
    """Async-symmetric counterpart to :func:`create_intent_classifier`.

    For classifier backends whose construction is asynchronous
    (embedding-model warm-up, network-bound vocabulary loading, …).
    Every built-in classifier constructs synchronously, so this
    function returns the same instance type as
    :func:`create_intent_classifier`; the surface is shipped for API
    symmetry and consumer-extensibility (an out-of-tree classifier's
    ``from_config_async`` is detected and awaited via
    :meth:`PluginRegistry.create_async`).

    Args:
        name: Registered backend name.
        config: Backend-specific config dict forwarded to the factory.

    Returns:
        The constructed :class:`IntentClassifier` instance.

    Raises:
        ValueError: Same as :func:`create_intent_classifier`.
        OperationError: Same as :func:`create_intent_classifier`.
    """
    return await create_surfacing_async(
        intent_classifier_backends,
        subject="intent classifier",
        key=name,
        config=dict(config) if config else {},
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

    Each child spec MUST declare ``classifier:`` (or its legacy
    alias ``name:``). Missing the discriminator raises ``ValueError``
    with the offending spec — silently dropping malformed entries
    would surface later as a misleading "requires at least one inner
    classifier" error pointing at the wrong root cause (a typo in
    ``classifier:`` would have rejected the entry without saying so).

    Sync-recursion limitation: child specs are resolved through the
    sync :func:`create_intent_classifier` unconditionally, even when
    this factory is invoked through :func:`create_intent_classifier_async`.
    Every built-in classifier (``"keyword"``, ``"llm"``, ``"composite"``)
    constructs synchronously, so the limitation is invisible today; an
    out-of-tree backend that requires the async construction path
    (factory exposing ``from_config_async``, or factory returning an
    awaitable) is NOT supported as a composite child — the sync call
    would discard the awaitable and produce an unawaited coroutine in
    place of the classifier. A consumer hitting this should wrap the
    async-constructed instance and pass it through a sync factory, or
    file an issue requesting structural async-aware composite recursion.
    """
    child_specs = config.get("classifiers", [])
    children: list[IntentClassifier] = []
    for spec in child_specs:
        if not isinstance(spec, Mapping):
            raise _written_by_dataknobs(
                ValueError(
                    f"composite classifier child spec must be a mapping, "
                    f"got {type(spec).__name__}: {spec!r}",
                )
            )
        name = spec.get("classifier") or spec.get("name")
        if name is None:
            raise _written_by_dataknobs(
                ValueError(
                    f"composite classifier child spec missing required "
                    f"'classifier' field (or legacy 'name'): {dict(spec)!r}",
                )
            )
        # Recursion preserved unchanged — uses the (now consolidated)
        # :func:`create_intent_classifier` shim.
        children.append(create_intent_classifier(name, spec.get("config", {})))
    return CompositeIntentClassifier(
        classifiers=children,
        strategy=config.get("strategy", "first_match"),
    )


intent_classifier_backends.register("keyword", _keyword_factory)
intent_classifier_backends.register("llm", _llm_factory)
intent_classifier_backends.register("composite", _composite_factory)


__all__ = [
    "IntentClassifierFactory",
    "create_intent_classifier",
    "create_intent_classifier_async",
    "intent_classifier_backends",
]
