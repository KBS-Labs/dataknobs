# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Backend registry and config-driven construction for knowledge bases.

Provides a plugin registry that lets 3rd parties register custom
knowledge-base backends without modifying core DynaBot code, plus the
public ``create_knowledge_base_from_config`` factory that dispatches a
config dict to the registered backend. The built-in ``rag`` backend is
registered lazily at first access.

Usage::

    from dataknobs_bots.knowledge import (
        create_knowledge_base_from_config,
        register_knowledge_base_backend,
    )

    register_knowledge_base_backend("my_kb", MyKnowledgeBase)
    kb = await create_knowledge_base_from_config({"type": "my_kb", ...})

The discriminator key is ``type`` (default ``rag``).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from dataknobs_common.exceptions import NotFoundError
from dataknobs_common.registry import PluginFactory, PluginRegistry, create_surfacing_async
from dataknobs_common.structured_config import (
    SKIP_VALIDATION,
    ConfigClassResolution,
    StructuredConfig,
    config_registries,
)

from .base import KnowledgeBase

logger = logging.getLogger(__name__)

# A knowledge-base factory is either a KnowledgeBase subclass (with
# ``from_config`` / ``from_config_async``) or a callable with the same
# dispatch signature.
#: Derived from the registry's own alias rather than restated, so this
#: surface widens with the registry instead of drifting behind it. The
#: hand-written copies omitted the asynchronous arm the registry has
#: always accepted at runtime.
KnowledgeBaseFactory = PluginFactory[KnowledgeBase]


def _register_builtins(registry: PluginRegistry[KnowledgeBase]) -> None:
    """Register the built-in knowledge-base backends (lazy, on first access).

    Imports are deferred to avoid circular imports.
    """
    from .rag import RAGKnowledgeBase

    registry.register("rag", RAGKnowledgeBase)


# Module-level singleton — data-driven dispatch replaces the former
# inline ``if/elif`` on ``config["type"]``.
knowledge_base_backends: PluginRegistry[KnowledgeBase] = PluginRegistry(
    "knowledge_base_backends",
    validate_type=KnowledgeBase,
    canonicalize_keys=True,
    config_key="type",
    config_key_default="rag",
    on_first_access=_register_builtins,
)


def _resolve_knowledge_base_config_cls(
    raw: Mapping[str, Any],
) -> ConfigClassResolution:
    """Resolve a ``knowledge_base`` section's dict to its config class.

    The resolver registered for the ``"knowledge_base"`` binding in
    :data:`~dataknobs_common.structured_config.config_registries`, used by
    :meth:`StructuredConfig.validate
    <dataknobs_common.structured_config.StructuredConfig.validate>` to
    validate a raw ``knowledge_base`` section without constructing the
    knowledge base. Because the resolved ``RAGKnowledgeBaseConfig`` itself
    carries the ``vector_store`` binding, one
    ``DynaBotConfig.from_dict(raw).validate()`` descends into the nested
    vector-store section too (the base's recursion through dry-run-built
    children).

    Delegates to ``knowledge_base_backends`` — the same registry the
    construction path uses — by reading ``CONFIG_CLS`` off the registered
    backend class for the ``"type"`` discriminator (defaulting to ``"rag"``,
    the factory's own default). Returns ``None`` for an unknown type (→
    ``ConfigurationError``); returns :data:`SKIP_VALIDATION` for a
    registered bare-callable backend with no ``CONFIG_CLS`` (see the memory
    resolver for the rationale).
    """
    backend_type = raw.get("type", "rag")  # registry's own default
    factory = knowledge_base_backends.get_factory(backend_type)
    if factory is None:
        return None  # unknown type -> validate() raises ConfigurationError
    config_cls = getattr(factory, "CONFIG_CLS", None)
    if isinstance(config_cls, type) and issubclass(config_cls, StructuredConfig):
        return config_cls
    return SKIP_VALIDATION  # registered bare callable, no CONFIG_CLS -> skip


# Eager registration (mirroring the memory resolver / ``dataknobs-data``'s
# ``vector_store``). The package ``__init__`` imports this module, so
# ``import dataknobs_bots.knowledge`` fires it. ``config_registries`` is a
# plain ``Registry`` -> ``allow_overwrite`` (NOT ``override=``, which is the
# ``PluginRegistry`` param) keeps re-import idempotent.
config_registries.register(
    "knowledge_base", _resolve_knowledge_base_config_cls, allow_overwrite=True
)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


async def create_knowledge_base_from_config(config: dict[str, Any]) -> KnowledgeBase:
    """Create a knowledge base from configuration.

    Args:
        config: Knowledge base configuration with:
            - type: Type of knowledge base (default ``rag``)
            - vector_store: Vector store configuration
            - embedding_provider: LLM provider for embeddings
            - embedding_model: Model to use for embeddings
            - chunking: Optional chunking configuration
            - documents_path: Optional path to load documents
            - document_pattern: Optional file pattern

    Returns:
        Configured knowledge base instance.

    Raises:
        ValueError: If the knowledge base type is not supported, or a
            backend reports a configuration problem. dataknobs' own refusals
            (an unknown vector backend or embedding provider) arrive as
            themselves, with their words. A ``ValueError`` a backend raised
            with its own words arrives as a bounded
            :class:`~dataknobs_common.exceptions.FactoryValueError` (still a
            ``ValueError``), because a backend's text can carry the
            connection URL it was handed; the original is on ``__cause__``.
        PluginConfigError: A key the backend's config class refuses (also a
            ``ValueError``), naming the key.
        ResourceError: An embedding or vector backend that failed to
            connect, bounded the same way; likewise ``TimeoutError``,
            ``ConcurrencyError``, ``SerializationError`` and
            ``ConfigurationError``. Any other failure arrives as a bounded
            ``OperationError``.

    Example:
        ```python
        kb = await create_knowledge_base_from_config({
            "type": "rag",
            "vector_store": {"backend": "memory", "dimensions": 384},
            "embedding_provider": "echo",
            "embedding_model": "test",
        })
        ```
    """
    kb_type = (config or {}).get("type", "rag")

    def unknown(_: NotFoundError) -> ValueError:
        return ValueError(
            f"Unknown knowledge base type: {kb_type}. "
            f"Available types: {', '.join(sorted(knowledge_base_backends.list_keys()))}"
        )

    return await create_surfacing_async(
        knowledge_base_backends,
        subject="knowledge base",
        unknown=unknown,
        config=config,
    )


def register_knowledge_base_backend(
    name: str,
    factory: KnowledgeBaseFactory,
    *,
    override: bool = False,
) -> None:
    """Register a custom knowledge-base backend.

    Args:
        name: Backend name (used in the ``type`` config field).
        factory: ``KnowledgeBase`` subclass or factory callable accepting
            ``(config, **collaborators)``.
        override: Replace an existing registration if ``True``.

    Raises:
        OperationError: If ``name`` is already registered and ``override``
            is ``False``.
        TypeError: If ``factory`` is not a ``KnowledgeBase`` subclass or callable.
    """
    knowledge_base_backends.register(name, factory, override=override)


def get_knowledge_base_backend_factory(name: str) -> KnowledgeBaseFactory | None:
    """Return the factory for a knowledge-base backend name, or ``None``."""
    return knowledge_base_backends.get_factory(name)


def is_knowledge_base_backend_registered(name: str) -> bool:
    """Check whether a knowledge-base backend name is registered."""
    return knowledge_base_backends.is_registered(name)


def list_knowledge_base_backends() -> list[str]:
    """Return a sorted list of all registered knowledge-base backend names."""
    return sorted(knowledge_base_backends.list_keys())
