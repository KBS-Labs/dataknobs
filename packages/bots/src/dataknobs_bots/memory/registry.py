"""Backend registry and config-driven construction for memory strategies.

Provides a plugin registry that lets 3rd parties register custom memory
backends without modifying core DynaBot code, plus the public
``create_memory_from_config`` factory that dispatches a config dict to the
registered backend. Built-in backends (buffer, vector, summary, composite)
are registered lazily at first access.

Usage::

    from dataknobs_bots.memory import (
        create_memory_from_config,
        register_memory_backend,
    )

    # Register a custom backend (class or factory callable)
    register_memory_backend("my_backend", MyMemory)

    # Then use it via config
    memory = await create_memory_from_config({"type": "my_backend", ...})

The discriminator key is ``type`` (default ``buffer``). Backends receive
the raw config dict positionally and the injected collaborators
(``llm_provider``, ``prompt_resolver``) as keyword arguments — a backend
that does not consume a given collaborator absorbs it with ``**_``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from dataknobs_common.exceptions import (
    DataknobsError,
    NotFoundError,
    OperationError,
)
from dataknobs_common.registry import PluginRegistry
from dataknobs_common.structured_config import (
    SKIP_VALIDATION,
    ConfigClassResolution,
    StructuredConfig,
    config_registries,
)

from .base import Memory
from .buffer import BufferMemory
from .composite import CompositeMemory
from .summary import SummaryMemory
from .vector import VectorMemory

if TYPE_CHECKING:
    from dataknobs_bots.prompts.resolver import PromptResolver

logger = logging.getLogger(__name__)

# A memory factory is either a Memory subclass (with ``from_config`` /
# ``from_config_async``) or a callable with the same dispatch signature.
MemoryFactory = type[Memory] | Callable[..., Memory]


# ------------------------------------------------------------------
# Built-in backend registration
# ------------------------------------------------------------------


def _register_builtins(registry: PluginRegistry[Memory]) -> None:
    """Register the built-in memory backends (lazy, on first access).

    Each backend is a ``StructuredConfigConsumer`` class registered
    directly: the registry's ``create_async`` calls each class's
    ``from_config_async`` and threads the injected ``llm_provider`` /
    ``prompt_resolver`` collaborators into its ``_ainit`` (buffer and
    vector ignore them; summary and composite consume them).
    """
    registry.register("buffer", BufferMemory)
    registry.register("vector", VectorMemory)
    registry.register("summary", SummaryMemory)
    registry.register("composite", CompositeMemory)


# Module-level singleton — data-driven dispatch replaces the former
# inline ``if/elif`` on ``config["type"]``.
memory_backends: PluginRegistry[Memory] = PluginRegistry(
    "memory_backends",
    validate_type=Memory,
    canonicalize_keys=True,
    config_key="type",
    config_key_default="buffer",
    on_first_access=_register_builtins,
)


def _resolve_memory_config_cls(
    raw: Mapping[str, Any],
) -> ConfigClassResolution:
    """Resolve a ``memory`` section's dict to its config class.

    The resolver registered for the ``"memory"`` binding in
    :data:`~dataknobs_common.structured_config.config_registries`, used by
    :meth:`StructuredConfig.validate
    <dataknobs_common.structured_config.StructuredConfig.validate>` to
    validate a raw ``memory`` section (and each element of a composite's
    ``strategies`` list) without constructing the memory backend.

    Delegates to ``memory_backends`` — the same registry the construction
    path uses — by reading ``CONFIG_CLS`` off the registered backend class
    for the ``"type"`` discriminator (defaulting to ``"buffer"``, the
    factory's own default). Holding no independent type→config-class table
    is the no-drift guarantee. Returns ``None`` for an unknown type, which
    ``validate`` surfaces as a ``ConfigurationError``.

    Unlike ``vector_backends`` (a closed set of config-bearing classes),
    ``register_memory_backend`` accepts a bare callable factory
    (``MemoryFactory = type[Memory] | Callable[..., Memory]``). Such a
    backend has no ``StructuredConfig`` ``CONFIG_CLS``, so the resolver
    returns :data:`SKIP_VALIDATION`: the backend is valid and constructible
    but has no typed schema to dry-run against, so ``validate`` skips it
    rather than false-positive-raising — distinct from the ``None`` return
    for a genuine typo'd discriminator.
    """
    backend_type = raw.get("type", "buffer")  # registry's own default
    factory = memory_backends.get_factory(backend_type)
    if factory is None:
        return None  # unknown type -> validate() raises ConfigurationError
    config_cls = getattr(factory, "CONFIG_CLS", None)
    if isinstance(config_cls, type) and issubclass(config_cls, StructuredConfig):
        return config_cls
    return SKIP_VALIDATION  # registered bare callable, no CONFIG_CLS -> skip


# Eager registration (mirroring ``dataknobs-data``'s ``vector_store``):
# importing this module is what makes the ``memory`` binding resolvable, and
# any parent config holding a memory section depends on this package. The
# package ``__init__`` imports this module, so ``import dataknobs_bots.memory``
# fires it. ``config_registries`` is a plain ``Registry`` -> ``allow_overwrite``
# (NOT ``override=``, which is ``memory_backends``'/``PluginRegistry``'s param;
# the two registries differ) keeps re-import idempotent.
config_registries.register(
    "memory", _resolve_memory_config_cls, allow_overwrite=True
)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


async def create_memory_from_config(
    config: dict[str, Any],
    llm_provider: Any | None = None,
    prompt_resolver: PromptResolver | None = None,
) -> Memory:
    """Create a memory instance from configuration.

    Args:
        config: Memory configuration with a ``type`` field (default
            ``buffer``) and type-specific params.
        llm_provider: Optional LLM provider, used by summary memory as a
            fallback when no dedicated ``llm`` section is configured.
        prompt_resolver: Optional PromptResolver for resolving memory prompts.

    Returns:
        Configured Memory instance.

    Raises:
        ValueError: If the memory type is not recognized, or a backend
            reports a configuration problem (e.g. summary memory without an
            LLM source, a composite with no strategies).

    Example:
        ```python
        # Buffer memory
        memory = await create_memory_from_config({"type": "buffer", "max_messages": 10})

        # Vector memory
        memory = await create_memory_from_config({
            "type": "vector",
            "backend": "faiss",
            "dimension": 768,
            "embedding_provider": "ollama",
            "embedding_model": "nomic-embed-text",
        })

        # Summary memory (uses bot's LLM as fallback)
        memory = await create_memory_from_config(
            {"type": "summary", "recent_window": 10}, llm_provider=llm,
        )

        # Summary memory with its own dedicated LLM
        memory = await create_memory_from_config({
            "type": "summary",
            "recent_window": 10,
            "llm": {"provider": "ollama", "model": "gemma3:1b"},
        })

        # Composite memory (multiple strategies)
        memory = await create_memory_from_config({
            "type": "composite",
            "strategies": [
                {"type": "buffer", "max_messages": 50},
                {"type": "vector", "backend": "memory", "dimension": 768,
                 "embedding_provider": "ollama", "embedding_model": "nomic-embed-text"},
            ],
            "primary": 0,
        })
        ```
    """
    try:
        return await memory_backends.create_async(
            config=config,
            llm_provider=llm_provider,
            prompt_resolver=prompt_resolver,
        )
    except NotFoundError as exc:
        memory_type = (config or {}).get("type", "buffer")
        raise ValueError(
            f"Unknown memory type: {memory_type}. "
            f"Available types: {', '.join(sorted(memory_backends.list_keys()))}"
        ) from exc
    except OperationError as exc:
        # The registry wraps a backend-raised exception in OperationError.
        # Surface the original cause unchanged: the public contract raises
        # ValueError for config problems (missing LLM source, empty/invalid
        # composite, unknown nested type), and a backend's native dataknobs
        # error (``ResourceError`` from a vector backend that fails to
        # connect, ``ConfigurationError``, …) should reach the caller as its
        # own type rather than as a generic OperationError. A wrapper with no
        # informative cause propagates as-is.
        cause = exc.__cause__
        if isinstance(cause, ValueError | DataknobsError):
            raise cause from cause.__cause__
        raise


def register_memory_backend(
    name: str,
    factory: MemoryFactory,
    *,
    override: bool = False,
) -> None:
    """Register a custom memory backend.

    Args:
        name: Backend name (used in the ``type`` config field).
        factory: ``Memory`` subclass or factory callable accepting
            ``(config, **collaborators)``.
        override: Replace an existing registration if ``True``.

    Raises:
        OperationError: If ``name`` is already registered and ``override``
            is ``False``.
        TypeError: If ``factory`` is not a ``Memory`` subclass or callable.
    """
    memory_backends.register(name, factory, override=override)


def get_memory_backend_factory(name: str) -> MemoryFactory | None:
    """Return the factory for a memory backend name, or ``None``."""
    return memory_backends.get_factory(name)


def is_memory_backend_registered(name: str) -> bool:
    """Check whether a memory backend name is registered."""
    return memory_backends.is_registered(name)


def list_memory_backends() -> list[str]:
    """Return a sorted list of all registered memory backend names."""
    return sorted(memory_backends.list_keys())
