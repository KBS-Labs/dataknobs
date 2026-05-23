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
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from dataknobs_common.exceptions import (
    DataknobsError,
    NotFoundError,
    OperationError,
)
from dataknobs_common.registry import PluginRegistry

from .base import Memory
from .buffer import BufferMemory
from .composite import CompositeMemory
from .config import BufferMemoryConfig, CompositeMemoryConfig, SummaryMemoryConfig
from .summary import SummaryMemory
from .vector import VectorMemory

if TYPE_CHECKING:
    from dataknobs_bots.prompts.resolver import PromptResolver

logger = logging.getLogger(__name__)

# A memory factory is either a Memory subclass (with ``from_config`` /
# ``from_config_async``) or a callable with the same dispatch signature.
MemoryFactory = type[Memory] | Callable[..., Memory]


# ------------------------------------------------------------------
# Built-in backend builders
# ------------------------------------------------------------------


def _build_buffer(config: dict[str, Any], **_: Any) -> Memory:
    """Build a :class:`BufferMemory` from config."""
    cfg = BufferMemoryConfig.from_dict(config)
    return BufferMemory(max_messages=cfg.max_messages)


async def _build_vector(config: dict[str, Any], **_: Any) -> Memory:
    """Build a :class:`VectorMemory` (async warmup) from config."""
    return await VectorMemory.from_config(config)


async def _build_summary(
    config: dict[str, Any],
    *,
    llm_provider: Any | None = None,
    prompt_resolver: PromptResolver | None = None,
    **_: Any,
) -> Memory:
    """Build a :class:`SummaryMemory` from config.

    Resolves the summary LLM from the optional dedicated ``llm`` config
    section, falling back to the injected ``llm_provider`` (the bot's main
    LLM). Ownership of the provider's lifecycle follows which source was
    used.
    """
    cfg = SummaryMemoryConfig.from_dict(config)
    has_dedicated_llm = cfg.llm is not None
    summary_llm = await _resolve_summary_llm(cfg.llm, llm_provider)
    return SummaryMemory(
        llm_provider=summary_llm,
        recent_window=cfg.recent_window,
        summary_prompt=cfg.summary_prompt,
        owns_llm_provider=has_dedicated_llm,
        prompt_resolver=prompt_resolver,
    )


async def _build_composite(
    config: dict[str, Any],
    *,
    llm_provider: Any | None = None,
    prompt_resolver: PromptResolver | None = None,
    **_: Any,
) -> Memory:
    """Build a :class:`CompositeMemory`, recursing over child strategies.

    Each child spec is dispatched through the public factory so it gets the
    same error contract and collaborator threading. On any failure the
    already-built strategies are closed before the error propagates.
    """
    cfg = CompositeMemoryConfig.from_dict(config)
    strategies: list[Memory] = []
    try:
        for child in cfg.strategies:
            strategies.append(
                await create_memory_from_config(
                    child, llm_provider, prompt_resolver=prompt_resolver,
                )
            )
        if not strategies:
            raise ValueError(
                "Composite memory requires at least one strategy "
                "in 'strategies' list"
            )
        return CompositeMemory(
            strategies=strategies,
            primary_index=cfg.primary_index,
        )
    except Exception:
        # Clean up any already-initialized strategies
        for s in strategies:
            try:
                await s.close()
            except Exception:
                logger.warning(
                    "Failed to close strategy during cleanup: %s",
                    type(s).__name__,
                    exc_info=True,
                )
        raise


async def _resolve_summary_llm(
    llm_config: dict[str, Any] | None,
    fallback_provider: Any | None,
) -> Any:
    """Resolve the LLM provider for summary memory.

    If ``llm_config`` is provided, a dedicated provider is created and
    initialized from it. Otherwise the ``fallback_provider`` (typically the
    bot's own LLM) is used.

    Args:
        llm_config: Optional dedicated LLM-provider config.
        fallback_provider: Provider to use when no dedicated LLM is configured.

    Returns:
        An initialised ``AsyncLLMProvider``.

    Raises:
        ValueError: If neither a dedicated LLM config nor a fallback is available.
    """
    if llm_config is not None:
        from dataknobs_llm.llm import LLMProviderFactory

        factory = LLMProviderFactory(is_async=True)
        provider = factory.create(llm_config)
        await provider.initialize()
        return provider

    if fallback_provider is not None:
        return fallback_provider

    raise ValueError(
        "Summary memory requires an LLM provider. Either include an 'llm' "
        "section in the memory config or pass llm_provider to "
        "create_memory_from_config()."
    )


def _register_builtins(registry: PluginRegistry[Memory]) -> None:
    """Register the built-in memory backends (lazy, on first access)."""
    registry.register("buffer", _build_buffer)
    registry.register("vector", _build_vector)
    registry.register("summary", _build_summary)
    registry.register("composite", _build_composite)


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
