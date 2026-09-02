"""LLM provider implementations.

This module provides implementations for various LLM providers.
Supports both direct instantiation and dataknobs Config-based factory pattern.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, overload

from dataknobs_common.registry import PluginRegistry
from dataknobs_common.structured_config import StructuredConfig, config_registries

from ..base import (
    AsyncLLMProvider,
    CompletionMode,
    LLMConfig,
    LLMMessage,
    LLMResponse,
    SyncLLMProvider,
    normalize_llm_config,
)

# Import adapters and providers
from .anthropic import AnthropicProvider
from .base import SyncProviderAdapter
from .bedrock import BedrockConverseAdapter, BedrockProvider
from .caching import (
    CachingEmbedProvider,
    EmbeddingCache,
    MemoryEmbeddingCache,
    create_caching_provider,
)
from .echo import EchoProvider
from .huggingface import HuggingFaceProvider
from .ollama import OllamaProvider
from .openai import OpenAIAdapter, OpenAIProvider

if TYPE_CHECKING:
    from dataknobs_config.config import Config

_logger = logging.getLogger(__name__)


def _register_builtin_providers(
    registry: PluginRegistry[AsyncLLMProvider],
) -> None:
    """Register all built-in LLM providers."""
    for name, cls in [
        ("openai", OpenAIProvider),
        ("anthropic", AnthropicProvider),
        ("bedrock", BedrockProvider),
        ("ollama", OllamaProvider),
        ("huggingface", HuggingFaceProvider),
        ("echo", EchoProvider),
    ]:
        registry.register(name, cls)


# Module-level provider registry. ``PluginRegistry[T]``'s parameter is what a
# registration *produces*, not what the mapping stores — a provider class is
# already a ``Callable[..., AsyncLLMProvider]``, which is the shape ``register``
# accepts and ``get_factory`` hands back. The parameter used to read
# ``type[AsyncLLMProvider]``, carried across verbatim from the plain
# ``dict[str, type[AsyncLLMProvider] | None]`` this replaced; that made the
# registry statically produce provider *classes*, so ``register`` wanted a
# factory returning a class, and calling what ``get_factory`` returned yielded
# a class rather than a provider. Six of this file's eight findings came from
# those two words alone — measured by changing only them.
#
# Only ``register`` / ``get_factory`` / ``list_keys`` / ``unregister`` are used
# here: instantiation is this module's job, because it — not the registry —
# knows whether the sync adapter goes on top.
_provider_registry: PluginRegistry[AsyncLLMProvider] = PluginRegistry(
    "llm_providers",
    canonicalize_keys=True,
    on_first_access=_register_builtin_providers,
)


def _resolve_llm_config_cls(
    raw: Mapping[str, Any],
) -> type[StructuredConfig] | None:
    """Resolve an ``llm`` config section to its ``StructuredConfig`` class.

    ``LLMConfig`` is a single config class keyed by ``provider`` — there are
    no per-provider config subclasses (provider-specific knobs live in
    ``options``). So the resolver returns ``LLMConfig`` for *any known*
    provider and ``None`` for an unknown/missing one (which makes
    ``StructuredConfig.validate()`` raise ``ConfigurationError``). The
    known-provider set is delegated to the provider registry — the same
    source the factory checks at construction — so validation and
    construction can never drift. ``get_factory`` triggers the registry's
    lazy ``on_first_access`` registration, so built-in providers are visible
    here even on a cold registry.

    No ``SKIP_VALIDATION`` path is needed: every registered provider has a
    typed config (``LLMConfig``) to validate against.
    """
    provider = raw.get("provider")
    if provider and _provider_registry.get_factory(provider) is not None:
        return LLMConfig
    return None


# Registered eagerly at import so ``StructuredConfig.validate()`` can resolve
# an ``llm`` section without the consumer importing ``dataknobs-llm`` config
# types directly (the binding name is a string). ``allow_overwrite=True``
# keeps re-import idempotent.
config_registries.register("llm", _resolve_llm_config_cls, allow_overwrite=True)

# An ``embedding`` section *is* an LLM-provider section: ``create_embedding_provider``
# rides the same ``_provider_registry`` and forces ``mode=embedding`` onto an
# ``LLMConfig`` (there is no separate embedding-provider registry or config
# family — every embedder config key, including ``dimensions``, is already an
# ``LLMConfig`` field). So the ``embedding`` section validates against
# ``LLMConfig`` via the *same* resolver — registering a parallel resolver with
# identical logic would only duplicate it. The binding name is deliberately
# distinct from ``"llm"`` so the section stays semantically separate; if an
# embed-specific config surface is ever wanted, only this registration changes
# (the ``"llm"`` binding is untouched).
#
# Both bindings delegate to ``_resolve_llm_config_cls`` → ``_provider_registry``,
# the same registry the construction factory consults, so the no-drift guard in
# ``tests/test_llm_config_resolver.py`` (which enumerates
# ``_provider_registry.list_keys()``) covers both bindings at once.
config_registries.register("embedding", _resolve_llm_config_cls, allow_overwrite=True)


class LLMProviderFactory:
    """Factory for creating LLM providers from configuration.

    This factory class integrates with the dataknobs Config system,
    allowing providers to be instantiated via Config.get_factory().

    Example:
        ```python
        from dataknobs_config import Config
        config = Config({
            "llm": [{
                "name": "gpt4",
                "provider": "openai",
                "model": "gpt-4",
                "factory": "dataknobs_llm.LLMProviderFactory"
            }]
        })
        factory = config.get_factory("llm", "gpt4")
        provider = factory.create(config.get("llm", "gpt4"))
        ```
    """

    def __init__(self, is_async: bool = True):
        """Initialize the factory.

        Args:
            is_async: Whether to create async providers (default: True)
        """
        self.is_async = is_async

    def create(
        self,
        config: LLMConfig | Config | dict[str, Any],
        **kwargs: Any,
    ) -> AsyncLLMProvider | SyncProviderAdapter:
        """Create an LLM provider from configuration.

        The return type is a union because ``is_async`` is a *constructor*
        flag, which no overload can discriminate — this method has to be
        callable through the ``Config`` factory protocol, where the caller
        holds a factory object and not the flag that built it. Code that
        knows which half it wants should call :func:`create_llm_provider`
        instead, whose ``is_async`` is an argument and is therefore typed.

        The second arm names :class:`SyncProviderAdapter` rather than
        ``SyncLLMProvider``: the adapter deliberately inherits from nothing,
        and no ``SyncLLMProvider`` subclass exists in tree, so the arm this
        replaces was uninhabited. A ``# type: ignore[return-value]`` on the
        sync branch had been holding the mismatch down since before the
        registry existed.

        Args:
            config: Configuration (LLMConfig, Config object, or dict)
            **kwargs: Additional arguments passed to provider constructor
                (e.g. ``prompt_builder=``). Every built-in provider takes
                ``(config, prompt_builder=None)``; ``EchoProvider`` takes
                several more.

        Returns:
            LLM provider instance

        Raises:
            ValueError: If provider type is unknown
        """
        # Normalize config to LLMConfig
        llm_config = normalize_llm_config(config)

        # Get provider class from registry
        provider_class = _provider_registry.get_factory(llm_config.provider)
        if not provider_class:
            # Through ``list_providers`` so the failure message and the
            # supported read-side query cannot disagree — and so the list is
            # sorted rather than in registration order.
            available = self.list_providers()
            raise ValueError(
                f"Unknown provider: {llm_config.provider}. Available providers: {available}"
            )

        # Create provider instance. ``**kwargs`` is forwarded — the signature
        # and docstring have promised that since before the registry existed
        # and neither branch did it, so a caller passing ``prompt_builder=``
        # or ``responses=`` got a default-built provider and no error saying so.
        async_provider = provider_class(llm_config, **kwargs)
        # This module instantiates the factory itself rather than through
        # ``create()``, so the registry's own refusal of an awaitable factory
        # never runs here. ``register_provider`` declares
        # ``type[AsyncLLMProvider]`` and is the only registration path, so
        # this cannot fire -- but the alternative to checking is a cast, and
        # a cast would turn a broken registration into a provider that is
        # silently a coroutine. Sync construction is this module's premise:
        # it decides whether ``SyncProviderAdapter`` goes on top, and it
        # cannot decide that about something it has not awaited.
        if inspect.isawaitable(async_provider):
            # Closed rather than dropped, for the same reason
            # ``PluginRegistry._refuse_awaitable`` closes: an un-awaited
            # coroutine left to be collected emits a ``RuntimeWarning`` at
            # interpreter shutdown, attributed to the factory rather than to
            # the refusal -- which is the confusion the raise below removes.
            close = getattr(async_provider, "close", None)
            if callable(close):
                close()
            raise TypeError(
                f"Provider '{llm_config.provider}' is registered as an "
                f"asynchronous factory; LLMProviderFactory constructs "
                f"providers synchronously. Register a provider class."
            )
        if self.is_async:
            return async_provider
        return SyncProviderAdapter(async_provider)

    @staticmethod
    def register_provider(
        name: str,
        provider_class: type[AsyncLLMProvider],
    ) -> None:
        """Register a custom provider class.

        Allows extending the factory with custom provider implementations.

        Args:
            name: Provider name (e.g., 'custom')
            provider_class: Provider class (must inherit from AsyncLLMProvider)

        Example:
            ```python
            class CustomProvider(AsyncLLMProvider):
                pass
            LLMProviderFactory.register_provider('custom', CustomProvider)
            ```
        """
        _provider_registry.register(name, provider_class, override=True)

    @staticmethod
    def list_providers() -> list[str]:
        """Every registered provider family key, sorted.

        The read-side counterpart to :meth:`register_provider`, and the
        supported way for a consumer to ask "what can ``provider:`` be set
        to?" — config validators, schema/documentation generators, and
        interactive config builders all need that list, and without this they
        transcribe one into a literal that cannot include anything registered
        later. Reflects consumer registrations, so it stays correct for
        providers DK has never heard of.

        Returns:
            Sorted canonical family keys (e.g. ``["anthropic", "bedrock", …]``).
        """
        return sorted(_provider_registry.list_keys())

    def __call__(
        self,
        config: LLMConfig | Config | dict[str, Any],
        **kwargs: Any,
    ) -> AsyncLLMProvider | SyncProviderAdapter:
        """Allow factory to be called directly.

        Makes the factory callable for convenience.

        Args:
            config: Configuration
            **kwargs: Additional arguments

        Returns:
            LLM provider instance
        """
        return self.create(config, **kwargs)


@overload
def create_llm_provider(
    config: LLMConfig | Config | dict[str, Any],
    is_async: Literal[True] = ...,
) -> AsyncLLMProvider: ...


@overload
def create_llm_provider(
    config: LLMConfig | Config | dict[str, Any],
    is_async: Literal[False],
) -> SyncProviderAdapter: ...


@overload
def create_llm_provider(
    config: LLMConfig | Config | dict[str, Any],
    is_async: bool,
) -> AsyncLLMProvider | SyncProviderAdapter: ...


def create_llm_provider(
    config: LLMConfig | Config | dict[str, Any],
    is_async: bool = True,
) -> AsyncLLMProvider | SyncProviderAdapter:
    """Create appropriate LLM provider based on configuration.

    Convenience function that uses LLMProviderFactory internally.
    Now supports LLMConfig, Config objects, and dictionaries.

    Prefer this over ``LLMProviderFactory(is_async=...).create(config)``
    when the mode is known at the call site. Here ``is_async`` is an
    argument, so the overloads above resolve the return type to the one
    provider the call can actually produce; on the factory it is a
    constructor flag, and :meth:`LLMProviderFactory.create` has to return
    the union whatever it was set to. Every caller of the union form then
    pays for an arm it cannot receive — by narrowing the value with a check
    that can never fail, or by erasing it to ``Any``.

    The third overload keeps a caller passing a runtime ``bool`` working:
    it matches neither ``Literal``, and without it the call would be an
    error rather than the honest union.

    Args:
        config: LLM configuration (LLMConfig, Config, or dict)
        is_async: Whether to create async provider

    Returns:
        LLM provider instance

    Example:
        ```python
        # Direct usage with dict
        provider = create_llm_provider({
            "provider": "openai",
            "model": "gpt-4",
            "api_key": "..."
        })

        # With Config object
        from dataknobs_config import Config
        config = Config({"llm": [{"provider": "openai", "model": "gpt-4"}]})
        provider = create_llm_provider(config)
        ```
    """
    factory = LLMProviderFactory(is_async=is_async)
    return factory.create(config)


async def create_embedding_provider(
    config: LLMConfig | dict[str, Any],
    *,
    default_provider: str = "ollama",
    default_model: str = "nomic-embed-text",
) -> AsyncLLMProvider:
    """Create and initialize an embedding provider from configuration.

    Accepts a typed ``LLMConfig`` or a dict (mirroring the data factories,
    which accept a typed config or a raw dict). An embedder config *is* an
    ``LLMConfig`` — embedding providers ride the same provider registry — so
    no separate config type is needed; ``mode=embedding`` is forced in every
    case (a caller-supplied ``mode`` is overridden).

    - **Typed ``LLMConfig``:** used directly. ``provider`` / ``model`` are
      already validated as required fields; ``mode`` is forced to
      :attr:`CompletionMode.EMBEDDING` (via ``clone`` — ``LLMConfig`` is
      frozen). *default_provider* / *default_model* are unused on this path.
    - **Nested dict:** ``{"embedding": {"provider": "ollama", "model": "..."}}``
      -- the ``"embedding"`` sub-dict is extracted and used.  All extra keys
      in the sub-dict (``api_base``, ``api_key``, ``dimensions``, etc.) are
      forwarded to the provider.
    - **Legacy prefix dict:** ``{"embedding_provider": "ollama",
      "embedding_model": "..."}`` -- ``embedding_`` prefixed keys at the
      top level.  ``api_base``, ``api_key``, and ``dimensions`` are also
      forwarded when present at the top level.

    When neither dict format is present, *default_provider* / *default_model*
    are used (``ollama`` / ``nomic-embed-text``).

    Args:
        config: A typed ``LLMConfig`` or a configuration dict.
        default_provider: Default provider if not specified (dict path only).
        default_model: Default model if not specified (dict path only).

    Returns:
        Initialized ``AsyncLLMProvider`` instance ready for ``embed()`` calls.

    Example:
        ```python
        # Typed config
        provider = await create_embedding_provider(
            LLMConfig(provider="ollama", model="nomic-embed-text")
        )
        # Or a dict
        provider = await create_embedding_provider({
            "embedding": {"provider": "ollama", "model": "nomic-embed-text"},
        })
        embedding = await provider.embed("hello world")
        ```
    """
    provider_config: LLMConfig | dict[str, Any]
    if isinstance(config, LLMConfig):
        # Typed path: force embedding mode (clone — LLMConfig is frozen).
        provider_config = (
            config
            if config.mode is CompletionMode.EMBEDDING
            else config.clone(mode=CompletionMode.EMBEDDING)
        )
    else:
        # Dict path. 1. Nested "embedding" sub-dict (preferred)
        extra: dict[str, Any]
        embedding_config = config.get("embedding", {})
        if embedding_config and isinstance(embedding_config, dict):
            provider_name = embedding_config.get("provider", default_provider)
            model_name = embedding_config.get("model", default_model)
            # Forward all extra keys (api_base, api_key, dimensions, etc.)
            extra = {k: v for k, v in embedding_config.items() if k not in ("provider", "model")}
        else:
            # 2. Legacy prefix format (embedding_provider / embedding_model)
            provider_name = config.get("embedding_provider", default_provider)
            model_name = config.get("embedding_model", default_model)
            extra = {}
            for passthrough in ("api_base", "api_key", "dimensions"):
                if passthrough in config:
                    extra[passthrough] = config[passthrough]
        provider_config = {
            "provider": provider_name,
            "model": model_name,
            **extra,
            "mode": "embedding",  # Always forced — must come after **extra
        }

    # Single log/error identity, read from the *resolved* ``provider_config``
    # (the typed path may have cloned it for embedding mode) so the success and
    # failure log sites share one source. ``provider_config`` is an
    # ``LLMConfig`` on the typed path, a built dict on the dict path.
    if isinstance(provider_config, LLMConfig):
        log_provider, log_model = provider_config.provider, provider_config.model
    else:
        log_provider, log_model = provider_config["provider"], provider_config["model"]

    try:
        provider = create_llm_provider(provider_config)
        await provider.initialize()
    except Exception:
        _logger.exception(
            "Failed to create embedding provider: %s/%s",
            log_provider,
            log_model,
        )
        raise

    _logger.info(
        "Embedding provider initialized: %s/%s",
        log_provider,
        log_model,
    )
    return provider


# Export all providers and factory for backward compatibility
__all__ = [
    # Base classes (re-exported for convenience)
    "AsyncLLMProvider",
    "SyncLLMProvider",
    "LLMConfig",
    "LLMMessage",
    "LLMResponse",
    # Adapters
    "SyncProviderAdapter",
    "OpenAIAdapter",
    "BedrockConverseAdapter",
    # Providers
    "OpenAIProvider",
    "AnthropicProvider",
    "BedrockProvider",
    "OllamaProvider",
    "HuggingFaceProvider",
    "EchoProvider",
    "CachingEmbedProvider",
    "EmbeddingCache",
    "MemoryEmbeddingCache",
    "create_caching_provider",
    # Factory
    "LLMProviderFactory",
    "create_llm_provider",
    "create_embedding_provider",
    "normalize_llm_config",
]
