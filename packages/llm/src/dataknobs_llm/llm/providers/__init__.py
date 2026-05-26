"""LLM provider implementations.

This module provides implementations for various LLM providers.
Supports both direct instantiation and dataknobs Config-based factory pattern.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

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
    registry: PluginRegistry[type[AsyncLLMProvider]],
) -> None:
    """Register all built-in LLM providers."""
    for name, cls in [
        ("openai", OpenAIProvider),
        ("anthropic", AnthropicProvider),
        ("ollama", OllamaProvider),
        ("huggingface", HuggingFaceProvider),
        ("echo", EchoProvider),
    ]:
        registry.register(name, cls)


# Module-level provider registry — PluginRegistry stores provider *classes*
# (not instances), so get() returns a class and create() is not used here.
_provider_registry: PluginRegistry[type[AsyncLLMProvider]] = PluginRegistry(
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
config_registries.register(
    "embedding", _resolve_llm_config_cls, allow_overwrite=True
)


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
    ) -> AsyncLLMProvider | SyncLLMProvider:
        """Create an LLM provider from configuration.

        Args:
            config: Configuration (LLMConfig, Config object, or dict)
            **kwargs: Additional arguments passed to provider constructor

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
            available = _provider_registry.list_keys()
            raise ValueError(
                f"Unknown provider: {llm_config.provider}. "
                f"Available providers: {available}"
            )

        # Create provider instance
        if self.is_async:
            return provider_class(llm_config)
        else:
            # Wrap in sync adapter
            async_provider = provider_class(llm_config)
            return SyncProviderAdapter(async_provider)  # type: ignore[return-value]

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

    def __call__(
        self,
        config: LLMConfig | Config | dict[str, Any],
        **kwargs: Any,
    ) -> AsyncLLMProvider | SyncLLMProvider:
        """Allow factory to be called directly.

        Makes the factory callable for convenience.

        Args:
            config: Configuration
            **kwargs: Additional arguments

        Returns:
            LLM provider instance
        """
        return self.create(config, **kwargs)


def create_llm_provider(
    config: LLMConfig | Config | dict[str, Any],
    is_async: bool = True,
) -> AsyncLLMProvider | SyncLLMProvider:
    """Create appropriate LLM provider based on configuration.

    Convenience function that uses LLMProviderFactory internally.
    Now supports LLMConfig, Config objects, and dictionaries.

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
            extra = {
                k: v for k, v in embedding_config.items()
                if k not in ("provider", "model")
            }
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

    factory = LLMProviderFactory(is_async=True)
    try:
        provider = factory.create(provider_config)
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
    'AsyncLLMProvider',
    'SyncLLMProvider',
    'LLMConfig',
    'LLMMessage',
    'LLMResponse',
    # Adapters
    'SyncProviderAdapter',
    'OpenAIAdapter',
    # Providers
    'OpenAIProvider',
    'AnthropicProvider',
    'OllamaProvider',
    'HuggingFaceProvider',
    'EchoProvider',
    'CachingEmbedProvider',
    'EmbeddingCache',
    'MemoryEmbeddingCache',
    'create_caching_provider',
    # Factory
    'LLMProviderFactory',
    'create_llm_provider',
    'create_embedding_provider',
    'normalize_llm_config',
]
