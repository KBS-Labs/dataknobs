"""LLM provider implementations.

This module provides implementations for various LLM providers.
Supports both direct instantiation and dataknobs Config-based factory pattern.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ..base import (
    LLMConfig, AsyncLLMProvider, SyncLLMProvider,
    normalize_llm_config, LLMMessage, LLMResponse
)

# Import adapters and providers
from .base import SyncProviderAdapter
from .openai import OpenAIAdapter, OpenAIProvider
from .anthropic import AnthropicProvider
from .ollama import OllamaProvider
from .huggingface import HuggingFaceProvider
from .caching import (
    CachingEmbedProvider,
    EmbeddingCache,
    MemoryEmbeddingCache,
    create_caching_provider,
)
from .echo import EchoProvider

if TYPE_CHECKING:
    from dataknobs_config.config import Config

_logger = logging.getLogger(__name__)


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

    # Registry of provider classes
    _providers: dict[str, type[AsyncLLMProvider] | None] = {
        'openai': None,  # Populated lazily
        'anthropic': None,
        'ollama': None,
        'huggingface': None,
        'echo': None,
    }

    def __init__(self, is_async: bool = True):
        """Initialize the factory.

        Args:
            is_async: Whether to create async providers (default: True)
        """
        self.is_async = is_async

        # Lazily populate provider registry
        if LLMProviderFactory._providers['openai'] is None:
            LLMProviderFactory._providers.update({
                'openai': OpenAIProvider,
                'anthropic': AnthropicProvider,
                'ollama': OllamaProvider,
                'huggingface': HuggingFaceProvider,
                'echo': EchoProvider,
            })

    def create(
        self,
        config: LLMConfig | Config | dict[str, Any],
        **kwargs: Any
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

        # Get provider class
        provider_class = self._providers.get(llm_config.provider.lower())
        if not provider_class:
            raise ValueError(
                f"Unknown provider: {llm_config.provider}. "
                f"Available providers: {list(self._providers.keys())}"
            )

        # Create provider instance
        if self.is_async:
            return provider_class(llm_config)
        else:
            # Wrap in sync adapter
            async_provider = provider_class(llm_config)
            return SyncProviderAdapter(async_provider)  # type: ignore

    @classmethod
    def register_provider(
        cls,
        name: str,
        provider_class: type[AsyncLLMProvider]
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
        cls._providers[name.lower()] = provider_class

    def __call__(
        self,
        config: LLMConfig | Config | dict[str, Any],
        **kwargs: Any
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
    is_async: bool = True
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
    config: dict[str, Any],
    *,
    default_provider: str = "ollama",
    default_model: str = "nomic-embed-text",
) -> AsyncLLMProvider:
    """Create and initialize an embedding provider from configuration.

    Normalizes configuration from two supported formats:

    - **Nested format:** ``{"embedding": {"provider": "ollama", "model": "..."}}``
      -- the ``"embedding"`` sub-dict is extracted and used.  All extra keys
      in the sub-dict (``api_base``, ``api_key``, ``dimensions``, etc.) are
      forwarded to the provider.
    - **Legacy prefix format:** ``{"embedding_provider": "ollama",
      "embedding_model": "..."}`` -- ``embedding_`` prefixed keys at the
      top level.  ``api_base``, ``api_key``, and ``dimensions`` are also
      forwarded when present at the top level.

    When neither format is present, *default_provider* / *default_model*
    are used (``ollama`` / ``nomic-embed-text``).

    Args:
        config: Configuration dict.
        default_provider: Default provider if not specified.
        default_model: Default model if not specified.

    Returns:
        Initialized ``AsyncLLMProvider`` instance ready for ``embed()`` calls.

    Example:
        ```python
        provider = await create_embedding_provider({
            "embedding": {
                "provider": "ollama",
                "model": "nomic-embed-text",
            },
        })
        embedding = await provider.embed("hello world")
        ```
    """
    # 1. Nested "embedding" sub-dict (preferred)
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

    factory = LLMProviderFactory(is_async=True)
    provider_config = {
        "provider": provider_name,
        "model": model_name,
        **extra,
        "mode": "embedding",  # Always forced — must come after **extra
    }
    try:
        provider = factory.create(provider_config)
        await provider.initialize()
    except Exception:
        _logger.exception(
            "Failed to create embedding provider: %s/%s",
            provider_name,
            model_name,
        )
        raise

    _logger.info(
        "Embedding provider initialized: %s/%s",
        provider_name,
        model_name,
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
