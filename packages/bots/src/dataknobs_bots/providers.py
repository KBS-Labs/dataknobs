"""Provider creation utilities for dataknobs-bots.

Shared helpers for creating and initializing LLM providers used across
bot subsystems (memory, knowledge base, reasoning).
"""

from __future__ import annotations

import logging
from typing import Any

from dataknobs_llm.llm import AsyncLLMProvider, LLMProviderFactory

logger = logging.getLogger(__name__)


async def create_embedding_provider(
    config: dict[str, Any],
    *,
    provider_key: str = "provider",
    model_key: str = "model",
    default_provider: str = "ollama",
    default_model: str = "nomic-embed-text",
) -> AsyncLLMProvider:
    """Create and initialize an embedding provider from configuration.

    Normalizes configuration from multiple formats:

    - **Nested format:** ``{"embedding": {"provider": "ollama", "model": "..."}}``
      -- the ``"embedding"`` sub-dict is extracted and used.
    - **Flat format:** ``{"provider": "ollama", "model": "..."}``
      -- keys are read directly from the top-level dict.
    - **Legacy flat format:** ``{"embedding_provider": "ollama", "embedding_model": "..."}``
      -- ``embedding_`` prefixed keys are tried as a fallback.

    Args:
        config: Configuration dict.
        provider_key: Key for provider name (default ``"provider"``).
        model_key: Key for model name (default ``"model"``).
        default_provider: Default provider if not specified.
        default_model: Default model if not specified.

    Returns:
        Initialized ``AsyncLLMProvider`` instance ready for ``embed()`` calls.
    """
    # Check for nested "embedding" sub-dict first
    embedding_config = config.get("embedding", {})
    if embedding_config and isinstance(embedding_config, dict):
        provider_name = embedding_config.get(provider_key, default_provider)
        model_name = embedding_config.get(model_key, default_model)
        # Pass through any extra keys (api_base, api_key, dimensions, etc.)
        extra = {
            k: v for k, v in embedding_config.items()
            if k not in (provider_key, model_key)
        }
    else:
        # Flat format: try direct keys, then legacy embedding_ prefix
        provider_name = config.get(
            provider_key,
            config.get("embedding_provider", default_provider),
        )
        model_name = config.get(
            model_key,
            config.get("embedding_model", default_model),
        )
        extra = {}
        # Forward api_base/api_key if present at top level
        for passthrough in ("api_base", "api_key", "dimensions"):
            if passthrough in config:
                extra[passthrough] = config[passthrough]

    factory = LLMProviderFactory(is_async=True)
    provider = factory.create({
        "provider": provider_name,
        "model": model_name,
        **extra,
    })
    await provider.initialize()

    logger.info(
        "Embedding provider initialized: %s/%s",
        provider_name,
        model_name,
    )
    return provider
