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
    """
    # 1. Nested "embedding" sub-dict (preferred)
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
        logger.exception(
            "Failed to create embedding provider: %s/%s",
            provider_name,
            model_name,
        )
        raise

    logger.info(
        "Embedding provider initialized: %s/%s",
        provider_name,
        model_name,
    )
    return provider
