"""Memory implementations for DynaBot."""

import logging
from typing import Any

from .artifact_bank import ArtifactBank
from .artifact_io import (
    append_to_book,
    list_book,
    load_artifact,
    load_from_book,
    save_artifact,
    save_book,
)
from .bank import (
    AsyncBankProtocol,
    AsyncMemoryBank,
    BankRecord,
    EmptyBankProxy,
    MemoryBank,
    SyncBankProtocol,
)
from .base import Memory
from .buffer import BufferMemory
from .catalog import ArtifactBankCatalog
from .composite import CompositeMemory
from .summary import SummaryMemory
from .vector import VectorMemory

logger = logging.getLogger(__name__)

__all__ = [
    "ArtifactBank",
    "ArtifactBankCatalog",
    "AsyncBankProtocol",
    "AsyncMemoryBank",
    "BankRecord",
    "BufferMemory",
    "CompositeMemory",
    "EmptyBankProxy",
    "Memory",
    "MemoryBank",
    "SummaryMemory",
    "SyncBankProtocol",
    "VectorMemory",
    "append_to_book",
    "create_memory_from_config",
    "list_book",
    "load_artifact",
    "load_from_book",
    "save_artifact",
    "save_book",
]


async def create_memory_from_config(
    config: dict[str, Any],
    llm_provider: Any | None = None,
) -> Memory:
    """Create memory instance from configuration.

    Args:
        config: Memory configuration with 'type' field and type-specific params
        llm_provider: Optional LLM provider instance, required for summary memory

    Returns:
        Configured Memory instance

    Raises:
        ValueError: If memory type is not recognized or required params missing

    Example:
        ```python
        # Buffer memory
        config = {
            "type": "buffer",
            "max_messages": 10
        }
        memory = await create_memory_from_config(config)

        # Vector memory
        config = {
            "type": "vector",
            "backend": "faiss",
            "dimension": 768,
            "embedding_provider": "ollama",
            "embedding_model": "nomic-embed-text"
        }
        memory = await create_memory_from_config(config)

        # Summary memory (uses bot's LLM as fallback)
        config = {
            "type": "summary",
            "recent_window": 10,
        }
        memory = await create_memory_from_config(config, llm_provider=llm)

        # Summary memory with its own dedicated LLM
        config = {
            "type": "summary",
            "recent_window": 10,
            "llm": {
                "provider": "ollama",
                "model": "gemma3:1b",
            },
        }
        memory = await create_memory_from_config(config)

        # Composite memory (multiple strategies)
        config = {
            "type": "composite",
            "strategies": [
                {"type": "buffer", "max_messages": 50},
                {
                    "type": "vector",
                    "backend": "memory",
                    "dimension": 768,
                    "embedding_provider": "ollama",
                    "embedding_model": "nomic-embed-text",
                },
            ],
            "primary": 0,
        }
        memory = await create_memory_from_config(config)
        ```
    """
    memory_type = config.get("type", "buffer").lower()

    if memory_type == "buffer":
        return BufferMemory(max_messages=config.get("max_messages", 10))

    elif memory_type == "vector":
        return await VectorMemory.from_config(config)

    elif memory_type == "summary":
        # Track whether a dedicated provider was created (owns lifecycle)
        # vs reusing the bot's main LLM (bot owns lifecycle)
        has_dedicated_llm = "llm" in config
        summary_llm = await _resolve_summary_llm(config, llm_provider)
        return SummaryMemory(
            llm_provider=summary_llm,
            recent_window=config.get("recent_window", 10),
            summary_prompt=config.get("summary_prompt"),
            owns_llm_provider=has_dedicated_llm,
        )

    elif memory_type == "composite":
        strategy_configs = config.get("strategies", [])
        strategies: list[Memory] = []
        try:
            for strategy_config in strategy_configs:
                strategy = await create_memory_from_config(
                    strategy_config, llm_provider
                )
                strategies.append(strategy)
            if not strategies:
                raise ValueError(
                    "Composite memory requires at least one strategy "
                    "in 'strategies' list"
                )
            return CompositeMemory(
                strategies=strategies,
                primary_index=config.get("primary", 0),
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

    else:
        raise ValueError(
            f"Unknown memory type: {memory_type}. "
            f"Available types: buffer, composite, summary, vector"
        )


async def _resolve_summary_llm(
    config: dict[str, Any],
    fallback_provider: Any | None,
) -> Any:
    """Resolve the LLM provider for summary memory.

    If the config contains an ``llm`` section, a dedicated provider is
    created from it.  Otherwise the ``fallback_provider`` (typically the
    bot's own LLM) is used.

    Args:
        config: Summary memory configuration, may contain an ``llm`` key
        fallback_provider: Provider to use when no dedicated LLM is configured

    Returns:
        An initialised ``AsyncLLMProvider``

    Raises:
        ValueError: If neither a dedicated LLM config nor a fallback is available
    """
    if "llm" in config:
        from dataknobs_llm.llm import LLMProviderFactory

        factory = LLMProviderFactory(is_async=True)
        provider = factory.create(config["llm"])
        await provider.initialize()
        return provider

    if fallback_provider is not None:
        return fallback_provider

    raise ValueError(
        "Summary memory requires an LLM provider. Either include an 'llm' "
        "section in the memory config or pass llm_provider to "
        "create_memory_from_config()."
    )
