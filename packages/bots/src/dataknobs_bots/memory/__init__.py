"""Memory implementations for DynaBot."""

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
from .bank import AsyncMemoryBank, BankRecord, EmptyBankProxy, MemoryBank
from .base import Memory
from .buffer import BufferMemory
from .summary import SummaryMemory
from .vector import VectorMemory

__all__ = [
    "ArtifactBank",
    "AsyncMemoryBank",
    "BankRecord",
    "BufferMemory",
    "EmptyBankProxy",
    "Memory",
    "MemoryBank",
    "SummaryMemory",
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
            "dimension": 1536,
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-small"
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
        ```
    """
    memory_type = config.get("type", "buffer").lower()

    if memory_type == "buffer":
        return BufferMemory(max_messages=config.get("max_messages", 10))

    elif memory_type == "vector":
        return await VectorMemory.from_config(config)

    elif memory_type == "summary":
        summary_llm = await _resolve_summary_llm(config, llm_provider)
        return SummaryMemory(
            llm_provider=summary_llm,
            recent_window=config.get("recent_window", 10),
            summary_prompt=config.get("summary_prompt"),
        )

    else:
        raise ValueError(
            f"Unknown memory type: {memory_type}. "
            f"Available types: buffer, summary, vector"
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
