"""Memory implementations for DynaBot."""

from typing import Any

from .base import Memory
from .buffer import BufferMemory
from .vector import VectorMemory

__all__ = ["Memory", "BufferMemory", "VectorMemory", "create_memory_from_config"]


async def create_memory_from_config(config: dict[str, Any]) -> Memory:
    """Create memory instance from configuration.

    Args:
        config: Memory configuration with 'type' field and type-specific params

    Returns:
        Configured Memory instance

    Raises:
        ValueError: If memory type is not recognized

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
        ```
    """
    memory_type = config.get("type", "buffer").lower()

    if memory_type == "buffer":
        return BufferMemory(max_messages=config.get("max_messages", 10))

    elif memory_type == "vector":
        return await VectorMemory.from_config(config)

    else:
        raise ValueError(
            f"Unknown memory type: {memory_type}. "
            f"Available types: buffer, vector"
        )
