"""Knowledge base implementations for DynaBot."""

from typing import Any

from .rag import RAGKnowledgeBase
from .retrieval import (
    ChunkMerger,
    ContextFormatter,
    FormatterConfig,
    MergedChunk,
    MergerConfig,
)
from .query import (
    ContextualExpander,
    Message,
    QueryTransformer,
    TransformerConfig,
    create_transformer,
    is_ambiguous_query,
)

__all__ = [
    # Main knowledge base
    "RAGKnowledgeBase",
    "create_knowledge_base_from_config",
    # Retrieval utilities
    "ChunkMerger",
    "MergedChunk",
    "MergerConfig",
    "ContextFormatter",
    "FormatterConfig",
    # Query utilities
    "QueryTransformer",
    "TransformerConfig",
    "create_transformer",
    "ContextualExpander",
    "Message",
    "is_ambiguous_query",
]


async def create_knowledge_base_from_config(config: dict[str, Any]) -> RAGKnowledgeBase:
    """Create knowledge base from configuration.

    Args:
        config: Knowledge base configuration with:
            - type: Type of knowledge base (currently only 'rag' supported)
            - vector_store: Vector store configuration
            - embedding_provider: LLM provider for embeddings
            - embedding_model: Model to use for embeddings
            - chunking: Optional chunking configuration
            - documents_path: Optional path to load documents
            - document_pattern: Optional file pattern

    Returns:
        Configured knowledge base instance

    Raises:
        ValueError: If knowledge base type is not supported

    Example:
        ```python
        config = {
            "type": "rag",
            "vector_store": {
                "backend": "memory",
                "dimensions": 384
            },
            "embedding_provider": "echo",
            "embedding_model": "test"
        }
        kb = await create_knowledge_base_from_config(config)
        ```
    """
    kb_type = config.get("type", "rag").lower()

    if kb_type == "rag":
        return await RAGKnowledgeBase.from_config(config)
    else:
        raise ValueError(
            f"Unknown knowledge base type: {kb_type}. " f"Available types: rag"
        )
