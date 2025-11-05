"""Knowledge base implementations for DynaBot."""

from typing import Any

from .rag import RAGKnowledgeBase

__all__ = ["RAGKnowledgeBase", "create_knowledge_base_from_config"]


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
