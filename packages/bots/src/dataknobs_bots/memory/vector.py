"""Vector-based semantic memory implementation."""

from datetime import datetime
from typing import Any
from uuid import uuid4

import numpy as np

from .base import Memory


class VectorMemory(Memory):
    """Vector-based semantic memory using dataknobs-data vector stores.

    This implementation stores messages with vector embeddings and retrieves
    relevant messages based on semantic similarity.

    Attributes:
        vector_store: Vector store backend from dataknobs_data.vector.stores
        embedding_provider: LLM provider for generating embeddings
        max_results: Maximum number of results to return
        similarity_threshold: Minimum similarity score for results
    """

    def __init__(
        self,
        vector_store: Any,
        embedding_provider: Any,
        max_results: int = 5,
        similarity_threshold: float = 0.7,
    ):
        """Initialize vector memory.

        Args:
            vector_store: Vector store backend instance
            embedding_provider: LLM provider with embed() method
            max_results: Maximum number of similar messages to return
            similarity_threshold: Minimum similarity score (0-1)
        """
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.max_results = max_results
        self.similarity_threshold = similarity_threshold

    @classmethod
    async def from_config(cls, config: dict[str, Any]) -> "VectorMemory":
        """Create VectorMemory from configuration.

        Args:
            config: Configuration dictionary with:
                - backend: Vector store backend type
                - dimension: Vector dimension (optional, depends on backend)
                - collection: Collection/index name (optional)
                - embedding_provider: LLM provider name for embeddings
                - embedding_model: Model to use for embeddings
                - max_results: Max results to return (default 5)
                - similarity_threshold: Min similarity score (default 0.7)

        Returns:
            Configured VectorMemory instance
        """
        from dataknobs_data.vector.stores import VectorStoreFactory
        from dataknobs_llm.llm import LLMProviderFactory

        # Create vector store
        store_config = {
            "backend": config.get("backend", "memory"),
            "dimensions": config.get("dimension", 1536),
        }

        # Add optional store parameters
        if "collection" in config:
            store_config["collection_name"] = config["collection"]
        if "persist_path" in config:
            store_config["persist_path"] = config["persist_path"]

        # Merge any additional store_params
        if "store_params" in config:
            store_config.update(config["store_params"])

        factory = VectorStoreFactory()
        vector_store = factory.create(**store_config)
        await vector_store.initialize()

        # Create embedding provider
        llm_factory = LLMProviderFactory(is_async=True)
        embedding_provider = llm_factory.create({
            "provider": config.get("embedding_provider", "openai"),
            "model": config.get("embedding_model", "text-embedding-ada-002"),
        })
        await embedding_provider.initialize()

        return cls(
            vector_store=vector_store,
            embedding_provider=embedding_provider,
            max_results=config.get("max_results", 5),
            similarity_threshold=config.get("similarity_threshold", 0.7),
        )

    async def add_message(
        self, content: str, role: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """Add message with vector embedding.

        Args:
            content: Message content
            role: Message role
            metadata: Optional metadata
        """
        # Generate embedding
        embedding = await self.embedding_provider.embed(content)

        # Convert to numpy array if needed
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding, dtype=np.float32)

        # Prepare metadata
        msg_metadata = {
            "content": content,
            "role": role,
            "timestamp": datetime.now().isoformat(),
            "id": str(uuid4()),
        }
        if metadata:
            msg_metadata.update(metadata)

        # Store in vector store
        await self.vector_store.add_vectors(
            vectors=[embedding], ids=[msg_metadata["id"]], metadata=[msg_metadata]
        )

    async def get_context(self, current_message: str) -> list[dict[str, Any]]:
        """Get semantically relevant messages.

        Args:
            current_message: Current message to find context for

        Returns:
            List of relevant message dictionaries sorted by similarity
        """
        # Generate query embedding
        query_embedding = await self.embedding_provider.embed(current_message)

        # Convert to numpy array if needed
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding, dtype=np.float32)

        # Search for similar vectors
        results = await self.vector_store.search(
            query_vector=query_embedding,
            k=self.max_results,
            include_metadata=True,
        )

        # Format results
        context = []
        for _vector_id, similarity, msg_metadata in results:
            if msg_metadata and similarity >= self.similarity_threshold:
                context.append(
                    {
                        "content": msg_metadata.get("content", ""),
                        "role": msg_metadata.get("role", ""),
                        "similarity": similarity,
                        "metadata": msg_metadata,
                    }
                )

        return context

    async def clear(self) -> None:
        """Clear all vectors from memory.

        Note: This deletes all vectors in the store. Use with caution
        if the store is shared across multiple memory instances.
        """
        # Get all vector IDs and delete them
        # Note: This is a simplified implementation
        # In production, you might want to track IDs separately
        # or use collection-level clearing if supported
        if hasattr(self.vector_store, "clear"):
            await self.vector_store.clear()
        else:
            # Fallback: delete individual vectors if we track them
            # For now, we'll raise an error suggesting to use a new instance
            raise NotImplementedError(
                "Vector store does not support clearing. "
                "Consider creating a new VectorMemory instance with a fresh collection."
            )
