"""Vector-based semantic memory implementation."""

import logging
from collections.abc import Iterable
from datetime import datetime
from typing import Any
from uuid import uuid4

import numpy as np

from dataknobs_common.metadata import enforce_immutable_keys

from .base import Memory

logger = logging.getLogger(__name__)


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
        default_metadata: dict[str, Any] | None = None,
        default_filter: dict[str, Any] | None = None,
        immutable_metadata_keys: Iterable[str] | None = None,
        owns_embedding_provider: bool = False,
        owns_vector_store: bool = False,
    ):
        """Initialize vector memory.

        Args:
            vector_store: Vector store backend instance
            embedding_provider: LLM provider with embed() method
            max_results: Maximum number of similar messages to return
            similarity_threshold: Minimum similarity score (0-1)
            default_metadata: Metadata merged into every ``add_message()``
                call. Caller-supplied metadata overrides these defaults
                unless the key is listed in ``immutable_metadata_keys``.
                Use for tenant scoping, e.g. ``{"user_id": "u123"}``.
            default_filter: Filter merged into every ``get_context()``
                search call. Use to scope reads to a tenant, e.g.
                ``{"user_id": "u123"}``.
            immutable_metadata_keys: Optional iterable of keys whose
                ``default_metadata`` values cannot be overridden by
                caller-supplied ``metadata`` on ``add_message()``.  Use
                for tenant-scoping identifiers paired with
                ``default_metadata`` — e.g.
                ``immutable_metadata_keys=["user_id"]`` paired with
                ``default_metadata={"user_id": "u123"}``.  Caller
                attempts to override an immutable key are logged as
                warnings and the configured value is preserved.
            owns_embedding_provider: If True, ``close()`` will close the
                embedding provider. Set by ``from_config`` for resources
                it creates. Default False for externally-injected providers.
            owns_vector_store: If True, ``close()`` will close the vector
                store. Set by ``from_config`` for resources it creates.
                Default False for externally-injected stores.
        """
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.max_results = max_results
        self.similarity_threshold = similarity_threshold
        self._default_metadata = default_metadata or {}
        self._default_filter = default_filter or {}
        self._immutable_keys: frozenset[str] = (
            frozenset(immutable_metadata_keys)
            if immutable_metadata_keys
            else frozenset()
        )
        self._owns_embedding_provider = owns_embedding_provider
        self._owns_vector_store = owns_vector_store

    @classmethod
    async def from_config(cls, config: dict[str, Any]) -> "VectorMemory":
        """Create VectorMemory from configuration.

        Args:
            config: Configuration dictionary with:
                - backend: Vector store backend type
                - dimension: Vector store dimension (singular; default 1536)
                - collection: Collection/index name (optional)
                - embedding: Nested embedding config dict (preferred), e.g.
                  ``{"provider": "ollama", "model": "nomic-embed-text",
                  "dimensions": 768}``
                - embedding_provider / embedding_model: Legacy flat keys.
                  Note: ``dimensions`` (plural) at the top level is forwarded
                  to the embedding provider, not the vector store.  Use
                  ``dimension`` (singular) for the vector store size.
                - max_results: Max results to return (default 5)
                - similarity_threshold: Min similarity score (default 0.7)

        Returns:
            Configured VectorMemory instance
        """
        from dataknobs_data.vector.stores import VectorStoreFactory

        from ..providers import create_embedding_provider

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
        embedding_provider = await create_embedding_provider(config)

        return cls(
            vector_store=vector_store,
            embedding_provider=embedding_provider,
            max_results=config.get("max_results", 5),
            similarity_threshold=config.get("similarity_threshold", 0.7),
            default_metadata=config.get("default_metadata"),
            default_filter=config.get("default_filter"),
            immutable_metadata_keys=config.get("immutable_metadata_keys"),
            owns_embedding_provider=True,
            owns_vector_store=True,
        )

    async def add_message(
        self, content: str, role: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """Add message with vector embedding.

        Args:
            content: Message content
            role: Message role
            metadata: Optional caller-supplied metadata. Merged after
                ``default_metadata`` (from init) and system base fields
                (``content``, ``role``, ``timestamp``, ``id``).
                Caller metadata has highest precedence, **except** for
                keys listed in ``immutable_metadata_keys`` (passed at
                construction): for those keys the value from
                ``default_metadata`` always wins and any caller-supplied
                value is discarded with a WARNING log naming the key.
                Used for tenant-scoping identifiers (e.g. ``user_id``,
                ``domain_id``) that callers must not be able to bypass.
        """
        # Generate embedding
        embedding = await self.embedding_provider.embed(content)

        # Convert to numpy array if needed
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding, dtype=np.float32)

        # Merge order: defaults < base fields (system-controlled) < caller metadata
        msg_metadata = dict(self._default_metadata)
        msg_metadata.update({
            "content": content,
            "role": role,
            "timestamp": datetime.now().isoformat(),
            "id": str(uuid4()),
        })
        if metadata:
            msg_metadata.update(metadata)

        # Enforce immutable keys after the layered merge — caller
        # values for keys named in ``immutable_metadata_keys`` are
        # discarded with a warning, restoring the ``default_metadata``
        # value.  Used for tenant-scoping identifiers that callers
        # must not be able to bypass.
        if self._immutable_keys:
            enforce_immutable_keys(
                target=msg_metadata,
                caller=metadata,
                source=self._default_metadata,
                keys=self._immutable_keys,
                logger=logger,
                context="VectorMemory.add_message",
            )

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
        search_kwargs: dict[str, Any] = {
            "query_vector": query_embedding,
            "k": self.max_results,
            "include_metadata": True,
        }
        if self._default_filter:
            search_kwargs["filter"] = dict(self._default_filter)

        results = await self.vector_store.search(**search_kwargs)

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

    def providers(self) -> dict[str, Any]:
        """Return the embedding provider, keyed by role."""
        from dataknobs_bots.bot.base import PROVIDER_ROLE_MEMORY_EMBEDDING

        if self.embedding_provider is not None:
            return {PROVIDER_ROLE_MEMORY_EMBEDDING: self.embedding_provider}
        return {}

    def set_provider(self, role: str, provider: Any) -> bool:
        """Replace the embedding provider if the role matches."""
        from dataknobs_bots.bot.base import PROVIDER_ROLE_MEMORY_EMBEDDING

        if role == PROVIDER_ROLE_MEMORY_EMBEDDING:
            self.embedding_provider = provider
            return True
        return False

    async def close(self) -> None:
        """Close owned resources.

        Only closes resources that this instance owns (created in
        ``from_config``). Externally-injected resources are left open
        for the caller to manage.
        """
        if (
            self._owns_embedding_provider
            and self.embedding_provider
            and hasattr(self.embedding_provider, "close")
        ):
            try:
                await self.embedding_provider.close()
            except Exception:
                logger.exception("Error closing embedding provider")

        if (
            self._owns_vector_store
            and self.vector_store
            and hasattr(self.vector_store, "close")
        ):
            try:
                await self.vector_store.close()
            except Exception:
                logger.exception("Error closing vector store")

    async def clear(
        self, filter_metadata: dict[str, Any] | None = None
    ) -> None:
        """Clear vectors from this memory.

        The effective filter AND-composes ``default_filter`` and
        ``filter_metadata`` so an explicit filter narrows WITHIN the
        tenant scope and never escapes it:

        * ``mem.clear()`` on a memory constructed with
          ``default_filter={"user_id": "u1"}`` removes only u1's
          records — symmetric with ``mem.get_context()``.
        * ``mem.clear(filter_metadata={"category": "A"})`` on the
          same memory removes records matching BOTH ``user_id=u1``
          AND ``category=A``.
        * ``mem.clear()`` on a memory with no ``default_filter``
          removes every vector in the backing store — the historical
          behavior.

        On key collision (e.g. caller passes a ``user_id`` that does
        not match the memory's tenant), the merged filter contains
        contradictory clauses and matches nothing — the clear is a
        no-op rather than a cross-tenant wipe.

        For consumers who genuinely want the all-tenants wipe on a
        ``VectorMemory`` constructed with ``default_filter``, call
        ``mem.vector_store.clear()`` directly to bypass the wrapper.

        Args:
            filter_metadata: Optional explicit filter.  AND-composed
                with ``default_filter`` rather than replacing it.

        Raises:
            NotImplementedError: If the backing vector store does
                not support ``clear()``.
        """
        if not hasattr(self.vector_store, "clear"):
            raise NotImplementedError(
                "Vector store does not support clearing. "
                "Consider creating a new VectorMemory instance with a "
                "fresh collection."
            )

        effective_filter: dict[str, Any] | None
        if self._default_filter and filter_metadata is not None:
            # AND-compose: caller-supplied keys narrow within the
            # tenant scope. If a caller key collides with a default
            # key, materialize a contradiction (a list of both
            # values) so the underlying ``_match_metadata_filter``
            # accepts only records matching BOTH (impossible when
            # values differ → no-op clear, never cross-tenant wipe).
            effective_filter = dict(self._default_filter)
            for key, value in filter_metadata.items():
                if key in effective_filter and effective_filter[key] != value:
                    # Contradiction: list-valued key matches nothing
                    # under four-quadrant semantics ([a, b] vs scalar
                    # x: x must be in [a, b]; we put both required
                    # values in the list, but the metadata is scalar
                    # and equals only one — so no record matches).
                    # Use empty list which never matches.
                    effective_filter[key] = []
                else:
                    effective_filter[key] = value
        elif filter_metadata is not None:
            effective_filter = dict(filter_metadata)
        elif self._default_filter:
            effective_filter = dict(self._default_filter)
        else:
            effective_filter = None

        await self.vector_store.clear(filter=effective_filter)
