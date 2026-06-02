"""Vector-based semantic memory implementation."""

import logging
from datetime import datetime
from typing import Any, ClassVar
from uuid import uuid4

import numpy as np

from dataknobs_common.metadata import enforce_immutable_keys
from dataknobs_common.structured_config import StructuredConfigConsumer

from .base import Memory, apply_history_redactions, compile_history_redactions
from .config import VectorMemoryConfig

logger = logging.getLogger(__name__)


class VectorMemory(StructuredConfigConsumer[VectorMemoryConfig], Memory):
    """Vector-based semantic memory using dataknobs-data vector stores.

    This implementation stores messages with vector embeddings and retrieves
    relevant messages based on semantic similarity.

    Construct from config (``await VectorMemory.from_config({...})``,
    which builds and owns the store + embedder) or from pre-built
    collaborators (``VectorMemory.from_components(vector_store=…,
    embedding_provider=…)``, which adopts caller-owned resources).

    Attributes:
        vector_store: Vector store backend from dataknobs_data.vector.stores
        embedding_provider: LLM provider for generating embeddings
        max_results: Maximum number of results to return
        similarity_threshold: Minimum similarity score for results
    """

    CONFIG_CLS: ClassVar[type[VectorMemoryConfig]] = VectorMemoryConfig

    def _setup(self) -> None:
        """Initialize config-derived knobs shared by both build paths.

        Collaborators (``vector_store`` / ``embedding_provider``) and the
        ownership flags are bound by :meth:`_ainit` (config-driven build,
        owns the resources it creates) or :meth:`_adopt_components`
        (pre-built injection, caller-owned).
        """
        self.max_results = self.config.max_results
        self.similarity_threshold = self.config.similarity_threshold
        self._default_metadata = self.config.default_metadata or {}
        self._default_filter = self.config.default_filter or {}
        self._immutable_keys: frozenset[str] = (
            frozenset(self.config.immutable_metadata_keys)
            if self.config.immutable_metadata_keys
            else frozenset()
        )
        self.vector_store: Any = None
        self.embedding_provider: Any = None
        self._owns_vector_store = False
        self._owns_embedding_provider = False
        self._compiled_redactions = compile_history_redactions(
            self.config.history_redactions
        )

    @classmethod
    async def from_config(  # type: ignore[override]
        cls, config: Any, **components: Any
    ) -> "VectorMemory":
        """Create VectorMemory from configuration (async warmup).

        Builds the vector store and embedding provider, then awaits the
        store's ``initialize()``. The instance owns both resources, so
        :meth:`close` closes them. Accepts a config dict or a typed
        :class:`VectorMemoryConfig`. The config keys are:

        - ``backend``: Vector store backend type
        - ``dimension``: Vector store dimension (singular; default 1536)
        - ``collection``: Collection/index name (optional)
        - ``embedding``: Nested embedding config dict (preferred), e.g.
          ``{"provider": "ollama", "model": "nomic-embed-text",
          "dimensions": 768}``
        - ``embedding_provider`` / ``embedding_model``: Legacy flat keys.
          Note: ``dimensions`` (plural) is forwarded to the embedding
          provider, not the vector store. Use ``dimension`` (singular)
          for the vector store size.
        - ``max_results``: Max results to return (default 5)
        - ``similarity_threshold``: Min similarity score (default 0.7)

        Returns:
            Configured VectorMemory instance.
        """
        return await cls.from_config_async(config, **components)

    async def _ainit(self, **_: Any) -> None:
        """Build and own the vector store + embedding provider from config."""
        if self._prebuilt:
            return
        from dataknobs_data.vector.stores import VectorStoreFactory

        from ..providers import build_embedding_config, create_embedding_provider

        store_config: dict[str, Any] = {
            "backend": self.config.backend,
            "dimensions": self.config.dimension,
        }
        if self.config.collection is not None:
            store_config["collection_name"] = self.config.collection
        if self.config.persist_path is not None:
            store_config["persist_path"] = self.config.persist_path
        if self.config.store_params:
            store_config.update(self.config.store_params)

        factory = VectorStoreFactory()
        self.vector_store = factory.create(**store_config)
        await self.vector_store.initialize()
        self._owns_vector_store = True

        self.embedding_provider = await create_embedding_provider(
            build_embedding_config(
                embedding=self.config.embedding,
                embedding_provider=self.config.embedding_provider,
                embedding_model=self.config.embedding_model,
                dimensions=self.config.dimensions,
                api_base=self.config.api_base,
                api_key=self.config.api_key,
            )
        )
        self._owns_embedding_provider = True

    def _adopt_components(
        self,
        *,
        vector_store: Any = None,
        embedding_provider: Any = None,
        **_: Any,
    ) -> None:
        """Adopt caller-owned store + embedder for ``from_components``."""
        if vector_store is None or embedding_provider is None:
            raise TypeError(
                "VectorMemory.from_components requires vector_store and "
                "embedding_provider"
            )
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self._owns_vector_store = False
        self._owns_embedding_provider = False

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

        Configured ``history_redactions`` are applied to assistant-role
        result rows AFTER the similarity search, so stored vectors and
        vector-store rows are untouched and scoring is unaffected. The
        non-content keys (``role``, ``similarity``, ``metadata``) carry
        over unchanged.

        Args:
            current_message: Current message to find context for

        Returns:
            List of relevant message dictionaries sorted by similarity,
            with assistant content redacted per the configured
            ``history_redactions`` (passthrough when none are configured).
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
        context: list[dict[str, Any]] = []
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

        # Read-time redaction on assistant-role results. Stored vectors
        # and vector-store rows are untouched; redaction happens AFTER
        # the similarity search so it does not perturb scoring. The
        # dict-shape helper copies only assistant rows and rewrites only
        # their ``content`` key, so ``similarity`` / ``metadata`` survive.
        return apply_history_redactions(context, self._compiled_redactions)

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
