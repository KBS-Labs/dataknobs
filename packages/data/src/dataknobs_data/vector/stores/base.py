"""Base class for specialized vector stores."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from ...fields import VectorField
from ...records import Record
from ..types import VectorSearchResult
from .common import VectorStoreBase

if TYPE_CHECKING:
    import numpy as np
    from collections.abc import Callable


class VectorStore(ABC, VectorStoreBase):
    """Abstract base class for specialized vector stores.

    This provides a dedicated vector storage backend that can be used
    independently or alongside traditional databases. It inherits from
    VectorStoreBase which provides common configuration parsing and
    utility methods.

    Async transport contract:
        The async methods (``initialize``, ``add_vectors``, ``search``,
        persist paths, ...) MUST NOT block the event loop. Use an async
        transport or offload blocking ``open()`` / ``pickle`` disk I/O via
        ``asyncio.to_thread``; ruff's ``ASYNC`` family enforces this and
        ``assert_no_blocking()`` proves it. See ``MemoryVectorStore.save``
        for the offloaded-persist reference.
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the vector store (create index, connect, etc.)."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close connections and clean up resources.

        Ownership contract for backing resources (connection pools,
        clients, sessions): a store that *built* its own backing resource
        — constructed from config, a connection string, or the
        ``VectorStoreFactory`` — owns that resource and closes it here. A
        store that was handed an *externally supplied* backing resource
        (e.g. a pool injected by a consumer that shares it across several
        stores) does **not** own that resource and must leave it open,
        releasing only per-store state. This lets a consumer share one
        pool across many stores and close each store independently without
        tearing down a resource the others still depend on.

        Backends with no *injectable* external resource satisfy this
        trivially: the in-memory, FAISS, and Chroma stores build their
        backing index/client internally, so they own whatever they hold and
        there is nothing a caller could share. Only a store that accepts an
        injected resource — today ``PgVectorStore`` with a caller-supplied
        asyncpg pool — needs an explicit ownership gate. A new backend that
        adds an injectable resource must follow the same pattern: track
        ownership at construction and skip teardown of caller-owned
        resources here.

        **This may raise.** A store with a ``persist_path`` persists here
        as well as releasing, and that write can fail — most notably with
        ``ConcurrencyError`` when another instance has written the file
        since (see ``VECTOR_FILTER_SEMANTICS.md``). Releasing and
        persisting are separate obligations, so an implementation must
        release regardless: the exception propagates, because rows are
        being lost, but a second ``close()`` must not raise again on a
        store that is already released.
        """
        pass

    @abstractmethod
    async def add_vectors(
        self,
        vectors: np.ndarray | list[np.ndarray],
        ids: list[str] | None = None,
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """Add vectors to the store, upserting on id conflict.

        An **empty batch is a no-op**, not an error: it writes nothing
        and returns ``[]``. An empty batch is something a caller
        produces rather than intends — a comprehension that filtered
        everything out, a chunker handed a blank document — so requiring
        an ``if items:`` guard at every call site only moves the check.
        Both ``[]`` and ``np.array([])`` count as empty. Implementations
        get this from ``VectorStoreBase._is_empty_batch``.

        A configured ``domain_id`` is defaulted into every row written
        that does not carry one of its own.

        Args:
            vectors: Vector(s) to add
            ids: Optional IDs for vectors (generated if not provided)
            metadata: Optional metadata for each vector

        Returns:
            List of IDs for the added vectors, empty for an empty batch
        """
        pass

    @abstractmethod
    async def get_vectors(
        self,
        ids: list[str],
        include_metadata: bool = True,
        include_timestamps: bool = False,
    ) -> list[tuple[np.ndarray | None, dict[str, Any] | None]]:
        """Retrieve vectors by ID.

        Args:
            ids: Vector IDs to retrieve
            include_metadata: Whether to include metadata
            include_timestamps: When True, inject ``_created_at`` and
                ``_updated_at`` (or configured keys) into each returned
                metadata dict, formatted per ``timestamps.format``
                config. Silently no-op when ``include_metadata=False``.

        Returns:
            One ``(vector, metadata)`` tuple per requested id, in the
            order asked for. An id the store does not hold yields
            ``(None, None)`` rather than being omitted, so the result
            stays positionally aligned with ``ids`` — which is why the
            vector is optional. Every backend already did this; the
            annotation said otherwise, and two of the four had widened it
            on their own.
        """
        pass

    @abstractmethod
    async def delete_vectors(self, ids: list[str]) -> int:
        """Delete vectors by ID.

        Args:
            ids: Vector IDs to delete

        Returns:
            Number of vectors deleted
        """
        pass

    @abstractmethod
    async def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        filter: dict[str, Any] | None = None,
        include_metadata: bool = True,
        include_timestamps: bool = False,
    ) -> list[tuple[str, float, dict[str, Any] | None]]:
        """Search for similar vectors.

        Args:
            query_vector: Query vector
            k: Number of results
            filter: Optional metadata filter
            include_metadata: Whether to include metadata
            include_timestamps: When True, inject ``_created_at`` and
                ``_updated_at`` (or configured keys) into each returned
                metadata dict, formatted per ``timestamps.format``
                config. Silently no-op when ``include_metadata=False``.

        Returns:
            List of (id, score, metadata) tuples
        """
        pass

    @abstractmethod
    async def update_metadata(
        self,
        ids: list[str],
        metadata: list[dict[str, Any]],
    ) -> int:
        """Replace metadata for existing vectors, by id.

        The supplied dict becomes the row's metadata **outright**: a key
        the row holds and the caller omits is *removed*, not retained.
        This is the opposite of :meth:`update_metadata_where`, whose
        contract is a merge, and the distinction is stated here rather
        than left to each backend because leaving it implicit is what
        allowed a shipped backend to read it as a merge — a key omitted
        from the same consumer code disappeared on three backends and
        survived on the fourth.

        A configured ``domain_id`` is preserved across the replacement
        rather than being one of the keys dropped, so a caller updating
        an unrelated field does not push the row out of its own scope.
        An id outside the configured scope is not updated and does not
        count toward the return value.

        Backends carrying metadata in a store with a narrower value
        domain than Python's may not round-trip every value; where that
        is so it is documented on the backend. ``ChromaVectorStore``
        cannot store a ``None`` value, because deleting a key and
        setting it to ``None`` are the same operation in chromadb's
        update API.

        Args:
            ids: Vector IDs to update
            metadata: The complete replacement metadata for each vector

        Returns:
            Number of vectors updated
        """
        pass

    @abstractmethod
    async def count(self, filter: dict[str, Any] | None = None) -> int:
        """Count vectors in the store.

        Args:
            filter: Optional metadata filter

        Returns:
            Number of vectors matching filter
        """
        pass

    @abstractmethod
    async def clear(self, filter: dict[str, Any] | None = None) -> None:
        """Clear vectors from the store.

        Args:
            filter: Optional metadata filter.  When ``None`` (default),
                all vectors are removed — preserving the historical
                unscoped behavior.  When provided, only vectors whose
                metadata matches the filter are removed; non-matching
                vectors are preserved.

        The filter shape is the same as for :meth:`search` and
        :meth:`count` — backend-specific operator support matches
        each backend's existing filter-translation capabilities.
        """
        pass

    async def update_metadata_where(
        self,
        filter: dict[str, Any] | None,
        set_: dict[str, Any],
    ) -> int:
        """Bulk-merge ``set_`` into the metadata of every matching vector.

        The filter-keyed sibling of :meth:`update_metadata` (which is
        id-keyed). ``filter`` has the same shape and four-quadrant
        semantics as :meth:`clear` / :meth:`count` / :meth:`search`;
        ``None`` matches every vector. ``set_`` is *merged* into each
        matched vector's existing metadata (existing keys overwritten,
        absent keys added) — it does not replace the metadata wholesale.

        Args:
            filter: Metadata filter selecting the rows to update.
            set_: Key/value pairs merged into each matched row's metadata.

        Returns:
            Number of vectors whose metadata was updated.

        Note:
            The default raises ``NotImplementedError``. This is the
            contract for **out-of-tree** vector stores only — it makes
            an unported backend fail loudly rather than silently skip a
            tombstone swap. Every in-tree store (Memory, FAISS,
            PgVector, Chroma) overrides this with a real implementation,
            so the default is never reached for a backend DataKnobs
            ships.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support update_metadata_where()")

    async def metadata_fields(self) -> set[str]:
        """Discover metadata field names present across stored vectors.

        Scans metadata for all stored vectors and returns the union of
        all field names.  Useful for introspection — e.g. detecting
        whether heading metadata (``headings``, ``heading_levels``) is
        available for topic-index construction.

        Returns:
            Set of metadata field names found across all vectors.

        Note:
            The default raises ``NotImplementedError`` rather than
            returning an empty set.  This is intentional: an empty set
            means "no metadata exists," while ``NotImplementedError``
            means "this backend cannot answer the question."  Returning
            ``set()`` would silently mislead consumers into concluding
            that heading metadata is absent when the backend simply
            doesn't support introspection.  Callers should catch
            ``NotImplementedError`` and decide what "unknown" means
            for their use case.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support metadata_fields()")

    async def update_vectors(
        self,
        vectors: np.ndarray | list[np.ndarray],
        ids: list[str],
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """Update existing vectors by ID.

        A thin alias for :meth:`add_vectors`, which upserts: on every
        shipping backend a same-id write replaces the row's metadata
        outright rather than merging into it, so re-adding *is* the
        update.

        This used to delete first. The delete was there to guarantee
        the replacement, which ``add_vectors`` already guarantees, and
        it cost two things to buy nothing:

        * **It discarded the row's ``created_at``.** Deleting takes the
          timestamp tracking with the row, so the re-add had nothing to
          preserve and stamped a fresh creation date — breaking the
          documented rule that ``created_at`` survives every write to a
          tracked id, and turning a re-ingest sweep into a rewrite of
          every row's creation date.
        * **It destroyed in-scope rows on a refused batch.** A scoped
          ``delete_vectors`` skips an out-of-domain id and deletes the
          rest; ``add_vectors`` raises on one. A mixed batch therefore
          deleted the caller's own row and then declined to restore it.

        Args:
            vectors: New vector values
            ids: IDs of vectors to update
            metadata: Optional new metadata

        Returns:
            List of updated IDs
        """
        return await self.add_vectors(vectors, ids, metadata)

    # Higher-level convenience methods

    async def add_records(
        self,
        records: list[Record],
        vector_field: str = "embedding",
        include_fields: list[str] | None = None,
    ) -> list[str]:
        """Add records with vector fields to the store.

        Args:
            records: Records containing vector fields
            vector_field: Name of the vector field
            include_fields: Fields to include in metadata

        Returns:
            List of IDs for added vectors
        """
        vectors = []
        ids = []
        metadatas = []

        for record in records:
            # Extract vector
            if vector_field not in record.fields:
                continue

            vector_obj = record.fields[vector_field]
            if not isinstance(vector_obj, VectorField):
                continue

            # Skip records without IDs
            if record.id is None:
                continue

            vectors.append(vector_obj.value)
            ids.append(record.id)

            # Build metadata
            metadata = {"record_id": record.id}

            # Add source field if present
            if vector_obj.source_field:
                metadata["source_field"] = vector_obj.source_field
                # Include source text if available
                if vector_obj.source_field in record.fields:
                    metadata["source_text"] = record.get_value(vector_obj.source_field)

            # Add model info if present
            if vector_obj.model_name:
                metadata["model_name"] = vector_obj.model_name
            if vector_obj.model_version:
                metadata["model_version"] = vector_obj.model_version

            # Add requested fields
            if include_fields:
                for field_name in include_fields:
                    if field_name in record.fields:
                        metadata[field_name] = record.get_value(field_name)

            metadatas.append(metadata)

        if vectors:
            return await self.add_vectors(vectors, ids=ids, metadata=metadatas)
        return []

    async def search_similar_records(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        filter: dict[str, Any] | None = None,
        fetch_records: Callable[[list[str]], list[Record]] | None = None,
    ) -> list[VectorSearchResult]:
        """Search and return results as VectorSearchResult objects.

        Args:
            query_vector: Query vector
            k: Number of results
            filter: Optional metadata filter
            fetch_records: Optional function to fetch full records

        Returns:
            List of VectorSearchResult objects
        """
        results = await self.search(query_vector, k=k, filter=filter, include_metadata=True)

        search_results = []
        record_ids = []

        for vector_id, _score, metadata in results:
            record_id = metadata.get("record_id", vector_id) if metadata else vector_id
            record_ids.append(record_id)

        # Fetch full records if function provided
        records_map = {}
        if fetch_records and record_ids:
            records = fetch_records(record_ids)
            records_map = {r.id: r for r in records}

        for vector_id, score, metadata in results:
            record_id = metadata.get("record_id", vector_id) if metadata else vector_id

            # Get or create record
            if record_id in records_map:
                record = records_map[record_id]
            else:
                # Create minimal record with metadata
                record = Record({"id": record_id})
                if metadata:
                    for key, value in metadata.items():
                        if key not in ["record_id", "source_text", "source_field"]:
                            record.fields[key] = value

            # Extract source text
            source_text = None
            if metadata:
                source_text = metadata.get("source_text")

            search_results.append(
                VectorSearchResult(
                    record=record,
                    score=score,
                    source_text=source_text,
                    vector_field=metadata.get("source_field") if metadata else None,
                    metadata=metadata or {},
                )
            )

        return search_results

    async def bulk_embed_and_store(
        self,
        texts: list[str],
        embedding_fn: Callable[[list[str]], np.ndarray],
        ids: list[str] | None = None,
        metadata: list[dict[str, Any]] | None = None,
        batch_size: int | None = None,
    ) -> list[str]:
        """Embed texts and store vectors.

        Args:
            texts: Texts to embed
            embedding_fn: Function to generate embeddings
            ids: Optional IDs for vectors
            metadata: Optional metadata for each vector
            batch_size: Batch size for embedding

        Returns:
            List of IDs for added vectors
        """
        batch_size = batch_size or self.batch_size
        all_ids = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_ids = ids[i : i + batch_size] if ids else None
            batch_metadata = metadata[i : i + batch_size] if metadata else None

            # Generate embeddings
            embeddings = embedding_fn(batch_texts)

            # Add source text to metadata
            if batch_metadata is None:
                batch_metadata = [{} for _ in batch_texts]

            for j, text in enumerate(batch_texts):
                batch_metadata[j]["source_text"] = text

            # Store vectors
            stored_ids = await self.add_vectors(embeddings, ids=batch_ids, metadata=batch_metadata)
            all_ids.extend(stored_ids)

        return all_ids
