"""Mixins and protocols for vector-capable databases."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Protocol

from ..fields import FieldType, VectorField
from .content import (
    CONTENT_HASH_KEY,
    DEFAULT_FIELD_SEPARATOR,
    assemble_source_text,
    compute_content_hash,
    content_hash_metadata,
    current_content_hash,
    stored_assembly,
)
from .hybrid import (
    FusionStrategy,
    HybridSearchConfig,
    HybridSearchResult,
    reciprocal_rank_fusion,
    weighted_score_fusion,
)
from .types import DistanceMetric, VectorSearchResult

if TYPE_CHECKING:
    import numpy as np
    from collections.abc import Callable
    from ..query import Query
    from ..records import Record


class VectorCapable(Protocol):
    """Protocol for backends that can handle vector operations."""

    async def has_vector_support(self) -> bool:
        """Check if backend has vector support available.

        Returns:
            True if vector operations are supported
        """
        ...

    async def enable_vector_support(self) -> bool:
        """Enable vector support (install extensions, configure indices, etc.).

        Returns:
            True if vector support was successfully enabled
        """
        ...

    async def detect_vector_fields(self, record: Record) -> list[str]:
        """Detect vector fields in a record.

        Args:
            record: Record to examine

        Returns:
            List of field names that contain vectors
        """
        return [
            field_name
            for field_name, field_obj in record.fields.items()
            if field_obj.type in (FieldType.VECTOR, FieldType.SPARSE_VECTOR)
        ]

    def get_vector_config(self) -> dict[str, Any]:
        """Get vector-specific configuration for this backend.

        Returns:
            Dictionary of vector configuration options
        """
        return {}


# --- the parts of the vector surface that do no awaiting ------------------
#
# There are two vector-operations mixins, sync and async, and the reason they
# are not one ~200-line pair of near-copies is the same reason `bulk_embed_mixin`
# is not: everything that is not the awaiting lives here, and each mixin is the
# thin driver that supplies it. What differs between the lanes is then visible
# as the whole of what differs.


def vector_field_for(
    vector_field: str,
    vector: np.ndarray | list[float],
    metadata: dict[str, Any] | None,
) -> VectorField:
    """The field :meth:`update_vector` writes, in either lane."""
    return VectorField(name=vector_field, value=vector, metadata=metadata)


def default_vector_index_stats(vector_field: str) -> dict[str, Any]:
    """What a backend with no index of its own reports about one."""
    return {"field": vector_field, "indexed": False, "vector_count": 0}


def resolve_hybrid_config(
    config: HybridSearchConfig | None,
    native_supported: bool,
) -> HybridSearchConfig:
    """Settle the fusion strategy before either lane runs a search.

    ``NATIVE`` is a request a backend may not be able to honour; asking it
    is the one awaited step, and is the caller's, so this takes the answer.
    """
    config = config or HybridSearchConfig()
    if config.fusion_strategy == FusionStrategy.NATIVE and not native_supported:
        return HybridSearchConfig(
            text_weight=config.text_weight,
            vector_weight=config.vector_weight,
            fusion_strategy=FusionStrategy.RRF,
            rrf_k=config.rrf_k,
            text_fields=config.text_fields,
        )
    return config


def hybrid_fetch_k(k: int) -> int:
    """How many results to fetch per arm, before fusion narrows to ``k``."""
    return min(k * 3, 100)


def hybrid_text_query(
    query_text: str,
    text_fields: list[str] | None,
    k: int,
    filter: Query | None = None,
) -> Query:
    """Build the text-arm query the default hybrid search runs.

    Only the first field is matched: this is the fallback for backends with
    no full-text search, and an OR across fields is what those backends
    should override this to express.
    """
    from ..query import Filter, Operator, Query

    query = filter.copy() if filter else Query()
    query.limit_value = k

    if text_fields:
        for field in text_fields[:1]:
            query.filters.append(
                Filter(field=field, operator=Operator.LIKE, value=f"%{query_text}%")
            )

    return query


def score_text_matches(
    records: list[Record],
    query_text: str,
    text_fields: list[str] | None,
) -> list[tuple[Record, float]]:
    """Assign the default relevance scores to a text-arm result set."""
    results: list[tuple[Record, float]] = []
    query_lower = query_text.lower()

    for i, record in enumerate(records):
        # Rank-based, then boosted for a substring and again for an exact match.
        score = 1.0 / (i + 1)

        for field in text_fields or []:
            value = record.get_value(field)
            if value and isinstance(value, str):
                if query_lower in value.lower():
                    score *= 1.5
                if query_lower == value.lower():
                    score *= 2.0

        results.append((record, min(score, 1.0)))

    return results


def fuse_hybrid_results(
    text_results: list[tuple[Record, float]],
    vector_results: list[VectorSearchResult],
    config: HybridSearchConfig,
    k: int,
) -> list[HybridSearchResult]:
    """Combine the two arms into one ranking. No I/O, so no lane."""
    records_by_id: dict[str, Record] = {}
    text_scores: list[tuple[str, float]] = []
    vector_scores: list[tuple[str, float]] = []

    for record, score in text_results:
        record_id = record.id or record.storage_id
        if record_id:
            records_by_id[record_id] = record
            text_scores.append((record_id, score))

    for result in vector_results:
        record_id = result.record.id or result.record.storage_id
        if record_id:
            records_by_id[record_id] = result.record
            vector_scores.append((record_id, result.score))

    if config.fusion_strategy == FusionStrategy.RRF:
        fused = reciprocal_rank_fusion(
            text_results=text_scores,
            vector_results=vector_scores,
            k=config.rrf_k,
            text_weight=config.text_weight,
            vector_weight=config.vector_weight,
        )
    else:  # WEIGHTED_SUM
        text_w, vector_w = config.normalize_weights()
        fused = weighted_score_fusion(
            text_results=text_scores,
            vector_results=vector_scores,
            text_weight=text_w,
            vector_weight=vector_w,
            normalize_scores=True,
        )

    text_score_map = dict(text_scores)
    vector_score_map = dict(vector_scores)
    text_rank_map = {rid: i + 1 for i, (rid, _) in enumerate(text_scores)}
    vector_rank_map = {rid: i + 1 for i, (rid, _) in enumerate(vector_scores)}

    results: list[HybridSearchResult] = []
    for record_id, combined_score in fused[:k]:
        if record_id not in records_by_id:
            continue

        results.append(
            HybridSearchResult(
                record=records_by_id[record_id],
                combined_score=combined_score,
                text_score=text_score_map.get(record_id),
                vector_score=vector_score_map.get(record_id),
                text_rank=text_rank_map.get(record_id),
                vector_rank=vector_rank_map.get(record_id),
                metadata={
                    "fusion_strategy": config.fusion_strategy.value,
                    "text_weight": config.text_weight,
                    "vector_weight": config.vector_weight,
                },
            )
        )

    return results


class SyncVectorOperationsMixin(ABC):
    """Vector operations for **synchronous** database backends.

    Mixed into a :class:`~dataknobs_data.database.SyncDatabase`. Its async
    twin is :class:`AsyncVectorOperationsMixin`, and picking the wrong one is
    not a style error: this lane's implemented methods call ``self.read`` /
    ``self.delete`` / ``self.search`` without awaiting them, so on an async
    backend each would hold a coroutine object where it expects a result ---
    truthy, never raised on, and silently wrong.

    There was one mixin for both lanes and it was this one's twin, so the
    five sync backends that mix in a vector surface --- memory, file, sqlite,
    s3 and postgres --- inherited ``async`` methods. Three of those methods
    raised ``TypeError: object NoneType can't be used in 'await' expression``
    on call; the two abstract ones were overridden with sync definitions,
    which nothing but a type checker reported.
    """

    @abstractmethod
    def vector_search(
        self,
        query_vector: np.ndarray | list[float],
        *,
        vector_field: str = "embedding",
        k: int = 10,
        metric: DistanceMetric = DistanceMetric.COSINE,
        filter: Query | None = None,
        include_source: bool = True,
        score_threshold: float | None = None,
    ) -> list[VectorSearchResult]:
        """Search for similar vectors.

        Everything after ``query_vector`` is keyword-only, and that is a
        refusal rather than a restriction. The twelve implementations do not
        agree on positional order --- most spell it ``(..., k, filter,
        metric)`` where this declares ``(..., k, metric, filter)`` --- so a
        fourth positional argument already means the metric on some backends
        and the filter on others. No concrete signature changes; what changes
        is that a call written against this declaration can no longer be
        written in the one form that was never portable.

        Args:
            query_vector: The vector to search for
            vector_field: Name of the vector field to search
            k: Number of results to return
            metric: Distance metric to use
            filter: Optional query filter to apply before vector search
            include_source: Whether to include source text in results
            score_threshold: Optional minimum similarity score

        Returns:
            List of search results ordered by similarity
        """

    @abstractmethod
    def bulk_embed_and_store(
        self,
        records: list[Record],
        text_field: str | list[str],
        vector_field: str = "embedding",
        embedding_fn: Callable[[list[str]], np.ndarray] | None = None,
        batch_size: int = 100,
        model_name: str | None = None,
        model_version: str | None = None,
    ) -> list[str]:
        """Embed text fields and store vectors with records.

        Args:
            records: Records to process
            text_field: Field name(s) containing text to embed
            vector_field: Field name to store vectors in
            embedding_fn: Function to generate embeddings
            batch_size: Number of records to process at once
            model_name: Name of the embedding model
            model_version: Version of the embedding model

        Returns:
            List of record IDs that were processed
        """

    def update_vector(
        self,
        record_id: str,
        vector_field: str,
        vector: np.ndarray | list[float],
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Update a vector field for a specific record.

        Args:
            record_id: ID of the record to update
            vector_field: Name of the vector field
            vector: New vector value
            metadata: Optional metadata to attach

        Returns:
            True if update was successful
        """
        record = self.read(record_id)  # type: ignore[attr-defined]
        if not record:
            return False

        record.fields[vector_field] = vector_field_for(vector_field, vector, metadata)

        return self.update(record_id, record) is not None  # type: ignore[attr-defined]

    def delete_from_index(self, record_id: str, vector_field: str = "embedding") -> bool:
        """Remove a record from the vector index.

        Args:
            record_id: ID of the record to remove
            vector_field: Name of the vector field

        Returns:
            True if deletion was successful
        """
        return self.delete(record_id)  # type: ignore[attr-defined,no-any-return]

    def create_vector_index(
        self,
        vector_field: str = "embedding",
        dimensions: int | None = None,
        metric: DistanceMetric = DistanceMetric.COSINE,
        index_type: str = "auto",
        **kwargs: Any,
    ) -> bool:
        """Create an index for vector similarity search.

        Args:
            vector_field: Name of the vector field to index
            dimensions: Number of dimensions (if known)
            metric: Distance metric for the index
            index_type: Type of index to create
            **kwargs: Backend-specific index parameters

        Returns:
            True if index was created successfully
        """
        return True

    def drop_vector_index(self, vector_field: str = "embedding") -> bool:
        """Drop a vector index.

        Args:
            vector_field: Name of the vector field

        Returns:
            True if index was dropped successfully
        """
        return True

    def get_vector_index_stats(self, vector_field: str = "embedding") -> dict[str, Any]:
        """Get statistics about a vector index.

        Args:
            vector_field: Name of the vector field

        Returns:
            Dictionary of index statistics
        """
        return default_vector_index_stats(vector_field)

    def hybrid_search(
        self,
        query_text: str,
        query_vector: np.ndarray | list[float],
        text_fields: list[str] | None = None,
        vector_field: str = "embedding",
        k: int = 10,
        config: HybridSearchConfig | None = None,
        filter: Query | None = None,
        metric: DistanceMetric = DistanceMetric.COSINE,
    ) -> list[HybridSearchResult]:
        """Perform hybrid search combining text and vector similarity.

        Runs a text search and a vector search and merges the two rankings
        with a configurable fusion strategy. Backends with native hybrid
        search (Elasticsearch, pgvector) override this for a single-query
        implementation; the fusion itself is :func:`fuse_hybrid_results`,
        which both lanes share.

        Args:
            query_text: Text query for keyword/text matching
            query_vector: Vector for semantic similarity search
            text_fields: Fields to search for text matching
            vector_field: Name of the vector field to search
            k: Number of results to return
            config: Hybrid search configuration (weights, fusion strategy)
            filter: Optional additional filters to apply
            metric: Distance metric for vector search

        Returns:
            List of HybridSearchResult ordered by combined score (descending)
        """
        config = resolve_hybrid_config(config, self._supports_native_hybrid())
        fetch_k = hybrid_fetch_k(k)

        text_results = self._text_search_for_hybrid(
            query_text=query_text,
            text_fields=config.text_fields or text_fields,
            k=fetch_k,
            filter=filter,
        )
        vector_results = self.vector_search(
            query_vector=query_vector,
            vector_field=vector_field,
            k=fetch_k,
            metric=metric,
            filter=filter,
        )

        return fuse_hybrid_results(text_results, vector_results, config, k)

    def _text_search_for_hybrid(
        self,
        query_text: str,
        text_fields: list[str] | None,
        k: int,
        filter: Query | None = None,
    ) -> list[tuple[Record, float]]:
        """Perform text search for hybrid search fusion.

        Default implementation uses a LIKE query on the first text field.
        Backends can override for better text search (e.g. full-text search).

        Args:
            query_text: Text to search for
            text_fields: Fields to search in
            k: Maximum results to return
            filter: Additional filters

        Returns:
            List of (record, score) tuples ordered by relevance
        """
        query = hybrid_text_query(query_text, text_fields, k, filter)
        records = self.search(query)  # type: ignore[attr-defined]

        return score_text_matches(records, query_text, text_fields)

    def _supports_native_hybrid(self) -> bool:
        """Check if this backend supports native hybrid search.

        Override in backends that have native hybrid search support
        (e.g., Elasticsearch with RRF).

        Returns:
            True if native hybrid search is supported
        """
        return False


class AsyncVectorOperationsMixin(ABC):
    """Vector operations for **asynchronous** database backends.

    Mixed into an :class:`~dataknobs_data.database.AsyncDatabase`. Its sync
    twin is :class:`SyncVectorOperationsMixin`; see there for what mixing in
    the wrong one costs.

    This is what ``VectorOperationsMixin`` has always been, and that name
    still resolves here.
    """

    @abstractmethod
    async def vector_search(
        self,
        query_vector: np.ndarray | list[float],
        *,
        vector_field: str = "embedding",
        k: int = 10,
        metric: DistanceMetric = DistanceMetric.COSINE,
        filter: Query | None = None,
        include_source: bool = True,
        score_threshold: float | None = None,
    ) -> list[VectorSearchResult]:
        """Search for similar vectors.

        Everything after ``query_vector`` is keyword-only, and that is a
        refusal rather than a restriction. The twelve implementations do not
        agree on positional order --- most spell it ``(..., k, filter,
        metric)`` where this declares ``(..., k, metric, filter)`` --- so a
        fourth positional argument already means the metric on some backends
        and the filter on others. No concrete signature changes; what changes
        is that a call written against this declaration can no longer be
        written in the one form that was never portable.

        Args:
            query_vector: The vector to search for
            vector_field: Name of the vector field to search
            k: Number of results to return
            metric: Distance metric to use
            filter: Optional query filter to apply before vector search
            include_source: Whether to include source text in results
            score_threshold: Optional minimum similarity score

        Returns:
            List of search results ordered by similarity
        """

    @abstractmethod
    async def bulk_embed_and_store(
        self,
        records: list[Record],
        text_field: str | list[str],
        vector_field: str = "embedding",
        embedding_fn: Callable[[list[str]], np.ndarray] | None = None,
        batch_size: int = 100,
        model_name: str | None = None,
        model_version: str | None = None,
    ) -> list[str]:
        """Embed text fields and store vectors with records.

        Args:
            records: Records to process
            text_field: Field name(s) containing text to embed
            vector_field: Field name to store vectors in
            embedding_fn: Function to generate embeddings
            batch_size: Number of records to process at once
            model_name: Name of the embedding model
            model_version: Version of the embedding model

        Returns:
            List of record IDs that were processed
        """

    async def update_vector(
        self,
        record_id: str,
        vector_field: str,
        vector: np.ndarray | list[float],
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Update a vector field for a specific record.

        Args:
            record_id: ID of the record to update
            vector_field: Name of the vector field
            vector: New vector value
            metadata: Optional metadata to attach

        Returns:
            True if update was successful
        """
        record = await self.read(record_id)  # type: ignore[attr-defined]
        if not record:
            return False

        record.fields[vector_field] = vector_field_for(vector_field, vector, metadata)

        return await self.update(record_id, record) is not None  # type: ignore[attr-defined]

    async def delete_from_index(self, record_id: str, vector_field: str = "embedding") -> bool:
        """Remove a record from the vector index.

        Args:
            record_id: ID of the record to remove
            vector_field: Name of the vector field

        Returns:
            True if deletion was successful
        """
        return await self.delete(record_id)  # type: ignore[attr-defined,no-any-return]

    async def create_vector_index(
        self,
        vector_field: str = "embedding",
        dimensions: int | None = None,
        metric: DistanceMetric = DistanceMetric.COSINE,
        index_type: str = "auto",
        **kwargs: Any,
    ) -> bool:
        """Create an index for vector similarity search.

        Args:
            vector_field: Name of the vector field to index
            dimensions: Number of dimensions (if known)
            metric: Distance metric for the index
            index_type: Type of index to create
            **kwargs: Backend-specific index parameters

        Returns:
            True if index was created successfully
        """
        return True

    async def drop_vector_index(self, vector_field: str = "embedding") -> bool:
        """Drop a vector index.

        Args:
            vector_field: Name of the vector field

        Returns:
            True if index was dropped successfully
        """
        return True

    async def get_vector_index_stats(self, vector_field: str = "embedding") -> dict[str, Any]:
        """Get statistics about a vector index.

        Args:
            vector_field: Name of the vector field

        Returns:
            Dictionary of index statistics
        """
        return default_vector_index_stats(vector_field)

    async def hybrid_search(
        self,
        query_text: str,
        query_vector: np.ndarray | list[float],
        text_fields: list[str] | None = None,
        vector_field: str = "embedding",
        k: int = 10,
        config: HybridSearchConfig | None = None,
        filter: Query | None = None,
        metric: DistanceMetric = DistanceMetric.COSINE,
    ) -> list[HybridSearchResult]:
        """Perform hybrid search combining text and vector similarity.

        Runs a text search and a vector search and merges the two rankings
        with a configurable fusion strategy. Backends with native hybrid
        search (Elasticsearch, pgvector) override this for a single-query
        implementation; the fusion itself is :func:`fuse_hybrid_results`,
        which both lanes share.

        Args:
            query_text: Text query for keyword/text matching
            query_vector: Vector for semantic similarity search
            text_fields: Fields to search for text matching
            vector_field: Name of the vector field to search
            k: Number of results to return
            config: Hybrid search configuration (weights, fusion strategy)
            filter: Optional additional filters to apply
            metric: Distance metric for vector search

        Returns:
            List of HybridSearchResult ordered by combined score (descending)
        """
        config = resolve_hybrid_config(config, await self._supports_native_hybrid())
        fetch_k = hybrid_fetch_k(k)

        text_results = await self._text_search_for_hybrid(
            query_text=query_text,
            text_fields=config.text_fields or text_fields,
            k=fetch_k,
            filter=filter,
        )
        vector_results = await self.vector_search(
            query_vector=query_vector,
            vector_field=vector_field,
            k=fetch_k,
            metric=metric,
            filter=filter,
        )

        return fuse_hybrid_results(text_results, vector_results, config, k)

    async def _text_search_for_hybrid(
        self,
        query_text: str,
        text_fields: list[str] | None,
        k: int,
        filter: Query | None = None,
    ) -> list[tuple[Record, float]]:
        """Perform text search for hybrid search fusion.

        Default implementation uses a LIKE query on the first text field.
        Backends can override for better text search (e.g. full-text search).

        Args:
            query_text: Text to search for
            text_fields: Fields to search in
            k: Maximum results to return
            filter: Additional filters

        Returns:
            List of (record, score) tuples ordered by relevance
        """
        query = hybrid_text_query(query_text, text_fields, k, filter)
        records = await self.search(query)  # type: ignore[attr-defined]

        return score_text_matches(records, query_text, text_fields)

    async def _supports_native_hybrid(self) -> bool:
        """Check if this backend supports native hybrid search.

        Override in backends that have native hybrid search support
        (e.g., Elasticsearch with RRF).

        Returns:
            True if native hybrid search is supported
        """
        return False


# The bare name has always meant the async lane, and every consumer who mixed
# it into an async backend is right to keep doing so. Kept as an alias rather
# than repointed at the sync lane, which would silently change what an
# existing subclass inherits.
VectorOperationsMixin = AsyncVectorOperationsMixin


class VectorSyncMixin:
    """Mixin for synchronizing vectors with source text."""

    async def sync_vectors_with_text(
        self,
        records: list[Record],
        text_fields: list[str],
        vector_field: str = "embedding",
        embedding_fn: Callable[[list[str]], np.ndarray] | None = None,
        force: bool = False,
        field_separator: str = DEFAULT_FIELD_SEPARATOR,
    ) -> int:
        """Synchronize vector embeddings with text content.

        Args:
            records: Records to synchronize
            text_fields: Text fields to generate vectors from
            vector_field: Vector field to update
            embedding_fn: Embedding function
            force: Force re-generation even if vectors exist
            field_separator: What to join ``text_fields`` on. Was hardcoded to
                a space, which is the value it still defaults to.

        Returns:
            Number of records updated
        """
        if not embedding_fn:
            raise ValueError("Embedding function is required for vector synchronization")

        updated = 0
        for record in records:
            # Check if vector needs update
            needs_update = force or vector_field not in record.fields

            if not needs_update:
                needs_update = self._text_vector_is_stale(
                    record, vector_field, text_fields, field_separator
                )

            if needs_update:
                text_content = assemble_source_text(record, text_fields, field_separator)

                # Generate embedding
                if text_content:
                    from ..fields import VectorField

                    result = embedding_fn([text_content])
                    # Handle both sync and async embedding functions
                    if hasattr(result, "__await__"):
                        embeddings = await result
                    else:
                        embeddings = result
                    record.fields[vector_field] = VectorField(
                        name=vector_field,
                        value=embeddings[0],
                        source_field=",".join(text_fields),
                        # Without this the field is unjudgeable: a
                        # `VectorTextSynchronizer` sweeping the same corpus
                        # finds no digest and treats it as current forever.
                        metadata=content_hash_metadata(
                            text_fields,
                            field_separator,
                            compute_content_hash(text_content),
                        ),
                    )
                    updated += 1

        return updated

    @staticmethod
    def _text_vector_is_stale(
        record: Record,
        vector_field: str,
        text_fields: list[str],
        field_separator: str,
    ) -> bool:
        """Whether an existing vector no longer matches the text it names.

        This used to compare the *set of source fields* and nothing else, so a
        vector went on being reported current after its text was edited --- the
        same omission `_has_current_vector` carried, in a second class. The
        digest closes it; the field-set comparison stays because a re-pointed
        `text_fields` changes what the vector means even when the digest
        cannot be read.
        """
        metadata = getattr(record.fields[vector_field], "metadata", None) or {}

        stored_fields, _stored_separator = stored_assembly(metadata)
        if stored_fields is None:
            # Records written before the assembly was described name their
            # sources in `source_field`, comma-joined --- and that key is
            # `None`, not absent, for a field built without one, so the old
            # `.get("source_field", "").split(",")` raised AttributeError
            # rather than defaulting.
            legacy = metadata.get("source_field")
            stored_fields = legacy.split(",") if isinstance(legacy, str) and legacy else []

        if set(stored_fields) != set(text_fields):
            return True

        stored_hash = metadata.get(CONTENT_HASH_KEY)
        if stored_hash is None:
            # Nothing to compare against; inventing a comparison would report
            # every hand-built field stale on the first sweep.
            return False

        current = current_content_hash(record, text_fields, field_separator)
        return current is not None and current != stored_hash
