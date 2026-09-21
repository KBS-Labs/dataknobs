# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Mixins and protocols for vector-capable databases."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Protocol

from ..fields import FieldType, VectorField
from .bulk_embed_mixin import attach_vector_field
from .content import (
    CONTENT_HASH_KEY,
    DEFAULT_FIELD_SEPARATOR,
    assemble_source_text,
    current_content_hash,
    derive_source_text,
    is_foreign_model,
    stored_assembly,
)
from .embedding import default_model_name, embed_texts, require_embedding_source
from .hybrid import (
    FusionStrategy,
    HybridSearchConfig,
    HybridSearchResult,
    reciprocal_rank_fusion,
    weighted_score_fusion,
)
from .types import BatchVectors, DistanceMetric, VectorSearchResult

if TYPE_CHECKING:
    import numpy as np
    from collections.abc import Callable
    from ..query import Query
    from ..records import Record
    from .embedding import TextEmbedder


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


def resolve_metric(database: object, metric: DistanceMetric | str | None) -> DistanceMetric:
    """Settle what metric a search runs under, once, before any backend sees it.

    Eight of the twelve implementations defaulted ``metric`` to ``None`` and
    resolved that against ``self.vector_metric``; the four with native vector
    support --- both Postgres and both Elasticsearch backends --- defaulted it
    to cosine and never consulted the database's configuration at all. So a
    database built with ``vector_metric="euclidean"`` searched under euclidean
    on memory, file, SQLite and S3 and under cosine on the four that could
    have used it. That is the same divergence as the two parameters this pass
    is about, and it closes the same way --- by deciding it above all twelve.

    What a *name* means is :meth:`DistanceMetric.resolve`'s question, and this
    asks it rather than answering it again. The two were briefly different
    answers: the aliases :meth:`DistanceMetric.get_aliases` published were
    declined here, because at the time nothing in the library resolved them
    --- ``"ip"`` happened to reach a pgvector operator table that knew it and
    raised on every other backend. The enum resolves them now, so the reason
    to decline is gone and there is one vocabulary instead of two.

    **The answer is canonical**, which is the half that makes every table
    below this safe. Resolving ``"l2"`` to ``L2`` settles the *spelling* and
    leaves the *aliasing* for each table to restate --- which is the
    divergence :meth:`DistanceMetric.canonical` exists to end, and restating
    it is what the tables were found doing. Two of them still are, one raising
    ``Unsupported metric: DistanceMetric.L2`` on every Python-path search and
    one answering an explicit ``l2`` with a cosine Elasticsearch mapping.
    Canonicalising once here covers all twelve backends and every table any of
    them reaches, including the ones a backend adds later.

    Args:
        database: The database whose configured metric ``None`` means.
        metric: A :class:`DistanceMetric`, a member value, or any published
            alias --- or ``None`` for the database's own setting.

    Returns:
        The canonical member for the metric to search under: one of
        ``COSINE``, ``EUCLIDEAN``, ``DOT_PRODUCT`` or ``L1``, never
        ``L2`` or ``INNER_PRODUCT``.

    Raises:
        ValueError: If ``metric`` is a string naming no metric, from
            :meth:`DistanceMetric.resolve`.
    """
    if metric is None:
        configured = getattr(database, "vector_metric", None)
        if isinstance(configured, DistanceMetric):
            return configured.canonical()
        return DistanceMetric.COSINE
    return DistanceMetric.resolve(metric).canonical()


def finish_vector_search(
    hits: list[VectorSearchResult],
    *,
    vector_field: str,
    include_source: bool,
    score_threshold: float | None,
) -> list[VectorSearchResult]:
    """Apply the two parameters the hook does not see.

    Shared by both lanes rather than written twice: the sync and async
    templates differ only in awaiting the hook, and everything after that
    ``await`` is identical. A twin pair that each implemented this would be
    the shape the twelve backends were already in.

    Args:
        hits: What the backend's k-nearest-neighbour search returned.
        vector_field: The field searched, which is where a hit's source
            description lives.
        include_source: Whether to derive ``source_text`` where it is absent.
        score_threshold: Drop hits scoring below this, if given.

    Returns:
        The hits that survive the threshold, in the order the backend gave.
    """
    results = []
    for hit in hits:
        if score_threshold is not None and hit.score < score_threshold:
            continue
        # Not overwritten: a backend that already knows the source text ---
        # one whose store holds it beside the vector --- has said something
        # the record cannot be re-read for.
        if include_source and hit.source_text is None:
            hit.source_text = derive_source_text(hit.record, vector_field)
        results.append(hit)
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

    def vector_search(
        self,
        query_vector: np.ndarray | list[float],
        *,
        vector_field: str = "embedding",
        k: int = 10,
        metric: DistanceMetric | str | None = None,
        filter: Query | None = None,
        include_source: bool = True,
        score_threshold: float | None = None,
    ) -> list[VectorSearchResult]:
        """Search for similar vectors.

        Everything after ``query_vector`` is keyword-only, and that is a
        refusal rather than a restriction. The twelve implementations did not
        agree on positional order --- most spelled it ``(..., k, filter,
        metric)`` where this declares ``(..., k, metric, filter)`` --- so a
        fourth positional argument meant the metric on some backends and the
        filter on others.

        **``score_threshold`` is a post-filter, so a call may return fewer
        than ``k`` results.** That is the defined behaviour rather than an
        artefact: it is what the Elasticsearch implementation always did, and
        over-fetching to refill ``k`` is a different promise that should not
        be adopted silently.

        Pushing the threshold into the query itself (``min_score`` on
        Elasticsearch, a ``WHERE`` on pgvector) is strictly better where
        available, and **no backend may reach it by overriding this method**
        --- ``test_no_backend_carries_its_own_vector_search`` forbids exactly
        that, because an override is how the twelve answers happened. An
        earlier draft of this paragraph offered the override as the way to do
        it, which the guard has never permitted. The hook does not see the
        threshold either, so push-down is not available today; adding it
        means giving ``_vector_search`` an explicit hint parameter here, on
        the mixin, with this post-filter still owning the contract and
        finding nothing left to drop.

        **The score's scale is the backend's**, and the threshold is compared
        against it unconverted. The ten Python-path and Postgres backends
        report a raw similarity; Elasticsearch reports its own ``_score``,
        which for a ``cosine`` mapping is ``(1 + cos) / 2``. So one threshold
        constant does not cut at the same place on every backend. Making it
        would mean converting Elasticsearch's score, and that needs the
        *mapping's* similarity rather than the requested metric --- see
        :meth:`SyncElasticsearchDatabase._vector_search` for why those are not
        the same thing.

        **``include_source`` does not decide whether the record comes back.**
        The record always does; :class:`VectorSearchResult` declares it
        required. What the knob decides is whether
        :func:`~dataknobs_data.vector.content.derive_source_text` runs ---
        cheap, but defensible to skip at large ``k`` over big text fields.

        Args:
            query_vector: The vector to search for
            vector_field: Name of the vector field to search
            k: Number of results to return
            metric: Distance metric, as a :class:`DistanceMetric` or its
                value. ``None`` means the metric this database was configured
                with, falling back to cosine.
            filter: Optional query filter to apply before vector search
            include_source: Whether to derive ``source_text`` onto each hit
            score_threshold: Drop hits scoring below this

        Returns:
            List of search results ordered by similarity
        """
        return finish_vector_search(
            self._vector_search(
                query_vector,
                vector_field=vector_field,
                k=k,
                metric=resolve_metric(self, metric),
                filter=filter,
            ),
            vector_field=vector_field,
            include_source=include_source,
            score_threshold=score_threshold,
        )

    @abstractmethod
    def _vector_search(
        self,
        query_vector: np.ndarray | list[float],
        *,
        vector_field: str,
        k: int,
        metric: DistanceMetric,
        filter: Query | None,
    ) -> list[VectorSearchResult]:
        """Raw k-nearest-neighbour search --- the part only a backend knows.

        No threshold and no source assembly: :meth:`vector_search` owns both,
        so that twelve backends cannot answer them thirteen ways again. A
        backend implements this and inherits the rest.

        ``metric`` arrives resolved and **canonical** --- never ``None``,
        never a string, and never one of the two alias members --- so an
        implementation may key a table on it with four entries rather than
        six, and cannot miss a spelling by writing the shorter table.

        Args:
            query_vector: The vector to search for
            vector_field: Name of the vector field to search
            k: Maximum number of results to return
            metric: Distance metric to use
            filter: Optional query filter to apply before vector search

        Returns:
            Search results ordered by similarity, at most ``k`` of them
        """

    @abstractmethod
    def bulk_embed_and_store(
        self,
        records: list[Record],
        text_field: str | list[str],
        vector_field: str = "embedding",
        embedding_fn: Callable[[list[str]], BatchVectors] | None = None,
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

        # `bool(...)`, not `is not None`: every backend's `update` returns
        # `bool`, and `False is not None` is `True` --- so this reported a
        # successful write for an update that did not happen.
        return bool(self.update(record_id, record))  # type: ignore[attr-defined]

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
        metric: DistanceMetric | str | None = None,
        index_type: str = "auto",
    ) -> bool:
        """Create an index for vector similarity search.

        Took ``**kwargs: Any`` for "backend-specific index parameters" and
        read it nowhere, which is the swallow removed from
        :meth:`vector_search` wearing a different name: a keyword bound here
        and went no further. A backend parameter belongs on that backend, as
        ``AsyncPostgresDatabase``'s ``lists`` already is.

        ``metric`` defaults to ``None`` --- the database's configured metric
        --- for the same reason :meth:`vector_search`'s does, and it was the
        last method on this surface still hardcoding cosine. An index built
        under a metric the searches do not use is an index the planner
        declines, so a database configured for euclidean was building one it
        could never read.

        Args:
            vector_field: Name of the vector field to index
            dimensions: Number of dimensions (if known)
            metric: Distance metric for the index. ``None`` means the metric
                this database was configured with, falling back to cosine.
            index_type: Type of index to create

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
        metric: DistanceMetric | str | None = None,
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
            metric: Distance metric for vector search, resolved by
                :meth:`vector_search`, which is all this does with it.
                ``None`` means the database's configured metric --- the same
                answer its own ``vector_search`` gives, which is what stops
                one object searching under two metrics.

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

    async def vector_search(
        self,
        query_vector: np.ndarray | list[float],
        *,
        vector_field: str = "embedding",
        k: int = 10,
        metric: DistanceMetric | str | None = None,
        filter: Query | None = None,
        include_source: bool = True,
        score_threshold: float | None = None,
    ) -> list[VectorSearchResult]:
        """Search for similar vectors.

        Everything after ``query_vector`` is keyword-only, and that is a
        refusal rather than a restriction. The twelve implementations did not
        agree on positional order --- most spelled it ``(..., k, filter,
        metric)`` where this declares ``(..., k, metric, filter)`` --- so a
        fourth positional argument meant the metric on some backends and the
        filter on others.

        **``score_threshold`` is a post-filter, so a call may return fewer
        than ``k`` results.** That is the defined behaviour rather than an
        artefact: it is what the Elasticsearch implementation always did, and
        over-fetching to refill ``k`` is a different promise that should not
        be adopted silently.

        Pushing the threshold into the query itself (``min_score`` on
        Elasticsearch, a ``WHERE`` on pgvector) is strictly better where
        available, and **no backend may reach it by overriding this method**
        --- ``test_no_backend_carries_its_own_vector_search`` forbids exactly
        that, because an override is how the twelve answers happened. An
        earlier draft of this paragraph offered the override as the way to do
        it, which the guard has never permitted. The hook does not see the
        threshold either, so push-down is not available today; adding it
        means giving ``_vector_search`` an explicit hint parameter here, on
        the mixin, with this post-filter still owning the contract and
        finding nothing left to drop.

        **The score's scale is the backend's**, and the threshold is compared
        against it unconverted. The ten Python-path and Postgres backends
        report a raw similarity; Elasticsearch reports its own ``_score``,
        which for a ``cosine`` mapping is ``(1 + cos) / 2``. So one threshold
        constant does not cut at the same place on every backend. Making it
        would mean converting Elasticsearch's score, and that needs the
        *mapping's* similarity rather than the requested metric --- see
        :meth:`SyncElasticsearchDatabase._vector_search` for why those are not
        the same thing.

        **``include_source`` does not decide whether the record comes back.**
        The record always does; :class:`VectorSearchResult` declares it
        required. What the knob decides is whether
        :func:`~dataknobs_data.vector.content.derive_source_text` runs ---
        cheap, but defensible to skip at large ``k`` over big text fields.

        Args:
            query_vector: The vector to search for
            vector_field: Name of the vector field to search
            k: Number of results to return
            metric: Distance metric, as a :class:`DistanceMetric` or its
                value. ``None`` means the metric this database was configured
                with, falling back to cosine.
            filter: Optional query filter to apply before vector search
            include_source: Whether to derive ``source_text`` onto each hit
            score_threshold: Drop hits scoring below this

        Returns:
            List of search results ordered by similarity
        """
        return finish_vector_search(
            await self._vector_search(
                query_vector,
                vector_field=vector_field,
                k=k,
                metric=resolve_metric(self, metric),
                filter=filter,
            ),
            vector_field=vector_field,
            include_source=include_source,
            score_threshold=score_threshold,
        )

    @abstractmethod
    async def _vector_search(
        self,
        query_vector: np.ndarray | list[float],
        *,
        vector_field: str,
        k: int,
        metric: DistanceMetric,
        filter: Query | None,
    ) -> list[VectorSearchResult]:
        """Raw k-nearest-neighbour search --- the part only a backend knows.

        No threshold and no source assembly: :meth:`vector_search` owns both,
        so that twelve backends cannot answer them thirteen ways again. A
        backend implements this and inherits the rest.

        ``metric`` arrives resolved and **canonical** --- never ``None``,
        never a string, and never one of the two alias members --- so an
        implementation may key a table on it with four entries rather than
        six, and cannot miss a spelling by writing the shorter table.

        Args:
            query_vector: The vector to search for
            vector_field: Name of the vector field to search
            k: Maximum number of results to return
            metric: Distance metric to use
            filter: Optional query filter to apply before vector search

        Returns:
            Search results ordered by similarity, at most ``k`` of them
        """

    @abstractmethod
    async def bulk_embed_and_store(
        self,
        records: list[Record],
        text_field: str | list[str],
        vector_field: str = "embedding",
        embedding_fn: Callable[[list[str]], BatchVectors] | None = None,
        batch_size: int = 100,
        model_name: str | None = None,
        model_version: str | None = None,
        *,
        embedder: TextEmbedder | None = None,
    ) -> list[str]:
        """Embed text fields and store vectors with records.

        Args:
            records: Records to process
            text_field: Field name(s) containing text to embed
            vector_field: Field name to store vectors in
            embedding_fn: Function to generate embeddings. Still accepted;
                prefer *embedder*.
            batch_size: Number of records to process at once
            model_name: Name of the embedding model
            model_version: Version of the embedding model
            embedder: A :class:`~dataknobs_data.vector.TextEmbedder`. Carries
                its own ``model_id``, so an implementation can fill
                *model_name* from the thing that produced the vectors rather
                than from a parameter the caller has to keep in step.

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

        # `bool(...)`, not `is not None`: every backend's `update` returns
        # `bool`, and `False is not None` is `True` --- so this reported a
        # successful write for an update that did not happen.
        return bool(await self.update(record_id, record))  # type: ignore[attr-defined]

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
        metric: DistanceMetric | str | None = None,
        index_type: str = "auto",
    ) -> bool:
        """Create an index for vector similarity search.

        Took ``**kwargs: Any`` for "backend-specific index parameters" and
        read it nowhere, which is the swallow removed from
        :meth:`vector_search` wearing a different name: a keyword bound here
        and went no further. A backend parameter belongs on that backend, as
        ``AsyncPostgresDatabase``'s ``lists`` already is.

        ``metric`` defaults to ``None`` --- the database's configured metric
        --- for the same reason :meth:`vector_search`'s does, and it was the
        last method on this surface still hardcoding cosine. An index built
        under a metric the searches do not use is an index the planner
        declines, so a database configured for euclidean was building one it
        could never read.

        Args:
            vector_field: Name of the vector field to index
            dimensions: Number of dimensions (if known)
            metric: Distance metric for the index. ``None`` means the metric
                this database was configured with, falling back to cosine.
            index_type: Type of index to create

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
        metric: DistanceMetric | str | None = None,
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
            metric: Distance metric for vector search, resolved by
                :meth:`vector_search`, which is all this does with it.
                ``None`` means the database's configured metric --- the same
                answer its own ``vector_search`` gives, which is what stops
                one object searching under two metrics.

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
        embedding_fn: Callable[[list[str]], BatchVectors] | None = None,
        force: bool = False,
        field_separator: str = DEFAULT_FIELD_SEPARATOR,
        *,
        embedder: TextEmbedder | None = None,
        model_name: str | None = None,
        model_version: str | None = None,
    ) -> int:
        """Synchronize vector embeddings with text content.

        Args:
            records: Records to synchronize
            text_fields: Text fields to generate vectors from
            vector_field: Vector field to update
            embedding_fn: Embedding function. Still accepted; prefer
                *embedder*.
            force: Force re-generation even if vectors exist
            field_separator: What to join ``text_fields`` on. Was hardcoded to
                a space, which is the value it still defaults to.
            embedder: A :class:`~dataknobs_data.vector.TextEmbedder`.
            model_name: Identity to store beside the vector, and to judge an
                existing one against. Defaults to the *embedder*'s own
                ``model_id``, which is what stops a caller naming one model
                while embedding with another.
            model_version: Version to store beside the vector. Not defaulted
                from *embedder*: a ``TextEmbedder`` carries an identity and no
                version, and inventing one here would write a value nothing
                produced.

        Returns:
            Number of records updated

        Raises:
            ValueError: Neither *embedding_fn* nor *embedder* was given, or
                both were.
        """
        require_embedding_source(embedder, embedding_fn)
        model_name = default_model_name(model_name, embedder)

        updated = 0
        for record in records:
            # Check if vector needs update
            needs_update = force or vector_field not in record.fields

            if not needs_update:
                needs_update = self._text_vector_is_stale(
                    record, vector_field, text_fields, field_separator, model_name
                )

            if needs_update:
                text_content = assemble_source_text(record, text_fields, field_separator)

                # Generate embedding
                if text_content:
                    embeddings = await embed_texts(
                        [text_content], embedder=embedder, embedding_fn=embedding_fn
                    )
                    # The shared builder rather than a sixth hand-rolled
                    # `VectorField`: the digest and the model identity are the
                    # two halves of what makes a stored vector judgeable, and
                    # this site had the first and not the second for as long
                    # as it built the field itself.
                    attach_vector_field(
                        record,
                        vector_field,
                        embeddings[0],
                        text_content,
                        text_fields,
                        field_separator,
                        model_name,
                        model_version,
                    )
                    updated += 1

        return updated

    @staticmethod
    def _text_vector_is_stale(
        record: Record,
        vector_field: str,
        text_fields: list[str],
        field_separator: str,
        model_name: str | None = None,
    ) -> bool:
        """Whether an existing vector no longer matches the text it names.

        This used to compare the *set of source fields* and nothing else, so a
        vector went on being reported current after its text was edited --- the
        same omission `_has_current_vector` carried, in a second class. The
        digest closes it; the field-set comparison stays because a re-pointed
        `text_fields` changes what the vector means even when the digest
        cannot be read.

        The model is the third comparison and answers a question the other two
        cannot: identical text through two models gives one digest and two
        incompatible vector spaces, so a swap is invisible to a check that
        only reads the text.
        """
        # Two-sided, and the same rule `_has_current_vector` applies -- now
        # by calling it rather than by restating it. A stored nothing means
        # the vector predates anything recording a name, not that it
        # disagrees; calling those stale would re-embed a whole corpus on the
        # first sweep after upgrading.
        stored_name = getattr(record.fields[vector_field], "model_name", None)
        if is_foreign_model(stored_name, model_name):
            return True

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
