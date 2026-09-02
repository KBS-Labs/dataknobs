"""Embedding-cluster topic index for structured content retrieval.

Clusters chunks by embedding similarity to identify topic regions.
This is the ``dataknobs-data`` implementation of the
:class:`~dataknobs_data.sources.topic_index.TopicIndex` protocol.

The index uses **lazy query-driven clustering**: no full corpus clustering
is pre-computed.  Each ``resolve()`` call drives a per-turn pipeline:

1. Fetch seed chunks via vector search relevant to this query.
2. Cluster the seed set by embedding similarity (small N, fast).
3. Match the query embedding against seed-set cluster centroids.
4. Return chunks from matched clusters within the seed pool.

For eager mode (when all chunks and embeddings are available upfront),
use :meth:`ClusterTopicIndex.from_chunks` to pre-cluster the full corpus.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..vector.embedding import TextEmbedder
from .base import RetrievalIntent, SourceResult, StrategyUnavailable
from .processing import agglomerative_cluster, cosine_similarity
from .topic_index import DEFAULT_HEADING_STOPWORDS

logger = logging.getLogger(__name__)

# ``EmbedFn`` and ``BatchEmbedFn`` used to be declared here --- the per-text and
# batch arities of "a thing that turns text into vectors", as two separate type
# aliases, one of which collided by name with a *third* alias of the same
# concept in ``processing.py``. Both are now ``TextEmbedder``, which is batch
# and carries a ``model_id``; the single-text arity is ``embed([t])[0]``.


class VectorQueryFn(Protocol):
    """Protocol for vector-search closures used by topic indices.

    Called by topic-index seeding to fetch candidate chunks via
    similarity search. Implementations forward ``filter_metadata`` to
    the underlying KB (scalar-equality semantics across built-in vector
    stores) so that ``RetrievalIntent.filters`` is honored on the
    topic-index path — not silently dropped.
    """

    async def __call__(
        self,
        query: str,
        top_k: int,
        *,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SourceResult]: ...


DEFAULT_LABEL_MIN_WORD_LENGTH: int = 3
"""Default minimum word length for auto-generated cluster labels."""

DEFAULT_LABEL_TOP_TERMS: int = 3
"""Default number of top terms used in auto-generated cluster labels."""


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------


@dataclass(frozen=True)
class ClusterTopicConfig:
    """Configuration for :class:`ClusterTopicIndex`.

    Attributes:
        cluster_threshold: Minimum similarity to merge two chunks into
            the same cluster.
        min_cluster_size: Minimum number of chunks to form a named
            cluster.  Smaller groups are assigned to cluster -1
            (unclustered).
        seed_max_results: Maximum seed chunks to fetch via vector search
            per query (lazy mode).
        seed_score_threshold: Drop vector seeds below this similarity
            (lazy mode).
        top_clusters: Maximum number of matching clusters to expand
            at query time.
        max_results_per_cluster: Maximum chunks to return from each
            matched cluster (ranked by within-cluster query similarity).
        max_total_results: Final cap on total returned chunks.
        centroid_score_threshold: Minimum centroid similarity to
            consider a cluster as a match.
        label_stopwords: Words to exclude when auto-generating cluster
            labels from content.
        label_min_word_length: Minimum word length for auto-label terms.
        label_top_terms: Number of top terms in auto-generated labels.
        scope_profiles: Per-scope parameter overrides keyed by scope
            name (same cascade as HeadingTreeIndex).
    """

    cluster_threshold: float = 0.7
    min_cluster_size: int = 2
    seed_max_results: int = 30
    seed_score_threshold: float = 0.2
    top_clusters: int = 3
    max_results_per_cluster: int = 20
    max_total_results: int = 50
    centroid_score_threshold: float = 0.2
    label_stopwords: frozenset[str] = DEFAULT_HEADING_STOPWORDS
    label_min_word_length: int = DEFAULT_LABEL_MIN_WORD_LENGTH
    label_top_terms: int = DEFAULT_LABEL_TOP_TERMS
    scope_profiles: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClusterTopicConfig:
        """Build from a config dict, ignoring unknown keys."""
        import dataclasses

        known = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}

        # Handle list -> frozenset for stopwords
        sw = filtered.get("label_stopwords")
        if isinstance(sw, list):
            filtered["label_stopwords"] = frozenset(sw)

        return cls(**filtered)


# ------------------------------------------------------------------
# Effective parameters (resolved per-query)
# ------------------------------------------------------------------


@dataclass
class _EffectiveParams:
    """Resolved parameters for a single resolve() call."""

    seed_max_results: int
    seed_score_threshold: float
    top_clusters: int
    max_results_per_cluster: int
    max_total_results: int
    centroid_score_threshold: float


def _resolve_params(
    config: ClusterTopicConfig,
    intent: RetrievalIntent | None,
) -> _EffectiveParams:
    """Resolve effective parameters via the cascade.

    Priority (highest to lowest):
    1. Explicit overrides in ``intent.raw_data["topic_index"]``
    2. Scope profile matching ``intent.scope``
    3. Config defaults
    """
    params: dict[str, Any] = {
        "seed_max_results": config.seed_max_results,
        "seed_score_threshold": config.seed_score_threshold,
        "top_clusters": config.top_clusters,
        "max_results_per_cluster": config.max_results_per_cluster,
        "max_total_results": config.max_total_results,
        "centroid_score_threshold": config.centroid_score_threshold,
    }

    if intent is not None:
        # Layer 2: scope profile
        scope = intent.scope
        if scope and scope in config.scope_profiles:
            profile = config.scope_profiles[scope]
            for k, v in profile.items():
                if k in params:
                    params[k] = v

        # Layer 1: explicit overrides
        overrides = intent.raw_data.get("topic_index")
        if isinstance(overrides, dict):
            for k, v in overrides.items():
                if k in params:
                    params[k] = v

    return _EffectiveParams(**params)


# ------------------------------------------------------------------
# Cluster data types
# ------------------------------------------------------------------


@dataclass
class _Cluster:
    """Internal cluster representation."""

    cluster_id: int
    label: str
    member_indices: list[int]
    centroid: list[float]


# ------------------------------------------------------------------
# ClusterTopicIndex
# ------------------------------------------------------------------


class ClusterTopicIndex:
    """Topic index backed by embedding clusters.

    Uses **lazy query-driven clustering**: each ``resolve()`` call
    fetches seed chunks via vector search, clusters the seed set by
    embedding similarity, matches the query to cluster centroids, and
    returns chunks from matched clusters.

    For eager mode, use :meth:`from_chunks` to pre-cluster a full
    corpus when all chunks and embeddings are available upfront.

    Args:
        embedder: Embedder used at resolve time for the query and for
            seed chunks. Optional, because an index built by
            :meth:`from_chunks` for inspection only never resolves.
        vector_query_fn: Async function for vector-based seeding.
            Accepts ``(query, top_k)`` and returns scored results.
            Required for lazy mode.
        source_name: Source name for logging and provenance.
        config: Cluster configuration.
    """

    def __init__(
        self,
        *,
        embedder: TextEmbedder | None = None,
        vector_query_fn: VectorQueryFn | None = None,
        source_name: str = "knowledge_base",
        config: ClusterTopicConfig | None = None,
    ) -> None:
        self._config = config or ClusterTopicConfig()
        self._source_name = source_name
        self._embedder = embedder
        self._vector_query_fn = vector_query_fn

        # Eager-mode state: populated by from_chunks(), None in lazy mode
        self._chunks: list[SourceResult] | None = None
        self._embeddings: list[list[float]] | None = None
        self._clusters: list[_Cluster] | None = None

    @classmethod
    def from_chunks(
        cls,
        chunks: list[SourceResult],
        embeddings: dict[str, list[float]],
        *,
        embedder: TextEmbedder | None = None,
        vector_query_fn: VectorQueryFn | None = None,
        source_name: str = "knowledge_base",
        config: ClusterTopicConfig | None = None,
        labels: dict[int, str] | None = None,
    ) -> ClusterTopicIndex:
        """Eagerly build from pre-loaded chunks and embeddings.

        Clusters the full corpus upfront.  Useful for testing or when
        all data is already available.  ``resolve()`` uses the pre-built
        clusters instead of per-turn construction.

        Args:
            chunks: Source result chunks.
            embeddings: Pre-computed embeddings keyed by chunk source_id.
            embedder: Embedder for query text at resolve time. Stays
                optional: building an index for inspection only --- to read
                its clusters and labels without ever calling ``resolve`` ---
                is a supported and tested construction.
            vector_query_fn: Unused in eager mode but stored for API
                consistency.
            source_name: Source name for provenance.
            config: Cluster configuration.
            labels: Optional user-supplied cluster labels keyed by
                cluster ID.
        """
        idx = cls(
            embedder=embedder,
            vector_query_fn=vector_query_fn,
            source_name=source_name,
            config=config,
        )

        # Filter to chunks with embeddings
        filtered_chunks: list[SourceResult] = []
        filtered_embeddings: list[list[float]] = []
        for chunk in chunks:
            emb = embeddings.get(chunk.source_id)
            if emb is not None:
                filtered_chunks.append(chunk)
                filtered_embeddings.append(emb)

        idx._chunks = filtered_chunks
        idx._embeddings = filtered_embeddings

        if filtered_chunks:
            idx._clusters = _build_clusters(
                filtered_chunks,
                filtered_embeddings,
                config=idx._config,
                labels=labels,
            )

        return idx

    @classmethod
    async def build(
        cls,
        chunks: list[SourceResult],
        embedder: TextEmbedder,
        *,
        vector_query_fn: VectorQueryFn | None = None,
        source_name: str = "knowledge_base",
        config: ClusterTopicConfig | None = None,
        labels: dict[int, str] | None = None,
    ) -> ClusterTopicIndex:
        """Embed chunks and eagerly cluster.  One-time construction cost.

        Args:
            chunks: Source result chunks.
            embedder: Embeds the corpus now and the query later. One
                parameter where there were two: ``batch_embed_fn`` and
                ``embed_fn`` were the same concept at two arities, and
                nothing stopped a caller passing embedders from different
                models for the two, which put the corpus and the queries
                searching it into different vector spaces.
            vector_query_fn: Stored for API consistency.
            source_name: Source name for provenance.
            config: Cluster configuration.
            labels: Optional user-supplied cluster labels.
        """
        texts = [c.content for c in chunks]
        if not texts:
            return cls(
                embedder=embedder,
                vector_query_fn=vector_query_fn,
                source_name=source_name,
                config=config,
            )

        all_embeddings = await embedder.embed(texts)
        embeddings_map = {
            chunk.source_id: emb for chunk, emb in zip(chunks, all_embeddings, strict=True)
        }
        return cls.from_chunks(
            chunks,
            embeddings_map,
            embedder=embedder,
            vector_query_fn=vector_query_fn,
            source_name=source_name,
            config=config,
            labels=labels,
        )

    async def resolve(
        self,
        query: str,
        *,
        context: str = "",
        llm: Any | None = None,
        top_k: int = 10,
        intent: RetrievalIntent | None = None,
    ) -> list[SourceResult]:
        """Embed query, cluster seeds, match centroids, retrieve chunks.

        Per-turn pipeline (lazy mode):

        1. Resolve effective parameters.
        2. Fetch seed chunks via vector search.
        3. Embed the query.
        4. Cluster seed chunks by their embeddings.
        5. Match query embedding against seed-set cluster centroids.
        6. Return chunks from matched clusters.

        In eager mode (constructed via :meth:`from_chunks`), steps 2
        and 4 use the pre-built clusters instead.

        Raises:
            StrategyUnavailable: this index cannot run at all --- no
                embedder, or lazy mode with no way to fetch seeds.
                Checked up front, before any work, so a lazy index with
                no seed source does not pay for a query embedding it
                cannot use.
            Exception: whatever ``embedder`` or ``vector_query_fn``
                raises, unchanged. These are not absorbed into an empty
                result, because a caller reads an empty topic index as a
                vocabulary gap worth retrying another way -- see the
                comments at each call site. An index that ran and matched
                nothing still returns an empty list.
        """
        embedder = self._require_embedder()
        if self._eager_state() is None:
            self._require_vector_query_fn()

        params = _resolve_params(self._config, intent)

        logger.info(
            "ClusterTopicIndex resolving for source '%s': top_clusters=%d, centroid_threshold=%.2f",
            self._source_name,
            params.top_clusters,
            params.centroid_score_threshold,
        )

        # Embed the query. Not wrapped: a query that could not be embedded
        # is not a query with no topics behind it, and returning an empty
        # list says the second. That answer is read, not just logged --
        # the grounded retrieval loop treats an empty topic index as a
        # vocabulary gap and falls back to plain text retrieval, so
        # absorbing this reroutes the turn and reports the wrong cause for
        # it. The loop already drops a source that raises, with its cause.
        query_embedding = (await embedder.embed([query]))[0]

        # Pick up the filter slice keyed by our source name and forward
        # it through vector-seed fetching, matching the convention the
        # main VectorKnowledgeSource.query path uses. Empty/missing slice
        # → no filter (preserves historical behavior).
        filter_metadata: dict[str, Any] | None = None
        if intent is not None:
            filter_metadata = intent.filters.get(self._source_name) or None

        # Get chunks, embeddings, and clusters for this turn
        chunks, embeddings, clusters = await self._get_clusters(
            query,
            params,
            filter_metadata=filter_metadata,
        )

        if not clusters:
            logger.info(
                "ClusterTopicIndex: no clusters formed for source '%s'",
                self._source_name,
            )
            return []

        logger.info(
            "ClusterTopicIndex: %d seeds -> %d clusters for source '%s'",
            len(chunks),
            len(clusters),
            self._source_name,
        )

        # Score clusters by centroid similarity
        cluster_scores: list[tuple[_Cluster, float]] = []
        for cluster in clusters:
            score = cosine_similarity(query_embedding, cluster.centroid)
            if score >= params.centroid_score_threshold:
                cluster_scores.append((cluster, score))

        # Sort by score descending, take top N
        cluster_scores.sort(key=lambda x: x[1], reverse=True)
        cluster_scores = cluster_scores[: params.top_clusters]

        if not cluster_scores:
            logger.info(
                "ClusterTopicIndex: no clusters matched query for source '%s' (threshold=%.2f)",
                self._source_name,
                params.centroid_score_threshold,
            )
            return []

        matched_labels = [c.label for c, _ in cluster_scores]
        logger.info(
            "ClusterTopicIndex: query matched %d clusters for source '%s': %s",
            len(cluster_scores),
            self._source_name,
            matched_labels,
        )

        # Collect chunks from matched clusters, ranked by query similarity
        all_results: list[SourceResult] = []
        seen_ids: set[str] = set()

        for cluster, _cluster_score in cluster_scores:
            # Rank members by query similarity
            member_scores: list[tuple[int, float]] = []
            for idx in cluster.member_indices:
                sim = cosine_similarity(query_embedding, embeddings[idx])
                member_scores.append((idx, sim))
            member_scores.sort(key=lambda x: x[1], reverse=True)

            count = 0
            for idx, _sim in member_scores:
                if count >= params.max_results_per_cluster:
                    break
                chunk = chunks[idx]
                if chunk.source_id not in seen_ids:
                    seen_ids.add(chunk.source_id)
                    all_results.append(chunk)
                    count += 1

        # Cap total results
        effective_max = min(top_k, params.max_total_results)
        if len(all_results) > effective_max:
            all_results = all_results[:effective_max]

        logger.info(
            "ClusterTopicIndex: %d matched clusters -> %d chunks for source '%s'",
            len(cluster_scores),
            len(all_results),
            self._source_name,
        )

        return all_results

    def topics(self) -> list[str]:
        """Return cluster labels.

        Only available in eager mode.  Returns ``[]`` in lazy mode.
        """
        if self._clusters is None:
            return []
        return [c.label for c in self._clusters]

    @property
    def cluster_info(self) -> list[dict[str, Any]]:
        """Return cluster info for introspection (eager mode only).

        Each dict has ``id``, ``label``, ``size``, and ``centroid``.
        """
        if self._clusters is None:
            return []
        return [
            {
                "id": c.cluster_id,
                "label": c.label,
                "size": len(c.member_indices),
                "centroid": c.centroid,
            }
            for c in self._clusters
        ]

    # ------------------------------------------------------------------
    # Private: what this index needs before it can answer
    # ------------------------------------------------------------------

    def _eager_state(
        self,
    ) -> tuple[list[SourceResult], list[list[float]], list[_Cluster]] | None:
        """Pre-built chunks, embeddings and clusters, or ``None`` if lazy.

        The one place the mode is decided.  ``resolve`` needs the answer
        to know whether ``vector_query_fn`` is required, and
        ``_get_clusters`` needs the state itself, so this hands back the
        state rather than a boolean --- a predicate would leave the
        caller to re-narrow three optionals the type checker cannot
        narrow through a ``bool``.

        Note this is not ``self._chunks is not None``:
        :meth:`from_chunks` sets ``_chunks`` unconditionally but builds
        ``_clusters`` only when at least one chunk had an embedding, so
        an index built from chunks with no embeddings is *lazy* and has
        to seed like one.
        """
        if self._chunks is not None and self._embeddings is not None and self._clusters is not None:
            return self._chunks, self._embeddings, self._clusters
        return None

    def _require_embedder(self) -> TextEmbedder:
        """The embedder, or say that this index cannot resolve at all.

        Raises rather than returning an empty result: an index with no
        embedder cannot run in either mode, and answering ``[]`` would
        make it indistinguishable from one that ran and matched nothing.
        See :meth:`TopicIndex.resolve` for the contract.
        """
        if self._embedder is None:
            raise StrategyUnavailable(
                f"ClusterTopicIndex on source '{self._source_name}' has no embedder, "
                "so it cannot resolve queries. Pass embedder= at construction. "
                "(topics() and cluster_info remain available without one, so an "
                "index built only for inspection is still valid.)"
            )
        return self._embedder

    def _require_vector_query_fn(self) -> VectorQueryFn:
        """The seed-fetching callable, or say that this index cannot resolve.

        Only lazy mode needs it: an eager index built by
        :meth:`from_chunks` has its pool already and never seeds.
        """
        if self._vector_query_fn is None:
            raise StrategyUnavailable(
                f"ClusterTopicIndex on source '{self._source_name}' is in lazy mode "
                "with no vector_query_fn, so it cannot fetch the seed chunks it "
                "clusters. Pass vector_query_fn= at construction, or build the "
                "index eagerly with from_chunks()."
            )
        return self._vector_query_fn

    # ------------------------------------------------------------------
    # Private: per-turn cluster construction
    # ------------------------------------------------------------------

    async def _get_clusters(
        self,
        query: str,
        params: _EffectiveParams,
        filter_metadata: dict[str, Any] | None = None,
    ) -> tuple[list[SourceResult], list[list[float]], list[_Cluster]]:
        """Get chunks, embeddings, and clusters for this turn.

        In eager mode, returns pre-built state.
        In lazy mode, fetches seeds and clusters them per-turn.
        """
        eager = self._eager_state()
        if eager is not None:
            return eager

        # Lazy mode: fetch seeds and cluster per-turn
        seed_results = await self._fetch_vector_seeds(
            query,
            params,
            filter_metadata=filter_metadata,
        )
        if not seed_results:
            return [], [], []

        # Get embeddings for seeds — use the seed results' relevance
        # metadata if available, otherwise embed via the embedder
        seed_chunks, seed_embeddings = await self._embed_seeds(seed_results)
        if not seed_chunks:
            return [], [], []

        clusters = _build_clusters(
            seed_chunks,
            seed_embeddings,
            config=self._config,
        )
        return seed_chunks, seed_embeddings, clusters

    async def _fetch_vector_seeds(
        self,
        query: str,
        params: _EffectiveParams,
        *,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SourceResult]:
        """Fetch seed results via vector search.

        The guard is unreachable once ``resolve`` has run --- it checks
        the same invariant up front.  Kept because the type checker
        cannot see that, and so that a future path reaching here without
        going through ``resolve`` fails by name rather than as
        ``'NoneType' object is not callable``.
        """
        vector_query_fn = self._require_vector_query_fn()

        # This is the index's retrieval call, and it is not wrapped for the
        # same reason the query embed above is not: a store that cannot be
        # reached is not a store with no seeds in it.
        results = await vector_query_fn(
            query,
            params.seed_max_results,
            filter_metadata=filter_metadata,
        )

        return [r for r in results if r.relevance >= params.seed_score_threshold]

    async def _embed_seeds(
        self,
        seeds: list[SourceResult],
    ) -> tuple[list[SourceResult], list[list[float]]]:
        """Embed seed chunks for clustering.

        Uses the configured embedder on each seed's content.

        One seed per call rather than one batch, which is deliberate: the
        ``except`` below drops a single chunk that will not embed and clusters
        the rest, and a batch call cannot say which text failed. Batching this
        is a real improvement *and* a real change to that failure semantics,
        so it is not made here as a side effect of a type change.
        """
        embedder = self._require_embedder()

        chunks: list[SourceResult] = []
        embeddings: list[list[float]] = []
        last_error: Exception | None = None
        for seed in seeds:
            try:
                emb = (await embedder.embed([seed.content]))[0]
            except Exception as exc:
                # One chunk that will not embed is dropped so the rest of
                # the pool can still cluster, which is what this catch is
                # for. Reported at WARNING rather than DEBUG: a pool
                # quietly losing chunks is a degraded answer, and at DEBUG
                # nobody sees it happening.
                last_error = exc
                logger.warning(
                    "Failed to embed seed chunk '%s' for source '%s', skipping",
                    seed.source_id,
                    self._source_name,
                    exc_info=True,
                )
                continue
            chunks.append(seed)
            embeddings.append(emb)

        if seeds and not chunks:
            # Every seed failed. The per-chunk tolerance above has stopped
            # describing what happened -- this is not a pool with nothing
            # worth clustering, it is an embedder that cannot embed -- and
            # an empty pool reaches the caller as the former.
            raise RuntimeError(
                f"Every one of the {len(seeds)} seed chunks for source "
                f"{self._source_name!r} failed to embed"
            ) from last_error

        return chunks, embeddings


# ------------------------------------------------------------------
# Module-level cluster construction
# ------------------------------------------------------------------


def _build_clusters(
    chunks: list[SourceResult],
    embeddings: list[list[float]],
    *,
    config: ClusterTopicConfig,
    labels: dict[int, str] | None = None,
) -> list[_Cluster]:
    """Cluster chunks by embedding similarity.

    Used by both eager construction and per-turn lazy clustering.
    """
    n = len(chunks)
    if n == 0:
        return []

    # Build similarity matrix
    sim_matrix: list[list[float]] = []
    for i in range(n):
        row: list[float] = []
        for j in range(n):
            if i == j:
                row.append(1.0)
            else:
                row.append(cosine_similarity(embeddings[i], embeddings[j]))
        sim_matrix.append(row)

    # Cluster
    assignments = agglomerative_cluster(
        sim_matrix,
        config.cluster_threshold,
        config.min_cluster_size,
    )

    # Build cluster objects
    cluster_members: dict[int, list[int]] = {}
    for idx, cid in enumerate(assignments):
        if cid >= 0:
            cluster_members.setdefault(cid, []).append(idx)

    result: list[_Cluster] = []
    for cid, members in sorted(cluster_members.items()):
        centroid = _compute_centroid(embeddings, members)

        if labels and cid in labels:
            label = labels[cid]
        else:
            label = _auto_label(chunks, members, config)

        result.append(
            _Cluster(
                cluster_id=cid,
                label=label,
                member_indices=members,
                centroid=centroid,
            )
        )

    return result


def _compute_centroid(
    embeddings: list[list[float]],
    member_indices: list[int],
) -> list[float]:
    """Compute the mean embedding for a set of members."""
    if not member_indices:
        return []

    dim = len(embeddings[member_indices[0]])
    centroid = [0.0] * dim
    for idx in member_indices:
        emb = embeddings[idx]
        for d in range(dim):
            centroid[d] += emb[d]

    count = len(member_indices)
    return [v / count for v in centroid]


def _auto_label(
    chunks: list[SourceResult],
    member_indices: list[int],
    config: ClusterTopicConfig,
) -> str:
    """Generate a cluster label from content keywords.

    Uses configurable stopwords, minimum word length, and number
    of top terms from ``config``.
    """
    words: Counter[str] = Counter()
    stopwords = config.label_stopwords
    min_len = config.label_min_word_length

    for idx in member_indices:
        text = chunks[idx].content.lower()
        for word in text.split():
            cleaned = "".join(c for c in word if c.isalpha())
            if len(cleaned) >= min_len and cleaned not in stopwords:
                words[cleaned] += 1

    top = [w for w, _ in words.most_common(config.label_top_terms)]
    return " ".join(top) if top else f"cluster_{member_indices[0]}"
