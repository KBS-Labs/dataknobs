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
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from .base import RetrievalIntent, SourceResult
from .processing import agglomerative_cluster, cosine_similarity
from .topic_index import DEFAULT_HEADING_STOPWORDS

logger = logging.getLogger(__name__)

# Type alias for an async embed function: text -> embedding vector.
EmbedFn = Callable[[str], Awaitable[list[float]]]

# Type alias for a batch embed function: texts -> list of embedding vectors.
BatchEmbedFn = Callable[[list[str]], Awaitable[list[list[float]]]]

# Type alias for a vector query function (same as HeadingTreeIndex).
# Accepts (query_text, top_k) and returns scored results.
VectorQueryFn = Callable[[str, int], Awaitable[list[SourceResult]]]

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
        embed_fn: Async function to embed a single text string.
            Used at resolve time to embed the query for centroid
            matching.
        vector_query_fn: Async function for vector-based seeding.
            Accepts ``(query, top_k)`` and returns scored results.
            Required for lazy mode.
        source_name: Source name for logging and provenance.
        config: Cluster configuration.
    """

    def __init__(
        self,
        *,
        embed_fn: EmbedFn | None = None,
        vector_query_fn: VectorQueryFn | None = None,
        source_name: str = "knowledge_base",
        config: ClusterTopicConfig | None = None,
    ) -> None:
        self._config = config or ClusterTopicConfig()
        self._source_name = source_name
        self._embed_fn = embed_fn
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
        embed_fn: EmbedFn | None = None,
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
            embed_fn: Async function to embed query text at resolve time.
            vector_query_fn: Unused in eager mode but stored for API
                consistency.
            source_name: Source name for provenance.
            config: Cluster configuration.
            labels: Optional user-supplied cluster labels keyed by
                cluster ID.
        """
        idx = cls(
            embed_fn=embed_fn,
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
                filtered_chunks, filtered_embeddings,
                config=idx._config, labels=labels,
            )

        return idx

    @classmethod
    async def build(
        cls,
        chunks: list[SourceResult],
        batch_embed_fn: BatchEmbedFn,
        *,
        embed_fn: EmbedFn | None = None,
        vector_query_fn: VectorQueryFn | None = None,
        source_name: str = "knowledge_base",
        config: ClusterTopicConfig | None = None,
        labels: dict[int, str] | None = None,
    ) -> ClusterTopicIndex:
        """Embed chunks and eagerly cluster.  One-time construction cost.

        Args:
            chunks: Source result chunks.
            batch_embed_fn: Async function that embeds a batch of texts.
            embed_fn: Async function to embed a single query at resolve time.
            vector_query_fn: Stored for API consistency.
            source_name: Source name for provenance.
            config: Cluster configuration.
            labels: Optional user-supplied cluster labels.
        """
        texts = [c.content for c in chunks]
        if not texts:
            return cls(
                embed_fn=embed_fn,
                vector_query_fn=vector_query_fn,
                source_name=source_name,
                config=config,
            )

        all_embeddings = await batch_embed_fn(texts)
        embeddings_map = {
            chunk.source_id: emb
            for chunk, emb in zip(chunks, all_embeddings)
        }
        return cls.from_chunks(
            chunks, embeddings_map,
            embed_fn=embed_fn,
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
        """
        if self._embed_fn is None:
            logger.warning(
                "No embed_fn configured for ClusterTopicIndex on source '%s' "
                "— cannot resolve queries. Provide embed_fn at construction.",
                self._source_name,
            )
            return []

        params = _resolve_params(self._config, intent)

        logger.info(
            "ClusterTopicIndex resolving for source '%s': "
            "top_clusters=%d, centroid_threshold=%.2f",
            self._source_name, params.top_clusters,
            params.centroid_score_threshold,
        )

        # Embed the query
        try:
            query_embedding = await self._embed_fn(query)
        except Exception:
            logger.warning(
                "Query embedding failed for source '%s'",
                self._source_name, exc_info=True,
            )
            return []

        # Get chunks, embeddings, and clusters for this turn
        chunks, embeddings, clusters = await self._get_clusters(query, params)

        if not clusters:
            logger.info(
                "ClusterTopicIndex: no clusters formed for source '%s'",
                self._source_name,
            )
            return []

        logger.info(
            "ClusterTopicIndex: %d seeds -> %d clusters for source '%s'",
            len(chunks), len(clusters), self._source_name,
        )

        # Score clusters by centroid similarity
        cluster_scores: list[tuple[_Cluster, float]] = []
        for cluster in clusters:
            score = cosine_similarity(query_embedding, cluster.centroid)
            if score >= params.centroid_score_threshold:
                cluster_scores.append((cluster, score))

        # Sort by score descending, take top N
        cluster_scores.sort(key=lambda x: x[1], reverse=True)
        cluster_scores = cluster_scores[:params.top_clusters]

        if not cluster_scores:
            logger.info(
                "ClusterTopicIndex: no clusters matched query for source '%s' "
                "(threshold=%.2f)",
                self._source_name, params.centroid_score_threshold,
            )
            return []

        matched_labels = [c.label for c, _ in cluster_scores]
        logger.info(
            "ClusterTopicIndex: query matched %d clusters for source '%s': %s",
            len(cluster_scores), self._source_name, matched_labels,
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
            "ClusterTopicIndex: %d matched clusters -> %d chunks "
            "for source '%s'",
            len(cluster_scores), len(all_results), self._source_name,
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
    # Private: per-turn cluster construction
    # ------------------------------------------------------------------

    async def _get_clusters(
        self,
        query: str,
        params: _EffectiveParams,
    ) -> tuple[list[SourceResult], list[list[float]], list[_Cluster]]:
        """Get chunks, embeddings, and clusters for this turn.

        In eager mode, returns pre-built state.
        In lazy mode, fetches seeds and clusters them per-turn.
        """
        if (
            self._chunks is not None
            and self._embeddings is not None
            and self._clusters is not None
        ):
            return self._chunks, self._embeddings, self._clusters

        # Lazy mode: fetch seeds and cluster per-turn
        seed_results = await self._fetch_vector_seeds(query, params)
        if not seed_results:
            return [], [], []

        # Get embeddings for seeds — use the seed results' relevance
        # metadata if available, otherwise embed via embed_fn
        seed_chunks, seed_embeddings = await self._embed_seeds(seed_results)
        if not seed_chunks:
            return [], [], []

        clusters = _build_clusters(
            seed_chunks, seed_embeddings, config=self._config,
        )
        return seed_chunks, seed_embeddings, clusters

    async def _fetch_vector_seeds(
        self,
        query: str,
        params: _EffectiveParams,
    ) -> list[SourceResult]:
        """Fetch seed results via vector search."""
        if self._vector_query_fn is None:
            logger.debug(
                "No vector_query_fn configured for source '%s', "
                "cannot fetch seeds in lazy mode",
                self._source_name,
            )
            return []

        try:
            results = await self._vector_query_fn(
                query, params.seed_max_results,
            )
        except Exception:
            logger.warning(
                "Vector query failed for source '%s'",
                self._source_name, exc_info=True,
            )
            return []

        return [
            r for r in results
            if r.relevance >= params.seed_score_threshold
        ]

    async def _embed_seeds(
        self,
        seeds: list[SourceResult],
    ) -> tuple[list[SourceResult], list[list[float]]]:
        """Embed seed chunks for clustering.

        Uses ``embed_fn`` to embed each seed's content.
        """
        if self._embed_fn is None:
            return [], []

        chunks: list[SourceResult] = []
        embeddings: list[list[float]] = []
        for seed in seeds:
            try:
                emb = await self._embed_fn(seed.content)
                chunks.append(seed)
                embeddings.append(emb)
            except Exception:
                logger.debug(
                    "Failed to embed seed chunk '%s', skipping",
                    seed.source_id,
                )
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

        result.append(_Cluster(
            cluster_id=cid,
            label=label,
            member_indices=members,
            centroid=centroid,
        ))

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
