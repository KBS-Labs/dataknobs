"""Result processing pipeline for grounded retrieval.

Provides configurable post-retrieval processing of
:class:`~dataknobs_data.sources.base.SourceResult` lists.  The pipeline
runs between merge and format in the retrieval flow, transforming raw
results into ranked, filtered, and optionally clustered output for
synthesis.

Key abstractions:

- :class:`ResultProcessor` -- protocol for a single processing stage.
- :class:`StrategyChain` -- ordered list of alternative processors;
  the first viable one wins.
- :class:`ResultPipeline` -- compose stages into an ordered pipeline.

Level 1 processors (no embedding dependency):

- :class:`CrossSourceNormalizer` -- normalize relevance scores across
  sources for cross-source comparability.
- :class:`RelativeRelevanceFilter` -- drop results significantly weaker
  than the best match.
- :class:`QueryRelevanceRanker` -- re-rank by term overlap with the
  user's original query.

Level 2-3 processors (clustering + query-cluster scoring):

- :class:`TermOverlapClusterer` -- group by shared significant terms.
- :class:`TfidfClusterer` -- TF-IDF vectors with cosine similarity.
- :class:`EmbeddingClusterer` -- semantic clustering via injected
  ``embed_fn`` callable.
- :class:`QueryClusterScorer` -- score clusters against the user's
  original query and re-order by cluster relevance.

This module is intentionally LLM-free -- it lives in ``dataknobs-data``
so any project can use it without depending on LLM or bots packages.
Embedding-based processors (Phase 3) accept an ``embed_fn`` callable
injected by the consumer.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter, defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from dataknobs_data.sources.base import RetrievalIntent, SourceResult

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Exceptions
# ------------------------------------------------------------------


class StrategyUnavailable(Exception):  # noqa: N818
    """A strategy cannot operate in the current context.

    Not an error -- signals the strategy chain to try the next
    alternative.  Distinct from actual processing errors, which
    propagate normally.

    A strategy that *can* run but *fails* raises a normal exception.
    ``StrategyUnavailable`` means "I'm not applicable", not "I broke."
    """


# ------------------------------------------------------------------
# Protocol
# ------------------------------------------------------------------


@runtime_checkable
class ResultProcessor(Protocol):
    """Single processing stage in the result pipeline.

    Raises :class:`StrategyUnavailable` if it cannot operate in the
    current context (e.g., embedding strategy with no embed_fn).
    The pipeline or strategy chain catches this and tries the next
    alternative.
    """

    async def process(
        self,
        results: list[SourceResult],
        intent: RetrievalIntent,
        user_message: str,
    ) -> list[SourceResult]:
        """Process results and return the (possibly modified) list."""
        ...


# ------------------------------------------------------------------
# Strategy chain and pipeline
# ------------------------------------------------------------------


@dataclass
class StrategyChain:
    """Ordered list of alternative strategies -- first viable one wins.

    The config author controls presence and order.  Each strategy is
    tried in sequence.  :class:`StrategyUnavailable` signals "try next."
    If all strategies are exhausted, results pass through unmodified
    (the config author chose not to add more fallbacks).

    Attributes:
        strategies: Ordered list of processors to attempt.
        name: Label for logging (e.g. ``"cluster"``, ``"normalize"``).
    """

    strategies: list[ResultProcessor]
    name: str = ""

    async def process(
        self,
        results: list[SourceResult],
        intent: RetrievalIntent,
        user_message: str,
    ) -> list[SourceResult]:
        """Try strategies in order; return first successful result."""
        for strategy in self.strategies:
            try:
                return await strategy.process(results, intent, user_message)
            except StrategyUnavailable as e:
                logger.info(
                    "%s strategy %s unavailable: %s",
                    self.name,
                    type(strategy).__name__,
                    e,
                )
                continue
        logger.warning(
            "%s: all strategies exhausted, results pass through unprocessed",
            self.name,
        )
        return results


@dataclass
class ResultPipeline:
    """Compose processing stages into an ordered pipeline.

    Each stage is either a single :class:`ResultProcessor` or a
    :class:`StrategyChain` of alternatives.  Stages run in sequence;
    within a chain, alternatives are tried in order.

    Attributes:
        stages: Ordered list of processing stages.
    """

    stages: list[ResultProcessor | StrategyChain] = field(default_factory=list)

    async def process(
        self,
        results: list[SourceResult],
        intent: RetrievalIntent,
        user_message: str,
    ) -> list[SourceResult]:
        """Run all stages in order."""
        for stage in self.stages:
            results = await stage.process(results, intent, user_message)
        return results


# ------------------------------------------------------------------
# Level 1 processors (no embedding dependency)
# ------------------------------------------------------------------


# Stopwords for query relevance -- lightweight set covering the most
# common English function words.
_QUERY_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "about", "between",
    "through", "during", "before", "after", "and", "but", "or", "not",
    "no", "if", "then", "than", "so", "that", "this", "it", "its",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "they",
    "what", "which", "who", "how", "when", "where", "why",
    "tell", "show", "give", "know", "want", "need", "like",
})

_WORD_PATTERN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Extract lowercase alphanumeric tokens."""
    return _WORD_PATTERN.findall(text.lower())


def _significant_tokens(text: str) -> set[str]:
    """Extract tokens minus stopwords."""
    return {t for t in _tokenize(text) if t not in _QUERY_STOPWORDS}


@dataclass
class CrossSourceNormalizer:
    """Normalize relevance scores across sources for comparability.

    Within each source, scores may have different scales and
    distributions (vector cosine similarity vs database term-coverage
    ratio).  Normalization makes them comparable for cross-source
    operations.

    Attributes:
        strategy: Normalization method.

            - ``"min_max"`` -- scale each source's scores to [0, 1]
            - ``"z_score"`` -- standardize by mean/stddev per source
            - ``"rank"`` -- replace scores with reciprocal rank position
    """

    strategy: str = "min_max"

    async def process(
        self,
        results: list[SourceResult],
        intent: RetrievalIntent,
        user_message: str,
    ) -> list[SourceResult]:
        """Normalize relevance scores across sources."""
        if not results:
            return results

        # Group by source
        by_source: dict[str, list[tuple[int, SourceResult]]] = defaultdict(list)
        for idx, r in enumerate(results):
            by_source[r.source_name].append((idx, r))

        # Single source -- no normalization needed
        if len(by_source) <= 1:
            return results

        normalized = list(results)  # shallow copy

        if self.strategy == "min_max":
            normalized = self._normalize_min_max(normalized, by_source)
        elif self.strategy == "z_score":
            normalized = self._normalize_z_score(normalized, by_source)
        elif self.strategy == "rank":
            normalized = self._normalize_rank(normalized, by_source)
        else:
            raise StrategyUnavailable(
                f"Unknown normalization strategy: {self.strategy!r}"
            )

        return normalized

    @staticmethod
    def _normalize_min_max(
        results: list[SourceResult],
        by_source: dict[str, list[tuple[int, SourceResult]]],
    ) -> list[SourceResult]:
        """Scale each source's scores to [0, 1] range."""
        out = list(results)
        for entries in by_source.values():
            scores = [r.relevance for _, r in entries]
            lo, hi = min(scores), max(scores)
            span = hi - lo
            for idx, r in entries:
                if span > 0:
                    norm = (r.relevance - lo) / span
                else:
                    norm = 1.0  # all same score -> treat as max
                out[idx] = SourceResult(
                    content=r.content,
                    source_id=r.source_id,
                    source_name=r.source_name,
                    source_type=r.source_type,
                    relevance=norm,
                    metadata={**r.metadata, "_raw_relevance": r.relevance},
                )
        return out

    @staticmethod
    def _normalize_z_score(
        results: list[SourceResult],
        by_source: dict[str, list[tuple[int, SourceResult]]],
    ) -> list[SourceResult]:
        """Standardize by mean/stddev, then clamp to [0, 1]."""
        out = list(results)
        for entries in by_source.values():
            scores = [r.relevance for _, r in entries]
            n = len(scores)
            mean = sum(scores) / n
            variance = sum((s - mean) ** 2 for s in scores) / n
            std = math.sqrt(variance) if variance > 0 else 1.0
            for idx, r in entries:
                z = (r.relevance - mean) / std
                # Map z-scores to [0, 1] via sigmoid-like clamping
                norm = max(0.0, min(1.0, 0.5 + z * 0.25))
                out[idx] = SourceResult(
                    content=r.content,
                    source_id=r.source_id,
                    source_name=r.source_name,
                    source_type=r.source_type,
                    relevance=norm,
                    metadata={**r.metadata, "_raw_relevance": r.relevance},
                )
        return out

    @staticmethod
    def _normalize_rank(
        results: list[SourceResult],
        by_source: dict[str, list[tuple[int, SourceResult]]],
    ) -> list[SourceResult]:
        """Replace scores with reciprocal rank position (1/rank)."""
        out = list(results)
        for entries in by_source.values():
            # Sort by relevance descending to assign ranks
            ranked = sorted(entries, key=lambda e: e[1].relevance, reverse=True)
            for rank_pos, (idx, r) in enumerate(ranked, start=1):
                norm = 1.0 / rank_pos
                out[idx] = SourceResult(
                    content=r.content,
                    source_id=r.source_id,
                    source_name=r.source_name,
                    source_type=r.source_type,
                    relevance=norm,
                    metadata={**r.metadata, "_raw_relevance": r.relevance},
                )
        return out


@dataclass
class RelativeRelevanceFilter:
    """Drop results significantly weaker than the best match.

    After normalization, removes results whose relevance is below
    ``threshold * max_relevance``.  This eliminates tangential results
    that technically exceeded the source's ``score_threshold`` but are
    far less relevant than the best matches.

    Attributes:
        threshold: Keep results >= this fraction of the best score.
        min_results: Never drop below this count regardless of scores.
    """

    threshold: float = 0.5
    min_results: int = 3

    async def process(
        self,
        results: list[SourceResult],
        intent: RetrievalIntent,
        user_message: str,
    ) -> list[SourceResult]:
        """Filter results below the relative threshold."""
        if not results:
            return results

        max_score = max(r.relevance for r in results)
        cutoff = self.threshold * max_score

        above = [r for r in results if r.relevance >= cutoff]

        # Respect min_results floor
        if len(above) >= self.min_results:
            return above

        # Take the top min_results by relevance
        by_score = sorted(results, key=lambda r: r.relevance, reverse=True)
        return by_score[: max(self.min_results, len(above))]


@dataclass
class QueryRelevanceRanker:
    """Re-rank results by term overlap with the user's original query.

    The generated search queries may diverge from the user's phrasing.
    This processor boosts results whose content contains terms from
    the original user message, ensuring query-relevant results surface.

    The final score is a weighted blend:
    ``(1 - boost_weight) * relevance + boost_weight * query_overlap``

    Attributes:
        boost_weight: Blend weight for query term overlap (0.0-1.0).
    """

    boost_weight: float = 0.3

    async def process(
        self,
        results: list[SourceResult],
        intent: RetrievalIntent,
        user_message: str,
    ) -> list[SourceResult]:
        """Re-rank results by blending relevance with query overlap."""
        if not results or not user_message:
            return results

        query_terms = _significant_tokens(user_message)
        if not query_terms:
            return results

        scored: list[tuple[float, SourceResult]] = []
        for r in results:
            content_tokens = _significant_tokens(r.content)
            if content_tokens:
                overlap = len(query_terms & content_tokens) / len(query_terms)
            else:
                overlap = 0.0

            blended = (1.0 - self.boost_weight) * r.relevance + self.boost_weight * overlap
            new_result = SourceResult(
                content=r.content,
                source_id=r.source_id,
                source_name=r.source_name,
                source_type=r.source_type,
                relevance=blended,
                metadata={**r.metadata, "_query_overlap": overlap},
            )
            scored.append((blended, new_result))

        # Sort descending by blended score
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored]


# ------------------------------------------------------------------
# Embedding function type
# ------------------------------------------------------------------

EmbedFn = Callable[[list[str]], Awaitable[list[list[float]]]]
"""Async callable that embeds a batch of texts into vectors.

Injected by the consumer (e.g. from a VectorKnowledgeSource's
embedding provider or a dedicated embedding model config).
Keeps this module free of LLM package dependencies.
"""


# ------------------------------------------------------------------
# Level 2-3 processors (clustering + query-cluster scoring)
# ------------------------------------------------------------------


def _auto_label(tokens_per_result: list[set[str]], member_indices: list[int]) -> str:
    """Generate a cluster label from the most frequent shared terms."""
    counts: Counter[str] = Counter()
    for idx in member_indices:
        counts.update(tokens_per_result[idx])
    # Top 3 most common terms across cluster members
    top = [word for word, _ in counts.most_common(3)]
    return " ".join(top) if top else "misc"


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _agglomerative_cluster(
    similarity_matrix: list[list[float]],
    threshold: float,
    min_cluster_size: int,
) -> list[int]:
    """Simple agglomerative clustering on a similarity matrix.

    Returns a list of cluster IDs (one per result).  Results that
    don't meet min_cluster_size are assigned cluster_id -1 (unclustered).
    """
    n = len(similarity_matrix)
    if n == 0:
        return []

    # Start: each result in its own cluster
    assignments = list(range(n))

    # Greedy merge: find most similar pair above threshold, merge
    while True:
        best_sim = -1.0
        best_i, best_j = -1, -1
        for i in range(n):
            for j in range(i + 1, n):
                if assignments[i] == assignments[j]:
                    continue
                if similarity_matrix[i][j] > best_sim:
                    best_sim = similarity_matrix[i][j]
                    best_i, best_j = i, j

        if best_sim < threshold:
            break

        # Merge: assign all of cluster_j to cluster_i
        old_cluster = assignments[best_j]
        new_cluster = assignments[best_i]
        for k in range(n):
            if assignments[k] == old_cluster:
                assignments[k] = new_cluster

    # Renumber clusters sequentially and mark small ones as -1
    cluster_members: dict[int, list[int]] = defaultdict(list)
    for idx, cid in enumerate(assignments):
        cluster_members[cid].append(idx)

    final = [-1] * n
    next_id = 0
    for members in cluster_members.values():
        if len(members) >= min_cluster_size:
            for idx in members:
                final[idx] = next_id
            next_id += 1

    return final


def _annotate_clusters(
    results: list[SourceResult],
    assignments: list[int],
    tokens_per_result: list[set[str]],
) -> list[SourceResult]:
    """Annotate results with cluster metadata."""
    # Compute cluster sizes and labels
    cluster_members: dict[int, list[int]] = defaultdict(list)
    for idx, cid in enumerate(assignments):
        if cid >= 0:
            cluster_members[cid].append(idx)

    labels: dict[int, str] = {}
    sizes: dict[int, int] = {}
    for cid, members in cluster_members.items():
        labels[cid] = _auto_label(tokens_per_result, members)
        sizes[cid] = len(members)

    out: list[SourceResult] = []
    for idx, r in enumerate(results):
        cid = assignments[idx]
        meta = {**r.metadata}
        if cid >= 0:
            meta["cluster_id"] = cid
            meta["cluster_label"] = labels[cid]
            meta["cluster_size"] = sizes[cid]
        else:
            meta["cluster_id"] = -1
            meta["cluster_label"] = "unclustered"
            meta["cluster_size"] = 1
        out.append(SourceResult(
            content=r.content,
            source_id=r.source_id,
            source_name=r.source_name,
            source_type=r.source_type,
            relevance=r.relevance,
            metadata=meta,
        ))
    return out


@dataclass
class TermOverlapClusterer:
    """Cluster results by shared significant terms.

    Groups results that share terms above a threshold ratio.
    Lightest-weight clustering -- fully deterministic, no external
    dependencies.  Ideal for tests.

    Attributes:
        similarity_threshold: Minimum Jaccard overlap to merge.
        min_cluster_size: Minimum results to form a cluster.
    """

    similarity_threshold: float = 0.3
    min_cluster_size: int = 2

    async def process(
        self,
        results: list[SourceResult],
        intent: RetrievalIntent,
        user_message: str,
    ) -> list[SourceResult]:
        """Cluster by term overlap."""
        if len(results) < self.min_cluster_size:
            return results

        tokens = [_significant_tokens(r.content) for r in results]
        n = len(results)

        # Build similarity matrix using Jaccard similarity
        sim: list[list[float]] = [[0.0] * n for _ in range(n)]
        for i in range(n):
            sim[i][i] = 1.0
            for j in range(i + 1, n):
                if tokens[i] and tokens[j]:
                    jaccard = len(tokens[i] & tokens[j]) / len(tokens[i] | tokens[j])
                else:
                    jaccard = 0.0
                sim[i][j] = jaccard
                sim[j][i] = jaccard

        assignments = _agglomerative_cluster(
            sim, self.similarity_threshold, self.min_cluster_size,
        )
        return _annotate_clusters(results, assignments, tokens)


@dataclass
class TfidfClusterer:
    """Cluster results by TF-IDF cosine similarity.

    Computes TF-IDF vectors from result content and clusters using
    cosine similarity.  No external model needed.  Deterministic.

    Attributes:
        similarity_threshold: Minimum cosine similarity to merge.
        min_cluster_size: Minimum results to form a cluster.
    """

    similarity_threshold: float = 0.5
    min_cluster_size: int = 2

    async def process(
        self,
        results: list[SourceResult],
        intent: RetrievalIntent,
        user_message: str,
    ) -> list[SourceResult]:
        """Cluster by TF-IDF cosine similarity."""
        if len(results) < self.min_cluster_size:
            return results

        tokens_per_doc = [_tokenize(r.content) for r in results]
        sig_tokens = [_significant_tokens(r.content) for r in results]

        # Build vocabulary
        vocab: dict[str, int] = {}
        for toks in tokens_per_doc:
            for t in toks:
                if t not in _QUERY_STOPWORDS and t not in vocab:
                    vocab[t] = len(vocab)

        if not vocab:
            return results

        n_docs = len(results)
        n_terms = len(vocab)

        # Compute TF-IDF vectors
        # Document frequency
        df: list[int] = [0] * n_terms
        for toks in tokens_per_doc:
            seen: set[str] = set()
            for t in toks:
                if t in vocab and t not in seen:
                    df[vocab[t]] += 1
                    seen.add(t)

        # TF-IDF per document
        vectors: list[list[float]] = []
        for toks in tokens_per_doc:
            tf_counts: Counter[str] = Counter(t for t in toks if t in vocab)
            vec = [0.0] * n_terms
            for term, count in tf_counts.items():
                idx = vocab[term]
                tf = count / len(toks) if toks else 0.0
                idf = math.log((n_docs + 1) / (df[idx] + 1)) + 1.0
                vec[idx] = tf * idf
            vectors.append(vec)

        # Build cosine similarity matrix
        sim: list[list[float]] = [[0.0] * n_docs for _ in range(n_docs)]
        for i in range(n_docs):
            sim[i][i] = 1.0
            for j in range(i + 1, n_docs):
                cs = _cosine_similarity(vectors[i], vectors[j])
                sim[i][j] = cs
                sim[j][i] = cs

        assignments = _agglomerative_cluster(
            sim, self.similarity_threshold, self.min_cluster_size,
        )
        return _annotate_clusters(results, assignments, sig_tokens)


@dataclass
class EmbeddingClusterer:
    """Cluster results by embedding similarity.

    Uses an injected ``embed_fn`` to compute embeddings for result
    content, then clusters using cosine similarity.  Highest quality
    but requires an embedding model.

    Raises :class:`StrategyUnavailable` if no ``embed_fn`` is set.

    Attributes:
        similarity_threshold: Minimum cosine similarity to merge.
        min_cluster_size: Minimum results to form a cluster.
        embed_fn: Async embedding function, injected at pipeline
            construction time.
    """

    similarity_threshold: float = 0.7
    min_cluster_size: int = 2
    embed_fn: EmbedFn | None = None

    async def process(
        self,
        results: list[SourceResult],
        intent: RetrievalIntent,
        user_message: str,
    ) -> list[SourceResult]:
        """Cluster by embedding cosine similarity."""
        if self.embed_fn is None:
            raise StrategyUnavailable("No embed_fn configured")

        if len(results) < self.min_cluster_size:
            return results

        # Compute embeddings for all result content
        texts = [r.content for r in results]
        embeddings = await self.embed_fn(texts)

        n = len(results)
        sig_tokens = [_significant_tokens(r.content) for r in results]

        # Build cosine similarity matrix
        sim: list[list[float]] = [[0.0] * n for _ in range(n)]
        for i in range(n):
            sim[i][i] = 1.0
            for j in range(i + 1, n):
                cs = _cosine_similarity(embeddings[i], embeddings[j])
                sim[i][j] = cs
                sim[j][i] = cs

        assignments = _agglomerative_cluster(
            sim, self.similarity_threshold, self.min_cluster_size,
        )
        return _annotate_clusters(results, assignments, sig_tokens)


@dataclass
class QueryClusterScorer:
    """Score clusters by relevance to the user's original query.

    For embedding-based scoring, computes a query embedding and scores
    each cluster's centroid (mean embedding of members) against it.
    For term-based scoring (when no embed_fn), uses term overlap
    between the query and concatenated cluster content.

    Annotates ``SourceResult.metadata`` with ``cluster_query_score``
    and re-orders results so higher-scoring clusters appear first.
    Within each cluster, original relevance order is preserved.

    Attributes:
        embed_fn: Optional async embedding function.  When ``None``,
            falls back to term-overlap scoring.
    """

    embed_fn: EmbedFn | None = None

    async def process(
        self,
        results: list[SourceResult],
        intent: RetrievalIntent,
        user_message: str,
    ) -> list[SourceResult]:
        """Score clusters against the query and re-order."""
        if not results or not user_message:
            return results

        # Group results by cluster_id
        clusters: dict[int, list[tuple[int, SourceResult]]] = defaultdict(list)
        for idx, r in enumerate(results):
            cid = r.metadata.get("cluster_id", -1)
            clusters[cid].append((idx, r))

        # Score each cluster against the query
        cluster_scores: dict[int, float] = {}

        if self.embed_fn is not None:
            cluster_scores = await self._score_with_embeddings(
                clusters, user_message,
            )
        else:
            cluster_scores = self._score_with_terms(clusters, user_message)

        # Annotate results with cluster_query_score
        annotated: list[SourceResult] = list(results)
        for cid, entries in clusters.items():
            score = cluster_scores.get(cid, 0.0)
            for idx, r in entries:
                annotated[idx] = SourceResult(
                    content=r.content,
                    source_id=r.source_id,
                    source_name=r.source_name,
                    source_type=r.source_type,
                    relevance=r.relevance,
                    metadata={**r.metadata, "cluster_query_score": score},
                )

        # Re-order: highest-scoring cluster first, within-cluster order preserved
        def sort_key(r: SourceResult) -> tuple[float, float]:
            return (
                -r.metadata.get("cluster_query_score", 0.0),
                -r.relevance,
            )

        annotated.sort(key=sort_key)
        return annotated

    async def _score_with_embeddings(
        self,
        clusters: dict[int, list[tuple[int, SourceResult]]],
        user_message: str,
    ) -> dict[int, float]:
        """Score clusters using embedding cosine similarity."""
        assert self.embed_fn is not None

        # Collect all texts to embed in one batch
        cluster_texts: dict[int, str] = {}
        for cid, entries in clusters.items():
            combined = " ".join(r.content for _, r in entries)
            cluster_texts[cid] = combined

        texts_to_embed = [user_message] + list(cluster_texts.values())
        embeddings = await self.embed_fn(texts_to_embed)

        query_emb = embeddings[0]
        scores: dict[int, float] = {}
        for i, cid in enumerate(cluster_texts.keys()):
            scores[cid] = _cosine_similarity(query_emb, embeddings[i + 1])

        return scores

    @staticmethod
    def _score_with_terms(
        clusters: dict[int, list[tuple[int, SourceResult]]],
        user_message: str,
    ) -> dict[int, float]:
        """Score clusters using term overlap with the query."""
        query_terms = _significant_tokens(user_message)
        if not query_terms:
            return dict.fromkeys(clusters, 0.0)

        scores: dict[int, float] = {}
        for cid, entries in clusters.items():
            combined = " ".join(r.content for _, r in entries)
            cluster_terms = _significant_tokens(combined)
            if cluster_terms:
                overlap = len(query_terms & cluster_terms) / len(query_terms)
            else:
                overlap = 0.0
            scores[cid] = overlap

        return scores


# ------------------------------------------------------------------
# Pipeline factory
# ------------------------------------------------------------------


def inject_embed_fn(pipeline: ResultPipeline, embed_fn: EmbedFn) -> None:
    """Inject an embedding function into all embedding-aware processors.

    Walks the pipeline stages (and strategy chains within them) and
    sets ``embed_fn`` on any :class:`EmbeddingClusterer` or
    :class:`QueryClusterScorer` that has ``embed_fn`` as an attribute.

    Call this after :func:`build_pipeline` when an embedding provider
    becomes available (e.g. from a ``VectorKnowledgeSource``).
    """
    for stage in pipeline.stages:
        if isinstance(stage, StrategyChain):
            for strategy in stage.strategies:
                if hasattr(strategy, "embed_fn"):
                    strategy.embed_fn = embed_fn  # type: ignore[union-attr]
        elif hasattr(stage, "embed_fn"):
            stage.embed_fn = embed_fn  # type: ignore[union-attr]


def build_pipeline(config: dict[str, Any] | None) -> ResultPipeline | None:
    """Build a :class:`ResultPipeline` from a config dict.

    Returns ``None`` if the config is empty or has no active stages.

    Config keys:

    - ``normalize_strategy`` -- ``str`` or ``list[dict]``
    - ``relative_threshold`` -- ``float``
    - ``min_results`` -- ``int`` (default 3)
    - ``query_rerank_weight`` -- ``float``
    - ``cluster_strategy`` -- ``str`` or ``list[dict]`` (Phase 3)
    - ``cluster_min_size`` -- ``int`` (Phase 3)
    - ``cluster_threshold`` -- ``float`` (Phase 3)
    """
    if not config:
        return None

    stages: list[ResultProcessor | StrategyChain] = []

    # Normalization
    norm_cfg = config.get("normalize_strategy")
    if norm_cfg is not None:
        norm_chain = _build_normalize_chain(norm_cfg)
        if norm_chain is not None:
            stages.append(norm_chain)

    # Relative relevance filter
    rel_threshold = config.get("relative_threshold")
    if rel_threshold is not None:
        min_results = config.get("min_results", 3)
        stages.append(RelativeRelevanceFilter(
            threshold=float(rel_threshold),
            min_results=int(min_results),
        ))

    # Query relevance ranker
    rerank_weight = config.get("query_rerank_weight")
    if rerank_weight is not None:
        stages.append(QueryRelevanceRanker(boost_weight=float(rerank_weight)))

    # Clustering strategy chain
    cluster_cfg = config.get("cluster_strategy")
    if cluster_cfg is not None:
        cluster_min = int(config.get("cluster_min_size", 2))
        cluster_thresh = float(config.get("cluster_threshold", 0.7))
        cluster_chain = _build_cluster_chain(
            cluster_cfg, cluster_min, cluster_thresh,
        )
        if cluster_chain is not None:
            stages.append(cluster_chain)
            # QueryClusterScorer runs after clustering
            stages.append(QueryClusterScorer())

    if not stages:
        return None

    return ResultPipeline(stages=stages)


def _build_normalize_chain(
    cfg: str | list[dict[str, Any]],
) -> StrategyChain | CrossSourceNormalizer:
    """Build normalization from config (shorthand or chain)."""
    if isinstance(cfg, str):
        return CrossSourceNormalizer(strategy=cfg)

    if isinstance(cfg, list):
        strategies: list[ResultProcessor] = []
        for entry in cfg:
            method = entry.get("method", "min_max")
            strategies.append(CrossSourceNormalizer(strategy=method))
        if len(strategies) == 1:
            return strategies[0]  # type: ignore[return-value]
        return StrategyChain(strategies=strategies, name="normalize")

    return CrossSourceNormalizer(strategy="min_max")


def _build_cluster_chain(
    cfg: str | list[dict[str, Any]],
    min_size: int,
    threshold: float,
) -> StrategyChain | ResultProcessor | None:
    """Build clustering from config (shorthand or chain)."""
    if isinstance(cfg, str):
        return _make_clusterer(cfg, min_size, threshold)

    if isinstance(cfg, list):
        strategies: list[ResultProcessor] = []
        for entry in cfg:
            method = entry.get("method", "term_overlap")
            c_min = int(entry.get("min_cluster_size", min_size))
            c_thresh = float(entry.get("cluster_threshold", threshold))
            clusterer = _make_clusterer(method, c_min, c_thresh)
            if clusterer is not None:
                strategies.append(clusterer)
        if not strategies:
            return None
        if len(strategies) == 1:
            return strategies[0]
        return StrategyChain(strategies=strategies, name="cluster")

    return None


def _make_clusterer(
    method: str,
    min_size: int,
    threshold: float,
) -> ResultProcessor | None:
    """Create a single clusterer by method name."""
    if method == "term_overlap":
        return TermOverlapClusterer(
            similarity_threshold=threshold,
            min_cluster_size=min_size,
        )
    if method == "tfidf":
        return TfidfClusterer(
            similarity_threshold=threshold,
            min_cluster_size=min_size,
        )
    if method == "embedding":
        # embed_fn must be injected separately; raise if not available
        return EmbeddingClusterer(
            similarity_threshold=threshold,
            min_cluster_size=min_size,
            embed_fn=None,  # Consumer must inject
        )
    logger.warning("Unknown cluster method: %s", method)
    return None
