"""Heading-tree topic index for structured document retrieval.

Uses heading metadata on chunks to identify and expand topic regions.
This is the ``dataknobs-bots`` implementation of the
:class:`~dataknobs_data.sources.topic_index.TopicIndex` protocol.

The index uses **lazy query-driven reconstruction**: no full heading tree
is pre-built.  Each ``resolve()`` call drives a per-turn pipeline:

1. Find seed chunks via vector search and/or heading-text matching.
2. Build a partial heading tree from the seeds' heading metadata.
3. Match query terms against the partial tree to identify regions.
4. Expand matched regions using chunks from the seed set.

Three entry strategies seed the heading identification:

- **both** (default): Merge seeds from heading-text matching AND
  vector search.  Covers vocabulary-aligned and semantic-gap queries.
- **heading_match**: Text-match query terms against heading labels.
  Avoids the "vector search prefers generic content" problem.
- **vector**: Vector search as seed, expand from hit metadata.
  Bridges vocabulary gaps (e.g. "safety" → security sections).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from dataknobs_data.sources.base import RetrievalIntent, SourceResult
from dataknobs_data.sources.topic_index import (
    HeadingMatchConfig,
    TopicNode,
    build_heading_tree,
    expand_region,
    find_heading_regions,
)
from dataknobs_llm.llm.base import LLMMessage

logger = logging.getLogger(__name__)


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
    ) -> list[SourceResult]:
        ...


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------


@dataclass(frozen=True)
class HeadingTreeConfig:
    """Configuration for :class:`HeadingTreeIndex`.

    All parameters have sensible defaults.  Config authors override
    per-source via YAML.  Scope profiles allow per-query overrides
    keyed by ``intent.scope``.

    Attributes:
        entry_strategy: How to find seed headings — ``"both"``,
            ``"heading_match"``, or ``"vector"``.
        seed_score_threshold: Drop vector seeds below this similarity.
        seed_max_results: Cap seed chunks before region identification.
        min_heading_depth: Exclude headings shallower than this level
            (structural stopword filter).
        expansion_mode: What descendants to include — ``"subtree"``,
            ``"children"``, or ``"leaves"``.
        max_expansion_depth: Levels below matched heading to traverse.
            ``None`` means unlimited.
        max_expanded_results: Final cap after all expansions.
        heading_selection: How to select among candidate headings —
            ``"score"`` (default, deterministic score-based ranking) or
            ``"llm"`` (LLM classification).  Score-based selection groups
            candidates by their ancestor at ``min_heading_depth``, ranks
            regions by max seed chunk relevance, and takes ``top_regions``.
        top_regions: Max heading regions to select (score-based mode).
            Analogous to ClusterTopicIndex's ``top_clusters``.
        resolution_prompt: Custom LLM prompt for heading selection.
            When ``None``, a built-in default is used if LLM is provided.
        max_headings_for_llm: Truncate heading list for LLM selection.
        heading_match: Configuration for heading-text matching.
        scope_profiles: Per-scope parameter overrides keyed by scope
            name (e.g. ``"focused"``, ``"broad"``).
        heading_selection_config: Optional LLM provider config dict for
            heading selection (e.g. ``{"provider": "ollama", "model":
            "phi4-mini"}``).  When set, a dedicated lightweight provider
            is used for LLM heading selection instead of the main model.
            Falls back to the main LLM when ``None``.
    """

    entry_strategy: str = "both"
    seed_score_threshold: float = 0.3
    seed_max_results: int = 10
    min_heading_depth: int = 1
    expansion_mode: str = "subtree"
    max_expansion_depth: int | None = None
    max_expanded_results: int = 50
    heading_selection: str = "score"
    top_regions: int = 3
    resolution_prompt: str | None = None
    max_headings_for_llm: int = 100
    heading_match: HeadingMatchConfig = field(default_factory=HeadingMatchConfig)
    scope_profiles: dict[str, dict[str, Any]] = field(default_factory=dict)
    heading_selection_config: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HeadingTreeConfig:
        """Build from a config dict, ignoring unknown keys."""
        import dataclasses

        known = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}

        # Handle nested HeadingMatchConfig
        hm_data = filtered.pop("heading_match", None)
        if isinstance(hm_data, dict):
            sw = hm_data.get("stopwords")
            if isinstance(sw, list):
                hm_data["stopwords"] = frozenset(sw)
            ep = hm_data.get("exclude_patterns")
            if isinstance(ep, list):
                hm_data["exclude_patterns"] = tuple(ep)
            filtered["heading_match"] = HeadingMatchConfig(**hm_data)

        return cls(**filtered)


# ------------------------------------------------------------------
# Effective parameters (resolved per-query)
# ------------------------------------------------------------------


@dataclass
class _EffectiveParams:
    """Resolved parameters for a single resolve() call.

    Built from: config defaults ← scope profile ← explicit overrides.
    """

    entry_strategy: str
    seed_score_threshold: float
    seed_max_results: int
    min_heading_depth: int
    expansion_mode: str
    max_expansion_depth: int | None
    max_expanded_results: int
    heading_selection: str
    top_regions: int


def _resolve_params(
    config: HeadingTreeConfig,
    intent: RetrievalIntent | None,
) -> _EffectiveParams:
    """Resolve effective parameters via the cascade.

    Priority (highest to lowest):
    1. Explicit overrides in ``intent.raw_data["topic_index"]``
    2. Scope profile matching ``intent.scope``
    3. Config defaults
    """
    params: dict[str, Any] = {
        "entry_strategy": config.entry_strategy,
        "seed_score_threshold": config.seed_score_threshold,
        "seed_max_results": config.seed_max_results,
        "min_heading_depth": config.min_heading_depth,
        "expansion_mode": config.expansion_mode,
        "max_expansion_depth": config.max_expansion_depth,
        "max_expanded_results": config.max_expanded_results,
        "heading_selection": config.heading_selection,
        "top_regions": config.top_regions,
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
# HeadingTreeIndex
# ------------------------------------------------------------------


class HeadingTreeIndex:
    """Topic index backed by a document's heading hierarchy.

    Uses heading metadata on chunks to identify and expand topic
    regions via **lazy query-driven reconstruction**.  No full heading
    tree is pre-built — each ``resolve()`` call builds a partial tree
    from query-driven seed results and expands within that seed set.

    Args:
        vector_query_fn: Async function for vector-based seeding.
            Accepts ``(query, top_k)`` and returns scored
            ``SourceResult`` instances.  Required for ``"vector"``
            and ``"both"`` entry strategies.
        source_name: Source name for logging.
        config: Full configuration.
        heading_selection_llm: Optional dedicated LLM provider for
            heading selection.  When set, ``_llm_select_headings``
            uses this instead of the main LLM passed to ``resolve()``.
    """

    def __init__(
        self,
        *,
        vector_query_fn: VectorQueryFn | None = None,
        source_name: str = "knowledge_base",
        config: HeadingTreeConfig | None = None,
        heading_selection_llm: Any | None = None,
    ) -> None:
        self._config = config or HeadingTreeConfig()
        self._source_name = source_name
        self._vector_query_fn = vector_query_fn
        self._heading_selection_llm = heading_selection_llm

        # Eager-mode state: populated by from_chunks(), None in lazy mode
        self._chunks_by_id: dict[str, SourceResult] | None = None
        self._tree: TopicNode | None = None

    @classmethod
    def from_chunks(
        cls,
        all_chunks: list[SourceResult],
        *,
        vector_query_fn: VectorQueryFn | None = None,
        source_name: str = "knowledge_base",
        config: HeadingTreeConfig | None = None,
        heading_selection_llm: Any | None = None,
    ) -> HeadingTreeIndex:
        """Eagerly build from a pre-loaded chunk set.

        Constructs the full heading tree upfront.  Useful for testing
        or when all chunks are already available.  ``resolve()`` still
        follows the same pipeline but uses the pre-built tree and
        chunk lookup instead of building them per-turn.
        """
        idx = cls(
            vector_query_fn=vector_query_fn,
            source_name=source_name,
            config=config,
            heading_selection_llm=heading_selection_llm,
        )
        idx._chunks_by_id = {c.source_id: c for c in all_chunks}
        idx._tree = build_heading_tree(all_chunks)
        return idx

    @classmethod
    def from_source_results(
        cls,
        results: list[SourceResult],
        *,
        vector_query_fn: VectorQueryFn | None = None,
        source_name: str = "knowledge_base",
        config: HeadingTreeConfig | None = None,
        heading_selection_llm: Any | None = None,
        min_heading_chunks: int = 3,
    ) -> HeadingTreeIndex | None:
        """Eagerly build from source results if heading metadata is present.

        Returns ``None`` if fewer than ``min_heading_chunks`` have
        heading metadata — the heading tree would be too sparse to
        be useful.
        """
        heading_count = sum(
            1 for r in results
            if r.metadata.get("headings")
        )
        if heading_count < min_heading_chunks:
            return None

        return cls.from_chunks(
            results,
            vector_query_fn=vector_query_fn,
            source_name=source_name,
            config=config,
            heading_selection_llm=heading_selection_llm,
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
        """Identify relevant heading regions, expand, collect chunks.

        Per-turn pipeline:

        1. Resolve effective parameters (config ← scope profile ← overrides).
        2. Fetch seed results via vector search (builds the chunk pool).
        3. Build a partial heading tree from the seeds' heading metadata.
        4. Find seed headings via heading-text matching against the
           partial tree and/or from the vector seed results directly.
        5. Filter by min_heading_depth.
        6. Optionally present to LLM for selection.
        7. Expand per expansion_mode / max_expansion_depth using the
           seed chunk pool.
        8. Cap at max_expanded_results.

        In eager mode (constructed via :meth:`from_chunks`), steps 2-3
        use the pre-built tree and chunk lookup instead.
        """
        params = _resolve_params(self._config, intent)

        logger.info(
            "HeadingTreeIndex resolving for source '%s': "
            "entry_strategy=%s, expansion_mode=%s",
            self._source_name, params.entry_strategy, params.expansion_mode,
        )

        # Pick up the filter slice keyed by our source name and forward
        # it through vector-seed fetching, matching the convention the
        # main VectorKnowledgeSource.query path uses. Empty/missing slice
        # → no filter (preserves historical behavior).
        filter_metadata: dict[str, Any] | None = None
        if intent is not None:
            filter_metadata = intent.filters.get(self._source_name) or None

        # Steps 2-3: Get chunk pool and heading tree for this turn
        chunks_by_id, tree = await self._get_tree_and_chunks(
            query, params, filter_metadata=filter_metadata,
        )

        if tree is None:
            logger.info(
                "HeadingTreeIndex: no heading tree for source '%s' "
                "(no seeds or no heading metadata)",
                self._source_name,
            )
            return []

        # Step 4: Find seed headings
        seed_nodes = self._find_seed_headings(
            query, params, tree, chunks_by_id,
        )

        if not seed_nodes:
            logger.info(
                "HeadingTreeIndex: no heading regions matched for source '%s'",
                self._source_name,
            )
            return []

        # Step 5: Filter by min_heading_depth
        seed_nodes = [
            n for n in seed_nodes
            if n.level >= params.min_heading_depth
        ]
        if not seed_nodes:
            return []

        matched_labels = [n.label for n in seed_nodes]
        logger.info(
            "HeadingTreeIndex: %d heading regions matched for source '%s': %s",
            len(seed_nodes), self._source_name, matched_labels,
        )

        # Step 6: Region selection — narrow candidates to top regions
        if len(seed_nodes) > 1:
            if params.heading_selection == "llm":
                # LLM-based selection (optional, explicit opt-in)
                selection_llm = self._heading_selection_llm or llm
                if selection_llm is not None:
                    seed_nodes = await self._llm_select_headings(
                        query, seed_nodes, selection_llm,
                    )
                else:
                    # No LLM available — fall through to score-based
                    seed_nodes = self._score_based_region_selection(
                        seed_nodes, chunks_by_id, params,
                    )
            else:
                # Score-based selection (default, deterministic)
                seed_nodes = self._score_based_region_selection(
                    seed_nodes, chunks_by_id, params,
                )

        # Step 7: Expand matched regions and collect chunks
        all_results: list[SourceResult] = []
        seen_ids: set[str] = set()

        for node in seed_nodes:
            expanded = expand_region(
                node,
                chunks_by_id,
                expansion_mode=params.expansion_mode,
                max_expansion_depth=params.max_expansion_depth,
            )
            for result in expanded:
                if result.source_id not in seen_ids:
                    seen_ids.add(result.source_id)
                    all_results.append(result)

        # Step 8: Cap results
        if len(all_results) > params.max_expanded_results:
            all_results = all_results[:params.max_expanded_results]

        logger.info(
            "HeadingTreeIndex: %d heading regions -> %d expanded chunks "
            "for source '%s'",
            len(seed_nodes), len(all_results), self._source_name,
        )

        return all_results

    def topics(self) -> list[str]:
        """Return unique heading labels from the pre-built tree.

        Only available in eager mode (constructed via :meth:`from_chunks`
        or :meth:`from_source_results`).  Returns ``[]`` in lazy mode.
        """
        if self._tree is None:
            return []
        labels: list[str] = []
        seen: set[str] = set()
        for node in self._tree.flatten():
            if node.label != "__root__" and node.label not in seen:
                seen.add(node.label)
                labels.append(node.label)
        return labels

    # ------------------------------------------------------------------
    # Private: per-turn tree and chunk pool
    # ------------------------------------------------------------------

    async def _get_tree_and_chunks(
        self,
        query: str,
        params: _EffectiveParams,
        filter_metadata: dict[str, Any] | None = None,
    ) -> tuple[dict[str, SourceResult], TopicNode | None]:
        """Get the heading tree and chunk pool for this turn.

        In eager mode, returns the pre-built tree and chunks.
        In lazy mode, fetches seeds via vector search and builds
        a partial tree from their heading metadata.
        """
        if self._tree is not None and self._chunks_by_id is not None:
            # Eager mode: use pre-built state
            return self._chunks_by_id, self._tree

        # Lazy mode: fetch seeds and build per-turn
        seed_results = await self._fetch_vector_seeds(
            query, params, filter_metadata=filter_metadata,
        )
        if not seed_results:
            return {}, None

        chunks_by_id = {r.source_id: r for r in seed_results}
        tree = build_heading_tree(seed_results)

        # Check if the tree has any real headings
        if not tree.children:
            return chunks_by_id, None

        return chunks_by_id, tree

    async def _fetch_vector_seeds(
        self,
        query: str,
        params: _EffectiveParams,
        *,
        filter_metadata: dict[str, Any] | None = None,
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
                query,
                params.seed_max_results,
                filter_metadata=filter_metadata,
            )
        except Exception:
            logger.warning(
                "Vector query failed for source '%s'",
                self._source_name, exc_info=True,
            )
            return []

        # Filter by score threshold
        return [
            r for r in results
            if r.relevance >= params.seed_score_threshold
        ]

    # ------------------------------------------------------------------
    # Private: seeding strategies
    # ------------------------------------------------------------------

    def _find_seed_headings(
        self,
        query: str,
        params: _EffectiveParams,
        tree: TopicNode,
        chunks_by_id: dict[str, SourceResult],
    ) -> list[TopicNode]:
        """Find seed heading nodes via the configured entry strategy."""
        strategy = params.entry_strategy

        if strategy == "heading_match":
            return self._heading_match_seeds(query, params, tree)

        if strategy == "vector":
            # In lazy mode, the tree was built from vector seeds —
            # all non-root nodes are vector-seeded headings.
            return self._vector_seed_nodes(tree, params)

        # "both" (default): merge heading-match and vector-seeded nodes
        heading_seeds = self._heading_match_seeds(query, params, tree)
        vector_seeds = self._vector_seed_nodes(tree, params)

        # Merge by node identity (label + level)
        seen = {(n.label, n.level) for n in heading_seeds}
        merged = list(heading_seeds)
        for node in vector_seeds:
            key = (node.label, node.level)
            if key not in seen:
                seen.add(key)
                merged.append(node)

        return merged

    def _heading_match_seeds(
        self,
        query: str,
        params: _EffectiveParams,
        tree: TopicNode,
    ) -> list[TopicNode]:
        """Find heading nodes matching query terms."""
        match_config = HeadingMatchConfig(
            stopwords=self._config.heading_match.stopwords,
            min_word_length=self._config.heading_match.min_word_length,
            min_heading_depth=params.min_heading_depth,
            exclude_patterns=self._config.heading_match.exclude_patterns,
        )
        return find_heading_regions(query, tree, config=match_config)

    def _vector_seed_nodes(
        self,
        tree: TopicNode,
        params: _EffectiveParams,
    ) -> list[TopicNode]:
        """Return non-root heading nodes from the tree.

        In lazy mode, the tree was built from vector search results —
        every node represents a heading found in at least one seed chunk.
        We return the deepest nodes (most specific headings) that have
        chunk content.
        """
        nodes: list[TopicNode] = []
        seen: set[tuple[int, str]] = set()
        for node in tree.flatten():
            if node.label == "__root__":
                continue
            key = (node.level, node.label)
            if key not in seen:
                seen.add(key)
                nodes.append(node)

        # Sort deepest first (most specific matches preferred)
        nodes.sort(key=lambda n: n.level, reverse=True)
        return nodes

    # ------------------------------------------------------------------
    # Private: score-based region selection
    # ------------------------------------------------------------------

    def _score_based_region_selection(
        self,
        candidates: list[TopicNode],
        chunks_by_id: dict[str, SourceResult],
        params: _EffectiveParams,
    ) -> list[TopicNode]:
        """Select top heading regions by seed chunk relevance score.

        Scores each candidate node by the max relevance of seed chunks
        it owns (directly or via descendants), ranks by score, and
        returns the top ``top_regions`` nodes.

        For nodes at ``min_heading_depth`` this naturally groups by
        top-level section.  For deeper nodes, each is scored
        independently — the expansion step handles subtree traversal.

        This is the default selection strategy — deterministic,
        zero-LLM-cost, and consistent with the principle of using
        deterministic algorithms for search ranking.
        """
        # Score each candidate by max relevance of chunks it contains
        scored: list[tuple[TopicNode, float]] = []
        for node in candidates:
            max_score = 0.0
            for chunk_id in node.descendant_chunk_ids():
                chunk = chunks_by_id.get(chunk_id)
                if chunk is not None:
                    max_score = max(max_score, chunk.relevance)
            # Also check direct chunk_ids (descendant_chunk_ids includes them)
            scored.append((node, max_score))

        # Rank by score descending, take top N
        scored.sort(key=lambda x: x[1], reverse=True)
        selected = scored[:params.top_regions]

        logger.info(
            "HeadingTreeIndex: score-based selection for source '%s': "
            "%d candidates -> top %d: %s",
            self._source_name, len(scored), len(selected),
            [(n.label, f"{s:.3f}") for n, s in selected],
        )

        return [node for node, _ in selected]

    # ------------------------------------------------------------------
    # Private: LLM heading selection
    # ------------------------------------------------------------------

    async def _llm_select_headings(
        self,
        query: str,
        candidates: list[TopicNode],
        llm: Any,
    ) -> list[TopicNode]:
        """Use LLM to select the most relevant headings from candidates.

        Falls back to returning all candidates if the LLM call fails
        or returns unparseable output.
        """
        max_headings = self._config.max_headings_for_llm
        truncated = candidates[:max_headings]

        # Use letter-based IDs to avoid collision with section numbers
        # in heading labels (e.g. "10. Security Considerations").
        heading_list = "\n".join(
            f"  ({_index_to_letter(i)}) {n.label}"
            for i, n in enumerate(truncated)
        )

        prompt = self._config.resolution_prompt or (
            "Given these document sections, select the ones most relevant "
            "to the user's question. Return ONLY the letter IDs of "
            "selected sections, one per line (e.g. (A), (C), (F))."
        )

        system_msg = f"{prompt}\n\nSections:\n{heading_list}"
        user_msg = query

        try:
            response = await llm.complete(
                messages=[
                    LLMMessage(role="system", content=system_msg),
                    LLMMessage(role="user", content=user_msg),
                ],
            )
            content = response.content if hasattr(response, "content") else str(response)
            selected_indices = _parse_letter_indices(content, len(truncated))
            if selected_indices:
                return [truncated[i] for i in selected_indices]
        except Exception:
            logger.warning(
                "LLM heading selection failed for source '%s', "
                "using all candidates",
                self._source_name, exc_info=True,
            )

        return candidates


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _index_to_letter(i: int) -> str:
    """Convert 0-based index to letter ID: 0→A, 1→B, …, 25→Z, 26→AA."""
    if i < 26:
        return chr(ord("A") + i)
    return chr(ord("A") + i // 26 - 1) + chr(ord("A") + i % 26)


def _parse_letter_indices(text: str, max_index: int) -> list[int]:
    """Parse ``(A), (C), (F)`` style letter IDs from LLM output."""
    import re

    indices: list[int] = []
    dropped: list[str] = []
    for match in re.finditer(r"\(([A-Z]{1,2})\)", text):
        letter = match.group(1)
        idx = _letter_to_index(letter)
        if 0 <= idx < max_index:
            if idx not in indices:
                indices.append(idx)
        else:
            dropped.append(letter)
    if dropped:
        logger.warning(
            "LLM heading selection: dropped out-of-range IDs %s "
            "(max valid: %s)",
            dropped, _index_to_letter(max_index - 1),
        )
    return indices


def _letter_to_index(letter: str) -> int:
    """Convert letter ID back to 0-based index: A→0, B→1, AA→26."""
    if len(letter) == 1:
        return ord(letter) - ord("A")
    return (ord(letter[0]) - ord("A") + 1) * 26 + ord(letter[1]) - ord("A")


def _parse_bracket_indices(text: str, max_index: int) -> list[int]:
    """Parse ``[0], [2], [5]`` style indices from LLM output.

    Kept for backward compatibility with custom resolution prompts
    that use bracket-number format.
    """
    import re

    indices: list[int] = []
    dropped: list[int] = []
    for match in re.finditer(r"\[(\d+)\]", text):
        idx = int(match.group(1))
        if 0 <= idx < max_index:
            if idx not in indices:
                indices.append(idx)
        else:
            dropped.append(idx)
    if dropped:
        logger.warning(
            "LLM heading selection: dropped out-of-range indices %s "
            "(max valid: %d)",
            dropped, max_index - 1,
        )
    return indices
