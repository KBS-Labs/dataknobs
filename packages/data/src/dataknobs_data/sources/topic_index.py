"""Topic-index abstractions for structured content retrieval.

A topic index maps user queries to content regions — heading subtrees,
embedding clusters, or any other structural grouping — and returns
chunks from matched regions.  This replaces the pattern of "generate
queries → vector similarity → hope for the best" with deterministic,
structure-aware retrieval.

The :class:`TopicIndex` protocol defines the contract.  Implementations
live in the package best suited to their dependencies:

- :class:`HeadingTreeIndex` (``dataknobs-bots``) — needs LLM for
  optional heading selection.
- :class:`ClusterTopicIndex` (``dataknobs-data``) — purely
  deterministic centroid matching.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .base import RetrievalIntent, SourceResult

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Defaults
# ------------------------------------------------------------------

DEFAULT_HEADING_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "can", "shall",
    "of", "in", "to", "for", "with", "on", "at", "by", "from",
    "as", "into", "about", "between", "through", "during", "after",
    "before", "above", "below", "up", "down", "out", "off", "over",
    "under", "again", "further", "then", "once", "and", "but", "or",
    "nor", "not", "so", "yet", "both", "each", "few", "more",
    "most", "other", "some", "such", "no", "only", "own", "same",
    "than", "too", "very", "just", "if", "when", "where", "how",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "i", "me", "my", "we", "our", "you", "your", "he", "him",
    "his", "she", "her", "it", "its", "they", "them", "their",
    "all", "any", "every", "tell", "show", "give", "get", "list",
    "describe", "explain", "find", "look", "want", "need",
})
"""Default stopwords filtered from queries during heading matching."""

DEFAULT_MIN_WORD_LENGTH: int = 2
"""Default minimum word length for heading matching (inclusive)."""


# ------------------------------------------------------------------
# Data types
# ------------------------------------------------------------------


@dataclass
class TopicNode:
    """A node in a topic hierarchy (heading tree or cluster tree).

    Attributes:
        label: Human-readable label (heading text, cluster name).
        level: Depth in the hierarchy (0 = root/title, 1 = top sections).
        children: Immediate child nodes.
        chunk_ids: IDs of chunks directly under this heading (not
            descendants).
        metadata: Arbitrary metadata (source file, heading number, etc.).
    """

    label: str
    level: int = 0
    children: list[TopicNode] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def flatten(self) -> list[TopicNode]:
        """Return self + all descendants in pre-order."""
        nodes = [self]
        for child in self.children:
            nodes.extend(child.flatten())
        return nodes

    def descendant_chunk_ids(self) -> list[str]:
        """All chunk IDs under this node (self + descendants)."""
        ids = list(self.chunk_ids)
        for child in self.children:
            ids.extend(child.descendant_chunk_ids())
        return ids

    def children_at_depth(self, depth: int) -> list[TopicNode]:
        """Return descendants exactly ``depth`` levels below this node.

        ``depth=1`` returns immediate children; ``depth=2`` returns
        grandchildren; etc.
        """
        if depth <= 0:
            return [self]
        result: list[TopicNode] = []
        for child in self.children:
            result.extend(child.children_at_depth(depth - 1))
        return result

    def leaves(self) -> list[TopicNode]:
        """Return leaf nodes (no children) under this node."""
        if not self.children:
            return [self]
        result: list[TopicNode] = []
        for child in self.children:
            result.extend(child.leaves())
        return result

    def descendants_to_depth(self, max_depth: int) -> list[TopicNode]:
        """Return all descendants up to ``max_depth`` levels below.

        ``max_depth=0`` returns only self.  ``max_depth=1`` returns
        self + immediate children.
        """
        nodes = [self]
        if max_depth <= 0:
            return nodes
        for child in self.children:
            nodes.extend(child.descendants_to_depth(max_depth - 1))
        return nodes


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------


@dataclass(frozen=True)
class HeadingMatchConfig:
    """Configuration for heading-text matching in :func:`find_heading_regions`.

    Attributes:
        stopwords: Words to filter from the query before matching.
        min_word_length: Minimum word length to keep (inclusive).
        min_heading_depth: Exclude headings shallower than this level.
            Depth 0 is the document title; depth 1 is top-level sections.
    """

    stopwords: frozenset[str] = DEFAULT_HEADING_STOPWORDS
    min_word_length: int = DEFAULT_MIN_WORD_LENGTH
    min_heading_depth: int = 1


# ------------------------------------------------------------------
# Protocol
# ------------------------------------------------------------------


@runtime_checkable
class TopicIndex(Protocol):
    """Topic-based content retrieval abstraction.

    Encapsulates topic resolution (identify relevant regions) and
    content collection (retrieve chunks from those regions) as a
    single operation.  Implementations own the topic structure and
    know how to resolve queries against it.
    """

    async def resolve(
        self,
        query: str,
        *,
        context: str = "",
        llm: Any | None = None,
        top_k: int = 10,
        intent: RetrievalIntent | None = None,
    ) -> list[SourceResult]:
        """Resolve a query to content via the topic index.

        Args:
            query: User message or search query.
            context: Optional conversation context for disambiguation.
            llm: LLM provider for strategies needing classification.
                ``None`` for purely deterministic strategies.
            top_k: Maximum results to return.
            intent: Resolved retrieval intent.  When present,
                ``intent.scope`` drives scope profile selection and
                ``intent.raw_data["topic_index"]`` provides explicit
                parameter overrides.

        Returns:
            Content chunks from matched topic regions.
        """
        ...

    def topics(self) -> list[str]:
        """Return available topic labels (headings, cluster names)."""
        ...


# ------------------------------------------------------------------
# Heading tree utilities
# ------------------------------------------------------------------


def build_heading_tree(
    chunks: list[SourceResult],
) -> TopicNode:
    """Reconstruct a heading tree from chunk heading metadata.

    Each chunk is expected to carry ``headings: list[str]`` and
    ``heading_levels: list[int]`` in its ``metadata``.  The deepest
    heading paths contain their ancestors, so heading redundancy across
    chunks is an asset — any chunk reveals its full lineage.

    Returns a synthetic root node (level -1) whose children are the
    top-level headings found across all chunks.
    """
    root = TopicNode(label="__root__", level=-1)
    # Map from (level, label) path tuples to nodes for dedup
    node_map: dict[tuple[tuple[int, str], ...], TopicNode] = {}

    for chunk in chunks:
        headings = chunk.metadata.get("headings", [])
        levels = chunk.metadata.get("heading_levels", [])

        if not headings or not levels:
            # Chunk has no heading metadata — attach to root
            root.chunk_ids.append(chunk.source_id)
            continue

        if len(headings) != len(levels):
            logger.warning(
                "Chunk %s has mismatched headings/levels lengths (%d vs %d), skipping",
                chunk.source_id, len(headings), len(levels),
            )
            root.chunk_ids.append(chunk.source_id)
            continue

        # Build/find each node along the heading path
        path: list[tuple[int, str]] = []
        parent = root
        for heading, level in zip(headings, levels):
            path.append((level, heading))
            path_key = tuple(path)

            if path_key not in node_map:
                node = TopicNode(label=heading, level=level)
                node_map[path_key] = node
                parent.children.append(node)
            parent = node_map[path_key]

        # Attach chunk to the deepest heading in its path
        parent.chunk_ids.append(chunk.source_id)

    return root


def find_heading_regions(
    query: str,
    tree: TopicNode,
    *,
    config: HeadingMatchConfig | None = None,
) -> list[TopicNode]:
    """Text-match query terms against heading labels in the tree.

    Performs case-insensitive word-boundary matching of query terms
    against heading labels.  Returns nodes whose labels match at least
    one query term, filtered by minimum heading depth.

    Args:
        query: User query string.
        tree: Root of the heading tree (from :func:`build_heading_tree`).
        config: Matching configuration.  When ``None``, uses defaults.

    Returns:
        Matching nodes sorted by depth (deepest first — more specific
        matches are preferred).
    """
    cfg = config or HeadingMatchConfig()
    query_words = extract_query_words(
        query,
        stopwords=cfg.stopwords,
        min_word_length=cfg.min_word_length,
    )
    if not query_words:
        return []

    matches: list[TopicNode] = []
    all_nodes = tree.flatten()

    for node in all_nodes:
        if node.level < cfg.min_heading_depth:
            continue
        label_lower = node.label.lower()
        for word in query_words:
            if re.search(rf"\b{re.escape(word)}\b", label_lower):
                matches.append(node)
                break

    # Sort deepest first — more specific matches preferred
    matches.sort(key=lambda n: n.level, reverse=True)
    return matches


def expand_region(
    node: TopicNode,
    chunks_by_id: dict[str, SourceResult],
    *,
    expansion_mode: str = "subtree",
    max_expansion_depth: int | None = None,
) -> list[SourceResult]:
    """Collect chunks from a heading region per expansion settings.

    Args:
        node: The matched heading node to expand.
        chunks_by_id: Lookup from chunk source_id to SourceResult.
        expansion_mode: What descendants to include:
            - ``"subtree"``: All descendants at every level.
            - ``"children"``: Only immediate children of the matched heading.
            - ``"leaves"``: Only the deepest nodes (no children).
        max_expansion_depth: How many levels below the matched heading
            to traverse.  ``None`` means unlimited.

    Returns:
        Chunks from the expanded region, deduplicated by source_id.
    """
    target_nodes = _select_expansion_nodes(
        node,
        expansion_mode=expansion_mode,
        max_expansion_depth=max_expansion_depth,
    )

    # Collect unique chunks
    seen: set[str] = set()
    results: list[SourceResult] = []
    for target in target_nodes:
        for cid in target.chunk_ids:
            if cid in seen:
                continue
            seen.add(cid)
            chunk = chunks_by_id.get(cid)
            if chunk is not None:
                results.append(chunk)

    return results


def extract_query_words(
    query: str,
    *,
    stopwords: frozenset[str] = DEFAULT_HEADING_STOPWORDS,
    min_word_length: int = DEFAULT_MIN_WORD_LENGTH,
) -> list[str]:
    """Extract significant words from a query for heading matching.

    Args:
        query: Raw query string.
        stopwords: Words to filter out.
        min_word_length: Minimum word length to keep (inclusive).

    Returns:
        Lowercased significant words.
    """
    words = re.findall(r"[a-z0-9]+", query.lower())
    return [
        w for w in words
        if w not in stopwords and len(w) >= min_word_length
    ]


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------


def _select_expansion_nodes(
    node: TopicNode,
    *,
    expansion_mode: str,
    max_expansion_depth: int | None,
) -> list[TopicNode]:
    """Determine which nodes to collect chunks from based on expansion settings."""
    if expansion_mode == "children":
        return [node] + list(node.children)

    if expansion_mode == "leaves":
        if max_expansion_depth is not None:
            # Find leaves within the depth-bounded subtree
            bounded = node.descendants_to_depth(max_expansion_depth)
            bounded_ids = {id(n) for n in bounded}
            return [
                n for n in bounded
                if not any(id(c) in bounded_ids for c in n.children)
            ]
        return node.leaves()

    # subtree (default)
    if max_expansion_depth is not None:
        return node.descendants_to_depth(max_expansion_depth)
    return node.flatten()
