# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

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

**Walking a topic tree.**  :class:`TopicNode`'s four walk methods —
:meth:`~TopicNode.flatten`, :meth:`~TopicNode.leaves`,
:meth:`~TopicNode.children_at_depth` and
:meth:`~TopicNode.descendants_to_depth` — are each one call into the generic
hierarchy walks in ``dataknobs-common``, over :class:`TopicNodeHierarchy`.  They
return what they always returned, in the order they always returned it; what
sharing buys is that a malformed tree terminates rather than recursing until the
stack is gone, and that the walks the family has and this class never had —
``ancestors``, ``paths_to_root``, ``deepest_common_ancestor`` — are available
over a topic tree by constructing the same adapter the methods construct.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol, runtime_checkable

from dataknobs_common import children_at_depth, descendants_to_depth, flatten, leaves

from .base import RetrievalIntent, SourceResult

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Defaults
# ------------------------------------------------------------------

DEFAULT_HEADING_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
        "shall",
        "of",
        "in",
        "to",
        "for",
        "with",
        "on",
        "at",
        "by",
        "from",
        "as",
        "into",
        "about",
        "between",
        "through",
        "during",
        "after",
        "before",
        "above",
        "below",
        "up",
        "down",
        "out",
        "off",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "and",
        "but",
        "or",
        "nor",
        "not",
        "so",
        "yet",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "only",
        "own",
        "same",
        "than",
        "too",
        "very",
        "just",
        "if",
        "when",
        "where",
        "how",
        "what",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "these",
        "those",
        "i",
        "me",
        "my",
        "we",
        "our",
        "you",
        "your",
        "he",
        "him",
        "his",
        "she",
        "her",
        "it",
        "its",
        "they",
        "them",
        "their",
        "all",
        "any",
        "every",
        "tell",
        "show",
        "give",
        "get",
        "list",
        "describe",
        "explain",
        "find",
        "look",
        "want",
        "need",
    }
)
"""Default stopwords filtered from queries during heading matching."""

DEFAULT_MIN_WORD_LENGTH: int = 2
"""Default minimum word length for heading matching (inclusive)."""

DEFAULT_HEADING_EXCLUDE_PATTERNS: tuple[str, ...] = (
    r"(?i)^references$",
    r"(?i)^informative\s+references$",
    r"(?i)^normative\s+references$",
    r"(?i)^appendix\b",
    r"(?i)^acknowledgements?$",
    r"(?i)^table\s+of\s+contents$",
    r"(?i)^bibliography$",
    r"(?i)^index$",
    r"(?i)^glossary$",
    r"(?i)^abstract$",
)
"""Default regex patterns for structural headings to exclude from matching.

These headings are navigational or organizational — they don't contain
topical content relevant to user queries.  Patterns are matched against
the heading label (after stripping leading section numbers like "12.2.").
"""


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
        axis = TopicNodeHierarchy(self)
        return axis.nodes(flatten(axis, from_id=axis.ANCHOR))

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
        axis = TopicNodeHierarchy(self)
        return axis.nodes(children_at_depth(axis, axis.ANCHOR, depth))

    def leaves(self) -> list[TopicNode]:
        """Return leaf nodes (no children) under this node."""
        axis = TopicNodeHierarchy(self)
        return axis.nodes(leaves(axis, under=axis.ANCHOR))

    def descendants_to_depth(self, max_depth: int) -> list[TopicNode]:
        """Return all descendants up to ``max_depth`` levels below.

        ``max_depth=0`` returns only self.  ``max_depth=1`` returns
        self + immediate children.
        """
        axis = TopicNodeHierarchy(self)
        return axis.nodes(descendants_to_depth(axis, axis.ANCHOR, max_depth))


#: A node's position in the tree, and the key :class:`TopicNodeHierarchy` uses.
#:
#: ``()`` is the anchor, ``(0,)`` its first child, ``(0, 1)`` the second child
#: of that.  The integers index ``TopicNode.children`` directly, so a key reads
#: as the route taken to reach the node.
TopicKey = tuple[int, ...]


@dataclass(frozen=True, eq=False)
class TopicNodeHierarchy:
    """A :class:`~dataknobs_common.hierarchy.Hierarchy` over a topic subtree.

    :class:`TopicNode` has no id.  Its fields are ``label``, ``level``,
    ``children``, ``chunk_ids`` and ``metadata``; ``label`` is not unique,
    because a document repeats its headings, and the class is a plain dataclass
    so two distinct nodes with the same fields compare equal.  There is
    therefore nothing on the node to key an axis by, and this adapter mints the
    key instead: **a node's position**, as a tuple of child indices from the
    anchor.

    A positional key is a value, it is hashable, and it fails legibly -- an
    unknown anchor is refused with ``no node (0, 1)`` rather than with an
    address nobody can look up.

    **Build one per call and discard it.**  The map is a snapshot, and
    :func:`build_heading_tree` grows a tree by appending to ``children``, so a
    key minted before an append can name a different node after one.  The four
    :class:`TopicNode` methods that use this build one, walk, project and drop
    it, which is why staleness cannot arise there.  A consumer holding one for
    longer owns that question, exactly as they would holding any index built
    from a mutable structure.

    What it buys a consumer is the rest of the family: ``paths_to_root``,
    ``ancestors`` and ``deepest_common_ancestor`` over a topic tree, none of
    which :class:`TopicNode` has.

    .. code-block:: python

        from dataknobs_common import ancestors, paths_to_root
        from dataknobs_data.sources import TopicNodeHierarchy

        axis = TopicNodeHierarchy(tree)
        deepest = axis.key_of(some_node)
        [n.label for n in axis.nodes(ancestors(axis, deepest))]

    **The axis is a tree projection of the node graph**, which is what makes
    the walks terminate.  Construction visits each node once, by identity, and
    a node reachable by two routes is placed at the first one found; the edge
    that would have reached it again is not in the axis.  So a ``children``
    list that forms a cycle is walked to its end instead of recursing until the
    stack is gone, and a subtree hung under two parents is walked once.
    Identity is used here and nowhere else, privately, for the one question it
    answers well: *have I already seen this object*.
    """

    #: The node the axis is rooted at.  It is the axis's only root.
    anchor: TopicNode

    #: Every node the anchor reaches, at the first key that reached it.
    nodes_by_key: Mapping[TopicKey, TopicNode] = field(init=False, repr=False)

    #: A key's children, in ``TopicNode.children`` order, skipping any edge
    #: that led to a node already placed.
    child_keys: Mapping[TopicKey, tuple[TopicKey, ...]] = field(init=False, repr=False)

    #: The anchor's key.  Named rather than spelled ``()`` at four call sites,
    #: because an empty tuple in an argument list reads as an oversight.
    ANCHOR: ClassVar[TopicKey] = ()

    def __post_init__(self) -> None:
        nodes: dict[TopicKey, TopicNode] = {self.ANCHOR: self.anchor}
        edges: dict[TopicKey, tuple[TopicKey, ...]] = {}
        seen: set[int] = {id(self.anchor)}
        stack: list[tuple[TopicKey, TopicNode]] = [(self.ANCHOR, self.anchor)]
        while stack:
            key, node = stack.pop()
            found: list[tuple[TopicKey, TopicNode]] = []
            for index, child in enumerate(node.children):
                if id(child) in seen:
                    continue
                seen.add(id(child))
                child_key = (*key, index)
                nodes[child_key] = child
                found.append((child_key, child))
            edges[key] = tuple(child_key for child_key, _ in found)
            stack.extend(reversed(found))
        object.__setattr__(self, "nodes_by_key", nodes)
        object.__setattr__(self, "child_keys", edges)

    # -- the protocol -------------------------------------------------

    def roots(self) -> Sequence[TopicKey]:
        """The anchor, and only ever the anchor."""
        return (self.ANCHOR,)

    def parents(self, node_id: TopicKey) -> Sequence[TopicKey]:
        """The key one step up, which a positional key already carries."""
        if not node_id or node_id not in self.nodes_by_key:
            return ()
        return (node_id[:-1],)

    def children(self, node_id: TopicKey) -> Sequence[TopicKey]:
        """The keys one step down, in ``children`` order."""
        return self.child_keys.get(node_id, ())

    def contains(self, node_id: TopicKey) -> bool:
        """Whether the anchor reaches that position at all."""
        return node_id in self.nodes_by_key

    # -- the projection -----------------------------------------------

    def node(self, node_id: TopicKey) -> TopicNode:
        """The node at ``node_id``.

        Raises:
            KeyError: if the axis does not reach that position.
        """
        return self.nodes_by_key[node_id]

    def nodes(self, node_ids: Sequence[TopicKey]) -> list[TopicNode]:
        """A walk's keys as the nodes they name, in the order given.

        Every walk over this axis answers in keys, and every caller here wants
        nodes.  One projection serves all of them, which is what keeps each
        delegating method to a single line.
        """
        return [self.nodes_by_key[node_id] for node_id in node_ids]

    def key_of(self, node: TopicNode) -> TopicKey:
        """Where ``node`` sits, by identity rather than by equality.

        Two distinct :class:`TopicNode` objects with the same fields compare
        equal, so a search by ``==`` would return whichever came first.  This
        answers about the object handed in.

        Raises:
            KeyError: if that object is not in this axis.
        """
        for key, candidate in self.nodes_by_key.items():
            if candidate is node:
                return key
        raise KeyError("that node is not in this hierarchy")


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
        exclude_patterns: Regex patterns for structural headings to exclude.
            Matched against the heading label after stripping leading
            section numbers (e.g. "12.2. Informative References" is tested
            as "Informative References").  Set to ``()`` to disable.
    """

    stopwords: frozenset[str] = DEFAULT_HEADING_STOPWORDS
    min_word_length: int = DEFAULT_MIN_WORD_LENGTH
    min_heading_depth: int = 1
    exclude_patterns: tuple[str, ...] = DEFAULT_HEADING_EXCLUDE_PATTERNS


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
            Content chunks from matched topic regions.  An empty list
            means the index ran and matched nothing --- a vocabulary
            gap, which callers may treat as a reason to try another
            retrieval route.

        Raises:
            StrategyUnavailable: (from
                :mod:`dataknobs_data.sources.base`) The index cannot run
                at all, because something it needs was never supplied
                (an embedder, a way to fetch seeds).  This is
                deliberately *not* an empty list: a caller that falls
                back on empty would otherwise take the same branch for
                a wiring fault as for a vocabulary gap, on every turn,
                and report the wrong cause.  Implementations must not
                absorb this condition.
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
                chunk.source_id,
                len(headings),
                len(levels),
            )
            root.chunk_ids.append(chunk.source_id)
            continue

        # Build/find each node along the heading path
        path: list[tuple[int, str]] = []
        parent = root
        for heading, level in zip(headings, levels, strict=True):
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
    one query term, filtered by minimum heading depth and exclusion
    patterns.

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

    # Pre-compile exclusion patterns
    compiled_excludes = [re.compile(p) for p in cfg.exclude_patterns]

    matches: list[TopicNode] = []
    all_nodes = tree.flatten()

    for node in all_nodes:
        if node.level < cfg.min_heading_depth:
            continue
        # Check exclusion patterns against label stripped of section numbers
        stripped_label = _strip_section_number(node.label)
        if _is_excluded(stripped_label, compiled_excludes):
            continue
        label_lower = node.label.lower()
        for word in query_words:
            if re.search(rf"\b{re.escape(word)}\b", label_lower):
                matches.append(node)
                break

    # Sort deepest first — more specific matches preferred
    matches.sort(key=lambda n: n.level, reverse=True)
    return matches


# Section number pattern: "10.", "10.2.", "10.2.1.", "A.", "C.1." etc.
_SECTION_NUMBER_RE = re.compile(r"^[A-Z0-9]+(?:\.[A-Z0-9]+)*\.?\s+")


def _strip_section_number(label: str) -> str:
    """Strip leading section number from a heading label.

    "10.2. Informative References" → "Informative References"
    "Appendix C. Acknowledgements" → "Acknowledgements"
    "Security Considerations" → "Security Considerations" (no change)
    """
    # Handle "Appendix X." prefix specially
    appendix_match = re.match(r"(?i)^appendix\s+", label)
    if appendix_match:
        return label  # Keep as-is; the exclude pattern matches "^appendix"
    return _SECTION_NUMBER_RE.sub("", label)


def _is_excluded(label: str, patterns: list[re.Pattern[str]]) -> bool:
    """Check if a label matches any exclusion pattern."""
    return any(p.search(label) for p in patterns)


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
    return [w for w in words if w not in stopwords and len(w) >= min_word_length]


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
            return [n for n in bounded if not any(id(c) in bounded_ids for c in n.children)]
        return node.leaves()

    # subtree (default)
    if max_expansion_depth is not None:
        return node.descendants_to_depth(max_expansion_depth)
    return node.flatten()
