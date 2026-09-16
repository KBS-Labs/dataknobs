"""Tests for topic-index abstractions and heading tree utilities."""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_data.sources.base import SourceResult
from dataknobs_common.exceptions import NotFoundError
from dataknobs_common import (
    Hierarchy,
    ancestors,
    deepest_common_ancestor,
    flatten,
    paths_to_root,
)
from dataknobs_data.sources.topic_index import (
    DEFAULT_HEADING_STOPWORDS,
    HeadingMatchConfig,
    TopicNode,
    TopicNodeHierarchy,
    build_heading_tree,
    expand_region,
    extract_query_words,
    find_heading_regions,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _chunk(
    source_id: str,
    headings: list[str] | None = None,
    heading_levels: list[int] | None = None,
    content: str = "",
) -> SourceResult:
    """Create a SourceResult with heading metadata."""
    metadata: dict[str, Any] = {}
    if headings is not None:
        metadata["headings"] = headings
    if heading_levels is not None:
        metadata["heading_levels"] = heading_levels
    return SourceResult(
        content=content or f"Content for {source_id}",
        source_id=source_id,
        source_name="test",
        source_type="vector_kb",
        metadata=metadata,
    )


def _chunks_by_id(chunks: list[SourceResult]) -> dict[str, SourceResult]:
    return {c.source_id: c for c in chunks}


# ------------------------------------------------------------------
# TopicNode
# ------------------------------------------------------------------


class TestTopicNode:
    """Tests for TopicNode data structure operations."""

    def test_flatten_single_node(self) -> None:
        node = TopicNode(label="root", level=0)
        assert len(node.flatten()) == 1

    def test_flatten_tree(self) -> None:
        child1 = TopicNode(label="c1", level=1)
        child2 = TopicNode(label="c2", level=1)
        grandchild = TopicNode(label="gc1", level=2)
        child1.children.append(grandchild)
        root = TopicNode(label="root", level=0, children=[child1, child2])
        flat = root.flatten()
        assert len(flat) == 4
        assert flat[0].label == "root"
        assert flat[1].label == "c1"
        assert flat[2].label == "gc1"
        assert flat[3].label == "c2"

    def test_descendant_chunk_ids(self) -> None:
        grandchild = TopicNode(label="gc", level=2, chunk_ids=["c3"])
        child = TopicNode(
            label="c",
            level=1,
            chunk_ids=["c1", "c2"],
            children=[grandchild],
        )
        root = TopicNode(label="r", level=0, children=[child])
        assert root.descendant_chunk_ids() == ["c1", "c2", "c3"]

    def test_children_at_depth(self) -> None:
        gc1 = TopicNode(label="gc1", level=2)
        gc2 = TopicNode(label="gc2", level=2)
        c1 = TopicNode(label="c1", level=1, children=[gc1])
        c2 = TopicNode(label="c2", level=1, children=[gc2])
        root = TopicNode(label="r", level=0, children=[c1, c2])
        depth2 = root.children_at_depth(2)
        assert [n.label for n in depth2] == ["gc1", "gc2"]

    def test_children_at_depth_zero(self) -> None:
        node = TopicNode(label="x", level=0)
        assert node.children_at_depth(0) == [node]

    def test_leaves_all_leaf(self) -> None:
        root = TopicNode(label="r", level=0, chunk_ids=["c1"])
        assert root.leaves() == [root]

    def test_leaves_mixed(self) -> None:
        leaf1 = TopicNode(label="l1", level=2)
        leaf2 = TopicNode(label="l2", level=2)
        branch = TopicNode(label="b", level=1, children=[leaf1])
        leaf_sibling = TopicNode(label="ls", level=1)
        root = TopicNode(label="r", level=0, children=[branch, leaf_sibling])
        leaves = root.leaves()
        assert [n.label for n in leaves] == ["l1", "ls"]
        # branch is not a leaf because it has children
        assert not any(n.label == "b" for n in leaves)
        # leaf2 is not under root
        assert leaf2 not in leaves

    def test_descendants_to_depth(self) -> None:
        ggc = TopicNode(label="ggc", level=3)
        gc = TopicNode(label="gc", level=2, children=[ggc])
        c = TopicNode(label="c", level=1, children=[gc])
        root = TopicNode(label="r", level=0, children=[c])
        # depth 2: root + c + gc (not ggc)
        nodes = root.descendants_to_depth(2)
        labels = [n.label for n in nodes]
        assert "ggc" not in labels
        assert "gc" in labels

    def test_descendants_to_depth_zero(self) -> None:
        child = TopicNode(label="c", level=1)
        root = TopicNode(label="r", level=0, children=[child])
        assert root.descendants_to_depth(0) == [root]


# ------------------------------------------------------------------
# build_heading_tree
# ------------------------------------------------------------------


class TestBuildHeadingTree:
    """Tests for heading tree reconstruction from chunk metadata."""

    def test_empty_chunks(self) -> None:
        tree = build_heading_tree([])
        assert tree.label == "__root__"
        assert tree.children == []

    def test_flat_headings(self) -> None:
        """Chunks with single-level headings become direct root children."""
        chunks = [
            _chunk("c1", ["Introduction"], [1]),
            _chunk("c2", ["Methods"], [1]),
            _chunk("c3", ["Conclusion"], [1]),
        ]
        tree = build_heading_tree(chunks)
        assert len(tree.children) == 3
        labels = [c.label for c in tree.children]
        assert labels == ["Introduction", "Methods", "Conclusion"]

    def test_nested_headings(self) -> None:
        """Multi-level heading paths create parent-child relationships."""
        chunks = [
            _chunk("c1", ["Security", "CSRF"], [1, 2]),
            _chunk("c2", ["Security", "Token Leakage"], [1, 2]),
            _chunk("c3", ["Introduction"], [1]),
        ]
        tree = build_heading_tree(chunks)
        assert len(tree.children) == 2  # Security, Introduction

        security = next(c for c in tree.children if c.label == "Security")
        assert len(security.children) == 2
        assert {c.label for c in security.children} == {"CSRF", "Token Leakage"}

    def test_deeply_nested(self) -> None:
        """Three-level nesting reconstructs correctly."""
        chunks = [
            _chunk("c1", ["10. Security", "10.12 CSRF", "10.12.1 Mitigation"], [1, 2, 3]),
            _chunk("c2", ["10. Security", "10.12 CSRF", "10.12.2 Notes"], [1, 2, 3]),
        ]
        tree = build_heading_tree(chunks)
        security = tree.children[0]
        assert security.label == "10. Security"
        csrf = security.children[0]
        assert csrf.label == "10.12 CSRF"
        assert len(csrf.children) == 2

    def test_heading_redundancy_dedup(self) -> None:
        """Multiple chunks sharing the same heading path don't create duplicate nodes."""
        chunks = [
            _chunk("c1", ["Security", "CSRF"], [1, 2]),
            _chunk("c2", ["Security", "CSRF"], [1, 2]),
        ]
        tree = build_heading_tree(chunks)
        security = tree.children[0]
        assert len(security.children) == 1  # Only one CSRF node
        csrf = security.children[0]
        assert set(csrf.chunk_ids) == {"c1", "c2"}

    def test_chunks_without_headings_attach_to_root(self) -> None:
        chunks = [
            _chunk("c1"),  # No heading metadata
            _chunk("c2", ["Section A"], [1]),
        ]
        tree = build_heading_tree(chunks)
        assert "c1" in tree.chunk_ids
        assert len(tree.children) == 1

    def test_mismatched_headings_levels_warns(self) -> None:
        """Chunks with mismatched headings/levels lengths go to root."""
        chunks = [
            _chunk("c1", ["A", "B"], [1]),  # 2 headings, 1 level
        ]
        tree = build_heading_tree(chunks)
        assert "c1" in tree.chunk_ids
        assert tree.children == []

    def test_mixed_depth_chunks(self) -> None:
        """Chunks at different depths in the same subtree."""
        chunks = [
            _chunk("c1", ["Security"], [1]),
            _chunk("c2", ["Security", "CSRF"], [1, 2]),
            _chunk("c3", ["Security", "CSRF", "Mitigation"], [1, 2, 3]),
        ]
        tree = build_heading_tree(chunks)
        security = tree.children[0]
        assert "c1" in security.chunk_ids
        csrf = security.children[0]
        assert "c2" in csrf.chunk_ids
        mitigation = csrf.children[0]
        assert "c3" in mitigation.chunk_ids


# ------------------------------------------------------------------
# find_heading_regions
# ------------------------------------------------------------------


class TestFindHeadingRegions:
    """Tests for heading-text matching."""

    def _build_rfc_tree(self) -> tuple[TopicNode, list[SourceResult]]:
        """Build a tree mimicking RFC 6749 structure."""
        chunks = [
            _chunk("c0", ["OAuth 2.0 Authorization Framework"], [0]),
            _chunk("c1", ["1. Introduction"], [1]),
            _chunk("c2", ["10. Security Considerations"], [1]),
            _chunk("c3", ["10. Security Considerations", "10.1 Client Authentication"], [1, 2]),
            _chunk("c4", ["10. Security Considerations", "10.12 CSRF"], [1, 2]),
            _chunk(
                "c5", ["10. Security Considerations", "10.12 CSRF", "10.12.1 Mitigation"], [1, 2, 3]
            ),
        ]
        tree = build_heading_tree(chunks)
        return tree, chunks

    def test_basic_match(self) -> None:
        tree, _ = self._build_rfc_tree()
        matches = find_heading_regions("security considerations", tree)
        assert any(n.label == "10. Security Considerations" for n in matches)

    def test_specific_match(self) -> None:
        tree, _ = self._build_rfc_tree()
        matches = find_heading_regions("CSRF protection", tree)
        labels = [n.label for n in matches]
        assert "10.12 CSRF" in labels

    def test_min_heading_depth_filters_title(self) -> None:
        tree, _ = self._build_rfc_tree()
        # "oauth" matches the title at depth 0 but should be filtered
        matches = find_heading_regions(
            "OAuth framework",
            tree,
            config=HeadingMatchConfig(min_heading_depth=1),
        )
        assert not any(n.level == 0 for n in matches)

    def test_min_heading_depth_zero_includes_title(self) -> None:
        tree, _ = self._build_rfc_tree()
        matches = find_heading_regions(
            "OAuth framework",
            tree,
            config=HeadingMatchConfig(min_heading_depth=0),
        )
        assert any(n.level == 0 for n in matches)

    def test_no_match(self) -> None:
        tree, _ = self._build_rfc_tree()
        matches = find_heading_regions("quantum computing", tree)
        assert matches == []

    def test_stopwords_filtered(self) -> None:
        tree, _ = self._build_rfc_tree()
        # "the" and "is" are stopwords, only "security" matches
        matches = find_heading_regions("the security is important", tree)
        assert len(matches) > 0

    def test_custom_stopwords(self) -> None:
        tree, _ = self._build_rfc_tree()
        # With "security" as a stopword, it won't match
        config = HeadingMatchConfig(
            stopwords=frozenset({"security"}) | DEFAULT_HEADING_STOPWORDS,
        )
        matches = find_heading_regions("security", tree, config=config)
        assert matches == []

    def test_sorted_deepest_first(self) -> None:
        tree, _ = self._build_rfc_tree()
        # "csrf" matches at depth 2 and depth 3 (mitigation has no csrf)
        matches = find_heading_regions("csrf", tree)
        assert matches[0].level >= matches[-1].level

    def test_empty_query(self) -> None:
        tree, _ = self._build_rfc_tree()
        matches = find_heading_regions("", tree)
        assert matches == []

    def test_stopword_only_query(self) -> None:
        tree, _ = self._build_rfc_tree()
        matches = find_heading_regions("the is a", tree)
        assert matches == []

    def test_default_exclude_patterns_filter_references(self) -> None:
        """Default exclude_patterns filter out References, Appendix, etc."""
        chunks = [
            _chunk("c1", ["10. Security Considerations"], [1]),
            _chunk("c2", ["12. References"], [1]),
            _chunk("c3", ["12. References", "12.2. Informative References"], [1, 2]),
            _chunk("c4", ["Appendix C. Acknowledgements"], [1]),
        ]
        tree = build_heading_tree(chunks)

        # "references" would normally match "12. References" and
        # "12.2. Informative References", but they're excluded
        matches = find_heading_regions("security references", tree)
        labels = [n.label for n in matches]
        assert "10. Security Considerations" in labels
        assert "12. References" not in labels
        assert "12.2. Informative References" not in labels

    def test_default_exclude_patterns_filter_appendix(self) -> None:
        chunks = [
            _chunk("c1", ["10. Security Considerations"], [1]),
            _chunk("c2", ["Appendix C. Acknowledgements"], [1]),
            _chunk("c3", ["Appendix A. Examples"], [1]),
        ]
        tree = build_heading_tree(chunks)
        # "appendix" as a query word would match, but the headings are excluded
        matches = find_heading_regions("appendix security", tree)
        labels = [n.label for n in matches]
        assert "10. Security Considerations" in labels
        assert "Appendix C. Acknowledgements" not in labels
        assert "Appendix A. Examples" not in labels

    def test_exclude_patterns_disabled(self) -> None:
        """Setting exclude_patterns=() disables exclusion."""
        chunks = [
            _chunk("c1", ["12. References"], [1]),
        ]
        tree = build_heading_tree(chunks)
        config = HeadingMatchConfig(exclude_patterns=())
        matches = find_heading_regions("references", tree, config=config)
        assert len(matches) == 1
        assert matches[0].label == "12. References"

    def test_custom_exclude_patterns(self) -> None:
        """Custom patterns replace the defaults."""
        chunks = [
            _chunk("c1", ["10. Security Considerations"], [1]),
            _chunk("c2", ["1. Introduction"], [1]),
        ]
        tree = build_heading_tree(chunks)
        # Exclude "Introduction" but not "References" (overriding defaults)
        config = HeadingMatchConfig(exclude_patterns=(r"(?i)^introduction$",))
        matches = find_heading_regions("introduction security", tree, config=config)
        labels = [n.label for n in matches]
        assert "10. Security Considerations" in labels
        assert "1. Introduction" not in labels

    def test_section_number_stripping_for_exclusion(self) -> None:
        """Section numbers are stripped before matching exclude patterns."""
        chunks = [
            _chunk("c1", ["12.2. Informative References"], [2]),
        ]
        tree = build_heading_tree(chunks)
        # "12.2." is stripped, "Informative References" matches the pattern
        matches = find_heading_regions(
            "informative references",
            tree,
            config=HeadingMatchConfig(min_heading_depth=0),
        )
        assert matches == []


# ------------------------------------------------------------------
# expand_region
# ------------------------------------------------------------------


class TestExpandRegion:
    """Tests for heading region expansion."""

    def _build_tree_with_chunks(self) -> tuple[TopicNode, dict[str, SourceResult]]:
        """Build a 3-level tree with chunks at each level."""
        chunks = [
            _chunk("root_c", ["Root"], [0]),
            _chunk("sec_c", ["Root", "Security"], [0, 1]),
            _chunk("csrf_c", ["Root", "Security", "CSRF"], [0, 1, 2]),
            _chunk("tok_c", ["Root", "Security", "Token"], [0, 1, 2]),
            _chunk("mit_c", ["Root", "Security", "CSRF", "Mitigation"], [0, 1, 2, 3]),
            _chunk("intro_c", ["Root", "Introduction"], [0, 1]),
        ]
        tree = build_heading_tree(chunks)
        by_id = _chunks_by_id(chunks)
        # Get the Security node for expansion tests
        root_node = tree.children[0]  # "Root"
        security = next(c for c in root_node.children if c.label == "Security")
        return security, by_id

    def test_subtree_default(self) -> None:
        security, by_id = self._build_tree_with_chunks()
        results = expand_region(security, by_id)
        ids = {r.source_id for r in results}
        assert ids == {"sec_c", "csrf_c", "tok_c", "mit_c"}

    def test_children_mode(self) -> None:
        security, by_id = self._build_tree_with_chunks()
        results = expand_region(security, by_id, expansion_mode="children")
        ids = {r.source_id for r in results}
        # Security node + immediate children (CSRF, Token) but not Mitigation
        assert "sec_c" in ids
        assert "csrf_c" in ids
        assert "tok_c" in ids
        assert "mit_c" not in ids

    def test_leaves_mode(self) -> None:
        security, by_id = self._build_tree_with_chunks()
        results = expand_region(security, by_id, expansion_mode="leaves")
        ids = {r.source_id for r in results}
        # Leaves: Mitigation (under CSRF) and Token (no children)
        assert "mit_c" in ids
        assert "tok_c" in ids
        # Security and CSRF are not leaves
        assert "sec_c" not in ids
        assert "csrf_c" not in ids

    def test_subtree_with_depth_limit(self) -> None:
        security, by_id = self._build_tree_with_chunks()
        results = expand_region(
            security,
            by_id,
            expansion_mode="subtree",
            max_expansion_depth=1,
        )
        ids = {r.source_id for r in results}
        # Security + children (CSRF, Token) but not grandchild (Mitigation)
        assert "sec_c" in ids
        assert "csrf_c" in ids
        assert "tok_c" in ids
        assert "mit_c" not in ids

    def test_leaves_with_depth_limit(self) -> None:
        security, by_id = self._build_tree_with_chunks()
        results = expand_region(
            security,
            by_id,
            expansion_mode="leaves",
            max_expansion_depth=1,
        )
        ids = {r.source_id for r in results}
        # Bounded to 1 level: Security + CSRF + Token.
        # Leaves within that bounded set: CSRF and Token (they have no
        # children *within the bounded set* — Mitigation is outside).
        assert "csrf_c" in ids
        assert "tok_c" in ids
        assert "mit_c" not in ids

    def test_deduplicates_chunk_ids(self) -> None:
        """Chunks appearing in multiple nodes are returned once."""
        node = TopicNode(
            label="parent",
            level=1,
            chunk_ids=["c1"],
            children=[TopicNode(label="child", level=2, chunk_ids=["c1", "c2"])],
        )
        by_id = {
            "c1": _chunk("c1"),
            "c2": _chunk("c2"),
        }
        results = expand_region(node, by_id)
        assert len(results) == 2

    def test_missing_chunk_skipped(self) -> None:
        """Chunk IDs not in the lookup are silently skipped."""
        node = TopicNode(label="n", level=1, chunk_ids=["exists", "missing"])
        by_id = {"exists": _chunk("exists")}
        results = expand_region(node, by_id)
        assert len(results) == 1


# ------------------------------------------------------------------
# extract_query_words
# ------------------------------------------------------------------


class TestExtractQueryWords:
    """Tests for query word extraction."""

    def test_basic(self) -> None:
        words = extract_query_words("What are the security considerations?")
        assert "security" in words
        assert "considerations" in words
        assert "what" not in words
        assert "the" not in words

    def test_short_words_filtered(self) -> None:
        words = extract_query_words("a b cd efg")
        assert "a" not in words
        assert "b" not in words
        assert "cd" in words
        assert "efg" in words

    def test_custom_stopwords(self) -> None:
        words = extract_query_words(
            "security risks",
            stopwords=frozenset({"security"}),
        )
        assert "security" not in words
        assert "risks" in words

    def test_custom_min_length(self) -> None:
        words = extract_query_words("ab cd", min_word_length=3)
        assert words == []

    def test_empty_string(self) -> None:
        assert extract_query_words("") == []

    def test_numbers_included(self) -> None:
        words = extract_query_words("section 10 considerations")
        assert "10" in words
        assert "considerations" in words


# ------------------------------------------------------------------
# HeadingMatchConfig
# ------------------------------------------------------------------


class TestHeadingMatchConfig:
    """Tests for configurable heading match parameters."""

    def test_defaults(self) -> None:
        cfg = HeadingMatchConfig()
        assert cfg.stopwords is DEFAULT_HEADING_STOPWORDS
        assert cfg.min_word_length == 2
        assert cfg.min_heading_depth == 1
        assert len(cfg.exclude_patterns) > 0

    def test_custom_config(self) -> None:
        custom_sw = frozenset({"security", "the"})
        cfg = HeadingMatchConfig(
            stopwords=custom_sw,
            min_word_length=4,
            min_heading_depth=2,
            exclude_patterns=(r"^test$",),
        )
        assert cfg.stopwords == custom_sw
        assert cfg.min_word_length == 4
        assert cfg.min_heading_depth == 2
        assert cfg.exclude_patterns == (r"^test$",)

    def test_frozen(self) -> None:
        cfg = HeadingMatchConfig()
        with pytest.raises(AttributeError):
            cfg.min_heading_depth = 5  # type: ignore[misc]


# ------------------------------------------------------------------
# Integration: build_heading_tree + find_heading_regions + expand_region
# ------------------------------------------------------------------


class TestHeadingPipelineIntegration:
    """End-to-end tests: build tree → find regions → expand."""

    def test_rfc_security_scenario(self) -> None:
        """Mimics the ay-04 scenario: query about security finds all subsections."""
        chunks = [
            _chunk("intro", ["1. Introduction"], [1]),
            _chunk("sec_overview", ["10. Security Considerations"], [1]),
            _chunk("csrf", ["10. Security Considerations", "10.12 CSRF"], [1, 2]),
            _chunk(
                "csrf_mit",
                ["10. Security Considerations", "10.12 CSRF", "10.12.1 Mitigation"],
                [1, 2, 3],
            ),
            _chunk("token", ["10. Security Considerations", "10.3 Token Leakage"], [1, 2]),
            _chunk("redirect", ["10. Security Considerations", "10.5 Redirect URI"], [1, 2]),
        ]
        tree = build_heading_tree(chunks)
        by_id = _chunks_by_id(chunks)

        # Find heading regions for "security"
        regions = find_heading_regions("security", tree)
        assert len(regions) > 0

        # Expand the broadest match (Security Considerations)
        sec_node = next(n for n in regions if "Security Considerations" in n.label)
        results = expand_region(sec_node, by_id, expansion_mode="subtree")
        result_ids = {r.source_id for r in results}

        # All security subsection chunks should be present
        assert "sec_overview" in result_ids
        assert "csrf" in result_ids
        assert "csrf_mit" in result_ids
        assert "token" in result_ids
        assert "redirect" in result_ids
        # Introduction should NOT be present
        assert "intro" not in result_ids

    def test_specific_query_narrows_to_subtopic(self) -> None:
        """A specific query like 'CSRF' finds only the CSRF subtree."""
        chunks = [
            _chunk("sec_overview", ["10. Security Considerations"], [1]),
            _chunk("csrf", ["10. Security Considerations", "10.12 CSRF"], [1, 2]),
            _chunk(
                "csrf_mit",
                ["10. Security Considerations", "10.12 CSRF", "10.12.1 Mitigation"],
                [1, 2, 3],
            ),
            _chunk("token", ["10. Security Considerations", "10.3 Token Leakage"], [1, 2]),
        ]
        tree = build_heading_tree(chunks)
        by_id = _chunks_by_id(chunks)

        regions = find_heading_regions("csrf", tree)
        # Deepest match first — 10.12 CSRF
        csrf_node = regions[0]
        assert "CSRF" in csrf_node.label

        results = expand_region(csrf_node, by_id, expansion_mode="subtree")
        result_ids = {r.source_id for r in results}
        assert "csrf" in result_ids
        assert "csrf_mit" in result_ids
        assert "token" not in result_ids


# ------------------------------------------------------------------
# The orders the walks emit, and the cycle that used to not terminate
# ------------------------------------------------------------------


def _level_order(root: TopicNode) -> list[str]:
    """The labels a level-synchronous descent would emit — the negative control.

    Not a walk anything ships, and deliberately so.  It exists to show that the
    assertions below are *capable* of failing: a fixture whose pre-order and
    level-order coincide would pass whichever discipline the walk used, which is
    a test that guards nothing while reporting green.
    """
    labels: list[str] = []
    frontier = [root]
    while frontier:
        labels.extend(node.label for node in frontier)
        frontier = [child for node in frontier for child in node.children]
    return labels


def _branching_tree() -> TopicNode:
    """A tree that branches at two levels, which is what tells the orders apart.

    ``root -> (c1 -> gc1, c2 -> gc2)``.  A chain cannot discriminate — every
    descent visits a chain in the same sequence — so the second branch is the
    whole point of the shape.
    """
    gc1 = TopicNode(label="gc1", level=2, chunk_ids=["x_gc1"])
    gc2 = TopicNode(label="gc2", level=2, chunk_ids=["x_gc2"])
    c1 = TopicNode(label="c1", level=1, chunk_ids=["x_c1"], children=[gc1])
    c2 = TopicNode(label="c2", level=1, chunk_ids=["x_c2"], children=[gc2])
    return TopicNode(label="root", level=0, chunk_ids=["x_root"], children=[c1, c2])


class TestWalkOrdersAreAsserted:
    """Each walk's emission order, pinned with a control that can tell it apart.

    The existing tests in :class:`TestTopicNode` assert membership, counts and —
    for ``flatten`` — one order over a fixture that branches once.  None of them
    can see the difference between *pre-order by discovery* and a
    level-synchronous descent on the walk whose order a consumer actually reads,
    which is ``descendants_to_depth``: its own test asserts only that ``ggc`` is
    absent and ``gc`` is present.

    That gap is what these assertions close.  They are the proof obligation for
    replacing four hand-written descents with delegations to one shared walk:
    the suite passed before the swap and would pass after a swap that changed
    the order, so it validates nothing about the swap on its own.
    """

    def test_flatten_is_pre_order_by_discovery(self) -> None:
        root = _branching_tree()

        assert [n.label for n in root.flatten()] == ["root", "c1", "gc1", "c2", "gc2"]
        assert _level_order(root) == ["root", "c1", "c2", "gc1", "gc2"]

    def test_descendants_to_depth_is_pre_order_by_discovery(self) -> None:
        """The one whose order reaches a consumer, and whose own test cannot see it."""
        root = _branching_tree()

        assert [n.label for n in root.descendants_to_depth(2)] == [
            "root",
            "c1",
            "gc1",
            "c2",
            "gc2",
        ]
        assert _level_order(root) == ["root", "c1", "c2", "gc1", "gc2"]

    def test_leaves_is_pre_order_by_discovery(self) -> None:
        """Mixed depth, because equal-depth leaves cannot tell the orders apart."""
        l1 = TopicNode(label="l1", level=2)
        branch = TopicNode(label="b", level=1, children=[l1])
        leaf_sibling = TopicNode(label="ls", level=1)
        root = TopicNode(label="r", level=0, children=[branch, leaf_sibling])

        assert [n.label for n in root.leaves()] == ["l1", "ls"]
        # The control: a level-synchronous descent reaches ``ls`` first.
        assert [label for label in _level_order(root) if label in {"l1", "ls"}] == [
            "ls",
            "l1",
        ]

    def test_children_at_depth_is_one_level_and_has_no_order_to_choose(self) -> None:
        """The one walk with nothing to discriminate, asserted as such.

        A level *is* the set of nodes at one depth, so every descent that
        reaches it reaches the same nodes in the order the parents were visited.
        Recorded here rather than left out, so that a reader looking for the
        fourth order assertion finds the reason there is none instead of a gap.
        """
        root = _branching_tree()

        assert [n.label for n in root.children_at_depth(2)] == ["gc1", "gc2"]
        assert [label for label in _level_order(root) if label.startswith("gc")] == [
            "gc1",
            "gc2",
        ]


class TestExpansionArmsAgree:
    """Bounded and unbounded expansion agree on order, not only on membership.

    ``expansion_mode`` has two arms per mode: a bounded one reached when
    ``max_expansion_depth`` is set, and an unbounded one when it is not.  They
    have always agreed, because each pair happened to be written in the same
    discipline — so ``max_expansion_depth`` is a depth bound rather than also a
    reordering switch.  Nothing said so, which made it true by coincidence.
    """

    def test_subtree_arms_agree_when_the_bound_reaches_the_whole_tree(self) -> None:
        root = _branching_tree()
        by_id = _chunks_by_id([_chunk(cid) for cid in ["x_root", "x_c1", "x_gc1", "x_c2", "x_gc2"]])

        unbounded = expand_region(root, by_id, expansion_mode="subtree")
        bounded = expand_region(root, by_id, expansion_mode="subtree", max_expansion_depth=2)

        assert [r.source_id for r in bounded] == [r.source_id for r in unbounded]
        assert [r.source_id for r in unbounded] == ["x_root", "x_c1", "x_gc1", "x_c2", "x_gc2"]

    def test_leaves_arms_agree_when_the_bound_reaches_the_whole_tree(self) -> None:
        root = _branching_tree()
        by_id = _chunks_by_id([_chunk(cid) for cid in ["x_root", "x_c1", "x_gc1", "x_c2", "x_gc2"]])

        unbounded = expand_region(root, by_id, expansion_mode="leaves")
        bounded = expand_region(root, by_id, expansion_mode="leaves", max_expansion_depth=2)

        assert [r.source_id for r in bounded] == [r.source_id for r in unbounded]
        assert [r.source_id for r in unbounded] == ["x_gc1", "x_gc2"]

    def test_the_bound_selects_rather_than_reorders(self) -> None:
        """A tighter bound drops a tail; it does not permute what remains."""
        root = _branching_tree()
        by_id = _chunks_by_id([_chunk(cid) for cid in ["x_root", "x_c1", "x_gc1", "x_c2", "x_gc2"]])

        shallow = expand_region(root, by_id, expansion_mode="subtree", max_expansion_depth=1)
        deep = expand_region(root, by_id, expansion_mode="subtree", max_expansion_depth=2)

        shallow_ids = [r.source_id for r in shallow]
        assert shallow_ids == ["x_root", "x_c1", "x_c2"]
        # Every chunk the shallow bound keeps appears in the deep one, in order.
        assert [i for i in (r.source_id for r in deep) if i in set(shallow_ids)] == shallow_ids


class TestAMalformedTreeTerminates:
    """A ``children`` list that is not a tree, and what the walks do with it.

    ``TopicNode`` is a public dataclass with a mutable ``children`` list, so
    nothing stops a caller from building a cycle or from hanging one subtree
    under two parents.  ``build_heading_tree`` never does either, which is why
    this went unnoticed: every walk descended with no record of where it had
    been, so a cycle recursed until the interpreter stopped it.

    These assert the behaviour *after* the walks are shared.  Before it,
    ``flatten`` and ``leaves`` raised ``RecursionError`` here — not a hang, and
    not a wrong answer, but an answer a consumer cannot act on.
    """

    @staticmethod
    def _cycle() -> TopicNode:
        """``a -> b -> a``. Two nodes, each listing the other as a child."""
        a = TopicNode(label="a", level=0)
        b = TopicNode(label="b", level=1)
        a.children.append(b)
        b.children.append(a)
        return a

    @staticmethod
    def _shared_subtree() -> TopicNode:
        """One subtree hung under two parents — a DAG, not a cycle."""
        shared = TopicNode(label="shared", level=2, children=[TopicNode(label="deep", level=3)])
        p1 = TopicNode(label="p1", level=1, children=[shared])
        p2 = TopicNode(label="p2", level=1, children=[shared])
        return TopicNode(label="root", level=0, children=[p1, p2])

    def test_flatten_terminates_on_a_cycle(self) -> None:
        assert [n.label for n in self._cycle().flatten()] == ["a", "b"]

    def test_leaves_terminates_on_a_cycle(self) -> None:
        """``b`` is a leaf, because the edge that closed the cycle is not walked.

        Nothing in a cycle is childless in the *graph*, so this answer is a
        property of the projection rather than of the input: the walk reaches
        ``b``, finds its only child already visited, and has nothing left to
        descend into.  It is the same rule that makes ``p2`` a leaf in
        :meth:`test_a_node_reachable_twice_is_emitted_once` — a node whose every
        edge led somewhere already seen.
        """
        assert [n.label for n in self._cycle().leaves()] == ["b"]

    def test_descendants_to_depth_stops_repeating_on_a_cycle(self) -> None:
        """It terminated before — by exhausting the depth bound — and repeated.

        The depth bound made this the one walk a cycle did not crash, and the
        answer it gave was ``a, b, a, b``: the same two nodes emitted once per
        level until the bound ran out.  A caller reading that as a region got
        each node as many times as the bound allowed.
        """
        assert [n.label for n in self._cycle().descendants_to_depth(3)] == ["a", "b"]

    def test_a_node_reachable_twice_is_emitted_once(self) -> None:
        """The other half of the same guard, and the one with no cycle in it.

        A subtree hung under two parents used to be walked twice, so a region
        built from it carried every chunk below the shared node twice over.
        ``expand_region`` deduplicates chunks by id and so hid this, but
        ``flatten()`` is public and its caller sees the repeat.
        """
        root = self._shared_subtree()

        assert [n.label for n in root.flatten()] == ["root", "p1", "shared", "deep", "p2"]
        assert [n.label for n in root.leaves()] == ["deep", "p2"]

    def test_the_second_parent_is_still_reached(self) -> None:
        """Dropping the repeat must not drop the branch that carried it.

        ``p2`` has exactly one child and it is the shared node, so a guard that
        skipped a seen node by skipping its parent's edge *and* its parent
        would lose ``p2`` itself — a node nothing else reaches.
        """
        root = self._shared_subtree()

        assert "p2" in [n.label for n in root.flatten()]
        assert [n.label for n in root.children_at_depth(1)] == ["p1", "p2"]


class TestTopicNodeHierarchy:
    """The adapter, as the public construct it is rather than as an internal.

    Its four protocol members are what the delegations above run on, so those
    are exercised heavily by every other test in this file.  What is asserted
    here is the part a consumer touches directly: the key, the projection back
    to nodes, and the walks ``TopicNode`` does not itself carry.
    """

    def test_it_satisfies_the_protocol(self) -> None:
        axis = TopicNodeHierarchy(_branching_tree())

        assert isinstance(axis, Hierarchy)

    def test_the_key_is_a_route_through_children(self) -> None:
        """``(0, 1)`` means ``children[0].children[1]``, and reads that way."""
        root = _branching_tree()
        axis = TopicNodeHierarchy(root)

        assert axis.roots() == ((),)
        assert axis.children(()) == ((0,), (1,))
        assert axis.children((0,)) == ((0, 0),)
        assert axis.node((0, 0)).label == "gc1"
        assert axis.node((1, 0)) is root.children[1].children[0]

    def test_a_position_the_tree_does_not_reach_is_absent_rather_than_empty(self) -> None:
        """``contains`` is what separates *nothing below* from *not here*."""
        axis = TopicNodeHierarchy(_branching_tree())

        assert axis.contains((0, 0)) is True
        assert axis.contains((9,)) is False
        assert axis.children((9,)) == ()
        assert axis.children((0, 0)) == ()

    def test_parents_is_the_prefix_and_the_anchor_has_none(self) -> None:
        axis = TopicNodeHierarchy(_branching_tree())

        assert axis.parents((0, 0)) == ((0,),)
        assert axis.parents((0,)) == ((),)
        assert axis.parents(()) == ()
        assert axis.parents((9,)) == ()

    def test_an_unknown_anchor_is_refused_legibly(self) -> None:
        """The reason the key is a value: the refusal names a position."""
        axis = TopicNodeHierarchy(_branching_tree())

        with pytest.raises(NotFoundError, match=r"no node \(9, 9\) in this hierarchy"):
            flatten(axis, from_id=(9, 9))

    def test_it_carries_the_walks_topicnode_does_not(self) -> None:
        """What making the adapter public is for."""
        root = _branching_tree()
        axis = TopicNodeHierarchy(root)
        gc2 = axis.key_of(root.children[1].children[0])

        assert axis.nodes(ancestors(axis, gc2)) == [root.children[1], root]
        assert paths_to_root(axis, gc2) == (((1, 0), (1,), ()),)
        assert deepest_common_ancestor(axis, (0, 0), (1, 0)) == ()

    def test_key_of_answers_by_identity_not_by_equality(self) -> None:
        """Two nodes with the same fields are ``==``; they are not the same node."""
        first = TopicNode(label="same", level=1)
        second = TopicNode(label="same", level=1)
        root = TopicNode(label="root", level=0, children=[first, second])
        axis = TopicNodeHierarchy(root)

        assert first == second
        assert axis.key_of(first) == (0,)
        assert axis.key_of(second) == (1,)

    def test_key_of_refuses_a_node_from_another_tree(self) -> None:
        axis = TopicNodeHierarchy(_branching_tree())

        with pytest.raises(KeyError):
            axis.key_of(TopicNode(label="elsewhere", level=0))

    def test_the_index_survives_a_skipped_edge(self) -> None:
        """A dropped duplicate edge leaves a gap in the indices, not a shift.

        ``p2``'s only child is already placed, so ``p2`` has no children here —
        but the *second* child of a node whose first was dropped must keep
        index 1, or the key stops naming a route through ``children``.
        """
        shared = TopicNode(label="shared", level=2)
        holder = TopicNode(label="holder", level=1, children=[shared])
        second = TopicNode(label="second", level=2)
        reuser = TopicNode(label="reuser", level=1, children=[shared, second])
        root = TopicNode(label="root", level=0, children=[holder, reuser])
        axis = TopicNodeHierarchy(root)

        assert axis.children((1,)) == ((1, 1),)
        assert axis.node((1, 1)) is second
