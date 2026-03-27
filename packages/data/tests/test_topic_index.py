"""Tests for topic-index abstractions and heading tree utilities."""

from __future__ import annotations

import pytest

from dataknobs_data.sources.base import SourceResult
from dataknobs_data.sources.topic_index import (
    DEFAULT_HEADING_STOPWORDS,
    HeadingMatchConfig,
    TopicNode,
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
    metadata: dict = {}
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
            label="c", level=1,
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
            _chunk("c5", ["10. Security Considerations", "10.12 CSRF", "10.12.1 Mitigation"], [1, 2, 3]),
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
            security, by_id,
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
            security, by_id,
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
            label="parent", level=1, chunk_ids=["c1"],
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
            _chunk("csrf_mit", ["10. Security Considerations", "10.12 CSRF", "10.12.1 Mitigation"], [1, 2, 3]),
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
            _chunk("csrf_mit", ["10. Security Considerations", "10.12 CSRF", "10.12.1 Mitigation"], [1, 2, 3]),
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
