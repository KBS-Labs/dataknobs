"""Tests for HeadingTreeIndex — heading-tree topic index for structured docs."""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_data.sources.base import RetrievalIntent, SourceResult
from dataknobs_data.sources.topic_index import DEFAULT_HEADING_STOPWORDS, HeadingMatchConfig

from dataknobs_bots.knowledge.sources.heading_tree import (
    HeadingTreeConfig,
    HeadingTreeIndex,
    _index_to_letter,
    _letter_to_index,
    _parse_bracket_indices,
    _parse_letter_indices,
)

from dataknobs_llm import EchoProvider
from dataknobs_llm.testing import text_response

_ECHO_CONFIG: dict[str, str] = {"provider": "echo", "model": "test"}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _chunk(
    source_id: str,
    headings: list[str] | None = None,
    heading_levels: list[int] | None = None,
    content: str = "",
    relevance: float = 1.0,
) -> SourceResult:
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
        relevance=relevance,
        metadata=metadata,
    )


def _rfc_chunks() -> list[SourceResult]:
    """Build chunks mimicking an RFC with security sections."""
    return [
        _chunk("title", ["OAuth 2.0 Authorization Framework"], [0],
               content="This document describes OAuth 2.0"),
        _chunk("intro", ["1. Introduction"], [1],
               content="OAuth provides authorization flows"),
        _chunk("sec_overview", ["10. Security Considerations"], [1],
               content="This section describes security considerations"),
        _chunk("csrf", ["10. Security Considerations", "10.12 CSRF"], [1, 2],
               content="CSRF attacks exploit trust in authorized users"),
        _chunk("csrf_mit", ["10. Security Considerations", "10.12 CSRF", "10.12.1 Mitigation"], [1, 2, 3],
               content="Use state parameter to prevent CSRF"),
        _chunk("token", ["10. Security Considerations", "10.3 Token Leakage"], [1, 2],
               content="Tokens may leak through logs or referrer headers"),
        _chunk("redirect", ["10. Security Considerations", "10.5 Redirect URI"], [1, 2],
               content="Open redirectors can be exploited for token theft"),
        _chunk("client_auth", ["10. Security Considerations", "10.1 Client Authentication"], [1, 2],
               content="Clients must authenticate to the authorization server"),
    ]


def _make_vector_fn(
    chunks: list[SourceResult],
) -> Any:
    """Create a simple vector query fn that returns chunks matching query words."""
    async def vector_fn(query: str, top_k: int) -> list[SourceResult]:
        query_lower = query.lower()
        matches = []
        for c in chunks:
            content_lower = c.content.lower()
            if any(w in content_lower for w in query_lower.split()):
                matches.append(c)
        return matches[:top_k]
    return vector_fn


# ------------------------------------------------------------------
# HeadingTreeIndex — construction
# ------------------------------------------------------------------


class TestHeadingTreeIndexConstruction:
    """Tests for index construction and topics()."""

    def test_topics_returns_unique_labels_eager(self) -> None:
        chunks = _rfc_chunks()
        index = HeadingTreeIndex.from_chunks(chunks)
        topics = index.topics()
        assert "10. Security Considerations" in topics
        assert "10.12 CSRF" in topics
        assert "1. Introduction" in topics
        assert "__root__" not in topics

    def test_topics_returns_empty_in_lazy_mode(self) -> None:
        index = HeadingTreeIndex()
        assert index.topics() == []

    def test_from_source_results_returns_none_when_sparse(self) -> None:
        chunks = [
            _chunk("c1"),  # No heading metadata
            _chunk("c2"),
        ]
        result = HeadingTreeIndex.from_source_results(chunks)
        assert result is None

    def test_from_source_results_returns_index_when_sufficient(self) -> None:
        chunks = _rfc_chunks()
        result = HeadingTreeIndex.from_source_results(chunks)
        assert result is not None
        assert isinstance(result, HeadingTreeIndex)


# ------------------------------------------------------------------
# HeadingTreeIndex — heading_match strategy (eager mode)
# ------------------------------------------------------------------


class TestHeadingMatchStrategy:
    """Tests for the heading_match entry strategy using eager mode."""

    @pytest.mark.asyncio
    async def test_security_query_expands_to_all_subsections(self) -> None:
        """The ay-04 scenario: 'security' finds all security subsections."""
        chunks = _rfc_chunks()
        config = HeadingTreeConfig(entry_strategy="heading_match")
        index = HeadingTreeIndex.from_chunks(chunks, config=config)

        results = await index.resolve("security considerations")
        ids = {r.source_id for r in results}

        assert "sec_overview" in ids
        assert "csrf" in ids
        assert "csrf_mit" in ids
        assert "token" in ids
        assert "redirect" in ids
        assert "client_auth" in ids
        assert "intro" not in ids

    @pytest.mark.asyncio
    async def test_specific_query_narrows_to_subtopic(self) -> None:
        chunks = _rfc_chunks()
        config = HeadingTreeConfig(entry_strategy="heading_match")
        index = HeadingTreeIndex.from_chunks(chunks, config=config)

        results = await index.resolve("CSRF protection")
        ids = {r.source_id for r in results}

        assert "csrf" in ids
        assert "csrf_mit" in ids
        # Other security subsections not included (CSRF node is deepest match)
        assert "token" not in ids

    @pytest.mark.asyncio
    async def test_no_match_returns_empty(self) -> None:
        chunks = _rfc_chunks()
        config = HeadingTreeConfig(entry_strategy="heading_match")
        index = HeadingTreeIndex.from_chunks(chunks, config=config)

        results = await index.resolve("quantum computing")
        assert results == []

    @pytest.mark.asyncio
    async def test_min_heading_depth_filters_title(self) -> None:
        chunks = _rfc_chunks()
        config = HeadingTreeConfig(
            entry_strategy="heading_match",
            min_heading_depth=1,
        )
        index = HeadingTreeIndex.from_chunks(chunks, config=config)

        # "oauth" matches the title at depth 0 — should be filtered
        results = await index.resolve("OAuth framework")
        ids = {r.source_id for r in results}
        assert "title" not in ids


# ------------------------------------------------------------------
# HeadingTreeIndex — vector strategy
# ------------------------------------------------------------------


class TestVectorStrategy:
    """Tests for the vector entry strategy."""

    @pytest.mark.asyncio
    async def test_vector_seeds_expand_to_heading_region(self) -> None:
        """Vector search finds CSRF chunks → expands to heading region."""
        chunks = _rfc_chunks()
        vector_fn = _make_vector_fn(chunks)
        config = HeadingTreeConfig(entry_strategy="vector")
        index = HeadingTreeIndex.from_chunks(
            chunks, vector_query_fn=vector_fn, config=config,
        )

        results = await index.resolve("CSRF attacks")
        ids = {r.source_id for r in results}

        assert "csrf" in ids
        assert "csrf_mit" in ids

    @pytest.mark.asyncio
    async def test_no_vector_fn_lazy_returns_empty(self) -> None:
        """In lazy mode with no vector_query_fn, resolve returns empty."""
        config = HeadingTreeConfig(entry_strategy="vector")
        index = HeadingTreeIndex(config=config)  # No vector_query_fn, lazy mode

        results = await index.resolve("security")
        assert results == []

    @pytest.mark.asyncio
    async def test_no_vector_fn_eager_still_uses_tree(self) -> None:
        """In eager mode, vector strategy uses the pre-built tree nodes."""
        config = HeadingTreeConfig(entry_strategy="vector")
        index = HeadingTreeIndex.from_chunks(
            _rfc_chunks(), config=config,
        )  # No vector_query_fn but tree is pre-built

        results = await index.resolve("security")
        # Eager tree has all nodes; vector strategy returns them
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_vocabulary_gap_bridged(self) -> None:
        """'safety' doesn't match 'Security' headings but vector search bridges the gap."""
        chunks = _rfc_chunks()

        async def safety_to_security(query: str, top_k: int) -> list[SourceResult]:
            # Simulates vector search bridging 'safety' → security content
            if "safety" in query.lower():
                return [c for c in chunks if "security" in c.content.lower()][:top_k]
            return []

        config = HeadingTreeConfig(entry_strategy="vector")
        index = HeadingTreeIndex.from_chunks(
            chunks, vector_query_fn=safety_to_security, config=config,
        )

        results = await index.resolve("safety concerns")
        ids = {r.source_id for r in results}
        # Should find security content through vector bridge
        assert len(results) > 0
        assert "sec_overview" in ids or "csrf" in ids or "token" in ids


# ------------------------------------------------------------------
# HeadingTreeIndex — lazy mode (per-turn tree construction)
# ------------------------------------------------------------------


class TestLazyMode:
    """Tests for lazy query-driven reconstruction."""

    @pytest.mark.asyncio
    async def test_lazy_resolve_builds_tree_per_turn(self) -> None:
        """Lazy mode builds the tree from vector search seeds."""
        chunks = _rfc_chunks()
        vector_fn = _make_vector_fn(chunks)
        config = HeadingTreeConfig(entry_strategy="vector")
        index = HeadingTreeIndex(
            vector_query_fn=vector_fn, config=config,
        )

        # No pre-built tree
        assert index.topics() == []

        results = await index.resolve("CSRF attacks")
        ids = {r.source_id for r in results}
        assert "csrf" in ids
        assert "csrf_mit" in ids

    @pytest.mark.asyncio
    async def test_lazy_heading_match_uses_seed_tree(self) -> None:
        """In lazy 'both' mode, heading_match runs against seed-built tree."""
        chunks = _rfc_chunks()
        vector_fn = _make_vector_fn(chunks)
        config = HeadingTreeConfig(entry_strategy="both")
        index = HeadingTreeIndex(
            vector_query_fn=vector_fn, config=config,
        )

        results = await index.resolve("security CSRF")
        ids = {r.source_id for r in results}
        assert "csrf" in ids

    @pytest.mark.asyncio
    async def test_lazy_no_vector_fn_returns_empty(self) -> None:
        config = HeadingTreeConfig(entry_strategy="both")
        index = HeadingTreeIndex(config=config)

        results = await index.resolve("security")
        assert results == []

    @pytest.mark.asyncio
    async def test_lazy_no_heading_metadata_returns_empty(self) -> None:
        """Seeds with no heading metadata produce no tree."""
        async def no_heading_fn(query: str, top_k: int) -> list[SourceResult]:
            return [_chunk("c1", content="some content")]

        index = HeadingTreeIndex(vector_query_fn=no_heading_fn)
        results = await index.resolve("anything")
        assert results == []


# ------------------------------------------------------------------
# HeadingTreeIndex — both strategy (default)
# ------------------------------------------------------------------


class TestBothStrategy:
    """Tests for the 'both' entry strategy (default)."""

    @pytest.mark.asyncio
    async def test_merges_heading_and_vector_seeds(self) -> None:
        """Both strategies contribute seed headings, merged by set union."""
        chunks = _rfc_chunks()
        vector_fn = _make_vector_fn(chunks)
        config = HeadingTreeConfig(entry_strategy="both")
        index = HeadingTreeIndex.from_chunks(
            chunks, vector_query_fn=vector_fn, config=config,
        )

        results = await index.resolve("security CSRF")
        ids = {r.source_id for r in results}

        assert "csrf" in ids
        assert "csrf_mit" in ids

    @pytest.mark.asyncio
    async def test_deduplicates_regions_from_both_strategies(self) -> None:
        """Same heading region found by both strategies is expanded once."""
        chunks = _rfc_chunks()
        vector_fn = _make_vector_fn(chunks)
        config = HeadingTreeConfig(entry_strategy="both")
        index = HeadingTreeIndex.from_chunks(
            chunks, vector_query_fn=vector_fn, config=config,
        )

        results = await index.resolve("CSRF")
        # No duplicate chunks even though both strategies find CSRF region
        ids = [r.source_id for r in results]
        assert len(ids) == len(set(ids))


# ------------------------------------------------------------------
# Expansion modes
# ------------------------------------------------------------------


class TestExpansionModes:
    """Tests for expansion_mode and max_expansion_depth."""

    @pytest.mark.asyncio
    async def test_children_mode(self) -> None:
        chunks = _rfc_chunks()
        config = HeadingTreeConfig(
            entry_strategy="heading_match",
            expansion_mode="children",
        )
        index = HeadingTreeIndex.from_chunks(chunks, config=config)

        results = await index.resolve("security considerations")
        ids = {r.source_id for r in results}

        assert "sec_overview" in ids
        assert "csrf" in ids
        assert "token" in ids
        assert "redirect" in ids
        assert "client_auth" in ids
        # Grandchild (10.12.1 Mitigation) NOT included in children mode
        assert "csrf_mit" not in ids

    @pytest.mark.asyncio
    async def test_leaves_mode(self) -> None:
        chunks = _rfc_chunks()
        config = HeadingTreeConfig(
            entry_strategy="heading_match",
            expansion_mode="leaves",
        )
        index = HeadingTreeIndex.from_chunks(chunks, config=config)

        results = await index.resolve("security considerations")
        ids = {r.source_id for r in results}

        assert "csrf_mit" in ids
        assert "token" in ids
        assert "redirect" in ids
        assert "client_auth" in ids
        assert "csrf" not in ids
        assert "sec_overview" not in ids

    @pytest.mark.asyncio
    async def test_subtree_with_depth_limit(self) -> None:
        chunks = _rfc_chunks()
        config = HeadingTreeConfig(
            entry_strategy="heading_match",
            expansion_mode="subtree",
            max_expansion_depth=1,
        )
        index = HeadingTreeIndex.from_chunks(chunks, config=config)

        results = await index.resolve("security considerations")
        ids = {r.source_id for r in results}

        assert "sec_overview" in ids
        assert "csrf" in ids
        assert "token" in ids
        # Depth 2 (grandchild): NOT included
        assert "csrf_mit" not in ids


# ------------------------------------------------------------------
# Scope profiles
# ------------------------------------------------------------------


class TestScopeProfiles:
    """Tests for dynamic parameter tuning via scope profiles."""

    @pytest.mark.asyncio
    async def test_scope_profile_overrides_expansion_mode(self) -> None:
        chunks = _rfc_chunks()
        config = HeadingTreeConfig(
            entry_strategy="heading_match",
            expansion_mode="subtree",
            scope_profiles={
                "focused": {"expansion_mode": "children"},
            },
        )
        index = HeadingTreeIndex.from_chunks(chunks, config=config)

        intent = RetrievalIntent(
            text_queries=["security"],
            scope="focused",
        )
        results = await index.resolve(
            "security considerations", intent=intent,
        )
        ids = {r.source_id for r in results}

        assert "csrf_mit" not in ids  # grandchild excluded

    @pytest.mark.asyncio
    async def test_explicit_override_beats_scope_profile(self) -> None:
        chunks = _rfc_chunks()
        config = HeadingTreeConfig(
            entry_strategy="heading_match",
            expansion_mode="subtree",
            scope_profiles={
                "focused": {"expansion_mode": "children"},
            },
        )
        index = HeadingTreeIndex.from_chunks(chunks, config=config)

        intent = RetrievalIntent(
            text_queries=["security"],
            scope="focused",
            raw_data={"topic_index": {"expansion_mode": "leaves"}},
        )
        results = await index.resolve(
            "security considerations", intent=intent,
        )
        ids = {r.source_id for r in results}

        assert "csrf_mit" in ids  # leaf node
        assert "sec_overview" not in ids  # non-leaf

    @pytest.mark.asyncio
    async def test_unknown_scope_uses_defaults(self) -> None:
        chunks = _rfc_chunks()
        config = HeadingTreeConfig(
            entry_strategy="heading_match",
            expansion_mode="subtree",
            scope_profiles={"focused": {"expansion_mode": "children"}},
        )
        index = HeadingTreeIndex.from_chunks(chunks, config=config)

        intent = RetrievalIntent(
            text_queries=["security"],
            scope="unknown_scope",
        )
        results = await index.resolve(
            "security considerations", intent=intent,
        )
        ids = {r.source_id for r in results}

        assert "csrf_mit" in ids


# ------------------------------------------------------------------
# LLM heading selection
# ------------------------------------------------------------------


class TestScoreBasedSelection:
    """Tests for default score-based heading region selection."""

    @pytest.mark.asyncio
    async def test_default_uses_score_selection(self) -> None:
        """Default heading_selection='score' — no LLM called."""
        chunks = _rfc_chunks()
        config = HeadingTreeConfig(entry_strategy="heading_match")
        index = HeadingTreeIndex.from_chunks(chunks, config=config)

        llm = EchoProvider(_ECHO_CONFIG)
        results = await index.resolve("CSRF security", llm=llm)
        assert llm.call_count == 0  # Score-based, no LLM needed
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_top_regions_limits_selection(self) -> None:
        """top_regions caps the number of selected heading regions."""
        chunks = _rfc_chunks()
        config = HeadingTreeConfig(
            entry_strategy="heading_match",
            top_regions=1,
        )
        index = HeadingTreeIndex.from_chunks(chunks, config=config)

        # "security" matches multiple headings; top_regions=1 keeps best
        results = await index.resolve("security considerations")
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_higher_scoring_region_preferred(self) -> None:
        """Regions with higher-scoring seed chunks are preferred."""
        chunks = [
            SourceResult(
                content="High-relevance security",
                source_id="sec_high",
                source_name="kb",
                source_type="vector_kb",
                relevance=0.9,
                metadata={"headings": ["Security"], "heading_levels": [1]},
            ),
            SourceResult(
                content="Low-relevance intro",
                source_id="intro_low",
                source_name="kb",
                source_type="vector_kb",
                relevance=0.3,
                metadata={"headings": ["Introduction"], "heading_levels": [1]},
            ),
        ]
        config = HeadingTreeConfig(
            entry_strategy="heading_match",
            top_regions=1,
            min_heading_depth=0,
        )
        index = HeadingTreeIndex.from_chunks(chunks, config=config)

        # Both "security" and "introduction" match, but top_regions=1
        # should prefer "Security" (0.9) over "Introduction" (0.3)
        results = await index.resolve("security introduction")
        ids = {r.source_id for r in results}
        assert "sec_high" in ids


class TestLLMHeadingSelection:
    """Tests for optional LLM-based heading selection (heading_selection='llm')."""

    @pytest.mark.asyncio
    async def test_llm_narrows_heading_candidates(self) -> None:
        chunks = _rfc_chunks()
        config = HeadingTreeConfig(
            entry_strategy="heading_match",
            heading_selection="llm",
        )
        index = HeadingTreeIndex.from_chunks(chunks, config=config)

        llm = EchoProvider(_ECHO_CONFIG)
        llm.set_responses([text_response("(A)")])

        results = await index.resolve("CSRF security", llm=llm)
        ids = {r.source_id for r in results}

        assert "csrf" in ids or "csrf_mit" in ids

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back_to_all_candidates(self) -> None:
        chunks = _rfc_chunks()
        config = HeadingTreeConfig(
            entry_strategy="heading_match",
            heading_selection="llm",
        )
        index = HeadingTreeIndex.from_chunks(chunks, config=config)

        llm = EchoProvider(_ECHO_CONFIG)
        llm.set_response_function(lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("LLM failed")))

        results = await index.resolve("security", llm=llm)
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_single_candidate_skips_llm(self) -> None:
        """When only one heading matches, LLM selection is skipped."""
        chunks = _rfc_chunks()
        config = HeadingTreeConfig(
            entry_strategy="heading_match",
            heading_selection="llm",
        )
        index = HeadingTreeIndex.from_chunks(chunks, config=config)

        llm = EchoProvider(_ECHO_CONFIG)
        results = await index.resolve("CSRF", llm=llm)
        assert llm.call_count == 0
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_llm_receives_llm_message_objects(self) -> None:
        """LLM heading selection sends LLMMessage objects, not dicts."""
        from dataknobs_llm.llm.base import LLMMessage

        chunks = _rfc_chunks()
        config = HeadingTreeConfig(
            entry_strategy="heading_match",
            heading_selection="llm",
        )
        index = HeadingTreeIndex.from_chunks(chunks, config=config)

        llm = EchoProvider(_ECHO_CONFIG)
        llm.set_responses([text_response("(A)")])

        await index.resolve("CSRF security", llm=llm)
        assert llm.call_count == 1
        last_call = llm.get_last_call()
        assert last_call is not None
        messages = last_call["messages"]
        assert all(isinstance(m, LLMMessage) for m in messages)
        assert messages[0].role == "system"
        assert messages[1].role == "user"

    @pytest.mark.asyncio
    async def test_dedicated_heading_selection_llm(self) -> None:
        """Dedicated heading_selection_llm is used instead of the main LLM."""
        chunks = _rfc_chunks()
        config = HeadingTreeConfig(
            entry_strategy="heading_match",
            heading_selection="llm",
        )

        # Dedicated LLM for heading selection
        dedicated_llm = EchoProvider(_ECHO_CONFIG)
        dedicated_llm.set_responses([text_response("(A)")])

        index = HeadingTreeIndex.from_chunks(
            chunks, config=config,
            heading_selection_llm=dedicated_llm,
        )

        # Main LLM should NOT be called
        main_llm = EchoProvider(_ECHO_CONFIG)

        results = await index.resolve("CSRF security", llm=main_llm)
        assert dedicated_llm.call_count == 1
        assert main_llm.call_count == 0
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_dedicated_llm_fallback_to_main(self) -> None:
        """Without dedicated LLM, falls back to main LLM parameter."""
        chunks = _rfc_chunks()
        config = HeadingTreeConfig(
            entry_strategy="heading_match",
            heading_selection="llm",
        )

        # No dedicated LLM
        index = HeadingTreeIndex.from_chunks(chunks, config=config)

        main_llm = EchoProvider(_ECHO_CONFIG)
        main_llm.set_responses([text_response("(A)")])

        results = await index.resolve("CSRF security", llm=main_llm)
        assert main_llm.call_count == 1
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_llm_mode_no_llm_falls_back_to_score(self) -> None:
        """heading_selection='llm' without any LLM falls back to score-based."""
        chunks = _rfc_chunks()
        config = HeadingTreeConfig(
            entry_strategy="heading_match",
            heading_selection="llm",
        )
        index = HeadingTreeIndex.from_chunks(chunks, config=config)

        # No LLM provided at all — should fall back to score-based
        results = await index.resolve("CSRF security")
        assert len(results) > 0


# ------------------------------------------------------------------
# Result limits
# ------------------------------------------------------------------


class TestResultLimits:
    """Tests for configurable result limits."""

    @pytest.mark.asyncio
    async def test_max_expanded_results(self) -> None:
        chunks = _rfc_chunks()
        config = HeadingTreeConfig(
            entry_strategy="heading_match",
            max_expanded_results=3,
        )
        index = HeadingTreeIndex.from_chunks(chunks, config=config)

        results = await index.resolve("security considerations")
        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_seed_score_threshold_filters_weak_seeds(self) -> None:
        """Vector seeds below the threshold are dropped."""
        chunks = _rfc_chunks()

        async def low_score_fn(query: str, top_k: int) -> list[SourceResult]:
            return [
                _chunk("weak", ["10. Security Considerations", "10.3 Token Leakage"], [1, 2],
                       relevance=0.1),  # Below threshold
            ]

        config = HeadingTreeConfig(
            entry_strategy="vector",
            seed_score_threshold=0.3,
        )
        index = HeadingTreeIndex(
            vector_query_fn=low_score_fn, config=config,
        )

        results = await index.resolve("token leakage")
        assert results == []  # Weak seed was filtered out


# ------------------------------------------------------------------
# HeadingTreeConfig
# ------------------------------------------------------------------


class TestHeadingTreeConfig:
    """Tests for configuration construction."""

    def test_from_dict_basic(self) -> None:
        cfg = HeadingTreeConfig.from_dict({
            "entry_strategy": "heading_match",
            "expansion_mode": "children",
            "max_expanded_results": 25,
        })
        assert cfg.entry_strategy == "heading_match"
        assert cfg.expansion_mode == "children"
        assert cfg.max_expanded_results == 25

    def test_from_dict_ignores_unknown_keys(self) -> None:
        cfg = HeadingTreeConfig.from_dict({
            "entry_strategy": "both",
            "unknown_key": "ignored",
        })
        assert cfg.entry_strategy == "both"

    def test_from_dict_nested_heading_match(self) -> None:
        cfg = HeadingTreeConfig.from_dict({
            "heading_match": {
                "stopwords": ["the", "a", "custom"],
                "min_word_length": 3,
                "min_heading_depth": 2,
            },
        })
        assert "custom" in cfg.heading_match.stopwords
        assert cfg.heading_match.min_word_length == 3
        assert cfg.heading_match.min_heading_depth == 2

    def test_from_dict_scope_profiles(self) -> None:
        cfg = HeadingTreeConfig.from_dict({
            "scope_profiles": {
                "focused": {"expansion_mode": "children"},
                "broad": {"max_expansion_depth": None},
            },
        })
        assert cfg.scope_profiles["focused"]["expansion_mode"] == "children"

    def test_defaults(self) -> None:
        cfg = HeadingTreeConfig()
        assert cfg.entry_strategy == "both"
        assert cfg.expansion_mode == "subtree"
        assert cfg.max_expansion_depth is None
        assert cfg.seed_score_threshold == 0.3
        assert cfg.heading_match.stopwords is DEFAULT_HEADING_STOPWORDS


# ------------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------------


class TestParseBracketIndices:
    """Tests for LLM output parsing."""

    def test_basic(self) -> None:
        assert _parse_bracket_indices("[0], [2], [5]", 10) == [0, 2, 5]

    def test_multiline(self) -> None:
        text = "[0]\n[3]\n[7]"
        assert _parse_bracket_indices(text, 10) == [0, 3, 7]

    def test_out_of_range_filtered(self) -> None:
        assert _parse_bracket_indices("[0], [99]", 10) == [0]

    def test_deduplicates(self) -> None:
        assert _parse_bracket_indices("[1], [1], [2]", 5) == [1, 2]

    def test_no_brackets(self) -> None:
        assert _parse_bracket_indices("no indices here", 5) == []

    def test_mixed_text(self) -> None:
        text = "I think sections [2] and [4] are most relevant."
        assert _parse_bracket_indices(text, 5) == [2, 4]


class TestParseLetterIndices:
    """Tests for letter-based LLM output parsing."""

    def test_basic(self) -> None:
        assert _parse_letter_indices("(A), (C), (F)", 10) == [0, 2, 5]

    def test_multiline(self) -> None:
        text = "(A)\n(D)\n(H)"
        assert _parse_letter_indices(text, 10) == [0, 3, 7]

    def test_out_of_range_filtered(self) -> None:
        # Only 5 candidates (A-E), (Z) is out of range
        assert _parse_letter_indices("(A), (Z)", 5) == [0]

    def test_deduplicates(self) -> None:
        assert _parse_letter_indices("(B), (B), (C)", 5) == [1, 2]

    def test_no_letters(self) -> None:
        assert _parse_letter_indices("no IDs here", 5) == []

    def test_mixed_text(self) -> None:
        text = "I think sections (B) and (D) are most relevant."
        assert _parse_letter_indices(text, 5) == [1, 3]

    def test_does_not_confuse_section_numbers(self) -> None:
        """Letter IDs avoid collision with section numbers like '10.'."""
        text = "Section 10 is relevant, selecting (A)"
        assert _parse_letter_indices(text, 5) == [0]


class TestIndexToLetter:
    """Tests for letter ID conversion."""

    def test_basic(self) -> None:
        assert _index_to_letter(0) == "A"
        assert _index_to_letter(25) == "Z"

    def test_double_letter(self) -> None:
        assert _index_to_letter(26) == "AA"
        assert _index_to_letter(27) == "AB"

    def test_roundtrip(self) -> None:
        for i in range(52):
            assert _letter_to_index(_index_to_letter(i)) == i
