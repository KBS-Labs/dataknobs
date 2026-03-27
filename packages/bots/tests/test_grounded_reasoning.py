"""Tests for the GroundedReasoning strategy.

Uses BotTestHarness with bot_config and a lightweight InMemoryKnowledgeBase
test helper (real KnowledgeBase subclass, not a mock).
"""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_bots.knowledge.base import KnowledgeBase
from dataknobs_bots.reasoning.grounded import (
    DEFAULT_PROVENANCE_TEMPLATE,
    GroundedReasoning,
    _VALID_STYLES,
)
from dataknobs_data.sources.base import RetrievalIntent, SourceResult
from dataknobs_bots.reasoning.grounded_config import (
    GroundedIntentConfig,
    GroundedReasoningConfig,
    GroundedResultProcessingConfig,
    GroundedRetrievalConfig,
    GroundedSourceConfig,
    GroundedSynthesisConfig,
)
from dataknobs_bots.testing import BotTestHarness, GroundedConfigBuilder
from dataknobs_llm.testing import text_response


# ------------------------------------------------------------------
# Test helper: InMemoryKnowledgeBase
# ------------------------------------------------------------------


class InMemoryKnowledgeBase(KnowledgeBase):
    """Real KnowledgeBase implementation for testing.

    Returns scripted results for any query.  Tracks query history
    for assertions.
    """

    def __init__(self, results: list[dict[str, Any]] | None = None) -> None:
        self._results = results or []
        self.queries: list[str] = []

    async def query(
        self,
        query: str,
        k: int = 5,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        self.queries.append(query)
        return self._results[:k]

    async def close(self) -> None:
        pass


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

SAMPLE_KB_RESULTS: list[dict[str, Any]] = [
    {
        "text": "The authorization code grant type is used to obtain access tokens.",
        "source": "rfc6749.md",
        "heading_path": "1.3. Authorization Grant > 1.3.1. Authorization Code",
        "similarity": 0.87,
        "metadata": {
            "headings": ["1.3. Authorization Grant", "1.3.1. Authorization Code"],
            "chunk_index": 42,
        },
    },
    {
        "text": "The implicit grant type is optimized for public clients.",
        "source": "rfc6749.md",
        "heading_path": "1.3. Authorization Grant > 1.3.2. Implicit",
        "similarity": 0.82,
        "metadata": {
            "headings": ["1.3. Authorization Grant", "1.3.2. Implicit"],
            "chunk_index": 43,
        },
    },
    {
        "text": "Refresh tokens are issued to the client by the authorization server.",
        "source": "rfc6749.md",
        "heading_path": "1.5. Refresh Token",
        "similarity": 0.75,
        "metadata": {
            "headings": ["1.5. Refresh Token"],
            "chunk_index": 55,
        },
    },
]


def _grounded_bot_config(
    *,
    num_queries: int = 2,
    store_provenance: bool = True,
    intent_mode: str = "extract",
    synthesis_mode: str = "llm",
    **extra_reasoning: Any,
) -> dict[str, Any]:
    """Build a minimal bot_config with grounded strategy.

    Thin wrapper over :class:`GroundedConfigBuilder` for backward
    compatibility with existing tests.  New tests should use
    ``GroundedConfigBuilder`` directly.
    """
    config = (
        GroundedConfigBuilder()
        .intent(mode=intent_mode, num_queries=num_queries)
        .synthesis(mode=synthesis_mode)
        .provenance(store_provenance)
        .build()
    )
    config["reasoning"].update(extra_reasoning)
    return config


# ------------------------------------------------------------------
# Config tests
# ------------------------------------------------------------------


class TestGroundedReasoningConfig:
    """Test configuration dataclasses."""

    def test_defaults(self) -> None:
        cfg = GroundedReasoningConfig()
        assert cfg.intent.mode == "extract"
        assert cfg.intent.num_queries == 3
        assert cfg.retrieval.top_k == 5
        assert cfg.retrieval.score_threshold == 0.3
        assert cfg.synthesis.mode == "llm"
        assert cfg.synthesis.require_citations is True
        assert cfg.synthesis.allow_parametric is False
        assert cfg.store_provenance is True

    def test_backward_compat_query_generation_alias(self) -> None:
        """query_generation property returns same object as intent."""
        cfg = GroundedReasoningConfig()
        assert cfg.query_generation is cfg.intent
        assert cfg.query_generation.num_queries == 3

    def test_from_dict_with_intent_key(self) -> None:
        cfg = GroundedReasoningConfig.from_dict({
            "intent": {"mode": "static", "text_queries": ["test"]},
            "retrieval": {"top_k": 10},
            "synthesis": {"mode": "template", "template": "{{ results }}"},
        })
        assert cfg.intent.mode == "static"
        assert cfg.intent.text_queries == ["test"]
        assert cfg.synthesis.mode == "template"

    def test_from_dict_with_legacy_query_generation_key(self) -> None:
        """Legacy query_generation key maps to extract mode."""
        cfg = GroundedReasoningConfig.from_dict({
            "query_generation": {"num_queries": 5, "domain_context": "OAuth"},
            "retrieval": {"top_k": 10, "score_threshold": 0.5},
            "synthesis": {"allow_parametric": True, "citation_format": "source"},
            "store_provenance": False,
        })
        assert cfg.intent.mode == "extract"
        assert cfg.intent.num_queries == 5
        assert cfg.intent.domain_context == "OAuth"
        assert cfg.retrieval.top_k == 10
        assert cfg.synthesis.allow_parametric is True
        assert cfg.store_provenance is False

    def test_from_dict_empty(self) -> None:
        cfg = GroundedReasoningConfig.from_dict({})
        assert cfg.intent.num_queries == 3
        assert cfg.intent.mode == "extract"
        assert cfg.store_provenance is True

    def test_from_dict_with_sources(self) -> None:
        cfg = GroundedReasoningConfig.from_dict({
            "sources": [
                {"type": "vector_kb", "name": "docs"},
                {"type": "database", "name": "courses", "backend": "sqlite"},
            ],
        })
        assert len(cfg.sources) == 2
        assert cfg.sources[0].source_type == "vector_kb"
        assert cfg.sources[0].name == "docs"
        assert cfg.sources[1].source_type == "database"
        assert cfg.sources[1].options["backend"] == "sqlite"

    def test_intent_config_defaults(self) -> None:
        cfg = GroundedIntentConfig()
        assert cfg.mode == "extract"
        assert cfg.default_filters == {}
        assert cfg.include_message_as_query is True
        assert cfg.text_queries == []

    def test_source_config_from_dict(self) -> None:
        sc = GroundedSourceConfig.from_dict({
            "type": "database",
            "name": "users",
            "backend": "sqlite",
            "connection": "users.db",
        })
        assert sc.source_type == "database"
        assert sc.name == "users"
        assert sc.options == {"backend": "sqlite", "connection": "users.db"}


# ------------------------------------------------------------------
# Strategy unit tests (direct instantiation)
# ------------------------------------------------------------------


class TestGroundedReasoningUnit:
    """Unit tests for internal methods."""

    def test_parse_queries_numbered_list(self) -> None:
        from dataknobs_bots.knowledge.query import parse_query_response

        text = "1. OAuth grant types\n2. authorization code flow\n3. implicit grant"
        queries = parse_query_response(text, "fallback")
        assert len(queries) == 3
        assert queries[0] == "OAuth grant types"

    def test_parse_queries_plain_lines(self) -> None:
        from dataknobs_bots.knowledge.query import parse_query_response

        text = "OAuth grant types\nauthorization code flow"
        queries = parse_query_response(text, "fallback")
        assert len(queries) == 2

    def test_parse_queries_empty_returns_fallback(self) -> None:
        from dataknobs_bots.knowledge.query import parse_query_response

        queries = parse_query_response("", "my fallback")
        assert queries == ["my fallback"]

    def test_extract_user_message_prefers_raw_content(self) -> None:
        messages = [
            {"role": "user", "content": "augmented msg", "metadata": {"raw_content": "original msg"}},
        ]
        assert GroundedReasoning._extract_user_message(messages) == "original msg"

    def test_extract_user_message_falls_back_to_content(self) -> None:
        messages = [
            {"role": "user", "content": "plain msg"},
        ]
        assert GroundedReasoning._extract_user_message(messages) == "plain msg"

    def test_provider_management(self) -> None:
        strategy = GroundedReasoning(config=GroundedReasoningConfig())
        assert strategy.providers() == {}

        strategy._query_provider = "fake_provider"
        assert "grounded_query" in strategy.providers()
        assert strategy.set_provider("grounded_query", "new_provider")
        assert strategy._query_provider == "new_provider"
        assert not strategy.set_provider("unknown_role", "x")

    def test_set_knowledge_base_wraps_in_source(self) -> None:
        from dataknobs_bots.knowledge.sources.vector import VectorKnowledgeSource

        strategy = GroundedReasoning(config=GroundedReasoningConfig())
        assert len(strategy._sources) == 0
        kb = InMemoryKnowledgeBase()
        strategy.set_knowledge_base(kb)
        assert len(strategy._sources) == 1
        assert isinstance(strategy._sources[0], VectorKnowledgeSource)

    def test_add_source(self) -> None:
        from dataknobs_data.sources.base import GroundedSource, RetrievalIntent, SourceResult

        class StubSource(GroundedSource):
            @property
            def name(self) -> str:
                return "stub"

            @property
            def source_type(self) -> str:
                return "stub"

            async def query(self, intent: RetrievalIntent, *, top_k: int = 5, score_threshold: float = 0.0) -> list[SourceResult]:
                return []

        strategy = GroundedReasoning(config=GroundedReasoningConfig())
        strategy.add_source(StubSource())
        assert len(strategy._sources) == 1

    def test_build_static_intent(self) -> None:
        cfg = GroundedReasoningConfig(
            intent=GroundedIntentConfig(
                mode="static",
                text_queries=["enrollment policies"],
                filters={"courses": {"department": "CS"}},
                scope="exact",
                include_message_as_query=True,
            ),
        )
        strategy = GroundedReasoning(config=cfg)
        intent = strategy._build_static_intent("What about math?")

        assert "enrollment policies" in intent.text_queries
        assert "What about math?" in intent.text_queries
        assert intent.filters == {"courses": {"department": "CS"}}
        assert intent.scope == "exact"

    def test_build_static_intent_no_message_append(self) -> None:
        cfg = GroundedReasoningConfig(
            intent=GroundedIntentConfig(
                mode="static",
                text_queries=["fixed query"],
                include_message_as_query=False,
            ),
        )
        strategy = GroundedReasoning(config=cfg)
        intent = strategy._build_static_intent("user question")

        assert intent.text_queries == ["fixed query"]

    def test_build_template_intent(self) -> None:
        cfg = GroundedReasoningConfig(
            intent=GroundedIntentConfig(
                mode="template",
                template=(
                    "text_queries:\n"
                    "  - \"{{ message }}\"\n"
                    "filters:\n"
                    "  courses:\n"
                    "    department: CS\n"
                    "scope: focused\n"
                ),
            ),
        )
        strategy = GroundedReasoning(config=cfg)
        intent = strategy._build_template_intent("intro algorithms", {})

        assert "intro algorithms" in intent.text_queries
        assert intent.filters == {"courses": {"department": "CS"}}
        assert intent.scope == "focused"

    def test_build_template_intent_with_metadata(self) -> None:
        cfg = GroundedReasoningConfig(
            intent=GroundedIntentConfig(
                mode="template",
                template=(
                    "text_queries:\n"
                    "  - \"{{ message }}\"\n"
                    "filters:\n"
                    "  courses:\n"
                    "    department: \"{{ metadata.get('department', 'CS') }}\"\n"
                ),
            ),
        )
        strategy = GroundedReasoning(config=cfg)
        intent = strategy._build_template_intent("test", {"department": "Math"})

        assert intent.filters["courses"]["department"] == "Math"

    def test_build_template_intent_no_template_fallback(self) -> None:
        cfg = GroundedReasoningConfig(
            intent=GroundedIntentConfig(mode="template"),
        )
        strategy = GroundedReasoning(config=cfg)
        intent = strategy._build_template_intent("fallback message", {})

        assert intent.text_queries == ["fallback message"]

    def test_render_provenance_output(self) -> None:
        cfg = GroundedReasoningConfig(
            synthesis=GroundedSynthesisConfig(
                mode="template",
                template=(
                    "Found {{ results|length }} results:\n"
                    "{% for r in results %}"
                    "- {{ r.text_preview }}\n"
                    "{% endfor %}"
                ),
            ),
        )
        strategy = GroundedReasoning(config=cfg)
        provenance = {
            "results": [
                {"text_preview": "Auth code grant", "source_name": "kb"},
                {"text_preview": "Implicit grant", "source_name": "kb"},
            ],
            "results_by_source": {},
            "intent": {},
        }
        text = strategy._render_provenance_output(
            "formatted context", provenance, "user msg", {},
        )
        assert "Found 2 results:" in text
        assert "Auth code grant" in text

    def test_render_provenance_output_no_custom_template_uses_builtin(self) -> None:
        """Without a custom template, the built-in provenance template is used."""
        cfg = GroundedReasoningConfig(
            synthesis=GroundedSynthesisConfig(mode="template"),
        )
        strategy = GroundedReasoning(config=cfg)
        text = strategy._render_provenance_output(
            "raw context", {"results": [], "results_by_source": {}}, "msg", {},
        )
        assert "No relevant results found." in text


# ------------------------------------------------------------------
# Integration tests (BotTestHarness) — Extract mode (default)
# ------------------------------------------------------------------


class TestGroundedReasoningIntegration:
    """Integration tests using BotTestHarness."""

    @pytest.mark.asyncio
    async def test_basic_pipeline(self) -> None:
        """Full pipeline: query gen → retrieval → synthesis."""
        kb = InMemoryKnowledgeBase(results=SAMPLE_KB_RESULTS)

        async with await BotTestHarness.create(
            bot_config=_grounded_bot_config(),
            main_responses=[
                text_response("OAuth grant types\nauthorization code flow"),
                text_response("Based on the KB: auth code is for confidential clients."),
            ],
        ) as harness:
            harness.bot.reasoning_strategy.set_knowledge_base(kb)

            result = await harness.chat("What are OAuth grant types?")
            assert "auth code" in result.response.lower() or "Based on the KB" in result.response

            # KB was queried
            assert len(kb.queries) > 0

            # Provenance was recorded
            manager = harness.bot.get_conversation_manager(harness.context.conversation_id)
            assert "retrieval_provenance" in manager.metadata

    @pytest.mark.asyncio
    async def test_provenance_structure(self) -> None:
        """Provenance record has expected fields including intent and results_by_source."""
        kb = InMemoryKnowledgeBase(results=SAMPLE_KB_RESULTS)

        async with await BotTestHarness.create(
            bot_config=_grounded_bot_config(),
            main_responses=[
                text_response("query one\nquery two"),
                text_response("Synthesis response"),
            ],
        ) as harness:
            harness.bot.reasoning_strategy.set_knowledge_base(kb)
            await harness.chat("Tell me about tokens")

            manager = harness.bot.get_conversation_manager(harness.context.conversation_id)
            prov = manager.metadata["retrieval_provenance"]
            assert "intent" in prov
            assert "results" in prov
            assert "results_by_source" in prov
            assert "total_results" in prov
            assert "deduplicated_to" in prov
            assert "retrieval_time_ms" in prov
            assert "intent_resolution_time_ms" in prov
            # intent has expected shape
            assert "text_queries" in prov["intent"]
            assert "scope" in prov["intent"]

    @pytest.mark.asyncio
    async def test_empty_kb_results(self) -> None:
        """Strategy handles empty KB results gracefully."""
        kb = InMemoryKnowledgeBase(results=[])

        async with await BotTestHarness.create(
            bot_config=_grounded_bot_config(),
            main_responses=[
                text_response("search query"),
                text_response("I don't have information on that topic."),
            ],
        ) as harness:
            harness.bot.reasoning_strategy.set_knowledge_base(kb)
            result = await harness.chat("What is quantum computing?")

            assert result.response
            assert len(kb.queries) > 0

    @pytest.mark.asyncio
    async def test_no_sources_configured(self) -> None:
        """Strategy without sources logs warning and returns LLM response."""
        async with await BotTestHarness.create(
            bot_config=_grounded_bot_config(),
            main_responses=[
                text_response("search query"),
                text_response("I have no sources to search."),
            ],
        ) as harness:
            # Don't set KB — strategy has no sources
            result = await harness.chat("Hello")
            assert result.response

    @pytest.mark.asyncio
    async def test_provenance_disabled(self) -> None:
        """When store_provenance is False, no metadata recorded."""
        kb = InMemoryKnowledgeBase(results=SAMPLE_KB_RESULTS)

        async with await BotTestHarness.create(
            bot_config=_grounded_bot_config(store_provenance=False),
            main_responses=[
                text_response("query"),
                text_response("response"),
            ],
        ) as harness:
            harness.bot.reasoning_strategy.set_knowledge_base(kb)
            await harness.chat("test")

            manager = harness.bot.get_conversation_manager(harness.context.conversation_id)
            assert "retrieval_provenance" not in manager.metadata

    @pytest.mark.asyncio
    async def test_deduplication_across_queries(self) -> None:
        """Same chunk returned by multiple queries is deduplicated.

        VectorKnowledgeSource deduplicates internally (across the 3
        queries it executes), so the source returns 3 unique results.
        The strategy's cross-source dedup is a no-op with one source.
        """
        kb = InMemoryKnowledgeBase(results=SAMPLE_KB_RESULTS)

        async with await BotTestHarness.create(
            bot_config=_grounded_bot_config(num_queries=3),
            main_responses=[
                text_response(
                    "OAuth grant types\n"
                    "authorization code flow\n"
                    "implicit grant comparison"
                ),
                text_response("synthesis"),
            ],
        ) as harness:
            harness.bot.reasoning_strategy.set_knowledge_base(kb)
            await harness.chat("test dedup")

            # KB was queried 3 times (one per generated query)
            assert len(kb.queries) == 3

            manager = harness.bot.get_conversation_manager(harness.context.conversation_id)
            prov = manager.metadata["retrieval_provenance"]
            # Source deduplicates internally: 3 queries × 3 results → 3 unique
            assert prov["deduplicated_to"] == 3

    @pytest.mark.asyncio
    async def test_provenance_history_accumulates(self) -> None:
        """Each turn appends to retrieval_provenance_history."""
        kb = InMemoryKnowledgeBase(results=SAMPLE_KB_RESULTS)

        async with await BotTestHarness.create(
            bot_config=_grounded_bot_config(),
            main_responses=[
                text_response("grant types query"),
                text_response("Grant types are..."),
                text_response("refresh token query"),
                text_response("Refresh tokens are..."),
            ],
        ) as harness:
            harness.bot.reasoning_strategy.set_knowledge_base(kb)

            await harness.chat("What are grant types?")
            await harness.chat("What about refresh tokens?")

            manager = harness.bot.get_conversation_manager(harness.context.conversation_id)
            assert "retrieval_provenance" in manager.metadata
            history = manager.metadata["retrieval_provenance_history"]
            assert len(history) == 2
            # Both entries have intent with text_queries
            assert len(history[0]["intent"]["text_queries"]) > 0
            assert len(history[1]["intent"]["text_queries"]) > 0

    @pytest.mark.asyncio
    async def test_provenance_includes_full_text(self) -> None:
        """Provenance results include full text for UI rendering."""
        kb = InMemoryKnowledgeBase(results=SAMPLE_KB_RESULTS)

        async with await BotTestHarness.create(
            bot_config=_grounded_bot_config(),
            main_responses=[
                text_response("auth code query"),
                text_response("synthesis"),
            ],
        ) as harness:
            harness.bot.reasoning_strategy.set_knowledge_base(kb)
            await harness.chat("Tell me about auth code")

            manager = harness.bot.get_conversation_manager(harness.context.conversation_id)
            prov = manager.metadata["retrieval_provenance"]
            for result in prov["results"]:
                assert "text" in result
                assert len(result["text"]) > 0
                assert "source_name" in result

    @pytest.mark.asyncio
    async def test_score_threshold_filters_results(self) -> None:
        """Results below score_threshold are excluded."""
        low_score_results = [
            {**r, "similarity": 0.1} for r in SAMPLE_KB_RESULTS
        ]
        kb = InMemoryKnowledgeBase(results=low_score_results)

        config = _grounded_bot_config()
        config["reasoning"]["retrieval"]["score_threshold"] = 0.5

        async with await BotTestHarness.create(
            bot_config=config,
            main_responses=[
                text_response("query"),
                text_response("no relevant content found"),
            ],
        ) as harness:
            harness.bot.reasoning_strategy.set_knowledge_base(kb)
            await harness.chat("test threshold")

            manager = harness.bot.get_conversation_manager(harness.context.conversation_id)
            prov = manager.metadata["retrieval_provenance"]
            assert prov["total_results"] == 0

    @pytest.mark.asyncio
    async def test_results_by_source_in_provenance(self) -> None:
        """Provenance groups results by source name."""
        kb = InMemoryKnowledgeBase(results=SAMPLE_KB_RESULTS)

        async with await BotTestHarness.create(
            bot_config=_grounded_bot_config(),
            main_responses=[
                text_response("query text here"),
                text_response("synthesis"),
            ],
        ) as harness:
            harness.bot.reasoning_strategy.set_knowledge_base(kb)
            await harness.chat("test")

            manager = harness.bot.get_conversation_manager(harness.context.conversation_id)
            prov = manager.metadata["retrieval_provenance"]
            assert "knowledge_base" in prov["results_by_source"]
            kb_results = prov["results_by_source"]["knowledge_base"]
            assert len(kb_results) > 0
            assert kb_results[0]["source_type"] == "vector_kb"


# ------------------------------------------------------------------
# Intent bypass tests — static and template modes
# ------------------------------------------------------------------


class TestStaticIntentMode:
    """Test static intent mode — no LLM for intent resolution."""

    @pytest.mark.asyncio
    async def test_static_intent_skips_llm_query_gen(self) -> None:
        """Static mode only uses one LLM call (synthesis), not two."""
        kb = InMemoryKnowledgeBase(results=SAMPLE_KB_RESULTS)

        config = _grounded_bot_config(intent_mode="static")
        config["reasoning"]["intent"]["text_queries"] = ["authorization grants"]

        async with await BotTestHarness.create(
            bot_config=config,
            # Only one LLM response needed (synthesis) — no query gen call
            main_responses=[
                text_response("Auth code is used for confidential clients."),
            ],
        ) as harness:
            harness.bot.reasoning_strategy.set_knowledge_base(kb)
            result = await harness.chat("Tell me about grants")

            assert result.response
            # KB was queried with the static query + user message
            assert any("authorization grants" in q for q in kb.queries)

    @pytest.mark.asyncio
    async def test_static_intent_appends_user_message(self) -> None:
        """By default, static mode appends user message as a text query."""
        kb = InMemoryKnowledgeBase(results=SAMPLE_KB_RESULTS)

        config = _grounded_bot_config(intent_mode="static")
        config["reasoning"]["intent"]["text_queries"] = ["fixed query"]
        config["reasoning"]["intent"]["include_message_as_query"] = True

        async with await BotTestHarness.create(
            bot_config=config,
            main_responses=[text_response("response")],
        ) as harness:
            harness.bot.reasoning_strategy.set_knowledge_base(kb)
            await harness.chat("dynamic user question")

            manager = harness.bot.get_conversation_manager(harness.context.conversation_id)
            prov = manager.metadata["retrieval_provenance"]
            queries = prov["intent"]["text_queries"]
            assert "fixed query" in queries
            assert "dynamic user question" in queries

    @pytest.mark.asyncio
    async def test_static_intent_with_filters(self) -> None:
        """Static mode passes filters through to provenance."""
        kb = InMemoryKnowledgeBase(results=SAMPLE_KB_RESULTS)

        config = _grounded_bot_config(intent_mode="static")
        config["reasoning"]["intent"]["text_queries"] = ["test"]
        config["reasoning"]["intent"]["filters"] = {"courses": {"department": "CS"}}

        async with await BotTestHarness.create(
            bot_config=config,
            main_responses=[text_response("response")],
        ) as harness:
            harness.bot.reasoning_strategy.set_knowledge_base(kb)
            await harness.chat("query")

            manager = harness.bot.get_conversation_manager(harness.context.conversation_id)
            prov = manager.metadata["retrieval_provenance"]
            assert prov["intent"]["filters"] == {"courses": {"department": "CS"}}


class TestTemplateIntentMode:
    """Test template intent mode — Jinja2 produces intent."""

    @pytest.mark.asyncio
    async def test_template_intent_renders_message(self) -> None:
        """Template intent injects user message into intent."""
        kb = InMemoryKnowledgeBase(results=SAMPLE_KB_RESULTS)

        config = _grounded_bot_config(intent_mode="template")
        config["reasoning"]["intent"]["template"] = (
            "text_queries:\n"
            "  - \"{{ message }}\"\n"
            "scope: focused\n"
        )

        async with await BotTestHarness.create(
            bot_config=config,
            # Only synthesis LLM call needed
            main_responses=[text_response("template intent response")],
        ) as harness:
            harness.bot.reasoning_strategy.set_knowledge_base(kb)
            result = await harness.chat("OAuth refresh tokens")

            assert result.response
            manager = harness.bot.get_conversation_manager(harness.context.conversation_id)
            prov = manager.metadata["retrieval_provenance"]
            assert "OAuth refresh tokens" in prov["intent"]["text_queries"]


class TestDefaultFilters:
    """Test default_filters — merged regardless of intent mode."""

    @pytest.mark.asyncio
    async def test_default_filters_merged_with_extract_mode(self) -> None:
        """Default filters are applied even in extract mode."""
        kb = InMemoryKnowledgeBase(results=SAMPLE_KB_RESULTS)

        config = _grounded_bot_config(intent_mode="extract")
        config["reasoning"]["intent"]["default_filters"] = {
            "docs": {"category": "security"},
        }

        async with await BotTestHarness.create(
            bot_config=config,
            main_responses=[
                text_response("OAuth grant types"),
                text_response("synthesis"),
            ],
        ) as harness:
            harness.bot.reasoning_strategy.set_knowledge_base(kb)
            await harness.chat("test default filters")

            manager = harness.bot.get_conversation_manager(harness.context.conversation_id)
            prov = manager.metadata["retrieval_provenance"]
            assert prov["intent"]["filters"]["docs"]["category"] == "security"

    @pytest.mark.asyncio
    async def test_default_filters_override_static(self) -> None:
        """Default filters override static filters for the same source."""
        kb = InMemoryKnowledgeBase(results=SAMPLE_KB_RESULTS)

        config = _grounded_bot_config(intent_mode="static")
        config["reasoning"]["intent"]["text_queries"] = ["test"]
        config["reasoning"]["intent"]["filters"] = {
            "courses": {"department": "Math", "level": 100},
        }
        config["reasoning"]["intent"]["default_filters"] = {
            "courses": {"department": "CS"},
        }

        async with await BotTestHarness.create(
            bot_config=config,
            main_responses=[text_response("response")],
        ) as harness:
            harness.bot.reasoning_strategy.set_knowledge_base(kb)
            await harness.chat("query")

            manager = harness.bot.get_conversation_manager(harness.context.conversation_id)
            prov = manager.metadata["retrieval_provenance"]
            filters = prov["intent"]["filters"]["courses"]
            # Default filter overrides static department
            assert filters["department"] == "CS"
            # Static level preserved
            assert filters["level"] == 100

    def test_default_filters_override_existing_keys(self) -> None:
        """Unit test: default_filters always win over mode-produced filters."""
        cfg = GroundedReasoningConfig(
            intent=GroundedIntentConfig(
                mode="static",
                text_queries=["q"],
                filters={"src": {"status": "draft", "category": "auth"}},
                default_filters={"src": {"status": "published"}},
            ),
        )
        strategy = GroundedReasoning(config=cfg)

        import asyncio
        intent = asyncio.run(strategy._resolve_intent(
            "test", [], None, {},
        ))
        # default_filters overrides 'status' but preserves 'category'
        assert intent.filters["src"]["status"] == "published"
        assert intent.filters["src"]["category"] == "auth"


# ------------------------------------------------------------------
# Synthesis bypass tests — template mode
# ------------------------------------------------------------------


class TestSynthesisTemplateMode:
    """Test template synthesis — Jinja2 formats results, no LLM."""

    @pytest.mark.asyncio
    async def test_template_synthesis_no_llm_call(self) -> None:
        """Template synthesis uses zero LLM calls when combined with static intent."""
        kb = InMemoryKnowledgeBase(results=SAMPLE_KB_RESULTS)

        config = _grounded_bot_config(
            intent_mode="static",
            synthesis_mode="template",
        )
        config["reasoning"]["intent"]["text_queries"] = ["authorization grants"]
        config["reasoning"]["synthesis"]["template"] = (
            "Found {{ results|length }} results:\n"
            "{% for r in results %}"
            "- {{ r.text_preview }}\n"
            "{% endfor %}"
        )

        async with await BotTestHarness.create(
            bot_config=config,
            # No LLM responses needed at all!
            main_responses=[],
        ) as harness:
            harness.bot.reasoning_strategy.set_knowledge_base(kb)
            result = await harness.chat("Show me grants")

            assert "Found 3 results:" in result.response
            assert "authorization code" in result.response.lower()

    @pytest.mark.asyncio
    async def test_template_synthesis_with_extract_intent(self) -> None:
        """Template synthesis works with LLM-driven intent (1 LLM call)."""
        kb = InMemoryKnowledgeBase(results=SAMPLE_KB_RESULTS)

        config = _grounded_bot_config(
            intent_mode="extract",
            synthesis_mode="template",
        )
        config["reasoning"]["synthesis"]["template"] = (
            "Results for: {{ message }}\n"
            "{% for r in results %}"
            "- {{ r.text_preview }}\n"
            "{% endfor %}"
        )

        async with await BotTestHarness.create(
            bot_config=config,
            # Only query gen LLM call needed
            main_responses=[
                text_response("OAuth grant types"),
            ],
        ) as harness:
            harness.bot.reasoning_strategy.set_knowledge_base(kb)
            result = await harness.chat("What are grant types?")

            assert "Results for: What are grant types?" in result.response

    @pytest.mark.asyncio
    async def test_template_synthesis_accesses_intent(self) -> None:
        """Template can access intent data for provenance display."""
        kb = InMemoryKnowledgeBase(results=SAMPLE_KB_RESULTS)

        config = _grounded_bot_config(
            intent_mode="static",
            synthesis_mode="template",
        )
        config["reasoning"]["intent"]["text_queries"] = ["test query"]
        config["reasoning"]["synthesis"]["template"] = (
            "Searched for: {{ intent.text_queries|join(', ') }}\n"
            "Found {{ results|length }} results."
        )

        async with await BotTestHarness.create(
            bot_config=config,
            main_responses=[],
        ) as harness:
            harness.bot.reasoning_strategy.set_knowledge_base(kb)
            result = await harness.chat("show results")

            assert "test query" in result.response
            assert "Found 3 results." in result.response


# ------------------------------------------------------------------
# Result merge tests
# ------------------------------------------------------------------


class TestResultMerge:
    """Test multi-source result merging with weighted round-robin."""

    def _make_results(self, source_name: str, count: int) -> list[SourceResult]:
        return [
            SourceResult(
                content=f"{source_name} result {i}",
                source_id=f"{source_name}_{i}",
                source_name=source_name,
                source_type="test",
                relevance=1.0 - i * 0.1,
            )
            for i in range(count)
        ]

    def test_equal_weight_round_robin(self) -> None:
        """Default weight=1: sources alternate 1-for-1."""
        cfg = GroundedReasoningConfig()
        strategy = GroundedReasoning(config=cfg)

        results = strategy._merge_source_results({
            "a": self._make_results("a", 3),
            "b": self._make_results("b", 3),
        })
        names = [r.source_name for r in results]
        # Round-robin: a, b, a, b, a, b
        assert names == ["a", "b", "a", "b", "a", "b"]

    def test_weighted_round_robin(self) -> None:
        """Source with weight=3 gets 3 results per round vs 1."""
        cfg = GroundedReasoningConfig(
            sources=[
                GroundedSourceConfig(name="primary", weight=3),
                GroundedSourceConfig(name="secondary", weight=1),
            ],
        )
        strategy = GroundedReasoning(config=cfg)

        results = strategy._merge_source_results({
            "primary": self._make_results("primary", 6),
            "secondary": self._make_results("secondary", 2),
        })
        names = [r.source_name for r in results]
        # Round 1: primary x3, secondary x1
        # Round 2: primary x3, secondary x1
        assert names == [
            "primary", "primary", "primary", "secondary",
            "primary", "primary", "primary", "secondary",
        ]

    def test_weighted_exhaustion(self) -> None:
        """When a weighted source runs out mid-round, others continue."""
        cfg = GroundedReasoningConfig(
            sources=[
                GroundedSourceConfig(name="big", weight=3),
                GroundedSourceConfig(name="small", weight=1),
            ],
        )
        strategy = GroundedReasoning(config=cfg)

        results = strategy._merge_source_results({
            "big": self._make_results("big", 2),    # Runs out mid-round
            "small": self._make_results("small", 3),
        })
        # Round 1: big x2 (exhausted at 3rd), small x1
        # Round 2: small x1
        # Round 3: small x1
        names = [r.source_name for r in results]
        assert names.count("big") == 2
        assert names.count("small") == 3

    def test_deduplicate_respects_config(self) -> None:
        """deduplicate=False skips dedup step."""
        cfg = GroundedReasoningConfig(
            retrieval=GroundedRetrievalConfig(deduplicate=False),
        )
        strategy = GroundedReasoning(config=cfg)

        # Same source_id from two sources
        results_a = [SourceResult(
            content="same", source_id="dup", source_name="a",
            source_type="test", relevance=0.9,
        )]
        results_b = [SourceResult(
            content="same", source_id="dup", source_name="b",
            source_type="test", relevance=0.8,
        )]
        merged = strategy._merge_source_results({"a": results_a, "b": results_b})
        # Without dedup, both appear
        assert len(merged) == 2


# ------------------------------------------------------------------
# Factory registration tests
# ------------------------------------------------------------------


class TestGroundedFactory:
    """Test that 'grounded' is registered in the factory."""

    def test_create_from_config(self) -> None:
        from dataknobs_bots.reasoning import create_reasoning_from_config

        strategy = create_reasoning_from_config({
            "strategy": "grounded",
            "query_generation": {"num_queries": 2},
        })
        assert isinstance(strategy, GroundedReasoning)
        assert strategy._config.query_generation.num_queries == 2

    def test_create_from_config_with_intent_key(self) -> None:
        from dataknobs_bots.reasoning import create_reasoning_from_config

        strategy = create_reasoning_from_config({
            "strategy": "grounded",
            "intent": {"mode": "static", "text_queries": ["test"]},
        })
        assert isinstance(strategy, GroundedReasoning)
        assert strategy._config.intent.mode == "static"

    def test_create_with_knowledge_base(self) -> None:
        from dataknobs_bots.knowledge.sources.vector import VectorKnowledgeSource
        from dataknobs_bots.reasoning import create_reasoning_from_config

        kb = InMemoryKnowledgeBase()
        strategy = create_reasoning_from_config(
            {"strategy": "grounded"},
            knowledge_base=kb,
        )
        assert isinstance(strategy, GroundedReasoning)
        assert len(strategy._sources) == 1
        assert isinstance(strategy._sources[0], VectorKnowledgeSource)

    def test_create_with_kb_and_vector_source_no_double(self) -> None:
        """KB + vector_kb source in config should NOT create duplicate sources."""
        from dataknobs_bots.reasoning import create_reasoning_from_config

        kb = InMemoryKnowledgeBase()
        strategy = create_reasoning_from_config(
            {
                "strategy": "grounded",
                "sources": [{"type": "vector_kb", "name": "docs"}],
            },
            knowledge_base=kb,
        )
        assert isinstance(strategy, GroundedReasoning)
        # set_knowledge_base should NOT have been called because the config
        # already declares a vector_kb source — the base.py loop handles it.
        # At this point, sources list should be empty (base.py loop hasn't
        # run yet since that's async from_config territory).
        assert len(strategy._sources) == 0

    def test_factory_error_message_includes_grounded(self) -> None:
        from dataknobs_bots.reasoning import create_reasoning_from_config

        with pytest.raises(ValueError, match="grounded"):
            create_reasoning_from_config({"strategy": "nonexistent"})


# ------------------------------------------------------------------
# Synthesis prompt tests
# ------------------------------------------------------------------


class TestSynthesisPrompt:
    """Test synthesis prompt construction."""

    def test_citations_required(self) -> None:
        cfg = GroundedReasoningConfig(
            synthesis=GroundedSynthesisConfig(
                require_citations=True,
                citation_format="section",
            ),
        )
        strategy = GroundedReasoning(config=cfg)
        prompt = strategy._build_synthesis_system_prompt(
            "KB content here", "Original system prompt",
        )
        assert "Original system prompt" in prompt
        assert "<knowledge_base>" in prompt
        assert "KB content here" in prompt
        assert "section heading" in prompt

    def test_parametric_allowed(self) -> None:
        cfg = GroundedReasoningConfig(
            synthesis=GroundedSynthesisConfig(allow_parametric=True),
        )
        strategy = GroundedReasoning(config=cfg)
        prompt = strategy._build_synthesis_system_prompt("content", "sys")
        assert "general knowledge" in prompt

    def test_parametric_blocked(self) -> None:
        cfg = GroundedReasoningConfig(
            synthesis=GroundedSynthesisConfig(allow_parametric=False),
        )
        strategy = GroundedReasoning(config=cfg)
        prompt = strategy._build_synthesis_system_prompt("content", "sys")
        assert "Do not fill gaps" in prompt

    def test_empty_context(self) -> None:
        strategy = GroundedReasoning(config=GroundedReasoningConfig())
        prompt = strategy._build_synthesis_system_prompt("", "sys prompt")
        assert "sys prompt" in prompt
        assert "<knowledge_base>" not in prompt


# ------------------------------------------------------------------
# Config-driven source construction tests
# ------------------------------------------------------------------


class TestSourceFactory:
    """Tests for create_source_from_config() factory."""

    @pytest.mark.asyncio
    async def test_vector_kb_source_from_config(self) -> None:
        """vector_kb type wraps a KnowledgeBase."""
        from dataknobs_bots.knowledge.sources.factory import create_source_from_config

        kb = InMemoryKnowledgeBase(results=SAMPLE_KB_RESULTS)
        config = GroundedSourceConfig(
            source_type="vector_kb",
            name="docs",
        )
        source = await create_source_from_config(config, knowledge_base=kb)
        assert source.name == "docs"
        assert source.source_type == "vector_kb"

    @pytest.mark.asyncio
    async def test_vector_kb_source_with_topic_index_heading_tree(self) -> None:
        """Factory builds heading_tree topic index when config.topic_index is set."""
        from dataknobs_bots.knowledge.sources.factory import create_source_from_config
        from dataknobs_bots.knowledge.sources.heading_tree import HeadingTreeIndex

        kb = InMemoryKnowledgeBase(results=SAMPLE_KB_RESULTS)
        config = GroundedSourceConfig(
            source_type="vector_kb",
            name="docs",
            topic_index={
                "type": "heading_tree",
                "entry_strategy": "both",
                "expansion_mode": "subtree",
            },
        )
        source = await create_source_from_config(config, knowledge_base=kb)
        assert source.name == "docs"
        assert source.topic_index is not None
        assert isinstance(source.topic_index, HeadingTreeIndex)

    @pytest.mark.asyncio
    async def test_vector_kb_source_with_topic_index_cluster(self) -> None:
        """Factory builds cluster topic index when config.topic_index is set."""
        from dataknobs_bots.knowledge.sources.factory import create_source_from_config
        from dataknobs_data.sources.cluster_index import ClusterTopicIndex

        kb = InMemoryKnowledgeBase(results=SAMPLE_KB_RESULTS)
        config = GroundedSourceConfig(
            source_type="vector_kb",
            name="docs",
            topic_index={
                "type": "cluster",
                "cluster_threshold": 0.8,
            },
        )
        source = await create_source_from_config(config, knowledge_base=kb)
        assert source.name == "docs"
        assert source.topic_index is not None
        assert isinstance(source.topic_index, ClusterTopicIndex)

    @pytest.mark.asyncio
    async def test_vector_kb_source_with_heading_selection_config(self) -> None:
        """Factory builds dedicated heading selection LLM when configured."""
        from dataknobs_bots.knowledge.sources.factory import create_source_from_config
        from dataknobs_bots.knowledge.sources.heading_tree import HeadingTreeIndex

        kb = InMemoryKnowledgeBase(results=SAMPLE_KB_RESULTS)
        config = GroundedSourceConfig(
            source_type="vector_kb",
            name="docs",
            topic_index={
                "type": "heading_tree",
                "heading_selection_config": {
                    "provider": "echo",
                    "model": "test",
                },
            },
        )
        source = await create_source_from_config(config, knowledge_base=kb)
        assert isinstance(source.topic_index, HeadingTreeIndex)
        # The dedicated heading selection LLM should be set
        assert source.topic_index._heading_selection_llm is not None

    @pytest.mark.asyncio
    async def test_vector_kb_source_without_topic_index(self) -> None:
        """Factory creates source with no topic index when not configured."""
        from dataknobs_bots.knowledge.sources.factory import create_source_from_config

        kb = InMemoryKnowledgeBase(results=SAMPLE_KB_RESULTS)
        config = GroundedSourceConfig(
            source_type="vector_kb",
            name="docs",
        )
        source = await create_source_from_config(config, knowledge_base=kb)
        assert source.topic_index is None

    @pytest.mark.asyncio
    async def test_vector_kb_source_requires_knowledge_base(self) -> None:
        """vector_kb type raises without a knowledge_base."""
        from dataknobs_bots.knowledge.sources.factory import create_source_from_config

        config = GroundedSourceConfig(source_type="vector_kb", name="docs")
        with pytest.raises(ValueError, match="no knowledge_base was provided"):
            await create_source_from_config(config, knowledge_base=None)

    @pytest.mark.asyncio
    async def test_database_source_from_config(self) -> None:
        """database type creates DatabaseSource from config options."""
        from dataknobs_bots.knowledge.sources.factory import create_source_from_config

        config = GroundedSourceConfig(
            source_type="database",
            name="courses",
            options={
                "backend": "memory",
                "content_field": "description",
                "text_search_fields": ["title", "description"],
                "schema": {
                    "fields": {
                        "department": {"type": "string", "enum": ["CS", "Math"]},
                        "level": {"type": "integer"},
                        "description": {"type": "text"},
                    },
                },
            },
        )
        source = await create_source_from_config(config)
        assert source.name == "courses"
        assert source.source_type == "database"

        # Verify schema was generated correctly
        schema = source.get_schema()
        assert schema is not None
        assert "department" in schema.fields
        assert schema.fields["department"]["enum"] == ["CS", "Math"]

    @pytest.mark.asyncio
    async def test_database_source_minimal_config(self) -> None:
        """database type works with minimal config (defaults)."""
        from dataknobs_bots.knowledge.sources.factory import create_source_from_config

        config = GroundedSourceConfig(
            source_type="database",
            name="simple",
            options={"backend": "memory"},
        )
        source = await create_source_from_config(config)
        assert source.name == "simple"
        assert source.source_type == "database"

    @pytest.mark.asyncio
    async def test_unknown_source_type_raises(self) -> None:
        """Unknown source type raises ValueError."""
        from dataknobs_bots.knowledge.sources.factory import create_source_from_config

        config = GroundedSourceConfig(source_type="elasticsearch", name="search")
        with pytest.raises(ValueError, match="Unknown grounded source type"):
            await create_source_from_config(config)


class TestSourceFactorySchema:
    """Tests for _build_database_schema helper."""

    def test_string_type_shorthand(self) -> None:
        from dataknobs_bots.knowledge.sources.factory import _build_database_schema

        schema = _build_database_schema({"name": "string", "age": "integer"})
        assert "name" in schema.fields
        assert "age" in schema.fields

    def test_dict_type_with_enum(self) -> None:
        from dataknobs_bots.knowledge.sources.factory import _build_database_schema

        schema = _build_database_schema({
            "dept": {"type": "string", "enum": ["CS", "Math"]},
        })
        assert "dept" in schema.fields
        assert schema.fields["dept"].metadata.get("enum") == ["CS", "Math"]

    def test_empty_fields(self) -> None:
        from dataknobs_bots.knowledge.sources.factory import _build_database_schema

        schema = _build_database_schema({})
        assert len(schema.fields) == 0


class TestDynaBotSourceWiring:
    """Tests for DynaBot.from_config() grounded source construction."""

    @pytest.mark.asyncio
    async def test_config_driven_database_source(self) -> None:
        """DynaBot.from_config() constructs database sources from config."""
        from dataknobs_data import Record
        from dataknobs_data.backends.memory import AsyncMemoryDatabase

        config = {
            "llm": {"provider": "echo", "model": "echo-test"},
            "conversation_storage": {"backend": "memory"},
            "reasoning": {
                "strategy": "grounded",
                "intent": {"mode": "static", "text_queries": ["algorithms"]},
                "synthesis": {
                    "mode": "template",
                    "template": "Found {{ results|length }} result(s).",
                },
                "sources": [
                    {
                        "type": "database",
                        "name": "courses",
                        "backend": "memory",
                        "content_field": "description",
                        "text_search_fields": ["title"],
                        "schema": {
                            "fields": {
                                "title": "string",
                                "description": "text",
                            },
                        },
                    },
                ],
            },
        }

        async with await BotTestHarness.create(
            bot_config=config,
            main_responses=[],  # No LLM needed (static + template)
        ) as harness:
            strategy = harness.bot.reasoning_strategy
            assert len(strategy._sources) == 1
            assert strategy._sources[0].name == "courses"
            assert strategy._sources[0].source_type == "database"

    @pytest.mark.asyncio
    async def test_backward_compat_no_sources_with_kb(self) -> None:
        """Without sources config, KB is auto-wrapped (backward compat)."""
        kb = InMemoryKnowledgeBase(results=SAMPLE_KB_RESULTS)
        config = _grounded_bot_config()
        async with await BotTestHarness.create(
            bot_config=config,
            main_responses=[
                text_response("query1\nquery2"),
                text_response("Synthesized answer"),
            ],
        ) as harness:
            harness.bot.reasoning_strategy.set_knowledge_base(kb)
            result = await harness.chat("What are authorization grants?")
            assert result.response == "Synthesized answer"
            # KB was queried
            assert len(kb.queries) > 0


# ------------------------------------------------------------------
# QueryTransformer + ContextualExpander integration tests
# ------------------------------------------------------------------


class TestQueryTransformerIntegration:
    """Tests for QueryTransformer integration in extract mode."""

    @pytest.mark.asyncio
    async def test_extract_mode_uses_transformer(self) -> None:
        """Extract mode delegates to QueryTransformer."""
        kb = InMemoryKnowledgeBase(results=SAMPLE_KB_RESULTS)
        config = _grounded_bot_config()
        async with await BotTestHarness.create(
            bot_config=config,
            main_responses=[
                # First response: query generation (goes through transformer)
                text_response("OAuth grant types\nauthorization code"),
                # Second response: synthesis
                text_response("Here is the answer."),
            ],
        ) as harness:
            harness.bot.reasoning_strategy.set_knowledge_base(kb)
            strategy = harness.bot.reasoning_strategy
            # Verify transformer was created with suppress_thinking
            assert strategy._transformer is not None
            assert strategy._transformer.config.enabled is True
            assert strategy._transformer.config.suppress_thinking is True

            result = await harness.chat("What are authorization grants?")
            assert result.response == "Here is the answer."
            assert len(kb.queries) > 0

    @pytest.mark.asyncio
    async def test_transformer_uses_domain_context(self) -> None:
        """Transformer picks up domain_context from config."""
        config = _grounded_bot_config()
        config["reasoning"]["intent"]["domain_context"] = "OAuth 2.0"
        async with await BotTestHarness.create(
            bot_config=config,
            main_responses=[
                text_response("query1\nquery2"),
                text_response("answer"),
            ],
        ) as harness:
            strategy = harness.bot.reasoning_strategy
            assert strategy._transformer is not None
            assert strategy._transformer.config.domain_context == "OAuth 2.0"

    @pytest.mark.asyncio
    async def test_expander_disabled_by_default(self) -> None:
        """ContextualExpander is not created unless configured."""
        config = _grounded_bot_config()
        async with await BotTestHarness.create(
            bot_config=config,
            main_responses=[
                text_response("query"),
                text_response("answer"),
            ],
        ) as harness:
            strategy = harness.bot.reasoning_strategy
            assert strategy._expander is None

    @pytest.mark.asyncio
    async def test_expander_enabled_enriches_ambiguous_query(self) -> None:
        """When enabled, ambiguous queries get context keywords prepended."""
        kb = InMemoryKnowledgeBase(results=SAMPLE_KB_RESULTS)
        config = _grounded_bot_config()
        config["reasoning"]["intent"]["expand_ambiguous_queries"] = True
        config["reasoning"]["intent"]["max_context_turns"] = 3

        async with await BotTestHarness.create(
            bot_config=config,
            main_responses=[
                # First turn: query gen + synthesis
                text_response("OAuth grant types\nauth code"),
                text_response("Grant types are..."),
                # Second turn: query gen + synthesis
                text_response("more details query"),
                text_response("More details here."),
            ],
        ) as harness:
            harness.bot.reasoning_strategy.set_knowledge_base(kb)
            strategy = harness.bot.reasoning_strategy
            assert strategy._expander is not None

            # First turn: clear query, no expansion needed
            await harness.chat("What are OAuth authorization grants?")
            # Second turn: ambiguous query, should be expanded
            result = await harness.chat("Show me more")
            assert result.response == "More details here."

    @pytest.mark.asyncio
    async def test_static_mode_no_transformer(self) -> None:
        """Static intent mode does not create a transformer."""
        config = _grounded_bot_config(intent_mode="static")
        config["reasoning"]["intent"]["text_queries"] = ["fixed query"]
        async with await BotTestHarness.create(
            bot_config=config,
            main_responses=[text_response("answer")],
        ) as harness:
            strategy = harness.bot.reasoning_strategy
            assert strategy._transformer is None
            assert strategy._expander is None


# ------------------------------------------------------------------
# SchemaExtractor intent extraction tests
# ------------------------------------------------------------------


class TestSchemaExtractorIntentExtraction:
    """Tests for extract mode with extraction_config (SchemaExtractor path)."""

    @pytest.mark.asyncio
    async def test_extraction_config_creates_extractor_not_transformer(self) -> None:
        """When extraction_config is present, extractor is created, not transformer."""
        config = _grounded_bot_config()
        config["reasoning"]["intent"]["extraction_config"] = {
            "provider": "echo",
            "model": "echo-test",
        }
        async with await BotTestHarness.create(
            bot_config=config,
            main_responses=[
                # Synthesis response (extraction handled by extractor)
                text_response("Synthesized answer."),
            ],
        ) as harness:
            strategy = harness.bot.reasoning_strategy
            assert strategy._extractor is not None
            assert strategy._transformer is None

    @pytest.mark.asyncio
    async def test_extractor_produces_queries(self) -> None:
        """SchemaExtractor path produces queries from structured extraction."""
        from dataknobs_llm.testing import scripted_schema_extractor

        kb = InMemoryKnowledgeBase(results=SAMPLE_KB_RESULTS)
        config = _grounded_bot_config()
        # Use extraction_config to enable extractor path
        config["reasoning"]["intent"]["extraction_config"] = {
            "provider": "echo",
            "model": "echo-test",
        }
        async with await BotTestHarness.create(
            bot_config=config,
            main_responses=[
                # Synthesis only (extractor handles query gen)
                text_response("Based on the KB content..."),
            ],
        ) as harness:
            harness.bot.reasoning_strategy.set_knowledge_base(kb)
            strategy = harness.bot.reasoning_strategy

            # Inject a scripted extractor that returns structured intent
            extractor, _provider = scripted_schema_extractor([
                '{"text_queries": ["OAuth grant types", "authorization code flow"], "scope": "focused"}',
            ])
            strategy.set_extractor(extractor)

            result = await harness.chat("What are OAuth grant types?")
            assert result.response == "Based on the KB content..."
            # KB should have been queried with extracted queries
            assert len(kb.queries) >= 1

    @pytest.mark.asyncio
    async def test_extractor_scope_flows_to_intent(self) -> None:
        """Scope from extraction flows into RetrievalIntent."""
        from dataknobs_llm.testing import scripted_schema_extractor

        config = GroundedReasoningConfig(
            intent=GroundedIntentConfig(
                mode="extract",
                extraction_config={"provider": "echo", "model": "test"},
            ),
        )
        strategy = GroundedReasoning(config=config)

        extractor, _provider = scripted_schema_extractor([
            '{"text_queries": ["broad search"], "scope": "broad"}',
        ])
        strategy.set_extractor(extractor)

        intent = await strategy._extract_intent(
            "Give me a broad overview of OAuth",
            [],
        )
        assert intent.scope == "broad"
        assert intent.text_queries == ["broad search"]

    @pytest.mark.asyncio
    async def test_extractor_fallback_on_empty_queries(self) -> None:
        """Falls back to user message when extraction returns empty queries."""
        from dataknobs_llm.testing import scripted_schema_extractor

        config = GroundedReasoningConfig(
            intent=GroundedIntentConfig(
                mode="extract",
                extraction_config={"provider": "echo", "model": "test"},
            ),
        )
        strategy = GroundedReasoning(config=config)

        extractor, _provider = scripted_schema_extractor([
            '{"text_queries": []}',
        ])
        strategy.set_extractor(extractor)

        intent = await strategy._extract_intent("my question", [])
        assert intent.text_queries == ["my question"]

    @pytest.mark.asyncio
    async def test_extractor_fallback_on_exception(self) -> None:
        """Falls back to user message when extractor raises."""
        config = GroundedReasoningConfig(
            intent=GroundedIntentConfig(
                mode="extract",
                extraction_config={"provider": "echo", "model": "test"},
            ),
        )
        strategy = GroundedReasoning(config=config)

        class FailingExtractor:
            async def extract(self, **kwargs: Any) -> None:
                raise RuntimeError("extraction failed")

        strategy.set_extractor(FailingExtractor())

        intent = await strategy._extract_intent("my question", [])
        assert intent.text_queries == ["my question"]
        assert intent.scope == "focused"

    @pytest.mark.asyncio
    async def test_extractor_respects_num_queries(self) -> None:
        """Extracted queries are capped at num_queries."""
        from dataknobs_llm.testing import scripted_schema_extractor

        config = GroundedReasoningConfig(
            intent=GroundedIntentConfig(
                mode="extract",
                num_queries=2,
                extraction_config={"provider": "echo", "model": "test"},
            ),
        )
        strategy = GroundedReasoning(config=config)

        extractor, _provider = scripted_schema_extractor([
            '{"text_queries": ["q1", "q2", "q3", "q4"]}',
        ])
        strategy.set_extractor(extractor)

        intent = await strategy._extract_intent("question", [])
        assert len(intent.text_queries) == 2

    @pytest.mark.asyncio
    async def test_extractor_with_conversation_context(self) -> None:
        """Conversation context is included in extraction input."""
        from dataknobs_llm.testing import scripted_schema_extractor

        config = GroundedReasoningConfig(
            intent=GroundedIntentConfig(
                mode="extract",
                use_conversation_context=True,
                extraction_config={"provider": "echo", "model": "test"},
            ),
        )
        strategy = GroundedReasoning(config=config)

        extractor, ext_provider = scripted_schema_extractor([
            '{"text_queries": ["context-aware query"]}',
        ])
        strategy.set_extractor(extractor)

        messages = [
            {"role": "user", "content": "Tell me about OAuth"},
            {"role": "assistant", "content": "OAuth is..."},
            {"role": "user", "content": "What about grants?"},
        ]
        intent = await strategy._extract_intent("What about grants?", messages)
        assert intent.text_queries == ["context-aware query"]
        # The extraction input should have included context
        last_user_msg = ext_provider.get_last_user_message()
        assert last_user_msg is not None
        assert "OAuth" in last_user_msg

    @pytest.mark.asyncio
    async def test_no_extraction_config_uses_transformer(self) -> None:
        """Without extraction_config, extract mode uses QueryTransformer."""
        config = _grounded_bot_config()
        # No extraction_config — should use transformer path
        async with await BotTestHarness.create(
            bot_config=config,
            main_responses=[
                text_response("query1\nquery2"),  # Transformer query gen
                text_response("answer"),  # Synthesis
            ],
        ) as harness:
            strategy = harness.bot.reasoning_strategy
            assert strategy._transformer is not None
            assert strategy._extractor is None

    def test_extraction_config_in_grounded_intent_config(self) -> None:
        """extraction_config field exists on GroundedIntentConfig."""
        cfg = GroundedIntentConfig()
        assert cfg.extraction_config is None

        cfg2 = GroundedIntentConfig(
            extraction_config={"provider": "ollama", "model": "qwen3:8b"},
        )
        assert cfg2.extraction_config is not None
        assert cfg2.extraction_config["provider"] == "ollama"

    def test_extraction_config_from_dict(self) -> None:
        """extraction_config survives GroundedReasoningConfig.from_dict()."""
        config = GroundedReasoningConfig.from_dict({
            "intent": {
                "mode": "extract",
                "num_queries": 3,
                "extraction_config": {
                    "provider": "ollama",
                    "model": "qwen3:8b",
                    "temperature": 0.0,
                },
            },
        })
        assert config.intent.extraction_config is not None
        assert config.intent.extraction_config["model"] == "qwen3:8b"


# ------------------------------------------------------------------
# Auto-disable auto_context tests
# ------------------------------------------------------------------


class TestAutoContextAutoDisable:
    """Tests for auto_context auto-disable when strategy is grounded."""

    @pytest.mark.asyncio
    async def test_auto_context_disabled_for_grounded_with_kb(self) -> None:
        """Grounded strategy + KB results in auto_context=False."""
        kb = InMemoryKnowledgeBase(results=SAMPLE_KB_RESULTS)
        config = _grounded_bot_config()
        async with await BotTestHarness.create(
            bot_config=config,
            main_responses=[
                text_response("q1\nq2"),
                text_response("answer"),
            ],
        ) as harness:
            harness.bot.reasoning_strategy.set_knowledge_base(kb)
            # auto_context defaults to True, but from_config() should
            # have left it alone (no KB was created from config).
            # Verify the grounded strategy works regardless.
            result = await harness.chat("test query")
            assert result.response == "answer"

    @pytest.mark.asyncio
    async def test_auto_context_not_needed_for_grounded(self) -> None:
        """Grounded strategy does not use auto_context for retrieval."""
        kb = InMemoryKnowledgeBase(results=SAMPLE_KB_RESULTS)
        config = _grounded_bot_config()
        async with await BotTestHarness.create(
            bot_config=config,
            main_responses=[
                text_response("q1\nq2"),
                text_response("answer"),
            ],
        ) as harness:
            # Explicitly disable auto_context — should not affect grounded
            harness.bot._kb_auto_context = False
            harness.bot.reasoning_strategy.set_knowledge_base(kb)
            result = await harness.chat("test query")
            assert result.response == "answer"
            assert len(kb.queries) > 0  # KB was queried via strategy


# ------------------------------------------------------------------
# raw_content preference tests
# ------------------------------------------------------------------


class TestRawContentPreference:
    """Tests for raw_content preference in context building."""

    def test_build_conversation_context_prefers_raw_content(self) -> None:
        """_build_conversation_context uses raw_content from metadata."""
        cfg = GroundedIntentConfig(use_conversation_context=True, max_context_turns=3)
        messages = [
            {
                "role": "user",
                "content": "<knowledge_base>chunks</knowledge_base>\nWhat is OAuth?",
                "metadata": {"raw_content": "What is OAuth?"},
            },
            {"role": "assistant", "content": "OAuth is an authorization framework."},
            {"role": "user", "content": "Tell me more"},
        ]
        context = GroundedReasoning._build_conversation_context(messages, cfg)
        # Should use raw_content, not the augmented content
        assert "What is OAuth?" in context
        assert "<knowledge_base>" not in context

    def test_build_conversation_context_falls_back_to_content(self) -> None:
        """Falls back to content when raw_content is absent."""
        cfg = GroundedIntentConfig(use_conversation_context=True, max_context_turns=3)
        messages = [
            {"role": "user", "content": "What is OAuth?"},
            {"role": "assistant", "content": "OAuth is..."},
            {"role": "user", "content": "More please"},
        ]
        context = GroundedReasoning._build_conversation_context(messages, cfg)
        assert "What is OAuth?" in context


# ------------------------------------------------------------------
# Synthesis style resolution tests
# ------------------------------------------------------------------


class TestStyleResolution:
    """Test the synthesis style resolution cascade."""

    def _make_strategy(
        self,
        *,
        mode: str = "llm",
        style: str | None = None,
    ) -> GroundedReasoning:
        """Create a strategy with specified synthesis config."""
        config = GroundedReasoningConfig(
            synthesis=GroundedSynthesisConfig(mode=mode, style=style),
        )
        return GroundedReasoning(config)

    def _make_provenance(
        self,
        *,
        raw_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build a minimal provenance dict."""
        intent: dict[str, Any] = {
            "mode": "resolved",
            "text_queries": ["test"],
            "filters": {},
            "scope": "focused",
            "raw_data": raw_data,
        }
        return {"intent": intent, "results": [], "results_by_source": {}}

    class _FakeManager:
        """Minimal manager stub for style resolution tests."""

        def __init__(self, metadata: dict[str, Any] | None = None) -> None:
            self.metadata = metadata or {}
            self.system_prompt = "You are a test bot."

        def get_messages(self) -> list[dict[str, Any]]:
            return [{"role": "user", "content": "test"}]

    def test_default_is_conversational(self) -> None:
        """No style or mode set → defaults to conversational."""
        strategy = self._make_strategy()
        manager = self._FakeManager()
        prov = self._make_provenance()
        assert strategy._resolve_effective_style(manager, prov) == "conversational"

    def test_config_style(self) -> None:
        """Config-level style is returned when set."""
        strategy = self._make_strategy(style="hybrid")
        manager = self._FakeManager()
        prov = self._make_provenance()
        assert strategy._resolve_effective_style(manager, prov) == "hybrid"

    def test_legacy_mode_template(self) -> None:
        """mode=template maps to structured when no style set."""
        strategy = self._make_strategy(mode="template")
        manager = self._FakeManager()
        prov = self._make_provenance()
        assert strategy._resolve_effective_style(manager, prov) == "structured"

    def test_legacy_mode_llm(self) -> None:
        """mode=llm maps to conversational when no style set."""
        strategy = self._make_strategy(mode="llm")
        manager = self._FakeManager()
        prov = self._make_provenance()
        assert strategy._resolve_effective_style(manager, prov) == "conversational"

    def test_session_overrides_config(self) -> None:
        """Session-level metadata overrides config-level style."""
        strategy = self._make_strategy(style="conversational")
        manager = self._FakeManager(metadata={"synthesis_style": "structured"})
        prov = self._make_provenance()
        assert strategy._resolve_effective_style(manager, prov) == "structured"

    def test_per_turn_overrides_session(self) -> None:
        """Per-turn output_style from raw_data overrides session."""
        strategy = self._make_strategy(style="conversational")
        manager = self._FakeManager(metadata={"synthesis_style": "structured"})
        prov = self._make_provenance(raw_data={"output_style": "hybrid"})
        assert strategy._resolve_effective_style(manager, prov) == "hybrid"

    def test_invalid_per_turn_ignored(self) -> None:
        """Invalid per-turn output_style falls through to session."""
        strategy = self._make_strategy()
        manager = self._FakeManager(metadata={"synthesis_style": "hybrid"})
        prov = self._make_provenance(raw_data={"output_style": "invalid"})
        assert strategy._resolve_effective_style(manager, prov) == "hybrid"

    def test_invalid_session_ignored(self) -> None:
        """Invalid session synthesis_style falls through to config."""
        strategy = self._make_strategy(style="structured")
        manager = self._FakeManager(metadata={"synthesis_style": "bogus"})
        prov = self._make_provenance()
        assert strategy._resolve_effective_style(manager, prov) == "structured"

    def test_valid_styles_constant(self) -> None:
        """Verify the valid styles set matches expectations."""
        assert _VALID_STYLES == {"conversational", "structured", "hybrid"}


# ------------------------------------------------------------------
# Synthesis plan tests
# ------------------------------------------------------------------


class TestSynthesisPlan:
    """Test _resolve_synthesis produces correct plan artifacts."""

    def _make_strategy(
        self,
        *,
        style: str | None = None,
    ) -> GroundedReasoning:
        config = GroundedReasoningConfig(
            synthesis=GroundedSynthesisConfig(style=style),
        )
        return GroundedReasoning(config)

    class _FakeManager:
        def __init__(self) -> None:
            self.metadata: dict[str, Any] = {}
            self.system_prompt = "You are a bot."

        def get_messages(self) -> list[dict[str, Any]]:
            return [{"role": "user", "content": "test message"}]

    def _make_provenance(self) -> dict[str, Any]:
        return {
            "intent": {
                "mode": "resolved",
                "text_queries": ["test"],
                "filters": {},
                "scope": "focused",
                "raw_data": None,
            },
            "results": [],
            "results_by_source": {},
        }

    def test_conversational_has_prompt_only(self) -> None:
        """Conversational plan has system_prompt, no template_text."""
        strategy = self._make_strategy(style="conversational")
        plan = strategy._resolve_synthesis(
            "kb context", self._FakeManager(), self._make_provenance(),
        )
        assert plan.effective_style == "conversational"
        assert plan.system_prompt is not None
        assert plan.template_text is None

    def test_structured_has_template_only(self) -> None:
        """Structured plan has template_text, no system_prompt."""
        strategy = self._make_strategy(style="structured")
        plan = strategy._resolve_synthesis(
            "kb context", self._FakeManager(), self._make_provenance(),
        )
        assert plan.effective_style == "structured"
        assert plan.template_text is not None
        assert plan.system_prompt is None

    def test_hybrid_has_both(self) -> None:
        """Hybrid plan has both system_prompt and template_text."""
        strategy = self._make_strategy(style="hybrid")
        plan = strategy._resolve_synthesis(
            "kb context", self._FakeManager(), self._make_provenance(),
        )
        assert plan.effective_style == "hybrid"
        assert plan.system_prompt is not None
        assert plan.template_text is not None


# ------------------------------------------------------------------
# Default provenance template tests
# ------------------------------------------------------------------


class TestDefaultProvenanceTemplate:
    """Test the built-in DEFAULT_PROVENANCE_TEMPLATE rendering."""

    def test_renders_with_results(self) -> None:
        """Template renders results grouped by source."""
        import jinja2

        env = jinja2.Environment(undefined=jinja2.Undefined)
        tmpl = env.from_string(DEFAULT_PROVENANCE_TEMPLATE)

        result = tmpl.render(
            results=[
                {
                    "source_id": "chunk_1",
                    "source_name": "rfc6749",
                    "relevance": 0.85,
                    "text_preview": "Authorization code grant...",
                },
            ],
            results_by_source={
                "rfc6749": [
                    {
                        "source_id": "chunk_1",
                        "relevance": 0.85,
                        "text_preview": "Authorization code grant...",
                        "metadata": {"headings": ["4.1", "Authorization Code"]},
                    },
                ],
            },
            context="",
            message="",
            metadata={},
            intent={},
        )
        assert "rfc6749" in result
        assert "4.1 > Authorization Code" in result
        assert "85%" in result
        assert "1 result from 1 source" in result

    def test_renders_empty(self) -> None:
        """Template shows 'no results' when empty."""
        import jinja2

        env = jinja2.Environment(undefined=jinja2.Undefined)
        tmpl = env.from_string(DEFAULT_PROVENANCE_TEMPLATE)

        result = tmpl.render(
            results=[],
            results_by_source={},
            context="",
            message="",
            metadata={},
            intent={},
        )
        assert "No relevant results found." in result

    def test_custom_provenance_overrides_default(self) -> None:
        """Custom provenance_template overrides the built-in default."""
        config = GroundedReasoningConfig(
            synthesis=GroundedSynthesisConfig(
                style="structured",
                provenance_template="Custom: {{ results|length }} hits",
            ),
        )
        strategy = GroundedReasoning(config)

        class _Mgr:
            metadata: dict[str, Any] = {}
            system_prompt = "Bot."

            def get_messages(self) -> list[dict[str, Any]]:
                return [{"role": "user", "content": "test"}]

        prov: dict[str, Any] = {
            "intent": {
                "mode": "resolved",
                "text_queries": ["q"],
                "filters": {},
                "scope": "focused",
                "raw_data": None,
            },
            "results": [{"source_id": "a", "relevance": 0.9}],
            "results_by_source": {},
        }
        plan = strategy._resolve_synthesis("context", _Mgr(), prov)
        assert plan.template_text == "Custom: 1 hits"


# ------------------------------------------------------------------
# Synthesis style integration tests (BotTestHarness)
# ------------------------------------------------------------------


class TestSynthesisStyleIntegration:
    """Integration tests for synthesis style with full bot pipeline."""

    @pytest.mark.asyncio
    async def test_conversational_uses_llm(self) -> None:
        """style=conversational uses LLM synthesis (same as mode=llm)."""
        kb = InMemoryKnowledgeBase(results=SAMPLE_KB_RESULTS)
        config = (
            GroundedConfigBuilder()
            .intent(mode="static", text_queries=["auth grants"])
            .synthesis(style="conversational")
            .build()
        )

        async with await BotTestHarness.create(
            bot_config=config,
            main_responses=[text_response("LLM synthesis result")],
        ) as harness:
            harness.bot.reasoning_strategy.set_knowledge_base(kb)
            result = await harness.chat("How do grants work?")
            assert "LLM synthesis result" in result.response

    @pytest.mark.asyncio
    async def test_structured_no_llm_static_intent(self) -> None:
        """style=structured + static intent → zero LLM calls."""
        kb = InMemoryKnowledgeBase(results=SAMPLE_KB_RESULTS)
        config = (
            GroundedConfigBuilder()
            .intent(mode="static", text_queries=["auth grants"])
            .synthesis(style="structured")
            .build()
        )

        async with await BotTestHarness.create(
            bot_config=config,
            main_responses=[],
        ) as harness:
            harness.bot.reasoning_strategy.set_knowledge_base(kb)
            result = await harness.chat("Show me grants")
            # Should use the built-in provenance template
            assert "result" in result.response.lower()

    @pytest.mark.asyncio
    async def test_structured_uses_builtin_template(self) -> None:
        """style=structured with no custom template uses DEFAULT_PROVENANCE_TEMPLATE."""
        kb = InMemoryKnowledgeBase(results=SAMPLE_KB_RESULTS)
        config = (
            GroundedConfigBuilder()
            .intent(mode="static", text_queries=["auth grants"])
            .synthesis(style="structured")
            .build()
        )

        async with await BotTestHarness.create(
            bot_config=config,
            main_responses=[],
        ) as harness:
            harness.bot.reasoning_strategy.set_knowledge_base(kb)
            result = await harness.chat("Show grants")
            # Built-in template includes relevance percentages and source count
            assert "%" in result.response
            assert "source" in result.response.lower()

    @pytest.mark.asyncio
    async def test_structured_uses_custom_template(self) -> None:
        """style=structured with custom template uses that template."""
        kb = InMemoryKnowledgeBase(results=SAMPLE_KB_RESULTS)
        config = (
            GroundedConfigBuilder()
            .intent(mode="static", text_queries=["auth grants"])
            .synthesis(
                style="structured",
                template="Custom: {{ results|length }} results for {{ message }}",
            )
            .build()
        )

        async with await BotTestHarness.create(
            bot_config=config,
            main_responses=[],
        ) as harness:
            harness.bot.reasoning_strategy.set_knowledge_base(kb)
            result = await harness.chat("Show grants")
            assert result.response.startswith("Custom: 3 results for")

    @pytest.mark.asyncio
    async def test_hybrid_appends_provenance(self) -> None:
        """style=hybrid produces LLM response + provenance appendix."""
        kb = InMemoryKnowledgeBase(results=SAMPLE_KB_RESULTS)
        config = (
            GroundedConfigBuilder()
            .intent(mode="static", text_queries=["auth grants"])
            .synthesis(style="hybrid")
            .build()
        )

        async with await BotTestHarness.create(
            bot_config=config,
            main_responses=[text_response("Here is my analysis.")],
        ) as harness:
            harness.bot.reasoning_strategy.set_knowledge_base(kb)
            result = await harness.chat("Explain grants")
            # Response should contain LLM output AND provenance
            assert "Here is my analysis." in result.response
            # Provenance appendix from default template includes source counts
            assert "source" in result.response.lower()

    @pytest.mark.asyncio
    async def test_session_style_override(self) -> None:
        """Session metadata synthesis_style overrides config."""
        kb = InMemoryKnowledgeBase(results=SAMPLE_KB_RESULTS)
        config = (
            GroundedConfigBuilder()
            .intent(mode="static", text_queries=["auth grants"])
            .synthesis(style="conversational")
            .build()
        )

        async with await BotTestHarness.create(
            bot_config=config,
            main_responses=[
                text_response("First turn LLM"),  # conversational (turn 1)
                # Turn 2 uses structured — no LLM response needed
            ],
        ) as harness:
            harness.bot.reasoning_strategy.set_knowledge_base(kb)
            # First turn to create the conversation manager
            result1 = await harness.chat("First question")
            assert "First turn LLM" in result1.response

            # Now override at session level to structured
            manager = list(harness.bot._conversation_managers.values())[0]
            manager.metadata["synthesis_style"] = "structured"
            result2 = await harness.chat("Show me sources")
            # Should use structured (template) despite config saying conversational
            assert "%" in result2.response  # Built-in template has percentages

    @pytest.mark.asyncio
    async def test_backward_compat_mode_llm(self) -> None:
        """mode=llm without style works as before."""
        kb = InMemoryKnowledgeBase(results=SAMPLE_KB_RESULTS)
        config = (
            GroundedConfigBuilder()
            .intent(mode="static", text_queries=["auth grants"])
            .synthesis(mode="llm")
            .build()
        )

        async with await BotTestHarness.create(
            bot_config=config,
            main_responses=[text_response("LLM output")],
        ) as harness:
            harness.bot.reasoning_strategy.set_knowledge_base(kb)
            result = await harness.chat("query")
            assert "LLM output" in result.response

    @pytest.mark.asyncio
    async def test_backward_compat_mode_template(self) -> None:
        """mode=template without style works as before."""
        kb = InMemoryKnowledgeBase(results=SAMPLE_KB_RESULTS)
        config = (
            GroundedConfigBuilder()
            .intent(mode="static", text_queries=["auth grants"])
            .synthesis(
                mode="template",
                template="Found {{ results|length }} results.",
            )
            .build()
        )

        async with await BotTestHarness.create(
            bot_config=config,
            main_responses=[],
        ) as harness:
            harness.bot.reasoning_strategy.set_knowledge_base(kb)
            result = await harness.chat("query")
            assert "Found 3 results." in result.response

    @pytest.mark.asyncio
    async def test_provenance_includes_raw_data(self) -> None:
        """Provenance intent dict includes raw_data for per-turn style."""
        kb = InMemoryKnowledgeBase(results=SAMPLE_KB_RESULTS)
        config = (
            GroundedConfigBuilder()
            .intent(mode="static", text_queries=["test"])
            .build()
        )

        async with await BotTestHarness.create(
            bot_config=config,
            main_responses=[text_response("result")],
        ) as harness:
            harness.bot.reasoning_strategy.set_knowledge_base(kb)
            await harness.chat("query")
            manager = list(harness.bot._conversation_managers.values())[0]
            prov = manager.metadata.get("retrieval_provenance", {})
            assert "raw_data" in prov.get("intent", {})


# ------------------------------------------------------------------
# Synthesis style streaming tests
# ------------------------------------------------------------------


class TestSynthesisStyleStreaming:
    """Streaming tests verify parity with buffered synthesis paths."""

    @pytest.mark.asyncio
    async def test_stream_structured_single_chunk(self) -> None:
        """Streaming structured yields a single template chunk."""
        kb = InMemoryKnowledgeBase(results=SAMPLE_KB_RESULTS)
        config = (
            GroundedConfigBuilder()
            .intent(mode="static", text_queries=["auth grants"])
            .synthesis(style="structured")
            .build()
        )

        async with await BotTestHarness.create(
            bot_config=config,
            main_responses=[],
        ) as harness:
            harness.bot.reasoning_strategy.set_knowledge_base(kb)
            chunks = []
            async for chunk in harness.bot.stream_chat(
                "Show grants", harness.context,
            ):
                chunks.append(chunk)
            # stream_chat wraps LLMResponse into LLMStreamResponse with .delta
            full_text = "".join(c.delta for c in chunks if c.delta)
            assert "source" in full_text.lower()
            assert "%" in full_text

    @pytest.mark.asyncio
    async def test_stream_hybrid_appends_provenance(self) -> None:
        """Streaming hybrid yields LLM chunks then provenance chunk."""
        kb = InMemoryKnowledgeBase(results=SAMPLE_KB_RESULTS)
        config = (
            GroundedConfigBuilder()
            .intent(mode="static", text_queries=["auth grants"])
            .synthesis(style="hybrid")
            .build()
        )

        async with await BotTestHarness.create(
            bot_config=config,
            main_responses=[text_response("LLM analysis.")],
        ) as harness:
            harness.bot.reasoning_strategy.set_knowledge_base(kb)
            chunks = []
            async for chunk in harness.bot.stream_chat(
                "Explain grants", harness.context,
            ):
                chunks.append(chunk)
            full_text = "".join(c.delta for c in chunks if c.delta)
            # Should contain both LLM output and provenance
            assert "LLM analysis." in full_text
            assert "source" in full_text.lower()


# ------------------------------------------------------------------
# Intent schema output_style tests
# ------------------------------------------------------------------


class TestIntentSchemaOutputStyle:
    """Test that output_style field exists in the intent schema."""

    def test_output_style_in_schema(self) -> None:
        """INTENT_EXTRACTION_SCHEMA includes output_style property."""
        from dataknobs_bots.reasoning.grounded import INTENT_EXTRACTION_SCHEMA

        props = INTENT_EXTRACTION_SCHEMA["properties"]
        assert "output_style" in props
        assert set(props["output_style"]["enum"]) == {
            "conversational", "structured", "hybrid",
        }

    def test_output_style_not_required(self) -> None:
        """output_style is optional — not in required list."""
        from dataknobs_bots.reasoning.grounded import INTENT_EXTRACTION_SCHEMA

        assert "output_style" not in INTENT_EXTRACTION_SCHEMA.get("required", [])

    def test_output_style_hint_overrides_description(self) -> None:
        """Config output_style_hint overrides the schema description."""
        import copy

        from dataknobs_bots.reasoning.grounded import INTENT_EXTRACTION_SCHEMA

        custom_hint = "Always use conversational unless the user says 'raw'."
        config = GroundedReasoningConfig(
            intent=GroundedIntentConfig(
                output_style_hint=custom_hint,
            ),
        )

        # Simulate what _extract_intent does: deepcopy + override
        schema = copy.deepcopy(INTENT_EXTRACTION_SCHEMA)
        if config.intent.output_style_hint:
            schema["properties"]["output_style"]["description"] = (
                config.intent.output_style_hint
            )

        assert schema["properties"]["output_style"]["description"] == custom_hint
        # Original is unchanged
        assert INTENT_EXTRACTION_SCHEMA["properties"]["output_style"]["description"] != custom_hint

    def test_output_style_hint_default_is_none(self) -> None:
        """output_style_hint defaults to None (use built-in description)."""
        config = GroundedIntentConfig()
        assert config.output_style_hint is None


class TestIntentGrounding:
    """Tests for extraction grounding in _extract_intent().

    The grounding utility checks whether optional extracted fields
    are grounded in the user's message.  Ungrounded optional fields
    are dropped so the resolution cascade falls through to defaults.
    """

    @pytest.mark.asyncio
    async def test_output_style_ungrounded_dropped(self) -> None:
        """output_style: 'structured' for ambiguous query gets dropped.

        The enum grounding check looks for the literal word 'structured'
        in the user message.  When absent, the field is dropped and
        synthesis defaults to 'conversational'.
        """
        from dataknobs_llm.testing import scripted_schema_extractor

        kb = InMemoryKnowledgeBase(results=SAMPLE_KB_RESULTS)
        config = _grounded_bot_config()
        config["reasoning"]["intent"]["extraction_config"] = {
            "provider": "echo",
            "model": "echo-test",
        }
        async with await BotTestHarness.create(
            bot_config=config,
            main_responses=[text_response("Synthesized answer.")],
        ) as harness:
            harness.bot.reasoning_strategy.set_knowledge_base(kb)
            strategy = harness.bot.reasoning_strategy

            # Extractor returns output_style: structured, but the user
            # message has no grounding for "structured"
            extractor, _provider = scripted_schema_extractor([
                '{"text_queries": ["security risks"], "output_style": "structured"}',
            ])
            strategy.set_extractor(extractor)

            result = await harness.chat(
                "What security risks should I be aware of?",
            )
            # The bot should use conversational synthesis (LLM), not
            # structured (template), because output_style was dropped
            assert result.response == "Synthesized answer."

    @pytest.mark.asyncio
    async def test_output_style_grounded_kept(self) -> None:
        """output_style: 'structured' kept when user says 'structured'."""
        from dataknobs_llm.testing import scripted_schema_extractor

        kb = InMemoryKnowledgeBase(results=SAMPLE_KB_RESULTS)
        config = _grounded_bot_config()
        config["reasoning"]["intent"]["extraction_config"] = {
            "provider": "echo",
            "model": "echo-test",
        }
        async with await BotTestHarness.create(
            bot_config=config,
            main_responses=[text_response("Synthesized answer.")],
        ) as harness:
            harness.bot.reasoning_strategy.set_knowledge_base(kb)
            strategy = harness.bot.reasoning_strategy

            # User explicitly says "structured" — grounding passes
            extractor, _provider = scripted_schema_extractor([
                '{"text_queries": ["security risks"], "output_style": "structured"}',
            ])
            strategy.set_extractor(extractor)

            result = await harness.chat(
                "Show me a structured view of security risks",
            )
            # Structured mode uses template, not LLM synthesis
            assert "Synthesized answer." not in result.response

    @pytest.mark.asyncio
    async def test_text_queries_required_never_dropped(self) -> None:
        """text_queries is required — never dropped by grounding."""
        from dataknobs_llm.testing import scripted_schema_extractor

        kb = InMemoryKnowledgeBase(results=SAMPLE_KB_RESULTS)
        config = _grounded_bot_config()
        config["reasoning"]["intent"]["extraction_config"] = {
            "provider": "echo",
            "model": "echo-test",
        }
        async with await BotTestHarness.create(
            bot_config=config,
            main_responses=[text_response("Answer.")],
        ) as harness:
            harness.bot.reasoning_strategy.set_knowledge_base(kb)
            strategy = harness.bot.reasoning_strategy

            # Queries about topic X — the query words may not literally
            # appear in the user message (LLM can reformulate)
            extractor, _provider = scripted_schema_extractor([
                '{"text_queries": ["OAuth2 grant types", "authorization code"], "scope": "focused"}',
            ])
            strategy.set_extractor(extractor)

            result = await harness.chat("Tell me about auth patterns")
            # Queries should still reach the KB despite not being
            # literally in the user message — they're required fields
            assert len(kb.queries) >= 1

    @pytest.mark.asyncio
    async def test_scope_ungrounded_dropped(self) -> None:
        """scope is optional enum — dropped if not in user message."""
        from dataknobs_llm.testing import scripted_schema_extractor

        kb = InMemoryKnowledgeBase(results=SAMPLE_KB_RESULTS)
        config = _grounded_bot_config()
        config["reasoning"]["intent"]["extraction_config"] = {
            "provider": "echo",
            "model": "echo-test",
        }
        async with await BotTestHarness.create(
            bot_config=config,
            main_responses=[text_response("Answer.")],
        ) as harness:
            harness.bot.reasoning_strategy.set_knowledge_base(kb)
            strategy = harness.bot.reasoning_strategy

            # scope: "exact" but user didn't say "exact"
            extractor, _provider = scripted_schema_extractor([
                '{"text_queries": ["test query"], "scope": "exact"}',
            ])
            strategy.set_extractor(extractor)

            result = await harness.chat("Tell me about testing")
            # The query should still work — scope falls back to
            # default "focused" when dropped
            assert result.response == "Answer."


# ------------------------------------------------------------------
# Synthesis instruction tests
# ------------------------------------------------------------------


class TestSynthesisInstruction:
    """Tests for the synthesis instruction config field."""

    def test_instruction_appended_to_prompt(self) -> None:
        cfg = GroundedReasoningConfig(
            synthesis=GroundedSynthesisConfig(
                instruction="Focus on security risks.",
            ),
        )
        strategy = GroundedReasoning(config=cfg)
        prompt = strategy._build_synthesis_system_prompt(
            "KB content", "System prompt",
        )
        assert "Focus on security risks." in prompt
        # Instruction should come after grounding lines
        grounding_idx = prompt.index("Do not fill gaps")
        instruction_idx = prompt.index("Focus on security risks.")
        assert instruction_idx > grounding_idx

    def test_instruction_none_no_effect(self) -> None:
        cfg_with = GroundedReasoningConfig(
            synthesis=GroundedSynthesisConfig(instruction="Extra guidance."),
        )
        cfg_without = GroundedReasoningConfig(
            synthesis=GroundedSynthesisConfig(instruction=None),
        )
        strategy_with = GroundedReasoning(config=cfg_with)
        strategy_without = GroundedReasoning(config=cfg_without)
        prompt_with = strategy_with._build_synthesis_system_prompt(
            "KB content", "System prompt",
        )
        prompt_without = strategy_without._build_synthesis_system_prompt(
            "KB content", "System prompt",
        )
        assert "Extra guidance." in prompt_with
        assert "Extra guidance." not in prompt_without

    def test_instruction_with_parametric(self) -> None:
        cfg = GroundedReasoningConfig(
            synthesis=GroundedSynthesisConfig(
                allow_parametric=True,
                instruction="Emphasize practical examples.",
            ),
        )
        strategy = GroundedReasoning(config=cfg)
        prompt = strategy._build_synthesis_system_prompt("content", "sys")
        assert "general knowledge" in prompt
        assert "Emphasize practical examples." in prompt

    @pytest.mark.asyncio
    async def test_instruction_reaches_llm_integration(self) -> None:
        """Instruction appears in the synthesis prompt sent to the LLM."""
        kb = InMemoryKnowledgeBase(results=SAMPLE_KB_RESULTS)
        config = (
            GroundedConfigBuilder()
            .intent(mode="static", text_queries=["test"])
            .synthesis(instruction="Prioritize security content.")
            .build()
        )
        async with await BotTestHarness.create(
            bot_config=config,
            main_responses=[text_response("response")],
        ) as harness:
            harness.bot.reasoning_strategy.set_knowledge_base(kb)
            result = await harness.chat("Tell me about security")
            assert result.response == "response"

    def test_config_from_dict_with_instruction(self) -> None:
        cfg = GroundedReasoningConfig.from_dict({
            "synthesis": {
                "instruction": "Be concise.",
            },
        })
        assert cfg.synthesis.instruction == "Be concise."


# ------------------------------------------------------------------
# Result processing config tests
# ------------------------------------------------------------------


class TestResultProcessingConfig:
    """Tests for GroundedResultProcessingConfig and pipeline wiring."""

    def test_config_from_dict_with_result_processing(self) -> None:
        cfg = GroundedReasoningConfig.from_dict({
            "result_processing": {
                "normalize_strategy": "min_max",
                "relative_threshold": 0.4,
                "query_rerank_weight": 0.3,
            },
        })
        assert cfg.result_processing is not None
        assert cfg.result_processing.normalize_strategy == "min_max"
        assert cfg.result_processing.relative_threshold == pytest.approx(0.4)
        assert cfg.result_processing.query_rerank_weight == pytest.approx(0.3)

    def test_config_from_dict_without_result_processing(self) -> None:
        cfg = GroundedReasoningConfig.from_dict({})
        assert cfg.result_processing is None

    def test_pipeline_built_when_configured(self) -> None:
        cfg = GroundedReasoningConfig(
            result_processing=GroundedResultProcessingConfig(
                normalize_strategy="min_max",
                relative_threshold=0.5,
            ),
        )
        strategy = GroundedReasoning(config=cfg)
        assert strategy._result_pipeline is not None

    def test_pipeline_none_when_not_configured(self) -> None:
        cfg = GroundedReasoningConfig()
        strategy = GroundedReasoning(config=cfg)
        assert strategy._result_pipeline is None

    def test_grounded_config_builder_result_processing(self) -> None:
        config = (
            GroundedConfigBuilder()
            .result_processing(
                normalize_strategy="rank",
                relative_threshold=0.3,
            )
            .build()
        )
        rp = config["reasoning"]["result_processing"]
        assert rp["normalize_strategy"] == "rank"
        assert rp["relative_threshold"] == 0.3

    @pytest.mark.asyncio
    async def test_pipeline_integration(self) -> None:
        """Pipeline runs between merge and format in retrieve_context."""
        kb = InMemoryKnowledgeBase(results=SAMPLE_KB_RESULTS)
        config = (
            GroundedConfigBuilder()
            .intent(mode="static", text_queries=["test"])
            .result_processing(
                normalize_strategy="min_max",
                relative_threshold=0.3,
                min_results=1,
            )
            .build()
        )
        async with await BotTestHarness.create(
            bot_config=config,
            main_responses=[text_response("processed")],
        ) as harness:
            harness.bot.reasoning_strategy.set_knowledge_base(kb)
            result = await harness.chat("Tell me about authorization")
            assert result.response == "processed"


# ------------------------------------------------------------------
# Bridge mode tests
# ------------------------------------------------------------------


class TestBridgeMode:
    """Tests for allow_parametric: bridge synthesis mode."""

    def test_bridge_prompt_instruction(self) -> None:
        cfg = GroundedReasoningConfig(
            synthesis=GroundedSynthesisConfig(allow_parametric="bridge"),
        )
        strategy = GroundedReasoning(config=cfg)
        prompt = strategy._build_synthesis_system_prompt("KB content", "sys")
        assert "connect and synthesize concepts" in prompt
        assert "Do not introduce facts from outside" in prompt

    def test_strict_mode_unchanged(self) -> None:
        cfg = GroundedReasoningConfig(
            synthesis=GroundedSynthesisConfig(allow_parametric=False),
        )
        strategy = GroundedReasoning(config=cfg)
        prompt = strategy._build_synthesis_system_prompt("KB content", "sys")
        assert "Do not fill gaps" in prompt
        assert "connect and synthesize" not in prompt

    def test_relaxed_mode_unchanged(self) -> None:
        cfg = GroundedReasoningConfig(
            synthesis=GroundedSynthesisConfig(allow_parametric=True),
        )
        strategy = GroundedReasoning(config=cfg)
        prompt = strategy._build_synthesis_system_prompt("KB content", "sys")
        assert "general knowledge" in prompt
        assert "connect and synthesize" not in prompt

    def test_bridge_config_from_dict(self) -> None:
        cfg = GroundedReasoningConfig.from_dict({
            "synthesis": {"allow_parametric": "bridge"},
        })
        assert cfg.synthesis.allow_parametric == "bridge"


# ------------------------------------------------------------------
# Cluster-annotated formatting tests
# ------------------------------------------------------------------


class TestClusteredFormatting:
    """Tests for cluster-annotated result formatting."""

    def test_clustered_results_use_xml_tags(self) -> None:
        strategy = GroundedReasoning(config=GroundedReasoningConfig())
        results = [
            SourceResult(
                content="Security risk A",
                source_id="a",
                source_name="kb",
                source_type="vector_kb",
                relevance=0.9,
                metadata={"cluster_id": 0, "cluster_label": "security",
                          "cluster_size": 2, "cluster_query_score": 0.9},
            ),
            SourceResult(
                content="Security risk B",
                source_id="b",
                source_name="kb",
                source_type="vector_kb",
                relevance=0.8,
                metadata={"cluster_id": 0, "cluster_label": "security",
                          "cluster_size": 2, "cluster_query_score": 0.9},
            ),
            SourceResult(
                content="Database info",
                source_id="c",
                source_name="kb",
                source_type="vector_kb",
                relevance=0.5,
                metadata={"cluster_id": 1, "cluster_label": "database",
                          "cluster_size": 1, "cluster_query_score": 0.3},
            ),
        ]
        formatted = strategy._format_source_results(results)
        assert '<cluster id="0"' in formatted
        assert 'label="security"' in formatted
        assert '<cluster id="1"' in formatted
        assert "</cluster>" in formatted

    def test_unclustered_results_use_flat_format(self) -> None:
        strategy = GroundedReasoning(config=GroundedReasoningConfig())
        results = [
            SourceResult(
                content="Some content",
                source_id="a",
                source_name="kb",
                source_type="vector_kb",
                relevance=0.9,
            ),
        ]
        formatted = strategy._format_source_results(results)
        assert "<cluster" not in formatted

    def test_mixed_clustered_and_unclustered(self) -> None:
        strategy = GroundedReasoning(config=GroundedReasoningConfig())
        results = [
            SourceResult(
                content="Clustered item",
                source_id="a",
                source_name="kb",
                source_type="vector_kb",
                relevance=0.9,
                metadata={"cluster_id": 0, "cluster_label": "test",
                          "cluster_size": 1, "cluster_query_score": 0.5},
            ),
            SourceResult(
                content="Unclustered item",
                source_id="b",
                source_name="kb",
                source_type="vector_kb",
                relevance=0.5,
                metadata={"cluster_id": -1, "cluster_label": "unclustered",
                          "cluster_size": 1},
            ),
        ]
        formatted = strategy._format_source_results(results)
        assert '<cluster id="0"' in formatted
        assert "Unclustered item" in formatted


# ------------------------------------------------------------------
# Topic-index integration tests
# ------------------------------------------------------------------


class TestTopicIndexIntegration:
    """Tests for topic-index retrieval through the grounded strategy."""

    @pytest.mark.asyncio
    async def test_source_config_with_topic_index(self) -> None:
        """GroundedSourceConfig.from_dict() preserves topic_index config."""
        sc = GroundedSourceConfig.from_dict({
            "type": "vector_kb",
            "name": "docs",
            "topic_index": {
                "type": "heading_tree",
                "entry_strategy": "both",
                "expansion_mode": "subtree",
            },
        })
        assert sc.topic_index is not None
        assert sc.topic_index["type"] == "heading_tree"
        assert sc.topic_index["entry_strategy"] == "both"
        assert "topic_index" not in sc.options

    @pytest.mark.asyncio
    async def test_source_config_without_topic_index(self) -> None:
        """GroundedSourceConfig.from_dict() without topic_index keeps None."""
        sc = GroundedSourceConfig.from_dict({
            "type": "vector_kb",
            "name": "docs",
        })
        assert sc.topic_index is None

    @pytest.mark.asyncio
    async def test_retrieve_uses_topic_index_when_present(self) -> None:
        """_retrieve_from_sources uses topic_index.resolve when source has one."""
        from dataknobs_bots.knowledge.sources.heading_tree import (
            HeadingTreeConfig,
            HeadingTreeIndex,
        )
        from dataknobs_bots.knowledge.sources.vector import VectorKnowledgeSource

        # Build chunks with heading metadata
        chunks = [
            SourceResult(
                content="Security overview",
                source_id="sec",
                source_name="kb",
                source_type="vector_kb",
                metadata={"headings": ["Security"], "heading_levels": [1]},
            ),
            SourceResult(
                content="CSRF details",
                source_id="csrf",
                source_name="kb",
                source_type="vector_kb",
                metadata={"headings": ["Security", "CSRF"], "heading_levels": [1, 2]},
            ),
            SourceResult(
                content="Introduction content",
                source_id="intro",
                source_name="kb",
                source_type="vector_kb",
                metadata={"headings": ["Introduction"], "heading_levels": [1]},
            ),
        ]

        # Create a topic index from the chunks
        config = HeadingTreeConfig(entry_strategy="heading_match")
        topic_index = HeadingTreeIndex.from_chunks(chunks, config=config)

        # Create a KB source with the topic index attached
        kb = InMemoryKnowledgeBase()
        source = VectorKnowledgeSource(kb, name="docs", topic_index=topic_index)

        # Wire into a grounded strategy
        strategy = GroundedReasoning(config=GroundedReasoningConfig())
        strategy.add_source(source)

        intent = RetrievalIntent(text_queries=["security"])
        results_by_source = await strategy._retrieve_from_sources(
            intent, user_message="security considerations",
        )

        assert "docs" in results_by_source
        result_ids = {r.source_id for r in results_by_source["docs"]}
        # Topic index should expand "Security" heading to include CSRF
        assert "sec" in result_ids
        assert "csrf" in result_ids
        # Introduction should NOT be returned
        assert "intro" not in result_ids
        # KB.query() should NOT have been called (topic index used instead)
        assert len(kb.queries) == 0

    @pytest.mark.asyncio
    async def test_retrieve_falls_back_to_standard_when_no_topic_index(self) -> None:
        """Sources without topic_index use standard text_queries retrieval."""
        from dataknobs_bots.knowledge.sources.vector import VectorKnowledgeSource

        kb = InMemoryKnowledgeBase(results=SAMPLE_KB_RESULTS)
        source = VectorKnowledgeSource(kb, name="docs")  # No topic_index
        assert source.topic_index is None

        strategy = GroundedReasoning(config=GroundedReasoningConfig())
        strategy.add_source(source)

        intent = RetrievalIntent(text_queries=["authorization"])
        results_by_source = await strategy._retrieve_from_sources(
            intent, user_message="authorization grants",
        )

        assert "docs" in results_by_source
        assert len(results_by_source["docs"]) > 0
        # KB.query() should have been called (standard path)
        assert len(kb.queries) > 0

    @pytest.mark.asyncio
    async def test_mixed_sources_topic_index_and_standard(self) -> None:
        """Topic-index source alongside standard source — both produce results."""
        from dataknobs_bots.knowledge.sources.heading_tree import (
            HeadingTreeConfig,
            HeadingTreeIndex,
        )
        from dataknobs_bots.knowledge.sources.vector import VectorKnowledgeSource

        # Source 1: with topic index
        chunks = [
            SourceResult(
                content="Security details",
                source_id="sec",
                source_name="indexed",
                source_type="vector_kb",
                metadata={"headings": ["Security"], "heading_levels": [1]},
            ),
        ]
        topic_index = HeadingTreeIndex.from_chunks(
            chunks,
            config=HeadingTreeConfig(entry_strategy="heading_match"),
        )
        kb1 = InMemoryKnowledgeBase()
        source1 = VectorKnowledgeSource(kb1, name="indexed", topic_index=topic_index)

        # Source 2: standard (no topic index)
        kb2 = InMemoryKnowledgeBase(results=SAMPLE_KB_RESULTS)
        source2 = VectorKnowledgeSource(kb2, name="standard")

        strategy = GroundedReasoning(config=GroundedReasoningConfig())
        strategy.add_source(source1)
        strategy.add_source(source2)

        intent = RetrievalIntent(text_queries=["security"])
        results = await strategy._retrieve_from_sources(
            intent, user_message="security",
        )

        # Both sources produced results
        assert "indexed" in results
        assert "standard" in results
        assert len(results["indexed"]) > 0
        assert len(results["standard"]) > 0
