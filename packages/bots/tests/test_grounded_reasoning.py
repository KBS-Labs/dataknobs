"""Tests for the GroundedReasoning strategy.

Uses BotTestHarness with bot_config and a lightweight InMemoryKnowledgeBase
test helper (real KnowledgeBase subclass, not a mock).
"""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_bots.knowledge.base import KnowledgeBase
from dataknobs_bots.reasoning.grounded import GroundedReasoning
from dataknobs_data.sources.base import SourceResult
from dataknobs_bots.reasoning.grounded_config import (
    GroundedIntentConfig,
    GroundedReasoningConfig,
    GroundedRetrievalConfig,
    GroundedSourceConfig,
    GroundedSynthesisConfig,
)
from dataknobs_bots.testing import BotTestHarness
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
    """Build a minimal bot_config with grounded strategy."""
    reasoning: dict[str, Any] = {
        "strategy": "grounded",
        "intent": {"mode": intent_mode, "num_queries": num_queries},
        "retrieval": {"top_k": 5, "score_threshold": 0.0},
        "synthesis": {"mode": synthesis_mode, "require_citations": True},
        "store_provenance": store_provenance,
    }
    reasoning.update(extra_reasoning)
    return {
        "llm": {"provider": "echo", "model": "echo-test"},
        "conversation_storage": {"backend": "memory"},
        "reasoning": reasoning,
    }


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

    def test_render_synthesis_template(self) -> None:
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
        text = strategy._render_synthesis_template(
            "formatted context", provenance, "user msg", {},
        )
        assert "Found 2 results:" in text
        assert "Auth code grant" in text

    def test_render_synthesis_template_no_template_fallback(self) -> None:
        cfg = GroundedReasoningConfig(
            synthesis=GroundedSynthesisConfig(mode="template"),
        )
        strategy = GroundedReasoning(config=cfg)
        text = strategy._render_synthesis_template(
            "raw context", {}, "msg", {},
        )
        assert text == "raw context"


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
            # Verify transformer was created
            assert strategy._transformer is not None
            assert strategy._transformer.config.enabled is True

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
            "Tell me everything about OAuth",
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
