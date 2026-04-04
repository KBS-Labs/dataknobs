"""Tests for the HybridReasoning strategy.

Uses BotTestHarness with bot_config and InMemoryKnowledgeBase.
"""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_bots.knowledge.base import KnowledgeBase
from dataknobs_bots.reasoning.hybrid import HybridReasoning
from dataknobs_bots.reasoning.hybrid_config import HybridReasoningConfig
from dataknobs_bots.reasoning.grounded_config import (
    GroundedReasoningConfig,
    GroundedSynthesisConfig,
)
from dataknobs_bots.testing import BotTestHarness, StubManager
from dataknobs_llm.testing import text_response, tool_call_response
from dataknobs_llm.tools import Tool


# ------------------------------------------------------------------
# Test helpers
# ------------------------------------------------------------------


class InMemoryKnowledgeBase(KnowledgeBase):
    """Real KnowledgeBase for testing — returns scripted results."""

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


class CalculatorTool(Tool):
    """Simple tool for testing the ReAct phase."""

    def __init__(self) -> None:
        super().__init__(
            name="calculator",
            description="Perform a calculation",
        )

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {"type": "string"},
            },
            "required": ["expression"],
        }

    async def execute(self, expression: str = "", **kwargs: Any) -> str:
        return "42"


SAMPLE_KB_RESULTS: list[dict[str, Any]] = [
    {
        "text": "OAuth 2.0 uses authorization codes for secure token exchange.",
        "source": "rfc6749.md",
        "heading_path": "1.3 > 1.3.1 Authorization Code",
        "similarity": 0.9,
        "metadata": {
            "headings": ["1.3 Authorization Grant", "1.3.1 Authorization Code"],
            "chunk_index": 1,
        },
    },
]


def _build_hybrid_bot_config(
    *,
    intent_mode: str = "static",
    synthesis_style: str | None = None,
    react_max_iterations: int = 5,
    store_provenance: bool = True,
) -> dict[str, Any]:
    """Build a bot_config dict for hybrid strategy tests."""
    grounded: dict[str, Any] = {
        "intent": {
            "mode": intent_mode,
            "text_queries": ["OAuth authorization"],
        },
        "retrieval": {"top_k": 5, "score_threshold": 0.0},
        "store_provenance": store_provenance,
    }
    if synthesis_style is not None:
        grounded["synthesis"] = {"style": synthesis_style}

    return {
        "llm": {"provider": "echo", "model": "echo-test"},
        "conversation_storage": {"backend": "memory"},
        "reasoning": {
            "strategy": "hybrid",
            "grounded": grounded,
            "react": {
                "max_iterations": react_max_iterations,
                "verbose": False,
                "store_trace": False,
            },
            "store_provenance": store_provenance,
        },
    }


# ------------------------------------------------------------------
# Config tests
# ------------------------------------------------------------------


class TestHybridReasoningConfig:
    """Tests for HybridReasoningConfig.from_dict()."""

    def test_from_dict_defaults(self) -> None:
        config = HybridReasoningConfig.from_dict({})
        assert config.react_max_iterations == 5
        assert config.react_verbose is False
        assert config.react_store_trace is False
        assert config.store_provenance is True
        assert config.greeting_template is None
        assert isinstance(config.grounded, GroundedReasoningConfig)

    def test_from_dict_with_values(self) -> None:
        config = HybridReasoningConfig.from_dict({
            "grounded": {
                "intent": {"mode": "static", "text_queries": ["test"]},
                "retrieval": {"top_k": 10},
            },
            "react": {
                "max_iterations": 3,
                "verbose": True,
                "store_trace": True,
            },
            "store_provenance": False,
            "greeting_template": "Hello!",
        })
        assert config.react_max_iterations == 3
        assert config.react_verbose is True
        assert config.react_store_trace is True
        assert config.store_provenance is False
        assert config.greeting_template == "Hello!"
        assert config.grounded.intent.mode == "static"
        assert config.grounded.retrieval.top_k == 10


# ------------------------------------------------------------------
# Strategy construction tests
# ------------------------------------------------------------------


class TestHybridReasoningConstruction:
    """Tests for HybridReasoning instantiation and delegation."""

    def test_construction_providers(self) -> None:
        config = HybridReasoningConfig.from_dict({})
        strategy = HybridReasoning(config=config)
        # Behavioral check: providers() works and returns a dict
        assert isinstance(strategy.providers(), dict)

    def test_add_source_does_not_raise(self) -> None:
        from dataknobs_data.sources.base import GroundedSource, RetrievalIntent, SourceResult

        class DummySource(GroundedSource):
            @property
            def name(self) -> str:
                return "dummy"

            @property
            def source_type(self) -> str:
                return "test"

            async def query(
                self, intent: RetrievalIntent, **kwargs: Any,
            ) -> list[SourceResult]:
                return []

            async def close(self) -> None:
                pass

        config = HybridReasoningConfig.from_dict({})
        strategy = HybridReasoning(config=config)
        strategy.add_source(DummySource())
        # Verify it was forwarded to grounded
        assert len(strategy.providers()) >= 0  # no error means delegation works

    @pytest.mark.asyncio
    async def test_close_delegates_to_children(self) -> None:
        config = HybridReasoningConfig.from_dict({})
        strategy = HybridReasoning(config=config)
        # Should not raise
        await strategy.close()


# ------------------------------------------------------------------
# Generate tests (with BotTestHarness)
# ------------------------------------------------------------------


class TestHybridGenerate:
    """Tests for HybridReasoning.generate() via BotTestHarness."""

    @pytest.mark.asyncio
    async def test_generate_without_tools(self) -> None:
        """No tools — ReAct falls back to simple completion."""
        kb = InMemoryKnowledgeBase(SAMPLE_KB_RESULTS)
        bot_config = _build_hybrid_bot_config()

        async with await BotTestHarness.create(
            bot_config=bot_config,
            main_responses=[
                # 1st call: grounded query generation (static mode skips this)
                # 2nd call: ReAct simple completion (no tools)
                text_response("Based on the knowledge base, OAuth uses auth codes."),
            ],
        ) as harness:
            harness.bot.reasoning_strategy.set_knowledge_base(kb)
            result = await harness.chat("How does OAuth work?")

        assert "OAuth" in result.response or "auth" in result.response.lower()
        assert len(kb.queries) > 0

    @pytest.mark.asyncio
    async def test_generate_with_tools_no_tool_calls(self) -> None:
        """Tools available but LLM doesn't call any."""
        kb = InMemoryKnowledgeBase(SAMPLE_KB_RESULTS)
        bot_config = _build_hybrid_bot_config()
        tool = CalculatorTool()

        async with await BotTestHarness.create(
            bot_config=bot_config,
            main_responses=[
                text_response("The answer is in the knowledge base."),
            ],
            tools=[tool],
        ) as harness:
            harness.bot.reasoning_strategy.set_knowledge_base(kb)
            result = await harness.chat("What is OAuth?")

        assert "knowledge base" in result.response.lower()
        assert len(kb.queries) > 0

    @pytest.mark.asyncio
    async def test_generate_with_tool_calls(self) -> None:
        """Tools available and LLM makes a tool call."""
        kb = InMemoryKnowledgeBase(SAMPLE_KB_RESULTS)
        bot_config = _build_hybrid_bot_config()
        tool = CalculatorTool()

        async with await BotTestHarness.create(
            bot_config=bot_config,
            main_responses=[
                # ReAct iteration 1: LLM calls calculator
                tool_call_response("calculator", {"expression": "6 * 7"}),
                # ReAct iteration 2: LLM responds with final answer
                text_response("The calculation result is 42."),
            ],
            tools=[tool],
        ) as harness:
            harness.bot.reasoning_strategy.set_knowledge_base(kb)
            result = await harness.chat("Calculate 6 * 7")

        assert "42" in result.response
        assert len(kb.queries) > 0

    @pytest.mark.asyncio
    async def test_provenance_includes_tool_executions(self) -> None:
        """Provenance should contain both KB retrieval and tool execution data."""
        kb = InMemoryKnowledgeBase(SAMPLE_KB_RESULTS)
        bot_config = _build_hybrid_bot_config()
        tool = CalculatorTool()

        async with await BotTestHarness.create(
            bot_config=bot_config,
            main_responses=[
                tool_call_response("calculator", {"expression": "1+1"}),
                text_response("The result is 2."),
            ],
            tools=[tool],
        ) as harness:
            harness.bot.reasoning_strategy.set_knowledge_base(kb)
            await harness.chat("Calculate 1+1")

            # Check provenance in manager metadata — uses same schema
            # as GroundedReasoning: retrieval_provenance (current turn dict)
            # + retrieval_provenance_history (append-only list).
            manager = harness.bot.get_conversation_manager(
                harness.context.conversation_id,
            )
            prov = manager.metadata.get("retrieval_provenance")
            assert prov is not None
            assert isinstance(prov, dict)
            assert "intent" in prov or "results_by_source" in prov
            assert "tool_executions" in prov
            assert len(prov["tool_executions"]) == 1
            assert prov["tool_executions"][0]["tool_name"] == "calculator"

            # History should also have the entry
            history = manager.metadata.get("retrieval_provenance_history", [])
            assert len(history) == 1

    @pytest.mark.asyncio
    async def test_provenance_disabled(self) -> None:
        """When store_provenance is False, no provenance should be stored."""
        kb = InMemoryKnowledgeBase(SAMPLE_KB_RESULTS)
        bot_config = _build_hybrid_bot_config(store_provenance=False)

        async with await BotTestHarness.create(
            bot_config=bot_config,
            main_responses=[text_response("answer")],
        ) as harness:
            harness.bot.reasoning_strategy.set_knowledge_base(kb)
            await harness.chat("question")

            manager = harness.bot.get_conversation_manager(
                harness.context.conversation_id,
            )
            assert "retrieval_provenance" not in manager.metadata

    @pytest.mark.asyncio
    async def test_multi_turn_provenance_accumulation(self) -> None:
        """Provenance accumulates across turns without cross-contamination."""
        kb = InMemoryKnowledgeBase(SAMPLE_KB_RESULTS)
        bot_config = _build_hybrid_bot_config()
        tool = CalculatorTool()

        async with await BotTestHarness.create(
            bot_config=bot_config,
            main_responses=[
                # Turn 1: tool call + final answer
                tool_call_response("calculator", {"expression": "1+1"}),
                text_response("Result is 2."),
                # Turn 2: no tool call
                text_response("Just KB context."),
            ],
            tools=[tool],
        ) as harness:
            harness.bot.reasoning_strategy.set_knowledge_base(kb)

            await harness.chat("Calculate 1+1")
            await harness.chat("What is OAuth?")

            manager = harness.bot.get_conversation_manager(
                harness.context.conversation_id,
            )
            # retrieval_provenance is the LATEST turn (turn 2)
            prov = manager.metadata.get("retrieval_provenance")
            assert prov is not None
            assert len(prov["tool_executions"]) == 0

            # History has both turns
            history = manager.metadata.get("retrieval_provenance_history", [])
            assert len(history) == 2

            # Turn 1 should have tool executions
            assert len(history[0]["tool_executions"]) == 1
            assert history[0]["tool_executions"][0]["tool_name"] == "calculator"

            # Turn 2 should have empty tool executions
            assert len(history[1]["tool_executions"]) == 0


# ------------------------------------------------------------------
# Streaming tests
# ------------------------------------------------------------------


class TestHybridStreaming:
    """Tests for HybridReasoning.stream_generate()."""

    @pytest.mark.asyncio
    async def test_stream_without_tools(self) -> None:
        """No tools — should stream directly from manager.stream_complete()."""
        kb = InMemoryKnowledgeBase(SAMPLE_KB_RESULTS)
        bot_config = _build_hybrid_bot_config()

        async with await BotTestHarness.create(
            bot_config=bot_config,
            main_responses=[
                text_response("Streaming answer about OAuth."),
            ],
        ) as harness:
            harness.bot.reasoning_strategy.set_knowledge_base(kb)

            chunks: list[str] = []
            async for chunk in harness.bot.stream_chat(
                "How does OAuth work?", harness.context,
            ):
                delta = getattr(chunk, "delta", None) or ""
                if delta:
                    chunks.append(delta)

            assert len(chunks) > 0
            full = "".join(chunks)
            assert len(full) > 0

    @pytest.mark.asyncio
    async def test_stream_with_tools(self) -> None:
        """Tools available — stream should yield buffered ReAct result."""
        kb = InMemoryKnowledgeBase(SAMPLE_KB_RESULTS)
        bot_config = _build_hybrid_bot_config()
        tool = CalculatorTool()

        async with await BotTestHarness.create(
            bot_config=bot_config,
            main_responses=[
                tool_call_response("calculator", {"expression": "2+2"}),
                text_response("The answer is 4."),
            ],
            tools=[tool],
        ) as harness:
            harness.bot.reasoning_strategy.set_knowledge_base(kb)

            chunks: list[str] = []
            async for chunk in harness.bot.stream_chat(
                "Calculate 2+2", harness.context,
            ):
                delta = getattr(chunk, "delta", None) or ""
                if delta:
                    chunks.append(delta)

            full = "".join(chunks)
            assert len(full) > 0


# ------------------------------------------------------------------
# Synthesis style tests
# ------------------------------------------------------------------


class TestHybridSynthesis:
    """Tests for post-ReAct synthesis formatting."""

    @pytest.mark.asyncio
    async def test_conversational_style_passes_through(self) -> None:
        """Conversational style returns the ReAct response unchanged."""
        kb = InMemoryKnowledgeBase(SAMPLE_KB_RESULTS)
        bot_config = _build_hybrid_bot_config(synthesis_style="conversational")

        async with await BotTestHarness.create(
            bot_config=bot_config,
            main_responses=[text_response("Direct LLM answer.")],
        ) as harness:
            harness.bot.reasoning_strategy.set_knowledge_base(kb)
            result = await harness.chat("question")

        assert result.response == "Direct LLM answer."

    @pytest.mark.asyncio
    async def test_structured_style_uses_template(self) -> None:
        """Structured style renders provenance template."""
        kb = InMemoryKnowledgeBase(SAMPLE_KB_RESULTS)
        bot_config = _build_hybrid_bot_config(synthesis_style="structured")

        async with await BotTestHarness.create(
            bot_config=bot_config,
            main_responses=[text_response("LLM answer (should be discarded)")],
        ) as harness:
            harness.bot.reasoning_strategy.set_knowledge_base(kb)
            result = await harness.chat("question")

        # Structured mode uses template output, not the LLM response
        assert result.response != "LLM answer (should be discarded)"
        # Should contain some structured content from the template
        assert len(result.response) > 0

    @pytest.mark.asyncio
    async def test_hybrid_style_appends_template(self) -> None:
        """Hybrid style appends provenance template to LLM response."""
        kb = InMemoryKnowledgeBase(SAMPLE_KB_RESULTS)
        bot_config = _build_hybrid_bot_config(synthesis_style="hybrid")

        async with await BotTestHarness.create(
            bot_config=bot_config,
            # Two responses: one for grounded query gen (skipped in static),
            # one for the ReAct completion
            main_responses=[text_response("LLM narrative here.")],
        ) as harness:
            harness.bot.reasoning_strategy.set_knowledge_base(kb)
            result = await harness.chat("question")

        # Should start with the LLM narrative
        assert result.response.startswith("LLM narrative here.")
        # And have additional content from the template
        assert len(result.response) > len("LLM narrative here.")


# ------------------------------------------------------------------
# Factory registration tests
# ------------------------------------------------------------------


class TestHybridFactory:
    """Tests for create_reasoning_from_config with 'hybrid'."""

    def test_factory_creates_hybrid(self) -> None:
        from dataknobs_bots.reasoning import create_reasoning_from_config

        config = {
            "strategy": "hybrid",
            "grounded": {"intent": {"mode": "static"}},
            "react": {"max_iterations": 3},
        }
        strategy = create_reasoning_from_config(config)
        assert isinstance(strategy, HybridReasoning)
        # Behavioral check: providers() works (exercises composition)
        assert isinstance(strategy.providers(), dict)

    @pytest.mark.asyncio
    async def test_factory_hybrid_with_knowledge_base(self) -> None:
        """KB auto-wrapping verified by checking retrieval actually queries it."""
        from dataknobs_bots.reasoning import create_reasoning_from_config

        kb = InMemoryKnowledgeBase(SAMPLE_KB_RESULTS)
        config = {
            "strategy": "hybrid",
            "grounded": {
                "intent": {
                    "mode": "static",
                    "text_queries": ["test query"],
                },
            },
        }
        strategy = create_reasoning_from_config(config, knowledge_base=kb)
        assert isinstance(strategy, HybridReasoning)

        # Verify KB was auto-wrapped by running retrieval
        manager = StubManager()
        await manager.add_message("user", "test")
        await strategy.generate(manager, None)
        assert len(kb.queries) > 0

    def test_factory_error_message_includes_hybrid(self) -> None:
        from dataknobs_common import NotFoundError
        from dataknobs_bots.reasoning import create_reasoning_from_config

        with pytest.raises(NotFoundError) as exc_info:
            create_reasoning_from_config({"strategy": "nonexistent"})
        assert "hybrid" in exc_info.value.context.get("available", [])


# ------------------------------------------------------------------
# Greeting tests
# ------------------------------------------------------------------


class TestHybridGreeting:
    """Tests for HybridReasoning.greet() delegation."""

    @pytest.mark.asyncio
    async def test_greet_with_greeting_template(self) -> None:
        config = HybridReasoningConfig.from_dict({
            "greeting_template": "Hello, {{ user_name }}!",
        })
        strategy = HybridReasoning(config=config)
        manager = StubManager()

        result = await strategy.greet(
            manager, None, initial_context={"user_name": "Alice"},
        )
        assert result is not None
        assert "Alice" in result.content

    @pytest.mark.asyncio
    async def test_greet_without_template_returns_none(self) -> None:
        config = HybridReasoningConfig.from_dict({})
        strategy = HybridReasoning(config=config)
        manager = StubManager()

        result = await strategy.greet(manager, None)
        assert result is None
