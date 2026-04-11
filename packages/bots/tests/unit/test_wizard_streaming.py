"""Tests for wizard streaming phased protocol (item 80).

Validates:
- WizardReasoning satisfies StreamingPhasedProtocol
- stream_finalize_turn yields LLMStreamResponse chunks
- Template-rendered stages produce a single chunk
- End-to-end stream_chat with wizard yields streaming chunks
- Early response paths emit single chunks
- Auto-advance messages prepended before streamed response
- Wizard state saved only after full stream consumption
- Non-streaming phased fallback still works via single-chunk path
"""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_bots.reasoning.base import (
    PhasedReasoningProtocol,
    StreamingPhasedProtocol,
)
from dataknobs_bots.testing import BotTestHarness, WizardConfigBuilder
from dataknobs_bots.tools.bank_tools import RestartWizardTool
from dataknobs_llm import LLMStreamResponse
from dataknobs_llm.testing import text_response, tool_call_response


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_wizard_config() -> dict[str, Any]:
    """Two-stage wizard: gather name, then done."""
    return (
        WizardConfigBuilder("streaming-test")
        .stage("gather", is_start=True, prompt="What is your name?")
        .field("name", field_type="string", required=True)
        .transition("done", "data.get('name')")
        .stage("done", is_end=True, prompt="All done, {{ name }}!")
        .build()
    )


@pytest.fixture
def template_wizard_config() -> dict[str, Any]:
    """Wizard with a template-rendered landing stage."""
    return (
        WizardConfigBuilder("template-test")
        .stage("gather", is_start=True, prompt="What is your name?")
        .field("name", field_type="string", required=True)
        .transition("review", "data.get('name')")
        .stage(
            "review",
            is_end=True,
            prompt="Review your data.",
            response_template="Hello {{ name }}, review complete.",
        )
        .build()
    )


@pytest.fixture
def auto_advance_wizard_config() -> dict[str, Any]:
    """Wizard with a message stage that auto-advances."""
    return (
        WizardConfigBuilder("auto-advance-test")
        .stage("gather", is_start=True, prompt="What is your name?")
        .field("name", field_type="string", required=True)
        .transition("confirmation", "data.get('name')")
        .stage(
            "confirmation",
            prompt="Noted.",
            auto_advance=True,
            response_template="Got it, {{ name }}!",
        )
        .transition("final", "true")
        .stage("final", is_end=True, prompt="All done!")
        .build()
    )


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestStreamingProtocolConformance:
    """WizardReasoning satisfies StreamingPhasedProtocol."""

    @pytest.mark.asyncio
    async def test_wizard_satisfies_streaming_protocol(
        self, simple_wizard_config: dict[str, Any]
    ) -> None:
        """WizardReasoning is an instance of StreamingPhasedProtocol."""
        async with await BotTestHarness.create(
            wizard_config=simple_wizard_config,
            main_responses=["Hi"],
        ) as harness:
            strategy = harness.bot.reasoning_strategy
            assert isinstance(strategy, StreamingPhasedProtocol)
            # Also satisfies the base protocol (subtype relationship)
            assert isinstance(strategy, PhasedReasoningProtocol)

    def test_non_wizard_not_streaming_protocol(self) -> None:
        """SimpleReasoning does not satisfy StreamingPhasedProtocol."""
        from dataknobs_bots.reasoning.simple import SimpleReasoning

        strategy = SimpleReasoning()
        assert not isinstance(strategy, StreamingPhasedProtocol)
        assert not isinstance(strategy, PhasedReasoningProtocol)


# ---------------------------------------------------------------------------
# End-to-end streaming via BotTestHarness.stream_chat()
# ---------------------------------------------------------------------------


class TestWizardStreamChat:
    """End-to-end streaming through DynaBot.stream_chat() with wizard."""

    @pytest.mark.asyncio
    async def test_stream_chat_wizard_yields_chunks(
        self, simple_wizard_config: dict[str, Any]
    ) -> None:
        """stream_chat with wizard yields LLMStreamResponse chunks."""
        async with await BotTestHarness.create(
            wizard_config=simple_wizard_config,
            main_responses=["Got it!"],
            extraction_results=[[{"name": "Alice"}]],
        ) as harness:
            chunks: list[LLMStreamResponse] = []
            async for chunk in harness.bot.stream_chat(
                "My name is Alice", harness.context
            ):
                chunks.append(chunk)

            assert len(chunks) > 0
            assert all(isinstance(c, LLMStreamResponse) for c in chunks)
            # At least the final chunk should be marked final
            assert any(c.is_final for c in chunks)
            # Full text is non-empty
            full_text = "".join(c.delta for c in chunks)
            assert len(full_text) > 0

    @pytest.mark.asyncio
    async def test_stream_chat_early_response_single_chunk(
        self, simple_wizard_config: dict[str, Any]
    ) -> None:
        """Clarification early return emits as single chunk.

        Empty extraction (no fields extracted) triggers clarification.
        The clarification LLM call needs a main_response, plus the
        extraction call itself needs a result entry.
        """
        async with await BotTestHarness.create(
            wizard_config=simple_wizard_config,
            # First response: clarification LLM call
            main_responses=["Could you clarify your name?"],
            # Empty dict = extraction found nothing for required field
            extraction_results=[[{}]],
        ) as harness:
            chunks: list[LLMStreamResponse] = []
            async for chunk in harness.bot.stream_chat(
                "hmm", harness.context
            ):
                chunks.append(chunk)

            # Clarification is a single chunk (early return, not streamed)
            assert len(chunks) == 1
            assert chunks[0].is_final is True
            assert len(chunks[0].delta) > 0

    @pytest.mark.asyncio
    async def test_stream_chat_template_single_chunk(
        self, template_wizard_config: dict[str, Any]
    ) -> None:
        """Template-rendered stage response emits as single chunk."""
        async with await BotTestHarness.create(
            wizard_config=template_wizard_config,
            main_responses=["Here we go"],
            extraction_results=[[{"name": "Bob"}]],
        ) as harness:
            chunks: list[LLMStreamResponse] = []
            async for chunk in harness.bot.stream_chat(
                "My name is Bob", harness.context
            ):
                chunks.append(chunk)

            # Template stage emits single chunk
            assert len(chunks) == 1
            assert chunks[0].is_final is True
            assert "Bob" in chunks[0].delta

    @pytest.mark.asyncio
    async def test_stream_chat_state_saved_after_consumption(
        self, simple_wizard_config: dict[str, Any]
    ) -> None:
        """Wizard state is saved after stream is fully consumed."""
        async with await BotTestHarness.create(
            wizard_config=simple_wizard_config,
            main_responses=["Got it!"],
            extraction_results=[[{"name": "Alice"}]],
        ) as harness:
            # Use harness.stream_chat() which consumes fully and
            # snapshots wizard state
            result = await harness.stream_chat("My name is Alice")

            assert result.wizard_data.get("name") == "Alice"
            assert result.wizard_stage == "done"
            # Also verify via harness properties
            assert harness.wizard_data.get("name") == "Alice"
            assert harness.wizard_stage == "done"
            # Verify chunks were collected
            assert len(result.chunks) > 0

    @pytest.mark.asyncio
    async def test_auto_advance_prepended_before_stream(
        self, auto_advance_wizard_config: dict[str, Any]
    ) -> None:
        """Auto-advance messages appear before the streamed response."""
        async with await BotTestHarness.create(
            wizard_config=auto_advance_wizard_config,
            main_responses=["Welcome to the final stage!"],
            extraction_results=[[{"name": "Carol"}]],
        ) as harness:
            result = await harness.stream_chat("My name is Carol")

            # Auto-advance message content should be present
            assert "Carol" in result.response

    @pytest.mark.asyncio
    async def test_stream_chat_greet_single_chunk(
        self, simple_wizard_config: dict[str, Any]
    ) -> None:
        """greet() still works (non-streaming path, single chunk)."""
        async with await BotTestHarness.create(
            wizard_config=simple_wizard_config,
            main_responses=["Welcome!"],
        ) as harness:
            result = await harness.greet()
            assert result.response is not None
            assert len(result.response) > 0


# ---------------------------------------------------------------------------
# Streaming with tool_result_mapping
# ---------------------------------------------------------------------------


class _ProductLookupTool:
    """Minimal tool for testing tool_result_mapping in streaming path."""

    name = "product_lookup"
    description = "Look up a product by name"

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }

    async def execute(self, query: str = "", **_kwargs: Any) -> dict[str, Any]:
        return {"product_id": "PROD-42", "category": "widgets"}


class TestStreamingToolResultMapping:
    """Streaming path with tool_result_mapping interleaving."""

    @pytest.mark.asyncio
    async def test_tool_result_mapping_via_stream_chat(self) -> None:
        """extraction → tool execution → result mapping works in streaming."""
        config = (
            WizardConfigBuilder("tool-stream-test")
            .stage(
                "lookup",
                is_start=True,
                prompt="What product?",
                tool_result_mapping=[
                    {
                        "tool": "product_lookup",
                        "params": {"query": "product_name"},
                        "mapping": {
                            "product_id": "product_id",
                            "category": "product_category",
                        },
                    },
                ],
            )
            .field("product_name", field_type="string", required=True)
            .transition("review", "has('product_id')")
            .stage("review", is_end=True, prompt="Details below.")
            .build()
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Looking up...", "Here are the details."],
            extraction_results=[[{"product_name": "Widget"}]],
            tools=[_ProductLookupTool()],
        ) as harness:
            result = await harness.stream_chat("I want Widget")

            # Tool result mapped into wizard state
            assert harness.wizard_data.get("product_id") == "PROD-42"
            assert harness.wizard_data.get("product_category") == "widgets"
            assert harness.wizard_data.get("product_name") == "Widget"
            # Transition fired
            assert harness.wizard_stage == "review"
            # Streamed response
            assert len(result.chunks) > 0


# ---------------------------------------------------------------------------
# Streaming restart cleanup (finding 2 fix)
# ---------------------------------------------------------------------------


class TestStreamingRestartCleanup:
    """Tool-initiated restart in streaming performs cleanup without extra yield."""

    @pytest.mark.asyncio
    async def test_restart_tool_resets_wizard_state_in_streaming(self) -> None:
        """RestartWizardTool during streaming cleans up state, FSM returns to start.

        In the non-streaming path, execute_restart replaces the response.
        In the streaming path, the stage response has already been yielded,
        so restart_cleanup runs without emitting a replacement — the next
        turn picks up the restarted wizard naturally.
        """
        config = (
            WizardConfigBuilder("restart-stream-test")
            .stage(
                "gather",
                is_start=True,
                prompt="What is your name?",
            )
            .field("name", field_type="string", required=True)
            .transition("review", "data.get('name')")
            .stage(
                "review",
                prompt="Let me review that.",
                reasoning="react",
                max_iterations=3,
                tools=["restart_wizard"],
            )
            .transition("done", "true")
            .stage("done", is_end=True, prompt="All done!")
            .build()
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=[
                # Turn 1: extraction succeeds, transitions to "review"
                # Review stage uses ReAct — EchoProvider produces tool call
                # then text response
                tool_call_response("restart_wizard", {}),
                text_response("Restarting..."),
            ],
            extraction_results=[[{"name": "Alice"}]],
            tools=[RestartWizardTool()],
        ) as harness:
            result = await harness.stream_chat("My name is Alice")

            # After restart cleanup, wizard is back at start stage
            assert harness.wizard_stage == "gather"
            # State was cleared by restart_cleanup
            assert harness.wizard_data.get("name") is None
            # Streaming produced output (the review stage response,
            # before restart was detected post-stream)
            assert len(result.chunks) > 0


# ---------------------------------------------------------------------------
# Non-streaming phased fallback
# ---------------------------------------------------------------------------


class TestNonStreamingPhasedFallback:
    """Non-streaming phased strategies fall back to single-chunk wrapping."""

    @pytest.mark.asyncio
    async def test_non_streaming_phased_via_chat(
        self, simple_wizard_config: dict[str, Any]
    ) -> None:
        """chat() (non-streaming) still works with phased wizard."""
        async with await BotTestHarness.create(
            wizard_config=simple_wizard_config,
            main_responses=["Got it!"],
            extraction_results=[[{"name": "Dave"}]],
        ) as harness:
            result = await harness.chat("My name is Dave")
            assert result.response is not None
            assert len(result.response) > 0
            assert harness.wizard_data.get("name") == "Dave"
