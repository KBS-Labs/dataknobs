"""Tests for reasoning strategies."""

import tempfile

import pytest
import yaml

from dataknobs_bots import BotContext, DynaBot
from dataknobs_bots.reasoning import (
    ReActReasoning,
    ReasoningStrategy,
    SimpleReasoning,
    WizardReasoning,
    create_reasoning_from_config,
)


class TestSimpleReasoning:
    """Tests for SimpleReasoning strategy."""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test simple reasoning initialization."""
        strategy = SimpleReasoning()
        assert isinstance(strategy, ReasoningStrategy)

    @pytest.mark.asyncio
    async def test_generate(self):
        """Test simple reasoning generation."""
        # Create a bot with simple reasoning
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "reasoning": {"strategy": "simple"},
        }

        bot = await DynaBot.from_config(config)
        context = BotContext(
            conversation_id="conv-simple", client_id="test-client"
        )

        # Generate response
        response = await bot.chat("Hello", context)
        assert response is not None
        assert isinstance(response, str)


class TestReActReasoning:
    """Tests for ReActReasoning strategy."""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test ReAct reasoning initialization."""
        strategy = ReActReasoning(max_iterations=3, verbose=False)
        assert isinstance(strategy, ReasoningStrategy)
        assert strategy.max_iterations == 3
        assert strategy.verbose is False

    @pytest.mark.asyncio
    async def test_default_initialization(self):
        """Test ReAct with default parameters."""
        strategy = ReActReasoning()
        assert strategy.max_iterations == 5
        assert strategy.verbose is False

    @pytest.mark.asyncio
    async def test_generate_without_tools(self):
        """Test ReAct falls back when no tools available."""
        # Create bot with ReAct but no tools
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "reasoning": {"strategy": "react", "max_iterations": 3},
        }

        bot = await DynaBot.from_config(config)
        context = BotContext(
            conversation_id="conv-react-no-tools", client_id="test-client"
        )

        # Should work fine without tools (falls back to simple)
        response = await bot.chat("Hello", context)
        assert response is not None

    @pytest.mark.asyncio
    async def test_verbose_mode(self):
        """Test verbose mode doesn't break execution."""
        strategy = ReActReasoning(max_iterations=2, verbose=True)
        assert strategy.verbose is True

        # Create bot with verbose ReAct
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "reasoning": {"strategy": "react", "verbose": True, "max_iterations": 2},
        }

        bot = await DynaBot.from_config(config)
        context = BotContext(
            conversation_id="conv-react-verbose", client_id="test-client"
        )

        # Should work with verbose output
        response = await bot.chat("Test", context)
        assert response is not None


class TestReasoningFactory:
    """Tests for reasoning factory function."""

    def test_create_simple_reasoning(self):
        """Test creating simple reasoning from config."""
        config = {"strategy": "simple"}
        strategy = create_reasoning_from_config(config)
        assert isinstance(strategy, SimpleReasoning)

    def test_create_react_reasoning(self):
        """Test creating ReAct reasoning from config."""
        config = {"strategy": "react", "max_iterations": 3, "verbose": True}
        strategy = create_reasoning_from_config(config)
        assert isinstance(strategy, ReActReasoning)
        assert strategy.max_iterations == 3
        assert strategy.verbose is True

    def test_default_strategy(self):
        """Test that default strategy is simple."""
        config = {}
        strategy = create_reasoning_from_config(config)
        assert isinstance(strategy, SimpleReasoning)

    def test_invalid_strategy(self):
        """Test error handling for invalid strategy."""
        config = {"strategy": "invalid"}
        with pytest.raises(ValueError, match="Unknown reasoning strategy"):
            create_reasoning_from_config(config)

    def test_create_wizard_reasoning(self):
        """Test creating wizard reasoning from config."""
        # Create temp wizard config file
        wizard_config = {
            "name": "test-wizard",
            "version": "1.0",
            "stages": [
                {
                    "name": "welcome",
                    "is_start": True,
                    "prompt": "What would you like to do?",
                    "transitions": [{"target": "complete"}],
                },
                {
                    "name": "complete",
                    "is_end": True,
                    "prompt": "All done!",
                },
            ],
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(wizard_config, f)
            config_path = f.name

        config = {
            "strategy": "wizard",
            "wizard_config": config_path,
            "strict_validation": True,
        }

        strategy = create_reasoning_from_config(config)
        assert isinstance(strategy, WizardReasoning)
        assert strategy._strict_validation is True


class TestReActLoopBehavior:
    """Tests for ReAct loop exit paths and trace storage.

    The ReAct loop has four exit paths, each producing distinct trace statuses:
    1. No tool calls    → "completed" (returns directly)
    2. Duplicate calls  → "duplicate_tool_calls_detected" (breaks loop)
    3. Max iterations   → "max_iterations_reached" (loop exhausted)
    4. Normal progress  → "continued" per iteration, "completed" on final
    """

    @staticmethod
    def _make_calculator_tool() -> "Tool":
        """Create a simple calculator tool for testing."""
        from typing import Any
        from dataknobs_llm.tools import Tool

        class CalculatorTool(Tool):
            def __init__(self) -> None:
                super().__init__(
                    name="calculator",
                    description="Performs arithmetic",
                )

            @property
            def schema(self) -> dict[str, Any]:
                return {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string"},
                    },
                }

            async def execute(self, **kwargs: Any) -> str:
                return "42"

        return CalculatorTool()

    @staticmethod
    async def _get_trace(bot: "DynaBot", conversation_id: str) -> list[dict]:
        """Retrieve the stored reasoning trace from conversation metadata."""
        conv = await bot.get_conversation(conversation_id)
        assert conv is not None, "Conversation not found"
        assert conv.metadata is not None, "Conversation metadata is empty"
        trace = conv.metadata.get("reasoning_trace")
        assert trace is not None, "reasoning_trace not in metadata"
        return trace

    # -- Exit path 1: No tool calls → "completed" ---

    @pytest.mark.asyncio
    async def test_no_tool_calls_trace_shows_completed(self):
        """When the LLM returns no tool calls, trace has a single 'completed' step."""
        from dataknobs_llm.testing import text_response

        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "reasoning": {
                "strategy": "react",
                "max_iterations": 5,
                "store_trace": True,
            },
        }

        bot = await DynaBot.from_config(config)
        bot.tool_registry.register_tool(self._make_calculator_tool())
        context = BotContext(
            conversation_id="conv-completed", client_id="test-client"
        )

        # LLM responds with text only (no tool calls) on the first iteration
        bot.llm.set_responses([text_response("The answer is 42")])

        response = await bot.chat("What is the answer?", context)
        assert "42" in response

        # Only 1 LLM call needed (no tool loop)
        assert bot.llm.call_count == 1

        trace = await self._get_trace(bot, "conv-completed")
        assert len(trace) == 1
        assert trace[0]["status"] == "completed"
        assert trace[0]["iteration"] == 1
        assert trace[0]["tool_calls"] == []

    # -- Exit path 2: Duplicate tool calls → "duplicate_tool_calls_detected" ---

    @pytest.mark.asyncio
    async def test_duplicate_tool_calls_break_loop(self):
        """Repeated identical tool calls break the loop early."""
        from dataknobs_llm.testing import text_response, tool_call_response

        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "reasoning": {
                "strategy": "react",
                "max_iterations": 5,
                "store_trace": True,
            },
        }

        bot = await DynaBot.from_config(config)
        bot.tool_registry.register_tool(self._make_calculator_tool())
        context = BotContext(
            conversation_id="conv-react-dup", client_id="test-client"
        )

        # Same tool call twice, then a final text response (post-loop)
        bot.llm.set_responses([
            tool_call_response("calculator", {"expression": "247 * 39"}),
            tool_call_response("calculator", {"expression": "247 * 39"}),
            text_response("The answer is 9633"),
        ])

        response = await bot.chat("What is 247 * 39?", context)
        assert response is not None
        assert "9633" in response

        # 1st tool call + 2nd (duplicate detected) + final text = 3 calls
        assert bot.llm.call_count == 3

    @pytest.mark.asyncio
    async def test_duplicate_detection_trace_status(self):
        """Duplicate detection sets 'duplicate_tool_calls_detected', not 'max_iterations_reached'."""
        from dataknobs_llm.testing import text_response, tool_call_response

        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "reasoning": {
                "strategy": "react",
                "max_iterations": 5,
                "store_trace": True,
            },
        }

        bot = await DynaBot.from_config(config)
        bot.tool_registry.register_tool(self._make_calculator_tool())
        context = BotContext(
            conversation_id="conv-dup-trace", client_id="test-client"
        )

        bot.llm.set_responses([
            tool_call_response("calculator", {"expression": "2+2"}),
            tool_call_response("calculator", {"expression": "2+2"}),
            text_response("4"),
        ])

        await bot.chat("What is 2+2?", context)

        trace = await self._get_trace(bot, "conv-dup-trace")

        # Iteration 1: tool executed normally → "continued"
        # Iteration 2: duplicate detected → "duplicate_tool_calls_detected"
        assert len(trace) == 2
        assert trace[0]["status"] == "continued"
        assert trace[0]["iteration"] == 1
        assert len(trace[0]["tool_calls"]) == 1
        assert trace[0]["tool_calls"][0]["name"] == "calculator"
        assert trace[0]["tool_calls"][0]["status"] == "success"

        assert trace[1]["status"] == "duplicate_tool_calls_detected"
        assert trace[1]["iteration"] == 2
        # No tool_calls executed on the duplicate iteration (break before execution)
        assert trace[1]["tool_calls"] == []

        # Crucially, "max_iterations_reached" must NOT appear anywhere
        all_statuses = [step.get("status") for step in trace]
        assert "max_iterations_reached" not in all_statuses

    # -- Exit path 3: Max iterations exhausted → "max_iterations_reached" ---

    @pytest.mark.asyncio
    async def test_max_iterations_trace_status(self):
        """When loop exhausts all iterations, trace ends with 'max_iterations_reached'."""
        from dataknobs_llm.testing import text_response, tool_call_response

        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "reasoning": {
                "strategy": "react",
                "max_iterations": 2,
                "store_trace": True,
            },
        }

        bot = await DynaBot.from_config(config)
        bot.tool_registry.register_tool(self._make_calculator_tool())
        context = BotContext(
            conversation_id="conv-max-iter", client_id="test-client"
        )

        # Different tool calls each iteration so duplicate detection doesn't fire,
        # exhausting max_iterations=2, then a final text response (post-loop)
        bot.llm.set_responses([
            tool_call_response("calculator", {"expression": "1+1"}),
            tool_call_response("calculator", {"expression": "2+2"}),
            text_response("Done"),
        ])

        response = await bot.chat("Keep calculating", context)
        assert response is not None

        # 2 tool iterations + 1 final completion = 3 calls
        assert bot.llm.call_count == 3

        trace = await self._get_trace(bot, "conv-max-iter")

        # 2 "continued" iterations + 1 "max_iterations_reached" sentinel
        assert len(trace) == 3
        assert trace[0]["status"] == "continued"
        assert trace[0]["iteration"] == 1
        assert trace[1]["status"] == "continued"
        assert trace[1]["iteration"] == 2
        assert trace[2] == {"status": "max_iterations_reached"}

        # "duplicate_tool_calls_detected" must NOT appear
        all_statuses = [step.get("status") for step in trace]
        assert "duplicate_tool_calls_detected" not in all_statuses

    # -- Exit path 4: Normal multi-step then completion ---

    @pytest.mark.asyncio
    async def test_normal_completion_after_tool_calls(self):
        """Normal flow: tool iterations followed by text response → 'completed'."""
        from dataknobs_llm.testing import text_response, tool_call_response

        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "reasoning": {
                "strategy": "react",
                "max_iterations": 5,
                "store_trace": True,
            },
        }

        bot = await DynaBot.from_config(config)
        bot.tool_registry.register_tool(self._make_calculator_tool())
        context = BotContext(
            conversation_id="conv-normal", client_id="test-client"
        )

        # Two different tool calls, then LLM returns text (no tools) → completed
        bot.llm.set_responses([
            tool_call_response("calculator", {"expression": "10 * 5"}),
            tool_call_response("calculator", {"expression": "50 / 2"}),
            text_response("The final answer is 25"),
        ])

        response = await bot.chat("Calculate 10*5 then halve it", context)
        assert "25" in response
        assert bot.llm.call_count == 3

        trace = await self._get_trace(bot, "conv-normal")

        # 2 "continued" iterations + 1 "completed" (when LLM returned text)
        assert len(trace) == 3
        assert trace[0]["status"] == "continued"
        assert trace[0]["iteration"] == 1
        assert len(trace[0]["tool_calls"]) == 1
        assert trace[1]["status"] == "continued"
        assert trace[1]["iteration"] == 2
        assert len(trace[1]["tool_calls"]) == 1
        assert trace[2]["status"] == "completed"
        assert trace[2]["iteration"] == 3
        assert trace[2]["tool_calls"] == []

    # -- Duplicate detection does not fire on different calls ---

    @pytest.mark.asyncio
    async def test_different_tool_calls_do_not_trigger_detection(self):
        """Different tool calls across iterations proceed without early break."""
        from dataknobs_llm.testing import text_response, tool_call_response

        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "reasoning": {
                "strategy": "react",
                "max_iterations": 5,
            },
        }

        bot = await DynaBot.from_config(config)
        bot.tool_registry.register_tool(self._make_calculator_tool())
        context = BotContext(
            conversation_id="conv-react-diff", client_id="test-client"
        )

        # Different tool calls each iteration, then a final text response
        bot.llm.set_responses([
            tool_call_response("calculator", {"expression": "247 * 39"}),
            tool_call_response("calculator", {"expression": "9633 / 3"}),
            text_response("The final answer is 3211"),
        ])

        response = await bot.chat("Calculate 247*39 then divide by 3", context)
        assert response is not None
        assert "3211" in response

        # Both tool calls executed + final response = 3 calls
        assert bot.llm.call_count == 3

    # -- Trace tool call details ---

    @pytest.mark.asyncio
    async def test_trace_records_tool_call_details(self):
        """Trace entries include tool name, parameters, status, and result."""
        from dataknobs_llm.testing import text_response, tool_call_response

        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "reasoning": {
                "strategy": "react",
                "max_iterations": 5,
                "store_trace": True,
            },
        }

        bot = await DynaBot.from_config(config)
        bot.tool_registry.register_tool(self._make_calculator_tool())
        context = BotContext(
            conversation_id="conv-tool-details", client_id="test-client"
        )

        bot.llm.set_responses([
            tool_call_response("calculator", {"expression": "7 * 6"}),
            text_response("42"),
        ])

        await bot.chat("What is 7*6?", context)

        trace = await self._get_trace(bot, "conv-tool-details")
        assert len(trace) == 2  # 1 continued + 1 completed

        tool_entry = trace[0]["tool_calls"][0]
        assert tool_entry["name"] == "calculator"
        assert tool_entry["parameters"] == {"expression": "7 * 6"}
        assert tool_entry["status"] == "success"
        assert tool_entry["result"] == "42"  # CalculatorTool always returns "42"


class TestReActExtraContext:
    """Tests for the extra_context parameter on ReActReasoning."""

    @pytest.mark.asyncio
    async def test_extra_context_propagates_to_tools(self):
        """extra_context values are available in ToolExecutionContext."""
        from typing import Any

        from dataknobs_llm.testing import text_response, tool_call_response
        from dataknobs_llm.tools import ContextAwareTool, ToolExecutionContext

        captured_contexts: list[ToolExecutionContext] = []

        class ContextCaptureTool(ContextAwareTool):
            """Tool that captures its execution context for inspection."""

            def __init__(self) -> None:
                super().__init__(
                    name="capture",
                    description="Captures execution context",
                )

            @property
            def schema(self) -> dict[str, Any]:
                return {"type": "object", "properties": {}}

            async def execute_with_context(
                self, context: ToolExecutionContext, **kwargs: Any
            ) -> str:
                captured_contexts.append(context)
                return "captured"

        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "reasoning": {
                "strategy": "react",
                "max_iterations": 3,
            },
        }

        bot = await DynaBot.from_config(config)
        bot.tool_registry.register_tool(ContextCaptureTool())

        # Inject extra_context into the reasoning strategy
        bot.reasoning_strategy._extra_context = {"my_key": "my_value"}

        context = BotContext(
            conversation_id="conv-extra-ctx", client_id="test-client"
        )

        bot.llm.set_responses([
            tool_call_response("capture", {}),
            text_response("Done"),
        ])

        await bot.chat("Capture context", context)

        assert len(captured_contexts) == 1
        tool_ctx = captured_contexts[0]
        assert tool_ctx.extra.get("my_key") == "my_value"

    @pytest.mark.asyncio
    async def test_extra_context_none_by_default(self):
        """Without extra_context, tools still get context (just no extras)."""
        strategy = ReActReasoning(max_iterations=3)
        assert strategy._extra_context is None


class TestReasoningIntegration:
    """Integration tests for reasoning with bots."""

    @pytest.mark.asyncio
    async def test_bot_with_simple_reasoning(self):
        """Test bot with explicit simple reasoning."""
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "reasoning": {"strategy": "simple"},
        }

        bot = await DynaBot.from_config(config)
        assert bot.reasoning_strategy is not None
        assert isinstance(bot.reasoning_strategy, SimpleReasoning)

    @pytest.mark.asyncio
    async def test_bot_with_react_reasoning(self):
        """Test bot with ReAct reasoning."""
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "reasoning": {"strategy": "react", "max_iterations": 3},
        }

        bot = await DynaBot.from_config(config)
        assert bot.reasoning_strategy is not None
        assert isinstance(bot.reasoning_strategy, ReActReasoning)

    @pytest.mark.asyncio
    async def test_bot_without_reasoning(self):
        """Test bot works without explicit reasoning strategy."""
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
        }

        bot = await DynaBot.from_config(config)
        # Without reasoning config, strategy should be None
        # and bot should use default ConversationManager.complete()
        assert bot.reasoning_strategy is None

        context = BotContext(
            conversation_id="conv-no-reasoning", client_id="test-client"
        )
        response = await bot.chat("Hello", context)
        assert response is not None

    @pytest.mark.asyncio
    async def test_reasoning_with_memory(self):
        """Test reasoning strategies work with memory."""
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "memory": {"type": "buffer", "max_messages": 5},
            "reasoning": {"strategy": "simple"},
        }

        bot = await DynaBot.from_config(config)
        context = BotContext(
            conversation_id="conv-reasoning-memory", client_id="test-client"
        )

        # Multiple interactions
        await bot.chat("First message", context)
        response = await bot.chat("Second message", context)

        assert response is not None
        # Check memory was updated
        memory_context = await bot.memory.get_context("test")
        assert len(memory_context) >= 2

    @pytest.mark.asyncio
    async def test_react_with_store_trace(self):
        """Test ReAct with store_trace writes trace to conversation metadata."""
        from dataknobs_llm.testing import text_response, tool_call_response

        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "reasoning": {
                "strategy": "react",
                "max_iterations": 5,
                "store_trace": True,
            },
        }

        bot = await DynaBot.from_config(config)
        assert bot.reasoning_strategy is not None
        assert isinstance(bot.reasoning_strategy, ReActReasoning)
        assert bot.reasoning_strategy.store_trace is True

        # Register a tool so the ReAct loop actually runs
        from dataknobs_llm.tools import Tool
        from typing import Any

        class EchoTool(Tool):
            def __init__(self) -> None:
                super().__init__(name="echo", description="Echoes input")

            @property
            def schema(self) -> dict[str, Any]:
                return {"type": "object", "properties": {"text": {"type": "string"}}}

            async def execute(self, **kwargs: Any) -> str:
                return kwargs.get("text", "")

        bot.tool_registry.register_tool(EchoTool())

        context = BotContext(
            conversation_id="conv-react-trace", client_id="test-client"
        )

        # One tool call then completion
        bot.llm.set_responses([
            tool_call_response("echo", {"text": "hello"}),
            text_response("Done"),
        ])

        response = await bot.chat("Test message", context)
        assert response is not None

        # Verify trace was stored in conversation metadata
        conv = await bot.get_conversation("conv-react-trace")
        assert conv is not None
        trace = conv.metadata.get("reasoning_trace")
        assert trace is not None
        assert len(trace) == 2  # 1 continued + 1 completed
        assert trace[0]["status"] == "continued"
        assert trace[1]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_react_verbose_mode(self):
        """Test ReAct verbose mode uses logging instead of print."""
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "reasoning": {
                "strategy": "react",
                "max_iterations": 2,
                "verbose": True,
            },
        }

        bot = await DynaBot.from_config(config)
        assert bot.reasoning_strategy.verbose is True

        context = BotContext(
            conversation_id="conv-react-verbose", client_id="test-client"
        )

        # This should generate log messages (not print to stdout)
        # In production, these would go to your logging infrastructure
        response = await bot.chat("Test message", context)
        assert response is not None

    @pytest.mark.asyncio
    async def test_bot_with_wizard_reasoning(self):
        """Test bot with wizard reasoning strategy."""
        # Create temp wizard config file
        wizard_config = {
            "name": "test-wizard",
            "version": "1.0",
            "description": "Test wizard for integration",
            "stages": [
                {
                    "name": "welcome",
                    "is_start": True,
                    "prompt": "What would you like to do?",
                    "schema": {
                        "type": "object",
                        "properties": {"intent": {"type": "string"}},
                    },
                    "suggestions": ["Create", "Edit"],
                    "transitions": [{"target": "complete"}],
                },
                {
                    "name": "complete",
                    "is_end": True,
                    "prompt": "All done!",
                },
            ],
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(wizard_config, f)
            config_path = f.name

        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "reasoning": {
                "strategy": "wizard",
                "wizard_config": config_path,
                "strict_validation": False,  # Disable for echo provider testing
            },
        }

        bot = await DynaBot.from_config(config)
        assert bot.reasoning_strategy is not None
        assert isinstance(bot.reasoning_strategy, WizardReasoning)

        context = BotContext(
            conversation_id="conv-wizard", client_id="test-client"
        )

        # Generate initial response - should start wizard flow
        response = await bot.chat("Hello", context)
        assert response is not None
        assert isinstance(response, str)

    @pytest.mark.asyncio
    async def test_wizard_state_persistence(self):
        """Test wizard state persists across conversation turns."""
        wizard_config = {
            "name": "test-wizard",
            "version": "1.0",
            "stages": [
                {
                    "name": "step1",
                    "is_start": True,
                    "prompt": "Step 1: Enter name",
                    "transitions": [{"target": "step2"}],
                },
                {
                    "name": "step2",
                    "prompt": "Step 2: Enter email",
                    "transitions": [{"target": "complete"}],
                },
                {
                    "name": "complete",
                    "is_end": True,
                    "prompt": "Complete!",
                },
            ],
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(wizard_config, f)
            config_path = f.name

        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "reasoning": {
                "strategy": "wizard",
                "wizard_config": config_path,
                "strict_validation": False,
            },
        }

        bot = await DynaBot.from_config(config)
        context = BotContext(
            conversation_id="conv-wizard-persist", client_id="test-client"
        )

        # First message - starts at step1
        response1 = await bot.chat("John Doe", context)
        assert response1 is not None

        # Second message - should be at step2
        response2 = await bot.chat("john@example.com", context)
        assert response2 is not None

        # Both responses should work without errors
        # The wizard maintains state through conversation metadata


class TestStreamChatWithReasoningStrategy:
    """Tests that stream_chat() correctly dispatches through reasoning strategies."""

    @pytest.mark.asyncio
    async def test_stream_chat_with_simple_reasoning(self):
        """stream_chat() with a reasoning strategy yields a single chunk from the strategy."""
        from dataknobs_llm import LLMStreamResponse
        from dataknobs_llm.testing import text_response

        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "reasoning": {"strategy": "simple"},
        }

        bot = await DynaBot.from_config(config)
        bot.llm.set_responses([text_response("Strategy response")])

        context = BotContext(
            conversation_id="conv-stream-strategy", client_id="test-client"
        )

        chunks: list[LLMStreamResponse] = []
        async for chunk in bot.stream_chat("Hello", context):
            chunks.append(chunk)

        # Reasoning strategy produces a single complete chunk
        assert len(chunks) == 1
        assert chunks[0].delta == "Strategy response"
        assert chunks[0].is_final is True
        assert chunks[0].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_stream_chat_without_reasoning_streams_normally(self):
        """stream_chat() without a reasoning strategy streams token by token."""
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
        }

        bot = await DynaBot.from_config(config)
        assert bot.reasoning_strategy is None

        context = BotContext(
            conversation_id="conv-stream-no-strategy", client_id="test-client"
        )

        chunks = []
        async for chunk in bot.stream_chat("Hello", context):
            chunks.append(chunk)

        # Without strategy, streaming comes from the LLM directly
        assert len(chunks) >= 1
        full_response = "".join(c.delta for c in chunks)
        assert len(full_response) > 0

    @pytest.mark.asyncio
    async def test_stream_chat_with_reasoning_updates_memory(self):
        """stream_chat() with a reasoning strategy still updates memory."""
        from dataknobs_llm.testing import text_response

        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "memory": {"type": "buffer", "max_messages": 10},
            "reasoning": {"strategy": "simple"},
        }

        bot = await DynaBot.from_config(config)
        bot.llm.set_responses([text_response("Memory check response")])

        context = BotContext(
            conversation_id="conv-stream-memory", client_id="test-client"
        )

        async for _ in bot.stream_chat("Test message", context):
            pass

        # Verify memory was updated with both user and assistant messages
        memory_context = await bot.memory.get_context("test")
        assert len(memory_context) >= 2
