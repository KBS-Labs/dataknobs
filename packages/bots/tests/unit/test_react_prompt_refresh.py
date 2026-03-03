"""Tests for ReAct prompt refresh between iterations.

Verifies that ReActReasoning can refresh the system_prompt_override
between iterations when a prompt_refresher callback is provided.
This prevents stale context after mutating tool calls (e.g.
load_from_catalog changing artifact state mid-loop).
"""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_bots.reasoning.react import ReActReasoning
from dataknobs_llm.conversations import ConversationManager
from dataknobs_llm.llm.providers.echo import EchoProvider
from dataknobs_llm.testing import text_response, tool_call_response
from dataknobs_llm.tools.base import Tool


# ---------------------------------------------------------------------------
# Test tool
# ---------------------------------------------------------------------------

class StateMutatingTool(Tool):
    """Test tool that tracks execution and mutates shared state."""

    def __init__(self, shared_state: dict[str, Any]) -> None:
        super().__init__(
            name="mutate_state",
            description="Mutate shared state for testing",
        )
        self._shared_state = shared_state
        self.call_count = 0

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "value": {"type": "string"},
            },
        }

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        self.call_count += 1
        self._shared_state["value"] = f"mutated-{self.call_count}"
        return {"success": True, "new_value": self._shared_state["value"]}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_system_prompt_from_call(
    provider: EchoProvider, call_index: int,
) -> str:
    """Extract the system prompt from a specific EchoProvider call."""
    call = provider.get_call(call_index)
    messages = call["messages"]
    for msg in messages:
        if hasattr(msg, "role") and msg.role == "system":
            return msg.content
        if isinstance(msg, dict) and msg.get("role") == "system":
            return msg.get("content", "")
    return ""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestReActPromptRefresh:
    """Tests for prompt_refresher callback in ReAct loop."""

    @pytest.mark.asyncio
    async def test_refresher_updates_prompt_between_iterations(
        self,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """After tool execution, the refresher is called and the next
        iteration uses the updated system prompt.

        Scenario: tool call on iteration 1, text response on iteration 2.
        The refresher should be called after iteration 1's tool execution,
        and iteration 2 should use the refreshed prompt.
        """
        manager, provider = conversation_manager_pair

        # Script: iteration 1 = tool call, iteration 2 = text response
        provider.set_responses([
            tool_call_response("mutate_state", {"value": "new"}),
            text_response("Done with refreshed context"),
        ])

        shared_state: dict[str, Any] = {"value": "initial"}
        tool = StateMutatingTool(shared_state)

        # Refresher reads live state — simulates wizard re-rendering
        refresh_count = 0

        def refresher() -> str:
            nonlocal refresh_count
            refresh_count += 1
            return f"Refreshed prompt (state={shared_state['value']})"

        react = ReActReasoning(
            max_iterations=5,
            prompt_refresher=refresher,
        )

        # Add a user message so the conversation has context
        await manager.add_message(content="Do something", role="user")

        response = await react.generate(
            manager=manager,
            llm=None,
            tools=[tool],
            system_prompt_override="Original prompt",
        )

        assert response.content == "Done with refreshed context"
        assert tool.call_count == 1
        assert refresh_count >= 1

        # Iteration 1 used the original prompt
        original_prompt = _get_system_prompt_from_call(provider, 0)
        assert original_prompt == "Original prompt"

        # Iteration 2 used the refreshed prompt (after tool mutated state)
        refreshed_prompt = _get_system_prompt_from_call(provider, 1)
        assert "Refreshed prompt" in refreshed_prompt
        assert "mutated-1" in refreshed_prompt

    @pytest.mark.asyncio
    async def test_no_refresher_keeps_original_prompt(
        self,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """Without prompt_refresher, all iterations use the original prompt.

        Backward compatibility: existing behavior is unchanged.
        """
        manager, provider = conversation_manager_pair

        provider.set_responses([
            tool_call_response("mutate_state", {"value": "new"}),
            text_response("Done"),
        ])

        shared_state: dict[str, Any] = {"value": "initial"}
        tool = StateMutatingTool(shared_state)

        react = ReActReasoning(max_iterations=5)

        await manager.add_message(content="Do something", role="user")

        await react.generate(
            manager=manager,
            llm=None,
            tools=[tool],
            system_prompt_override="Original prompt",
        )

        assert provider.call_count == 2

        # Both iterations used the same original prompt
        prompt_1 = _get_system_prompt_from_call(provider, 0)
        prompt_2 = _get_system_prompt_from_call(provider, 1)
        assert prompt_1 == "Original prompt"
        assert prompt_2 == "Original prompt"

    @pytest.mark.asyncio
    async def test_refresher_called_before_max_iterations_fallback(
        self,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """When max iterations are exhausted, the final complete() call
        also uses the refreshed prompt.
        """
        manager, provider = conversation_manager_pair

        # All responses are tool calls — will exhaust max_iterations=2
        # Then the fallback manager.complete() is called (response 3)
        provider.set_responses([
            tool_call_response("mutate_state", {"value": "v1"}),
            tool_call_response("mutate_state", {"value": "v2"}),
            text_response("Final after max iterations"),
        ])

        shared_state: dict[str, Any] = {"value": "initial"}
        tool = StateMutatingTool(shared_state)

        refresh_count = 0

        def refresher() -> str:
            nonlocal refresh_count
            refresh_count += 1
            return f"Refreshed-{refresh_count} (state={shared_state['value']})"

        react = ReActReasoning(
            max_iterations=2,
            prompt_refresher=refresher,
        )

        await manager.add_message(content="Do something", role="user")

        response = await react.generate(
            manager=manager,
            llm=None,
            tools=[tool],
            system_prompt_override="Original prompt",
        )

        assert response.content == "Final after max iterations"
        assert tool.call_count == 2

        # The final call (after max iterations) should use refreshed prompt
        # provider.call_count == 3: two tool-call iterations + one final
        assert provider.call_count == 3
        final_prompt = _get_system_prompt_from_call(provider, 2)
        assert "Refreshed" in final_prompt
        assert "mutated-2" in final_prompt

    @pytest.mark.asyncio
    async def test_refresher_not_called_when_no_tool_calls(
        self,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """When the LLM responds with text immediately (no tool calls),
        the refresher is never called.
        """
        manager, provider = conversation_manager_pair

        provider.set_responses([
            text_response("Immediate answer"),
        ])

        shared_state: dict[str, Any] = {"value": "initial"}
        tool = StateMutatingTool(shared_state)

        refresh_count = 0

        def refresher() -> str:
            nonlocal refresh_count
            refresh_count += 1
            return "Should not be called"

        react = ReActReasoning(
            max_iterations=5,
            prompt_refresher=refresher,
        )

        await manager.add_message(content="Quick question", role="user")

        response = await react.generate(
            manager=manager,
            llm=None,
            tools=[tool],
            system_prompt_override="Original prompt",
        )

        assert response.content == "Immediate answer"
        assert refresh_count == 0
        assert provider.call_count == 1

        # The single call used the original prompt
        prompt = _get_system_prompt_from_call(provider, 0)
        assert prompt == "Original prompt"
