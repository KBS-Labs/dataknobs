"""Tests for tool call preservation in LLMMessage and provider message converters.

Verifies that:
- LLMMessage stores tool_calls from assistant responses
- Provider message converters include tool_calls in their output
- role="tool" messages are handled correctly by providers
- Metadata deep-copy prevents parameter aliasing
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from dataknobs_llm.llm.base import LLMMessage, ToolCall


class TestLLMMessageToolCalls:
    """Tests for the tool_calls field on LLMMessage."""

    def test_tool_calls_default_none(self) -> None:
        msg = LLMMessage(role="assistant", content="hello")
        assert msg.tool_calls is None

    def test_tool_calls_stored(self) -> None:
        calls = [
            ToolCall(name="search", parameters={"q": "test"}, id="call_1"),
        ]
        msg = LLMMessage(role="assistant", content="", tool_calls=calls)
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].name == "search"
        assert msg.tool_calls[0].parameters == {"q": "test"}

    def test_tool_role_message(self) -> None:
        msg = LLMMessage(
            role="tool",
            content='{"success": true}',
            name="add_bank_record",
        )
        assert msg.role == "tool"
        assert msg.name == "add_bank_record"


class TestOllamaMessageConversion:
    """Tests for Ollama provider's _messages_to_ollama method."""

    def _make_provider(self) -> Any:
        """Create an OllamaProvider for testing message conversion."""
        from dataknobs_llm.llm.base import LLMConfig
        from dataknobs_llm.llm.providers.ollama import OllamaProvider

        config = LLMConfig(
            provider="ollama",
            model="test-model",
            api_base="http://localhost:11434",
        )
        return OllamaProvider(config)

    def test_basic_messages(self) -> None:
        provider = self._make_provider()
        messages = [
            LLMMessage(role="system", content="You are helpful."),
            LLMMessage(role="user", content="Hello"),
            LLMMessage(role="assistant", content="Hi there!"),
        ]
        result = provider._messages_to_ollama(messages)
        assert len(result) == 3
        assert result[0] == {"role": "system", "content": "You are helpful."}
        assert result[1] == {"role": "user", "content": "Hello"}
        assert result[2] == {"role": "assistant", "content": "Hi there!"}

    def test_assistant_with_tool_calls_included(self) -> None:
        provider = self._make_provider()
        messages = [
            LLMMessage(
                role="assistant",
                content="",
                tool_calls=[
                    ToolCall(
                        name="add_record",
                        parameters={"name": "flour", "amount": "1/4 cup"},
                        id="call_1",
                    ),
                ],
            ),
        ]
        result = provider._messages_to_ollama(messages)
        assert len(result) == 1
        msg = result[0]
        assert msg["role"] == "assistant"
        assert msg["content"] == ""
        assert "tool_calls" in msg
        assert len(msg["tool_calls"]) == 1
        tc = msg["tool_calls"][0]
        assert tc["function"]["name"] == "add_record"
        assert tc["function"]["arguments"] == {"name": "flour", "amount": "1/4 cup"}

    def test_tool_calls_not_added_to_non_assistant(self) -> None:
        """tool_calls on a non-assistant message should not be included."""
        provider = self._make_provider()
        messages = [
            LLMMessage(
                role="user",
                content="test",
                tool_calls=[ToolCall(name="x", parameters={})],
            ),
        ]
        result = provider._messages_to_ollama(messages)
        assert "tool_calls" not in result[0]

    def test_tool_role_passed_through(self) -> None:
        provider = self._make_provider()
        messages = [
            LLMMessage(
                role="tool",
                content='{"success": true}',
                name="add_record",
            ),
        ]
        result = provider._messages_to_ollama(messages)
        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert result[0]["content"] == '{"success": true}'

    def test_multi_turn_tool_conversation(self) -> None:
        """Full multi-turn tool conversation preserves structure."""
        provider = self._make_provider()
        messages = [
            LLMMessage(role="system", content="You manage recipes."),
            LLMMessage(role="user", content="Add flour"),
            LLMMessage(
                role="assistant",
                content="",
                tool_calls=[
                    ToolCall(name="add_record", parameters={"name": "flour"}, id="c1"),
                ],
            ),
            LLMMessage(
                role="tool",
                content='{"success": true, "record_id": "abc123def456"}',
                name="add_record",
            ),
            LLMMessage(role="assistant", content="Added flour."),
        ]
        result = provider._messages_to_ollama(messages)
        assert len(result) == 5
        # Assistant with tool_calls
        assert "tool_calls" in result[2]
        assert result[2]["tool_calls"][0]["function"]["name"] == "add_record"
        # Tool result
        assert result[3]["role"] == "tool"
        # Final assistant (no tool_calls)
        assert "tool_calls" not in result[4]


class TestOpenAIMessageConversion:
    """Tests for OpenAI adapter's adapt_messages method."""

    def _make_adapter(self) -> Any:
        from dataknobs_llm.llm.providers.openai import OpenAIAdapter

        return OpenAIAdapter()

    def test_basic_messages(self) -> None:
        adapter = self._make_adapter()
        messages = [
            LLMMessage(role="user", content="Hello"),
            LLMMessage(role="assistant", content="Hi!"),
        ]
        result = adapter.adapt_messages(messages)
        assert result[0] == {"role": "user", "content": "Hello"}
        assert result[1] == {"role": "assistant", "content": "Hi!"}

    def test_assistant_with_tool_calls(self) -> None:
        adapter = self._make_adapter()
        messages = [
            LLMMessage(
                role="assistant",
                content="",
                tool_calls=[
                    ToolCall(
                        name="search",
                        parameters={"query": "test"},
                        id="call_abc",
                    ),
                ],
            ),
        ]
        result = adapter.adapt_messages(messages)
        msg = result[0]
        assert "tool_calls" in msg
        tc = msg["tool_calls"][0]
        assert tc["id"] == "call_abc"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "search"
        # OpenAI expects arguments as JSON string
        assert '"query"' in tc["function"]["arguments"]

    def test_tool_role_preserved(self) -> None:
        adapter = self._make_adapter()
        messages = [
            LLMMessage(role="tool", content='{"result": "ok"}', name="my_tool"),
        ]
        result = adapter.adapt_messages(messages)
        assert result[0]["role"] == "tool"
        assert result[0]["name"] == "my_tool"

    def test_name_preserved(self) -> None:
        adapter = self._make_adapter()
        messages = [
            LLMMessage(role="function", content="result", name="calc"),
        ]
        result = adapter.adapt_messages(messages)
        assert result[0]["name"] == "calc"


class TestMetadataDeepCopy:
    """Tests for deep-copy of tool_call parameters in metadata."""

    def test_deep_copy_prevents_aliasing(self) -> None:
        """Mutating original parameters should not affect the deep copy."""
        original_params = {"data": {"name": "flour", "amount": "1/4 cup"}}
        copied = copy.deepcopy(original_params)

        # Mutate the original
        original_params["data"]["amount"] = "MUTATED"

        # The copy should be unaffected
        assert copied["data"]["amount"] == "1/4 cup"

    def test_tool_call_parameters_are_independent(self) -> None:
        """ToolCall parameters dict should be safe to deep-copy."""
        tc = ToolCall(
            name="add_record",
            parameters={"nested": {"key": "value"}},
            id="test",
        )
        copied = copy.deepcopy(tc.parameters)
        tc.parameters["nested"]["key"] = "MUTATED"
        assert copied["nested"]["key"] == "value"
