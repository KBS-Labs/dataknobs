"""Tests for capture-replay infrastructure in dataknobs_llm.testing.

Tests serialization round-trips and CapturingProvider recording behavior.
No Ollama dependency — uses EchoProvider as the "real" delegate.
"""

import pytest

from dataknobs_llm import EchoProvider
from dataknobs_llm.llm.base import LLMMessage, LLMResponse, ToolCall
from dataknobs_llm.testing import (
    CapturedCall,
    CapturingProvider,
    llm_message_from_dict,
    llm_message_to_dict,
    llm_response_from_dict,
    llm_response_to_dict,
    text_response,
    tool_call_from_dict,
    tool_call_response,
    tool_call_to_dict,
)


# =============================================================================
# Serialization round-trip tests
# =============================================================================


class TestToolCallSerialization:
    """ToolCall to_dict/from_dict round-trips."""

    def test_basic_tool_call(self):
        tc = ToolCall(name="search", parameters={"query": "test"})
        d = tool_call_to_dict(tc)
        assert d == {"name": "search", "parameters": {"query": "test"}}

        restored = tool_call_from_dict(d)
        assert restored.name == tc.name
        assert restored.parameters == tc.parameters
        assert restored.id is None

    def test_tool_call_with_id(self):
        tc = ToolCall(name="get_weather", parameters={"city": "NYC"}, id="tc-123")
        d = tool_call_to_dict(tc)
        assert d["id"] == "tc-123"

        restored = tool_call_from_dict(d)
        assert restored.id == "tc-123"

    def test_empty_parameters(self):
        tc = ToolCall(name="list_all", parameters={})
        d = tool_call_to_dict(tc)
        restored = tool_call_from_dict(d)
        assert restored.parameters == {}

    def test_id_omitted_when_none(self):
        tc = ToolCall(name="test", parameters={})
        d = tool_call_to_dict(tc)
        assert "id" not in d


class TestLLMMessageSerialization:
    """LLMMessage to_dict/from_dict round-trips."""

    def test_simple_user_message(self):
        msg = LLMMessage(role="user", content="Hello")
        d = llm_message_to_dict(msg)
        assert d == {"role": "user", "content": "Hello"}

        restored = llm_message_from_dict(d)
        assert restored.role == "user"
        assert restored.content == "Hello"
        assert restored.name is None
        assert restored.tool_calls is None

    def test_system_message(self):
        msg = LLMMessage(role="system", content="You are helpful")
        d = llm_message_to_dict(msg)
        restored = llm_message_from_dict(d)
        assert restored.role == "system"
        assert restored.content == "You are helpful"

    def test_message_with_name(self):
        msg = LLMMessage(role="function", content='{"result": 42}', name="calculator")
        d = llm_message_to_dict(msg)
        assert d["name"] == "calculator"

        restored = llm_message_from_dict(d)
        assert restored.name == "calculator"

    def test_message_with_tool_calls(self):
        msg = LLMMessage(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(name="search", parameters={"q": "test"}, id="tc-1"),
                ToolCall(name="fetch", parameters={"url": "http://x"}, id="tc-2"),
            ],
        )
        d = llm_message_to_dict(msg)
        assert len(d["tool_calls"]) == 2
        assert d["tool_calls"][0]["name"] == "search"

        restored = llm_message_from_dict(d)
        assert len(restored.tool_calls) == 2
        assert restored.tool_calls[0].name == "search"
        assert restored.tool_calls[1].id == "tc-2"

    def test_message_with_metadata(self):
        msg = LLMMessage(role="user", content="Hi", metadata={"timestamp": "2026-01-01"})
        d = llm_message_to_dict(msg)
        assert d["metadata"] == {"timestamp": "2026-01-01"}

        restored = llm_message_from_dict(d)
        assert restored.metadata == {"timestamp": "2026-01-01"}

    def test_empty_metadata_omitted(self):
        msg = LLMMessage(role="user", content="Hi")
        d = llm_message_to_dict(msg)
        assert "metadata" not in d

    def test_message_with_function_call(self):
        msg = LLMMessage(
            role="assistant",
            content="",
            function_call={"name": "calc", "arguments": '{"x": 1}'},
        )
        d = llm_message_to_dict(msg)
        assert d["function_call"]["name"] == "calc"

        restored = llm_message_from_dict(d)
        assert restored.function_call["name"] == "calc"


class TestLLMResponseSerialization:
    """LLMResponse to_dict/from_dict round-trips."""

    def test_simple_response(self):
        resp = LLMResponse(content="Hello!", model="test-model")
        d = llm_response_to_dict(resp)
        assert d["content"] == "Hello!"
        assert d["model"] == "test-model"
        # Runtime fields excluded
        assert "created_at" not in d
        assert "cumulative_cost_usd" not in d

        restored = llm_response_from_dict(d)
        assert restored.content == "Hello!"
        assert restored.model == "test-model"

    def test_response_with_usage(self):
        resp = LLMResponse(
            content="Hi",
            model="test",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )
        d = llm_response_to_dict(resp)
        assert d["usage"]["total_tokens"] == 15

        restored = llm_response_from_dict(d)
        assert restored.usage["total_tokens"] == 15

    def test_response_with_tool_calls(self):
        resp = LLMResponse(
            content="",
            model="test",
            finish_reason="tool_calls",
            tool_calls=[ToolCall(name="search", parameters={"q": "test"}, id="tc-1")],
        )
        d = llm_response_to_dict(resp)
        assert d["finish_reason"] == "tool_calls"
        assert len(d["tool_calls"]) == 1

        restored = llm_response_from_dict(d)
        assert restored.finish_reason == "tool_calls"
        assert restored.tool_calls[0].name == "search"

    def test_response_with_cost(self):
        resp = LLMResponse(content="Hi", model="test", cost_usd=0.001)
        d = llm_response_to_dict(resp)
        assert d["cost_usd"] == 0.001

        restored = llm_response_from_dict(d)
        assert restored.cost_usd == 0.001

    def test_none_fields_omitted(self):
        resp = LLMResponse(content="Hi", model="test")
        d = llm_response_to_dict(resp)
        assert "finish_reason" not in d
        assert "usage" not in d
        assert "tool_calls" not in d
        assert "cost_usd" not in d
        assert "function_call" not in d

    def test_response_with_metadata(self):
        resp = LLMResponse(content="Hi", model="test", metadata={"provider": "ollama"})
        d = llm_response_to_dict(resp)
        assert d["metadata"] == {"provider": "ollama"}

        restored = llm_response_from_dict(d)
        assert restored.metadata == {"provider": "ollama"}


# =============================================================================
# CapturingProvider tests
# =============================================================================


class TestCapturingProvider:
    """CapturingProvider records calls correctly using EchoProvider as delegate."""

    @pytest.fixture()
    def echo_provider(self):
        """EchoProvider with scripted responses."""
        provider = EchoProvider({"provider": "echo", "model": "test"})
        provider.set_responses([
            text_response("First response"),
            text_response("Second response"),
        ])
        return provider

    @pytest.mark.asyncio()
    async def test_complete_passes_through(self, echo_provider):
        """CapturingProvider returns the delegate's response unchanged."""
        capturing = CapturingProvider(echo_provider, role="main")
        response = await capturing.complete("Hello")
        assert response.content == "First response"

    @pytest.mark.asyncio()
    async def test_captures_call(self, echo_provider):
        """CapturingProvider records the call after completion."""
        capturing = CapturingProvider(echo_provider, role="main")
        await capturing.complete("Hello")

        assert capturing.call_count == 1
        call = capturing.captured_calls[0]
        assert isinstance(call, CapturedCall)
        assert call.role == "main"
        assert call.call_index == 0
        assert call.duration_seconds >= 0

    @pytest.mark.asyncio()
    async def test_captures_messages(self, echo_provider):
        """CapturedCall contains serialized messages."""
        capturing = CapturingProvider(echo_provider, role="extraction")

        messages = [
            LLMMessage(role="system", content="You are helpful"),
            LLMMessage(role="user", content="Extract data"),
        ]
        await capturing.complete(messages)

        call = capturing.captured_calls[0]
        assert len(call.messages) == 2
        assert call.messages[0]["role"] == "system"
        assert call.messages[1]["content"] == "Extract data"

    @pytest.mark.asyncio()
    async def test_captures_response(self, echo_provider):
        """CapturedCall contains serialized response."""
        capturing = CapturingProvider(echo_provider)
        await capturing.complete("Hello")

        call = capturing.captured_calls[0]
        assert call.response["content"] == "First response"
        assert call.response["model"] == "test-model"

    @pytest.mark.asyncio()
    async def test_captures_config_overrides(self, echo_provider):
        """CapturedCall records config overrides."""
        capturing = CapturingProvider(echo_provider)
        await capturing.complete("Hello", config_overrides={"temperature": 0.5})

        call = capturing.captured_calls[0]
        assert call.config_overrides == {"temperature": 0.5}

    @pytest.mark.asyncio()
    async def test_multiple_calls_indexed(self, echo_provider):
        """Multiple calls get sequential call_index values."""
        capturing = CapturingProvider(echo_provider)
        await capturing.complete("First")
        await capturing.complete("Second")

        assert capturing.call_count == 2
        assert capturing.captured_calls[0].call_index == 0
        assert capturing.captured_calls[1].call_index == 1

    @pytest.mark.asyncio()
    async def test_role_tag(self):
        """Role tag is preserved on captured calls."""
        provider = EchoProvider({"provider": "echo", "model": "test"})
        provider.set_responses([text_response("ok")])

        capturing = CapturingProvider(provider, role="extraction")
        await capturing.complete("test")

        assert capturing.captured_calls[0].role == "extraction"

    @pytest.mark.asyncio()
    async def test_string_message_serialized(self):
        """String messages are serialized as user messages."""
        provider = EchoProvider({"provider": "echo", "model": "test"})
        provider.set_responses([text_response("ok")])

        capturing = CapturingProvider(provider)
        await capturing.complete("hello world")

        call = capturing.captured_calls[0]
        assert call.messages == [{"role": "user", "content": "hello world"}]

    @pytest.mark.asyncio()
    async def test_stream_complete_captures(self):
        """stream_complete yields chunks and captures the assembled response."""
        provider = EchoProvider({"provider": "echo", "model": "test"})
        provider.set_responses([text_response("streamed response")])

        capturing = CapturingProvider(provider)
        chunks = []
        async for chunk in capturing.stream_complete("test"):
            chunks.append(chunk)

        assert len(chunks) > 0
        assert capturing.call_count == 1
        call = capturing.captured_calls[0]
        # Assembled content should contain the full response
        assert "streamed response" in call.response["content"]

    @pytest.mark.asyncio()
    async def test_delegates_capabilities(self):
        """get_capabilities delegates to the wrapped provider."""
        provider = EchoProvider({"provider": "echo", "model": "test"})
        capturing = CapturingProvider(provider)
        caps = capturing.get_capabilities()
        assert caps == provider.get_capabilities()

    @pytest.mark.asyncio()
    async def test_tool_call_response_captured(self):
        """Tool call responses are captured with tool_calls in the response dict."""
        provider = EchoProvider({"provider": "echo", "model": "test"})
        provider.set_responses([
            tool_call_response("search", {"query": "test"})
        ])

        capturing = CapturingProvider(provider)
        response = await capturing.complete("search for test")

        assert response.tool_calls is not None
        call = capturing.captured_calls[0]
        assert call.response["tool_calls"][0]["name"] == "search"
