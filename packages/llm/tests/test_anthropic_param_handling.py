"""Tests for Anthropic provider parameter handling.

Bug: Anthropic provider unconditionally sends both temperature and top_p.
The Anthropic API rejects requests with both:
  400 Bad Request: temperature and top_p cannot both be specified

Root cause: LLMConfig defaults temperature=0.7 and top_p=1.0 (non-None),
making "not set" indistinguishable from "explicitly set."
"""

from __future__ import annotations

import pytest

from dataknobs_llm.llm.base import LLMConfig, LLMMessage, ToolCall


# ---------------------------------------------------------------------------
# LLMConfig.generation_params() tests
# ---------------------------------------------------------------------------


class TestGenerationParams:
    """Test LLMConfig.generation_params() returns only explicitly-set values."""

    def test_default_config_returns_empty(self):
        """Default LLMConfig should return no generation params."""
        config = LLMConfig(provider="anthropic", model="claude-3-haiku")
        params = config.generation_params()
        assert "temperature" not in params
        assert "top_p" not in params
        assert "frequency_penalty" not in params
        assert "presence_penalty" not in params

    def test_only_temperature_set(self):
        """Only temperature should appear when explicitly set."""
        config = LLMConfig(provider="anthropic", model="claude-3-haiku", temperature=0.5)
        params = config.generation_params()
        assert params["temperature"] == 0.5
        assert "top_p" not in params

    def test_only_top_p_set(self):
        """Only top_p should appear when explicitly set."""
        config = LLMConfig(provider="anthropic", model="claude-3-haiku", top_p=0.9)
        params = config.generation_params()
        assert params["top_p"] == 0.9
        assert "temperature" not in params

    def test_both_set(self):
        """Both should appear when both explicitly set."""
        config = LLMConfig(
            provider="anthropic", model="claude-3-haiku",
            temperature=0.5, top_p=0.9,
        )
        params = config.generation_params()
        assert params["temperature"] == 0.5
        assert params["top_p"] == 0.9

    def test_max_tokens_included(self):
        """max_tokens should appear when set."""
        config = LLMConfig(provider="anthropic", model="claude-3-haiku", max_tokens=1000)
        params = config.generation_params()
        assert params["max_tokens"] == 1000

    def test_stop_sequences_included(self):
        """stop_sequences should appear when set."""
        config = LLMConfig(
            provider="anthropic", model="claude-3-haiku",
            stop_sequences=["STOP"],
        )
        params = config.generation_params()
        assert params["stop_sequences"] == ["STOP"]

    def test_seed_included(self):
        """seed should appear when set."""
        config = LLMConfig(provider="anthropic", model="claude-3-haiku", seed=42)
        params = config.generation_params()
        assert params["seed"] == 42


# ---------------------------------------------------------------------------
# Anthropic provider _build_api_params() tests
# ---------------------------------------------------------------------------


class TestAnthropicBuildApiParams:
    """Test that Anthropic provider builds correct API params.

    _build_api_params() is a pure method (LLMConfig -> dict) that does not
    require an Anthropic client, so we call it directly on an uninitialised
    provider instance — no mocks needed.
    """

    def _make_provider(self, **config_kwargs):
        """Create an AnthropicProvider without initialising the client."""
        from dataknobs_llm.llm.providers.anthropic import AnthropicProvider

        config = LLMConfig(provider="anthropic", **config_kwargs)
        return AnthropicProvider(config)

    def test_default_config_sends_neither_temp_nor_top_p(self):
        """Default config should not send temperature or top_p."""
        provider = self._make_provider(model="claude-3-haiku-20240307")
        params = provider.adapter.adapt_config(provider.config)
        assert "temperature" not in params
        assert "top_p" not in params

    def test_only_temperature_sent_when_set(self):
        """Only temperature should be sent when explicitly set."""
        provider = self._make_provider(
            model="claude-3-haiku-20240307", temperature=0.5,
        )
        params = provider.adapter.adapt_config(provider.config)
        assert params["temperature"] == 0.5
        assert "top_p" not in params

    def test_only_top_p_sent_when_set(self):
        """Only top_p should be sent when explicitly set."""
        provider = self._make_provider(
            model="claude-3-haiku-20240307", top_p=0.9,
        )
        params = provider.adapter.adapt_config(provider.config)
        assert params["top_p"] == 0.9
        assert "temperature" not in params

    def test_both_sent_when_both_explicitly_set(self):
        """Both should be sent when both explicitly set."""
        provider = self._make_provider(
            model="claude-3-haiku-20240307", temperature=0.5, top_p=0.9,
        )
        params = provider.adapter.adapt_config(provider.config)
        assert params["temperature"] == 0.5
        assert params["top_p"] == 0.9

    def test_max_tokens_defaults_to_1024(self):
        """max_tokens should default to 1024 when not set."""
        provider = self._make_provider(model="claude-3-haiku-20240307")
        params = provider.adapter.adapt_config(provider.config)
        assert params["max_tokens"] == 1024

    def test_max_tokens_uses_explicit_value(self):
        """max_tokens should use explicitly set value."""
        provider = self._make_provider(
            model="claude-3-haiku-20240307", max_tokens=2048,
        )
        params = provider.adapter.adapt_config(provider.config)
        assert params["max_tokens"] == 2048

    def test_stop_sequences_included(self):
        """stop_sequences should be forwarded to API params."""
        provider = self._make_provider(
            model="claude-3-haiku-20240307", stop_sequences=["STOP", "END"],
        )
        params = provider.adapter.adapt_config(provider.config)
        assert params["stop_sequences"] == ["STOP", "END"]

    def test_model_always_included(self):
        """model should always be in API params."""
        provider = self._make_provider(model="claude-3-haiku-20240307")
        params = provider.adapter.adapt_config(provider.config)
        assert params["model"] == "claude-3-haiku-20240307"


# ---------------------------------------------------------------------------
# AnthropicAdapter.adapt_messages() tests
# ---------------------------------------------------------------------------


class TestAnthropicAdaptMessages:
    """Test AnthropicAdapter.adapt_messages() structured message conversion."""

    def _adapter(self):
        from dataknobs_llm.llm.providers.anthropic import AnthropicAdapter
        return AnthropicAdapter()

    def test_system_messages_extracted(self):
        """System messages should be extracted into the system parameter."""
        adapter = self._adapter()
        messages = [
            LLMMessage(role="system", content="You are helpful."),
            LLMMessage(role="user", content="Hi"),
        ]
        system, msgs = adapter.adapt_messages(messages)
        assert system == "You are helpful."
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"

    def test_system_merged_with_config_prompt(self):
        """System messages merge with the config system_prompt."""
        adapter = self._adapter()
        messages = [
            LLMMessage(role="system", content="Extra instructions."),
            LLMMessage(role="user", content="Hi"),
        ]
        system, msgs = adapter.adapt_messages(messages, system_prompt="Base prompt.")
        assert "Base prompt." in system
        assert "Extra instructions." in system

    def test_user_and_assistant_passthrough(self):
        """Plain user/assistant messages pass through unchanged."""
        adapter = self._adapter()
        messages = [
            LLMMessage(role="user", content="Hello"),
            LLMMessage(role="assistant", content="Hi there"),
        ]
        system, msgs = adapter.adapt_messages(messages)
        assert system == ""
        assert msgs[0] == {"role": "user", "content": "Hello"}
        assert msgs[1] == {"role": "assistant", "content": "Hi there"}

    def test_assistant_tool_calls_become_content_blocks(self):
        """Assistant messages with tool_calls become tool_use content blocks."""
        adapter = self._adapter()
        messages = [
            LLMMessage(
                role="assistant",
                content="Let me search.",
                tool_calls=[
                    ToolCall(name="search", parameters={"q": "test"}, id="toolu_123"),
                ],
            ),
        ]
        _, msgs = adapter.adapt_messages(messages)
        assert msgs[0]["role"] == "assistant"
        blocks = msgs[0]["content"]
        assert blocks[0] == {"type": "text", "text": "Let me search."}
        assert blocks[1]["type"] == "tool_use"
        assert blocks[1]["id"] == "toolu_123"
        assert blocks[1]["name"] == "search"
        assert blocks[1]["input"] == {"q": "test"}

    def test_tool_result_uses_tool_call_id(self):
        """Tool result messages should use tool_call_id as tool_use_id."""
        adapter = self._adapter()
        messages = [
            LLMMessage(
                role="tool",
                content='{"result": "found"}',
                name="search",
                tool_call_id="toolu_123",
            ),
        ]
        _, msgs = adapter.adapt_messages(messages)
        assert msgs[0]["role"] == "user"
        block = msgs[0]["content"][0]
        assert block["type"] == "tool_result"
        assert block["tool_use_id"] == "toolu_123"

    def test_tool_result_falls_back_to_name(self):
        """Tool result should fall back to name when tool_call_id is absent."""
        adapter = self._adapter()
        messages = [
            LLMMessage(role="tool", content="result", name="search"),
        ]
        _, msgs = adapter.adapt_messages(messages)
        block = msgs[0]["content"][0]
        assert block["tool_use_id"] == "search"

    def test_tool_result_falls_back_to_unknown(self):
        """Tool result should fall back to 'unknown' when both are absent."""
        adapter = self._adapter()
        messages = [
            LLMMessage(role="tool", content="result"),
        ]
        _, msgs = adapter.adapt_messages(messages)
        block = msgs[0]["content"][0]
        assert block["tool_use_id"] == "unknown"

    def test_consecutive_tool_results_consolidated(self):
        """Multiple tool results should be consolidated into one user message."""
        adapter = self._adapter()
        messages = [
            LLMMessage(role="tool", content="result1", name="search", tool_call_id="t1"),
            LLMMessage(role="tool", content="result2", name="calc", tool_call_id="t2"),
        ]
        _, msgs = adapter.adapt_messages(messages)
        # Should be a single user message, not two
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert len(msgs[0]["content"]) == 2
        assert msgs[0]["content"][0]["tool_use_id"] == "t1"
        assert msgs[0]["content"][1]["tool_use_id"] == "t2"


# ---------------------------------------------------------------------------
# AnthropicAdapter.adapt_response() tests
# ---------------------------------------------------------------------------


class TestAnthropicAdaptResponse:
    """Test AnthropicAdapter.adapt_response() content block parsing.

    Uses simple namespace objects to simulate Anthropic SDK response types.
    """

    def _adapter(self):
        from dataknobs_llm.llm.providers.anthropic import AnthropicAdapter
        return AnthropicAdapter()

    def _make_response(self, content_blocks, model="claude-3", stop_reason="end_turn"):
        """Build a fake Anthropic Message-like object."""
        class Block:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        class Usage:
            def __init__(self):
                self.input_tokens = 10
                self.output_tokens = 20

        class Response:
            def __init__(self):
                self.content = [Block(**b) for b in content_blocks]
                self.model = model
                self.stop_reason = stop_reason
                self.usage = Usage()

        return Response()

    def test_text_only_response(self):
        """Single text block should produce content string."""
        adapter = self._adapter()
        resp = self._make_response([{"type": "text", "text": "Hello!"}])
        parsed = adapter.adapt_response(resp)
        assert parsed.content == "Hello!"
        assert parsed.tool_calls is None

    def test_tool_use_only_response(self):
        """Single tool_use block (no text) should not crash."""
        adapter = self._adapter()
        resp = self._make_response([{
            "type": "tool_use",
            "id": "toolu_abc",
            "name": "search",
            "input": {"q": "test"},
        }])
        parsed = adapter.adapt_response(resp)
        assert parsed.content == ""
        assert len(parsed.tool_calls) == 1
        tc = parsed.tool_calls[0]
        assert tc.name == "search"
        assert tc.parameters == {"q": "test"}
        assert tc.id == "toolu_abc"

    def test_mixed_text_and_tool_use(self):
        """Mixed response should capture both text and tool calls."""
        adapter = self._adapter()
        resp = self._make_response([
            {"type": "text", "text": "I'll search for that."},
            {"type": "tool_use", "id": "toolu_1", "name": "search", "input": {"q": "x"}},
            {"type": "tool_use", "id": "toolu_2", "name": "calc", "input": {"expr": "1+1"}},
        ])
        parsed = adapter.adapt_response(resp)
        assert parsed.content == "I'll search for that."
        assert len(parsed.tool_calls) == 2
        assert parsed.tool_calls[0].name == "search"
        assert parsed.tool_calls[1].name == "calc"

    def test_usage_extracted(self):
        """Usage stats should be extracted from the response."""
        adapter = self._adapter()
        resp = self._make_response([{"type": "text", "text": "Hi"}])
        parsed = adapter.adapt_response(resp)
        assert parsed.usage["prompt_tokens"] == 10
        assert parsed.usage["completion_tokens"] == 20
        assert parsed.usage["total_tokens"] == 30


# ---------------------------------------------------------------------------
# AnthropicAdapter.adapt_tools() tests
# ---------------------------------------------------------------------------


class TestAnthropicAdaptTools:
    """Test AnthropicAdapter.adapt_tools() tool schema conversion."""

    def _adapter(self):
        from dataknobs_llm.llm.providers.anthropic import AnthropicAdapter
        return AnthropicAdapter()

    def test_converts_tool_objects(self):
        """Tool objects should be converted to Anthropic format."""
        adapter = self._adapter()

        class FakeTool:
            name = "search"
            description = "Search the web"
            schema = {"type": "object", "properties": {"q": {"type": "string"}}}

        result = adapter.adapt_tools([FakeTool()])
        assert len(result) == 1
        assert result[0]["name"] == "search"
        assert result[0]["description"] == "Search the web"
        assert result[0]["input_schema"]["type"] == "object"
