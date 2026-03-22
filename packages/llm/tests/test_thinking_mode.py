"""Tests for thinking-mode detection in _analyze_response().

Verifies that:
- Empty content + high completion tokens → metadata["thinking_only"] = True
- Normal responses pass through unchanged
- Tool-call responses (empty content but has tool_calls) are not flagged
- Ollama <think> tag parsing separates thinking from visible content
- Ollama 'think' parameter forwarding
- EchoProvider calls _analyze_response for consistency
"""

from __future__ import annotations

import pytest

from dataknobs_llm import EchoProvider, LLMMessage
from dataknobs_llm.llm.base import LLMResponse, ToolCall
from dataknobs_llm.testing import text_response, tool_call_response


def _msg(content: str = "hi") -> list[LLMMessage]:
    return [LLMMessage(role="user", content=content)]


@pytest.fixture()
def provider() -> EchoProvider:
    return EchoProvider({"provider": "echo", "model": "test"})


class TestAnalyzeResponseThinkingDetection:
    """Tests for base-class _analyze_response thinking-only detection."""

    def test_detects_thinking_only(self, provider: EchoProvider) -> None:
        """Empty content + high completion tokens → thinking_only flag."""
        response = LLMResponse(
            content="",
            model="test",
            finish_reason="stop",
            usage={"prompt_tokens": 50, "completion_tokens": 200, "total_tokens": 250},
        )
        result = provider._analyze_response(response)
        assert result.metadata.get("thinking_only") is True

    def test_normal_response_unchanged(self, provider: EchoProvider) -> None:
        """Response with content passes through without thinking_only flag."""
        response = LLMResponse(
            content="Hello, world!",
            model="test",
            finish_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )
        result = provider._analyze_response(response)
        assert "thinking_only" not in result.metadata
        assert result.content == "Hello, world!"

    def test_tool_call_response_not_flagged(self, provider: EchoProvider) -> None:
        """Empty content with tool_calls is NOT thinking-only."""
        response = LLMResponse(
            content="",
            model="test",
            finish_reason="tool_calls",
            usage={"prompt_tokens": 50, "completion_tokens": 100, "total_tokens": 150},
            tool_calls=[ToolCall(name="search", parameters={"q": "test"})],
        )
        result = provider._analyze_response(response)
        assert "thinking_only" not in result.metadata

    def test_low_token_empty_not_flagged(self, provider: EchoProvider) -> None:
        """Empty content with low completion tokens is NOT thinking-only."""
        response = LLMResponse(
            content="",
            model="test",
            finish_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )
        result = provider._analyze_response(response)
        assert "thinking_only" not in result.metadata

    def test_no_usage_empty_not_flagged(self, provider: EchoProvider) -> None:
        """Empty content with no usage info is NOT thinking-only."""
        response = LLMResponse(
            content="",
            model="test",
            finish_reason="stop",
            usage=None,
        )
        result = provider._analyze_response(response)
        assert "thinking_only" not in result.metadata

    def test_threshold_boundary(self, provider: EchoProvider) -> None:
        """Exactly 50 completion tokens does NOT trigger (> 50 required)."""
        response = LLMResponse(
            content="",
            model="test",
            finish_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 50, "total_tokens": 60},
        )
        result = provider._analyze_response(response)
        assert "thinking_only" not in result.metadata

    def test_threshold_boundary_above(self, provider: EchoProvider) -> None:
        """51 completion tokens with empty content triggers detection."""
        response = LLMResponse(
            content="",
            model="test",
            finish_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 51, "total_tokens": 61},
        )
        result = provider._analyze_response(response)
        assert result.metadata.get("thinking_only") is True


class TestOllamaThinkTagParsing:
    """Tests for Ollama-specific <think> tag parsing."""

    def test_think_tags_extracted(self) -> None:
        """<think> tags are parsed into metadata['thinking']."""
        from dataknobs_llm.llm.providers.ollama import OllamaProvider

        provider = OllamaProvider({"provider": "ollama", "model": "qwen3"})
        response = LLMResponse(
            content="<think>Let me reason about this.</think>The answer is 42.",
            model="qwen3",
            finish_reason="stop",
            usage={"prompt_tokens": 20, "completion_tokens": 30, "total_tokens": 50},
        )
        result = provider._analyze_response(response)
        assert result.metadata["thinking"] == "Let me reason about this."
        assert result.content == "The answer is 42."
        assert "thinking_only" not in result.metadata

    def test_think_tags_multiline(self) -> None:
        """Multi-line thinking content is handled."""
        from dataknobs_llm.llm.providers.ollama import OllamaProvider

        provider = OllamaProvider({"provider": "ollama", "model": "deepseek-r1"})
        thinking = "Step 1: Consider X.\nStep 2: Consider Y.\nStep 3: Conclude Z."
        response = LLMResponse(
            content=f"<think>{thinking}</think>Final answer: Z",
            model="deepseek-r1",
            finish_reason="stop",
            usage={"prompt_tokens": 20, "completion_tokens": 60, "total_tokens": 80},
        )
        result = provider._analyze_response(response)
        assert result.metadata["thinking"] == thinking
        assert result.content == "Final answer: Z"

    def test_think_tags_empty_visible_triggers_thinking_only(self) -> None:
        """<think> block with no visible answer → thinking_only flag."""
        from dataknobs_llm.llm.providers.ollama import OllamaProvider

        provider = OllamaProvider({"provider": "ollama", "model": "qwen3"})
        response = LLMResponse(
            content="<think>Long chain of thought here...</think>",
            model="qwen3",
            finish_reason="stop",
            usage={"prompt_tokens": 20, "completion_tokens": 200, "total_tokens": 220},
        )
        result = provider._analyze_response(response)
        assert result.metadata["thinking"] == "Long chain of thought here..."
        assert result.content == ""
        assert result.metadata.get("thinking_only") is True

    def test_no_think_tags_unchanged(self) -> None:
        """Normal content without <think> tags passes through unchanged."""
        from dataknobs_llm.llm.providers.ollama import OllamaProvider

        provider = OllamaProvider({"provider": "ollama", "model": "llama3.2"})
        response = LLMResponse(
            content="Just a normal response.",
            model="llama3.2",
            finish_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
        )
        result = provider._analyze_response(response)
        assert result.content == "Just a normal response."
        assert "thinking" not in result.metadata
        assert "thinking_only" not in result.metadata

    def test_think_parameter_forwarded_in_payload(self) -> None:
        """'think' option is forwarded to the Ollama API payload."""
        from dataknobs_llm.llm.providers.ollama import OllamaProvider

        provider = OllamaProvider({
            "provider": "ollama",
            "model": "qwen3",
            "options": {"think": True},
        })
        runtime_config = provider._get_runtime_config(None)
        assert runtime_config.options.get("think") is True


class TestEchoProviderAnalyzeResponse:
    """Tests that EchoProvider calls _analyze_response for consistency."""

    @pytest.mark.asyncio
    async def test_echo_calls_analyze_response(self) -> None:
        """EchoProvider.complete() runs _analyze_response on the response."""
        provider = EchoProvider({"provider": "echo", "model": "test"})
        # Create a response that would trigger thinking-only detection:
        # empty content, high completion tokens
        thinking_response = LLMResponse(
            content="",
            model="test",
            finish_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 200, "total_tokens": 210},
        )
        provider.set_responses([thinking_response])
        await provider.initialize()
        result = await provider.complete(_msg())
        assert result.metadata.get("thinking_only") is True

    @pytest.mark.asyncio
    async def test_echo_normal_response_not_flagged(self) -> None:
        """Normal EchoProvider responses don't get thinking_only flag."""
        provider = EchoProvider({"provider": "echo", "model": "test"})
        provider.set_responses([text_response("hello")])
        await provider.initialize()
        result = await provider.complete(_msg())
        assert "thinking_only" not in result.metadata
        assert result.content == "hello"

    @pytest.mark.asyncio
    async def test_echo_tool_call_response_not_flagged(self) -> None:
        """EchoProvider tool call responses don't get thinking_only flag."""
        provider = EchoProvider({"provider": "echo", "model": "test"})
        provider.set_responses([
            tool_call_response("search", {"query": "test"}),
        ])
        await provider.initialize()
        result = await provider.complete(_msg(), tools=["search"])
        assert "thinking_only" not in result.metadata
