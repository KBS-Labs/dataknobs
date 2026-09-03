"""Every adapter builds tool calls the same way, or the shape is not settled.

``LLMAdapter.tool_call_parameters()`` normalizes one provider's raw arguments.
It was adopted by Ollama alone, which left the question it answers open at the
five other sites that build a :class:`ToolCall`, and they answered it three
different ways:

===============================  ==========================================
Site                             What it did
===============================  ==========================================
``openai.py`` ``adapt_response`` never built ``tool_calls`` at all
``openai.py`` stream             ``json.loads(...)`` unguarded
``anthropic.py`` adapt_response  ``input if isinstance(input, dict) else {}``
``bedrock.py`` adapt_response    ``input if isinstance(input, dict) else {}``
``bedrock.py`` stream            ``json.loads(...)`` unguarded
``ollama.py`` adapt_response     ``tool_call_parameters()``
===============================  ==========================================

Three distinct defects follow, and this module reproduces each:

**The buffered OpenAI path dropped every tool call.** ``adapt_response`` built
an ``LLMResponse`` with no ``tool_calls`` argument, so it defaulted to ``None``
while ``openai.py`` advertised ``ModelCapability.FUNCTION_CALLING`` for the
model family. Only the streaming path surfaced calls. Two base-class hooks read
``response.tool_calls`` and so were silently disarmed on that path:
``_analyze_response`` mislabels a tool-call-only turn ``thinking_only``, and
``_warn_if_truncated`` downgrades the dangerous truncated-mid-tool-call case to
``info``.

**Unguarded ``json.loads``** raises ``json.JSONDecodeError`` through the
provider abstraction rather than the ``ValidationError`` every other parse
failure in this package raises.

**Substituting ``{}``** for arguments that cannot be read executes the tool
with no arguments at all, which is indistinguishable from a tool that takes
none.

Failing before the fix: every test in ``TestOpenAIBufferedPath``,
``TestUnreadableArgumentsRaise`` (the Anthropic, both Bedrock and the OpenAI
stream cases), and ``TestDisarmedBaseHooks``. ``TestParity`` passes for Ollama
before the fix and for all four after it.
"""

from __future__ import annotations

import json
import logging
import types
from typing import Any

import pytest
from dataknobs_common.exceptions import ValidationError
from dataknobs_llm.llm.base import LLMConfig
from dataknobs_llm.llm.providers.anthropic import AnthropicAdapter
from dataknobs_llm.llm.providers.bedrock import BedrockConverseAdapter
from dataknobs_llm.llm.providers.echo import EchoProvider
from dataknobs_llm.llm.providers.ollama import OllamaAdapter
from dataknobs_llm.llm.providers.openai import OpenAIAdapter, OpenAIProvider

from _anthropic_stubs import make_anthropic_response
from _bedrock_stubs import _StubBedrockClient, _stub_provider

# ---------------------------------------------------------------------------
# OpenAI SDK stand-ins (sanctioned: no local emulator for a paid vendor API)
# ---------------------------------------------------------------------------


def make_openai_response(
    *,
    content: str = "",
    tool_calls: list[tuple[str, Any, str]] | None = None,
    finish_reason: str = "stop",
    model: str = "gpt-4",
    completion_tokens: int = 7,
) -> object:
    """Build an object with the OpenAI chat-completion attribute shape.

    ``tool_calls`` entries are ``(name, arguments, id)``; the SDK sends
    ``arguments`` as a JSON-encoded string.
    """
    calls = [
        types.SimpleNamespace(
            id=call_id,
            type="function",
            function=types.SimpleNamespace(name=name, arguments=arguments),
        )
        for name, arguments, call_id in (tool_calls or [])
    ]
    message = types.SimpleNamespace(content=content, tool_calls=calls or None)
    choice = types.SimpleNamespace(message=message, finish_reason=finish_reason)
    usage = types.SimpleNamespace(
        prompt_tokens=5, completion_tokens=completion_tokens, total_tokens=5 + completion_tokens
    )
    return types.SimpleNamespace(choices=[choice], model=model, usage=usage)


def _delta_chunk(
    *,
    content: str = "",
    tool_deltas: list[tuple[int, str, str, str]] | None = None,
    finish_reason: str | None = None,
) -> object:
    """One streaming chunk. ``tool_deltas`` are ``(index, id, name, args)``."""
    calls = [
        types.SimpleNamespace(
            index=index,
            id=call_id,
            function=types.SimpleNamespace(name=name, arguments=arguments),
        )
        for index, call_id, name, arguments in (tool_deltas or [])
    ]
    delta = types.SimpleNamespace(content=content or None, tool_calls=calls or None)
    choice = types.SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return types.SimpleNamespace(choices=[choice])


class _Stream:
    """Async iterator over scripted chunks."""

    def __init__(self, chunks: list[object]) -> None:
        self._chunks = list(chunks)

    def __aiter__(self) -> _Stream:
        return self

    async def __anext__(self) -> object:
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


class _StreamingClient:
    """Minimal client whose ``chat.completions.create`` returns *stream*."""

    def __init__(self, stream: Any) -> None:
        async def create(**kwargs: Any) -> Any:
            return stream

        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=create))


def _openai_provider(client: Any) -> OpenAIProvider:
    provider = OpenAIProvider(LLMConfig(provider="openai", model="gpt-4"))
    provider._client = client
    provider._is_initialized = True
    return provider


def _bedrock_provider(client: _StubBedrockClient):  # type: ignore[no-untyped-def]
    return _stub_provider(
        LLMConfig(provider="bedrock", model="anthropic.claude-3-haiku-20240307-v1:0", timeout=5.0),
        client,
    )


def _bedrock_tool_stream(raw_input: str) -> list[dict[str, Any]]:
    return [
        {
            "contentBlockStart": {
                "contentBlockIndex": 0,
                "start": {"toolUse": {"toolUseId": "u1", "name": "search"}},
            }
        },
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"toolUse": {"input": raw_input}}}},
        {"messageStop": {"stopReason": "tool_use"}},
    ]


# ---------------------------------------------------------------------------
# The buffered OpenAI path built no tool calls at all
# ---------------------------------------------------------------------------


class TestOpenAIBufferedPath:
    """``complete(tools=...)`` surfaces the calls the model asked for."""

    def test_buffered_response_carries_the_tool_call(self) -> None:
        parsed = OpenAIAdapter().adapt_response(
            make_openai_response(
                tool_calls=[("search", '{"query": "cats"}', "call_1")],
                finish_reason="tool_calls",
            )
        )
        assert parsed.tool_calls is not None
        assert len(parsed.tool_calls) == 1
        assert parsed.tool_calls[0].name == "search"
        assert parsed.tool_calls[0].parameters == {"query": "cats"}
        assert parsed.tool_calls[0].id == "call_1"

    def test_every_call_of_a_parallel_turn_survives(self) -> None:
        parsed = OpenAIAdapter().adapt_response(
            make_openai_response(
                tool_calls=[
                    ("search", '{"query": "cats"}', "call_1"),
                    ("lookup", '{"id": 7}', "call_2"),
                ],
                finish_reason="tool_calls",
            )
        )
        assert parsed.tool_calls is not None
        assert [tc.name for tc in parsed.tool_calls] == ["search", "lookup"]

    def test_parameters_are_splattable(self) -> None:
        """Consumers call ``tool.execute(**tool_call.parameters)``."""
        parsed = OpenAIAdapter().adapt_response(
            make_openai_response(
                tool_calls=[("search", '{"query": "cats"}', "call_1")],
                finish_reason="tool_calls",
            )
        )
        assert parsed.tool_calls is not None
        assert dict(**parsed.tool_calls[0].parameters) == {"query": "cats"}

    def test_a_text_turn_still_has_no_tool_calls(self) -> None:
        parsed = OpenAIAdapter().adapt_response(make_openai_response(content="hello"))
        assert parsed.tool_calls is None

    def test_a_tool_taking_no_arguments_gets_an_empty_mapping(self) -> None:
        parsed = OpenAIAdapter().adapt_response(
            make_openai_response(tool_calls=[("ping", "", "call_1")], finish_reason="tool_calls")
        )
        assert parsed.tool_calls is not None
        assert parsed.tool_calls[0].parameters == {}


# ---------------------------------------------------------------------------
# Base hooks read response.tool_calls, so dropping them disarmed both
# ---------------------------------------------------------------------------


class TestDisarmedBaseHooks:
    """The two base hooks that read ``tool_calls`` see the OpenAI ones now."""

    def _provider(self) -> EchoProvider:
        return EchoProvider(LLMConfig(provider="echo", model="echo-model"))

    def test_tool_call_turn_is_not_mislabelled_thinking_only(self) -> None:
        """Empty content plus real tool calls is a tool turn, not a think turn."""
        parsed = OpenAIAdapter().adapt_response(
            make_openai_response(
                content="",
                tool_calls=[("search", '{"query": "cats"}', "call_1")],
                finish_reason="tool_calls",
                completion_tokens=120,
            )
        )
        analyzed = self._provider()._analyze_response(parsed)
        assert "thinking_only" not in analyzed.metadata

    def test_truncated_mid_tool_call_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """The dangerous case: incomplete arguments that look well-formed."""
        parsed = OpenAIAdapter().adapt_response(
            make_openai_response(
                tool_calls=[("submit", '{"body": "half a sen"}', "call_1")],
                finish_reason="length",
            )
        )
        assert parsed.truncated is True
        with caplog.at_level(logging.WARNING, logger="dataknobs_llm.llm.base"):
            self._provider()._analyze_response(parsed)
        assert any(
            "mid tool-call" in r.message and r.levelno == logging.WARNING for r in caplog.records
        )


# ---------------------------------------------------------------------------
# Arguments that cannot be read are reported, not silently replaced
# ---------------------------------------------------------------------------


class TestUnreadableArgumentsRaise:
    """Every site raises ValidationError; none substitutes ``{}`` or leaks."""

    def test_anthropic_non_object_input(self) -> None:
        response = make_anthropic_response(
            [{"type": "tool_use", "id": "t1", "name": "submit", "input": "not an object"}],
            stop_reason="tool_use",
        )
        with pytest.raises(ValidationError, match="submit"):
            AnthropicAdapter().adapt_response(response)

    def test_bedrock_buffered_non_object_input(self) -> None:
        payload = {
            "output": {
                "message": {
                    "content": [{"toolUse": {"toolUseId": "u1", "name": "search", "input": "nope"}}]
                }
            },
            "stopReason": "tool_use",
        }
        with pytest.raises(ValidationError, match="search"):
            BedrockConverseAdapter().adapt_response(
                payload, model="anthropic.claude-3-haiku-20240307-v1:0"
            )

    async def test_bedrock_stream_malformed_json(self) -> None:
        provider = _bedrock_provider(
            _StubBedrockClient(stream_events=_bedrock_tool_stream('{"query": '))
        )
        with pytest.raises(ValidationError, match="search"):
            async for _ in provider.stream_complete("hi"):
                pass

    async def test_openai_stream_malformed_json(self) -> None:
        provider = _openai_provider(
            _StreamingClient(
                _Stream(
                    [
                        _delta_chunk(tool_deltas=[(0, "call_1", "search", '{"query": ')]),
                        _delta_chunk(finish_reason="tool_calls"),
                    ]
                )
            )
        )
        with pytest.raises(ValidationError, match="search"):
            async for _ in provider.stream_complete("hi"):
                pass

    def test_openai_buffered_malformed_json(self) -> None:
        with pytest.raises(ValidationError, match="search"):
            OpenAIAdapter().adapt_response(
                make_openai_response(
                    tool_calls=[("search", '{"query": ', "call_1")], finish_reason="tool_calls"
                )
            )

    def test_the_raised_error_is_never_a_bare_json_error(self) -> None:
        """A JSONDecodeError reaching a consumer bypasses the package's hierarchy."""
        with pytest.raises(ValidationError) as excinfo:
            OpenAIAdapter().adapt_response(
                make_openai_response(
                    tool_calls=[("search", "{oops", "call_1")], finish_reason="tool_calls"
                )
            )
        assert not isinstance(excinfo.value, json.JSONDecodeError)
        assert isinstance(excinfo.value.__cause__, json.JSONDecodeError)


# ---------------------------------------------------------------------------
# One shape, four wire formats
# ---------------------------------------------------------------------------


class TestParity:
    """Each adapter turns its own wire format into the same canonical shape."""

    def test_openai(self) -> None:
        parsed = OpenAIAdapter().adapt_response(
            make_openai_response(
                tool_calls=[("search", '{"query": "cats"}', "call_1")],
                finish_reason="tool_calls",
            )
        )
        assert parsed.tool_calls is not None
        assert parsed.tool_calls[0].parameters == {"query": "cats"}

    def test_anthropic(self) -> None:
        parsed = AnthropicAdapter().adapt_response(
            make_anthropic_response(
                [
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "search",
                        "input": {"query": "cats"},
                    }
                ],
                stop_reason="tool_use",
            )
        )
        assert parsed.tool_calls is not None
        assert parsed.tool_calls[0].parameters == {"query": "cats"}

    def test_bedrock(self) -> None:
        parsed = BedrockConverseAdapter().adapt_response(
            {
                "output": {
                    "message": {
                        "content": [
                            {
                                "toolUse": {
                                    "toolUseId": "u1",
                                    "name": "search",
                                    "input": {"query": "cats"},
                                }
                            }
                        ]
                    }
                },
                "stopReason": "tool_use",
            },
            model="anthropic.claude-3-haiku-20240307-v1:0",
        )
        assert parsed.tool_calls is not None
        assert parsed.tool_calls[0].parameters == {"query": "cats"}

    def test_ollama(self) -> None:
        parsed = OllamaAdapter().adapt_response(
            {
                "model": "llama3.2",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "search", "arguments": {"query": "cats"}}}
                    ],
                },
                "done": True,
                "done_reason": "stop",
            }
        )
        assert parsed.tool_calls is not None
        assert parsed.tool_calls[0].parameters == {"query": "cats"}

    def test_an_empty_id_becomes_none_everywhere(self) -> None:
        """An empty-string id carries no more information than its absence."""
        parsed = OpenAIAdapter().adapt_response(
            make_openai_response(tool_calls=[("search", "{}", "")], finish_reason="tool_calls")
        )
        assert parsed.tool_calls is not None
        assert parsed.tool_calls[0].id is None
