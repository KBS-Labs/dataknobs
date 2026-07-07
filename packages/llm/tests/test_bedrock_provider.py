"""Tests for the Amazon Bedrock provider.

Coverage is layered per the testing hierarchy:

1. :class:`BedrockConverseAdapter` is pure request/response mapping — tested
   directly with no client and no mock (the bulk of coverage).
2. ``complete()`` / ``stream_complete()`` / ``embed()`` are exercised
   end-to-end against a thin async stub at the ``session.client(
   "bedrock-runtime")`` boundary. Bedrock is a paid external API with no
   faithful local emulator (LocalStack community / moto do not implement
   ``bedrock-runtime`` inference), so a boundary stub is the sanctioned last
   resort the mock-prohibition explicitly allows. The stub uses **async**
   method signatures matching aioboto3 so it cannot hide a missing-``await``.
3. An ``assert_no_blocking()`` test proves ``initialize()`` + a stubbed
   ``complete()`` do not block the event loop.

Live behavioural coverage lives in ``integration/test_bedrock_live.py``
behind the ``requires_bedrock`` marker.
"""

from __future__ import annotations

import json
from typing import Any, Self

import pytest

from dataknobs_llm.llm.base import (
    LLMConfig,
    LLMMessage,
    ModelCapability,
    ToolCall,
)
from dataknobs_llm.llm.providers.bedrock import (
    BedrockConverseAdapter,
    BedrockProvider,
    _canonical_model_id,
    _estimate_cost,
)


class _Tool:
    """Minimal Tool-like object for adapter tests."""

    def __init__(self, name: str = "search") -> None:
        self.name = name
        self.description = f"A tool called {name}"
        self.schema = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }


# ---------------------------------------------------------------------------
# Adapter tests (pure — no client, no mock)
# ---------------------------------------------------------------------------


class TestAdaptMessages:
    def test_plain_user_and_assistant(self) -> None:
        adapter = BedrockConverseAdapter()
        system, messages = adapter.adapt_messages([
            LLMMessage(role="user", content="hello"),
            LLMMessage(role="assistant", content="hi there"),
        ])
        assert system == []
        assert messages == [
            {"role": "user", "content": [{"text": "hello"}]},
            {"role": "assistant", "content": [{"text": "hi there"}]},
        ]

    def test_system_message_and_prompt_merge(self) -> None:
        adapter = BedrockConverseAdapter()
        system, messages = adapter.adapt_messages(
            [
                LLMMessage(role="system", content="be terse"),
                LLMMessage(role="user", content="hello"),
            ],
            system_prompt="you are helpful",
        )
        # system_prompt first, then in-list system message
        assert system == [{"text": "you are helpful"}, {"text": "be terse"}]
        assert messages == [
            {"role": "user", "content": [{"text": "hello"}]},
        ]

    def test_assistant_tool_calls(self) -> None:
        adapter = BedrockConverseAdapter()
        _, messages = adapter.adapt_messages([
            LLMMessage(
                role="assistant",
                content="let me look that up",
                tool_calls=[
                    ToolCall(name="search", parameters={"query": "x"}, id="t1"),
                ],
            ),
        ])
        assert messages == [
            {
                "role": "assistant",
                "content": [
                    {"text": "let me look that up"},
                    {
                        "toolUse": {
                            "toolUseId": "t1",
                            "name": "search",
                            "input": {"query": "x"},
                        }
                    },
                ],
            }
        ]

    def test_tool_call_without_id_falls_back_to_name(self) -> None:
        adapter = BedrockConverseAdapter()
        _, messages = adapter.adapt_messages([
            LLMMessage(
                role="assistant",
                content="",
                tool_calls=[ToolCall(name="search", parameters={})],
            ),
        ])
        tool_use = messages[0]["content"][0]["toolUse"]
        assert tool_use["toolUseId"] == "search"

    def test_consecutive_tool_results_consolidated(self) -> None:
        adapter = BedrockConverseAdapter()
        _, messages = adapter.adapt_messages([
            LLMMessage(role="tool", content="result A", tool_call_id="t1"),
            LLMMessage(role="tool", content="result B", tool_call_id="t2"),
        ])
        # Converse rejects consecutive same-role messages: both tool results
        # must land in a single user message.
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == [
            {"toolResult": {"toolUseId": "t1", "content": [{"text": "result A"}]}},
            {"toolResult": {"toolUseId": "t2", "content": [{"text": "result B"}]}},
        ]

    def test_tool_result_after_user_starts_new_message(self) -> None:
        adapter = BedrockConverseAdapter()
        _, messages = adapter.adapt_messages([
            LLMMessage(role="user", content="hi"),
            LLMMessage(role="tool", content="result", tool_call_id="t1"),
        ])
        assert len(messages) == 2
        assert messages[0]["content"] == [{"text": "hi"}]
        assert messages[1]["content"][0]["toolResult"]["toolUseId"] == "t1"


class TestAdaptConfig:
    def test_only_set_keys_emitted(self) -> None:
        adapter = BedrockConverseAdapter()
        config = LLMConfig(
            provider="bedrock",
            model="anthropic.claude-3-haiku-20240307-v1:0",
            temperature=0.5,
            max_tokens=256,
            top_p=0.9,
            stop_sequences=["END"],
        )
        params = adapter.adapt_config(config)
        assert params == {
            "modelId": "anthropic.claude-3-haiku-20240307-v1:0",
            "inferenceConfig": {
                "temperature": 0.5,
                "topP": 0.9,
                "maxTokens": 256,
                "stopSequences": ["END"],
            },
        }

    def test_no_generation_params_omits_inference_config(self) -> None:
        adapter = BedrockConverseAdapter()
        config = LLMConfig(provider="bedrock", model="amazon.nova-lite-v1:0")
        params = adapter.adapt_config(config)
        assert params == {"modelId": "amazon.nova-lite-v1:0"}


class TestAdaptTools:
    def test_tools_to_toolspec(self) -> None:
        adapter = BedrockConverseAdapter()
        specs = adapter.adapt_tools([_Tool("search")])
        assert specs == [
            {
                "toolSpec": {
                    "name": "search",
                    "description": "A tool called search",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                        }
                    },
                }
            }
        ]

    def test_raw_functions_to_toolspec(self) -> None:
        adapter = BedrockConverseAdapter()
        specs = adapter.adapt_raw_functions([
            {
                "name": "lookup",
                "description": "look it up",
                "parameters": {"type": "object", "properties": {}},
            }
        ])
        assert specs[0]["toolSpec"]["name"] == "lookup"
        assert specs[0]["toolSpec"]["inputSchema"]["json"] == {
            "type": "object",
            "properties": {},
        }


class TestAdaptResponse:
    def test_text_and_usage(self) -> None:
        adapter = BedrockConverseAdapter()
        response = {
            "output": {"message": {"content": [{"text": "the answer"}]}},
            "stopReason": "end_turn",
            "usage": {
                "inputTokens": 10,
                "outputTokens": 5,
                "totalTokens": 15,
            },
        }
        parsed = adapter.adapt_response(
            response, model="anthropic.claude-3-haiku-20240307-v1:0"
        )
        assert parsed.content == "the answer"
        assert parsed.finish_reason == "end_turn"
        assert parsed.usage == {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }
        assert parsed.model == "anthropic.claude-3-haiku-20240307-v1:0"
        assert parsed.tool_calls is None
        # cost estimated from the static price map
        assert parsed.cost_usd == pytest.approx(
            (10 / 1000) * 0.00025 + (5 / 1000) * 0.00125
        )

    def test_tool_use_blocks(self) -> None:
        adapter = BedrockConverseAdapter()
        response = {
            "output": {
                "message": {
                    "content": [
                        {"text": "let me search"},
                        {
                            "toolUse": {
                                "toolUseId": "abc",
                                "name": "search",
                                "input": {"query": "x"},
                            }
                        },
                    ]
                }
            },
            "stopReason": "tool_use",
        }
        parsed = adapter.adapt_response(response, model="unknown.model")
        assert parsed.content == "let me search"
        assert parsed.tool_calls is not None
        assert parsed.tool_calls[0].name == "search"
        assert parsed.tool_calls[0].parameters == {"query": "x"}
        assert parsed.tool_calls[0].id == "abc"
        # unknown model → no cost
        assert parsed.cost_usd is None


class TestHelpers:
    def test_canonical_model_id_strips_region(self) -> None:
        assert (
            _canonical_model_id("us.anthropic.claude-3-haiku-20240307-v1:0")
            == "anthropic.claude-3-haiku-20240307-v1:0"
        )
        assert (
            _canonical_model_id("amazon.titan-embed-text-v2:0")
            == "amazon.titan-embed-text-v2:0"
        )

    def test_estimate_cost_none_for_unknown(self) -> None:
        assert _estimate_cost("unknown.model", {"prompt_tokens": 1}) is None
        assert _estimate_cost("amazon.nova-lite-v1:0", None) is None


class TestCapabilities:
    def test_chat_model_has_function_calling(self) -> None:
        provider = BedrockProvider(
            LLMConfig(
                provider="bedrock",
                model="anthropic.claude-3-5-sonnet-20240620-v1:0",
            )
        )
        caps = provider.get_capabilities()
        assert ModelCapability.FUNCTION_CALLING in caps
        assert ModelCapability.VISION in caps
        assert ModelCapability.EMBEDDINGS not in caps

    def test_embedding_model_has_embeddings(self) -> None:
        provider = BedrockProvider(
            LLMConfig(provider="bedrock", model="amazon.titan-embed-text-v2:0")
        )
        caps = provider.get_capabilities()
        assert ModelCapability.EMBEDDINGS in caps
        assert ModelCapability.FUNCTION_CALLING not in caps

    @pytest.mark.asyncio
    async def test_validate_model_heuristic(self) -> None:
        good = BedrockProvider(
            LLMConfig(provider="bedrock", model="meta.llama3-8b-instruct-v1:0")
        )
        bad = BedrockProvider(
            LLMConfig(provider="bedrock", model="gpt-4")
        )
        assert await good.validate_model() is True
        assert await bad.validate_model() is False


class TestGuardrailConfig:
    def test_guardrail_applied_when_both_set(self) -> None:
        provider = BedrockProvider(
            LLMConfig(
                provider="bedrock",
                model="anthropic.claude-3-haiku-20240307-v1:0",
                options={
                    "guardrail_identifier": "gr-1",
                    "guardrail_version": "DRAFT",
                    "guardrail_trace": "enabled",
                },
            )
        )
        request = provider._build_converse_request(
            "hi", provider.config, None
        )
        assert request["guardrailConfig"] == {
            "guardrailIdentifier": "gr-1",
            "guardrailVersion": "DRAFT",
            "trace": "enabled",
        }

    def test_no_guardrail_when_incomplete(self) -> None:
        provider = BedrockProvider(
            LLMConfig(
                provider="bedrock",
                model="anthropic.claude-3-haiku-20240307-v1:0",
                options={"guardrail_identifier": "gr-1"},
            )
        )
        request = provider._build_converse_request("hi", provider.config, None)
        assert "guardrailConfig" not in request


# ---------------------------------------------------------------------------
# Provider-boundary stub — the sanctioned last-resort mock
# ---------------------------------------------------------------------------
#
# Bedrock is a paid external API with no faithful local emulator. This thin
# stub sits exactly at the ``session.client("bedrock-runtime")`` boundary
# and returns canned payloads, so complete()/stream_complete()/embed() run
# through their REAL code paths (request build, adapt, response parse). All
# methods are ``async``/async-context-manager so the stub cannot mask a
# missing ``await`` (the guardrail from testing-practices.md).


class _StubBody:
    """Async reader mimicking an aiobotocore streaming body."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._data = json.dumps(payload).encode("utf-8")

    async def read(self) -> bytes:
        return self._data


class _StubBedrockClient:
    """Async stub matching the aioboto3 bedrock-runtime client surface."""

    def __init__(
        self,
        *,
        converse_response: dict[str, Any] | None = None,
        stream_events: list[dict[str, Any]] | None = None,
        invoke_payloads: list[dict[str, Any]] | None = None,
    ) -> None:
        self._converse_response = converse_response
        self._stream_events = stream_events or []
        self._invoke_payloads = invoke_payloads or []
        self.converse_calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []
        self.invoke_calls: list[dict[str, Any]] = []

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def converse(self, **kwargs: Any) -> dict[str, Any]:
        self.converse_calls.append(kwargs)
        assert self._converse_response is not None
        return self._converse_response

    async def converse_stream(self, **kwargs: Any) -> dict[str, Any]:
        self.stream_calls.append(kwargs)

        async def _gen() -> Any:
            for event in self._stream_events:
                yield event

        return {"stream": _gen()}

    async def invoke_model(self, **kwargs: Any) -> dict[str, Any]:
        self.invoke_calls.append(kwargs)
        payload = self._invoke_payloads[len(self.invoke_calls) - 1]
        return {"body": _StubBody(payload)}


class _StubSession:
    """aioboto3.Session stub returning a fixed bedrock-runtime client."""

    def __init__(self, client: _StubBedrockClient) -> None:
        self._client = client
        self.client_calls: list[tuple[str, dict[str, Any]]] = []

    def client(self, service: str, **kwargs: Any) -> _StubBedrockClient:
        self.client_calls.append((service, kwargs))
        return self._client


def _stub_provider(
    config: LLMConfig, client: _StubBedrockClient
) -> BedrockProvider:
    """Build a provider with its session pre-wired to a stub (no AWS)."""
    provider = BedrockProvider(config)
    provider._session = _StubSession(client)
    provider._endpoint_url = None
    provider._is_initialized = True
    return provider


class TestCompleteBoundary:
    @pytest.mark.asyncio
    async def test_complete_runs_real_path(self) -> None:
        client = _StubBedrockClient(
            converse_response={
                "output": {"message": {"content": [{"text": "hi!"}]}},
                "stopReason": "end_turn",
                "usage": {
                    "inputTokens": 3,
                    "outputTokens": 2,
                    "totalTokens": 5,
                },
            }
        )
        provider = _stub_provider(
            LLMConfig(
                provider="bedrock",
                model="anthropic.claude-3-haiku-20240307-v1:0",
                temperature=0.2,
            ),
            client,
        )
        response = await provider.complete("hello")
        assert response.content == "hi!"
        assert response.finish_reason == "end_turn"
        assert response.usage["total_tokens"] == 5
        # request was built through the real path
        sent = client.converse_calls[0]
        assert sent["modelId"] == "anthropic.claude-3-haiku-20240307-v1:0"
        assert sent["messages"] == [
            {"role": "user", "content": [{"text": "hello"}]}
        ]
        assert sent["inferenceConfig"]["temperature"] == 0.2

    @pytest.mark.asyncio
    async def test_complete_with_tools(self) -> None:
        client = _StubBedrockClient(
            converse_response={
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
            }
        )
        provider = _stub_provider(
            LLMConfig(
                provider="bedrock",
                model="anthropic.claude-3-haiku-20240307-v1:0",
            ),
            client,
        )
        response = await provider.complete("find cats", tools=[_Tool("search")])
        assert response.tool_calls[0].name == "search"
        assert client.converse_calls[0]["toolConfig"]["tools"][0][
            "toolSpec"
        ]["name"] == "search"


class TestStreamBoundary:
    @pytest.mark.asyncio
    async def test_stream_text_and_final(self) -> None:
        client = _StubBedrockClient(
            stream_events=[
                {"contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": {"text": "Hel"},
                }},
                {"contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": {"text": "lo"},
                }},
                {"messageStop": {"stopReason": "end_turn"}},
                {"metadata": {"usage": {
                    "inputTokens": 4,
                    "outputTokens": 1,
                    "totalTokens": 5,
                }}},
            ]
        )
        provider = _stub_provider(
            LLMConfig(
                provider="bedrock",
                model="anthropic.claude-3-haiku-20240307-v1:0",
            ),
            client,
        )
        chunks = [c async for c in provider.stream_complete("hi")]
        text = "".join(c.delta for c in chunks if not c.is_final)
        assert text == "Hello"
        final = chunks[-1]
        assert final.is_final is True
        assert final.finish_reason == "end_turn"
        assert final.usage["total_tokens"] == 5
        assert final.model == "anthropic.claude-3-haiku-20240307-v1:0"

    @pytest.mark.asyncio
    async def test_stream_tool_use_accumulation(self) -> None:
        client = _StubBedrockClient(
            stream_events=[
                {"contentBlockStart": {
                    "contentBlockIndex": 0,
                    "start": {"toolUse": {"toolUseId": "u1", "name": "search"}},
                }},
                {"contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": {"toolUse": {"input": '{"query":'}},
                }},
                {"contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": {"toolUse": {"input": '"cats"}'}},
                }},
                {"messageStop": {"stopReason": "tool_use"}},
            ]
        )
        provider = _stub_provider(
            LLMConfig(
                provider="bedrock",
                model="anthropic.claude-3-haiku-20240307-v1:0",
            ),
            client,
        )
        chunks = [c async for c in provider.stream_complete("find cats")]
        final = chunks[-1]
        assert final.tool_calls is not None
        assert final.tool_calls[0].name == "search"
        assert final.tool_calls[0].parameters == {"query": "cats"}
        assert final.tool_calls[0].id == "u1"


class TestEmbedBoundary:
    @pytest.mark.asyncio
    async def test_titan_single(self) -> None:
        client = _StubBedrockClient(
            invoke_payloads=[{"embedding": [0.1, 0.2, 0.3]}]
        )
        provider = _stub_provider(
            LLMConfig(
                provider="bedrock",
                model="amazon.titan-embed-text-v2:0",
                dimensions=3,
            ),
            client,
        )
        vector = await provider.embed("hello")
        assert vector == [0.1, 0.2, 0.3]
        body = json.loads(client.invoke_calls[0]["body"])
        assert body["inputText"] == "hello"
        assert body["dimensions"] == 3
        assert body["normalize"] is True

    @pytest.mark.asyncio
    async def test_titan_list_gathers(self) -> None:
        client = _StubBedrockClient(
            invoke_payloads=[
                {"embedding": [0.1]},
                {"embedding": [0.2]},
            ]
        )
        provider = _stub_provider(
            LLMConfig(
                provider="bedrock", model="amazon.titan-embed-text-v2:0"
            ),
            client,
        )
        vectors = await provider.embed(["a", "b"])
        assert vectors == [[0.1], [0.2]]
        assert len(client.invoke_calls) == 2

    @pytest.mark.asyncio
    async def test_cohere_batch(self) -> None:
        client = _StubBedrockClient(
            invoke_payloads=[{"embeddings": [[0.1], [0.2]]}]
        )
        provider = _stub_provider(
            LLMConfig(
                provider="bedrock", model="cohere.embed-english-v3"
            ),
            client,
        )
        vectors = await provider.embed(["a", "b"])
        assert vectors == [[0.1], [0.2]]
        # one call for the whole list
        assert len(client.invoke_calls) == 1
        body = json.loads(client.invoke_calls[0]["body"])
        assert body["texts"] == ["a", "b"]

    @pytest.mark.asyncio
    async def test_unknown_embedding_family_raises(self) -> None:
        provider = _stub_provider(
            LLMConfig(
                provider="bedrock", model="anthropic.claude-3-haiku-20240307-v1:0"
            ),
            _StubBedrockClient(),
        )
        with pytest.raises(ValueError, match="Unsupported Bedrock embedding"):
            await provider.embed("hello")


class TestFunctionCallDeprecated:
    @pytest.mark.asyncio
    async def test_function_call_extracts_first_tool(self) -> None:
        client = _StubBedrockClient(
            converse_response={
                "output": {
                    "message": {
                        "content": [
                            {
                                "toolUse": {
                                    "toolUseId": "u1",
                                    "name": "lookup",
                                    "input": {"q": "x"},
                                }
                            }
                        ]
                    }
                },
                "stopReason": "tool_use",
            }
        )
        provider = _stub_provider(
            LLMConfig(
                provider="bedrock",
                model="anthropic.claude-3-haiku-20240307-v1:0",
            ),
            client,
        )
        with pytest.warns(DeprecationWarning):
            response = await provider.function_call(
                [LLMMessage(role="user", content="hi")],
                [{"name": "lookup", "description": "d", "parameters": {}}],
            )
        assert response.function_call == {"name": "lookup", "arguments": {"q": "x"}}


# ---------------------------------------------------------------------------
# Async-correctness proof
# ---------------------------------------------------------------------------


class TestNoBlocking:
    @pytest.mark.asyncio
    async def test_initialize_and_complete_do_not_block(self) -> None:
        pytest.importorskip("aioboto3")
        from dataknobs_common.testing import assert_no_blocking

        client = _StubBedrockClient(
            converse_response={
                "output": {"message": {"content": [{"text": "ok"}]}},
                "stopReason": "end_turn",
            }
        )
        provider = BedrockProvider(
            LLMConfig(
                provider="bedrock",
                model="anthropic.claude-3-haiku-20240307-v1:0",
                options={"region_name": "us-east-1"},
            )
        )
        # initialize() offloads session construction off the loop via the
        # shared factory; assert the whole path stays non-blocking.
        with assert_no_blocking():
            await provider.initialize()
            # Swap in the boundary stub for the request itself (the real
            # aioboto3 call would need AWS); the loop-safety of the session
            # build is what initialize() proves above.
            provider._session = _StubSession(client)
            await provider.complete("hello")
