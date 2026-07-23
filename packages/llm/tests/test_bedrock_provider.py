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
import logging
from typing import Any, Self

import pytest
from dataknobs_common.exceptions import ConfigurationError

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
        # Bedrock runs Claude → finish_reason normalized onto the canonical
        # vocabulary, raw stopReason preserved on metadata.
        assert parsed.finish_reason == "stop"
        assert parsed.metadata["raw_finish_reason"] == "end_turn"
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
        # tool_use normalized onto the canonical vocabulary.
        assert parsed.finish_reason == "tool_calls"
        assert parsed.metadata["raw_finish_reason"] == "tool_use"
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

    def test_embedding_model_excludes_chat_capabilities(self) -> None:
        """An embed-only model must NOT advertise chat/stream/generation.

        Reproduces the gap where these were appended unconditionally, so
        ``amazon.titan-embed-*`` reported CHAT/STREAMING/TEXT_GENERATION it
        cannot serve — capability-driven routing could then send a chat
        request to an embedding model.
        """
        provider = BedrockProvider(
            LLMConfig(provider="bedrock", model="amazon.titan-embed-text-v2:0")
        )
        caps = provider.get_capabilities()
        assert ModelCapability.CHAT not in caps
        assert ModelCapability.STREAMING not in caps
        assert ModelCapability.TEXT_GENERATION not in caps
        assert ModelCapability.VISION not in caps
        # embeddings is the sole capability
        assert caps == [ModelCapability.EMBEDDINGS]

    def test_cohere_embed_model_excludes_chat_capabilities(self) -> None:
        """Cohere embed models are embed-only too (``-embed-`` / prefix)."""
        provider = BedrockProvider(
            LLMConfig(provider="bedrock", model="cohere.embed-english-v3")
        )
        caps = provider.get_capabilities()
        assert caps == [ModelCapability.EMBEDDINGS]

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
    """Build a provider with its session pre-wired to a stub (no AWS).

    ``_session_config`` is built from ``config.options`` in ``__init__``,
    so ``_client_kwargs`` works without a real ``initialize()``.
    """
    provider = BedrockProvider(config)
    provider._session = _StubSession(client)
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
        # finish_reason normalized onto the canonical vocabulary (Claude family).
        assert response.finish_reason == "stop"
        assert response.metadata["raw_finish_reason"] == "end_turn"
        assert response.usage["total_tokens"] == 5
        # request was built through the real path
        sent = client.converse_calls[0]
        assert sent["modelId"] == "anthropic.claude-3-haiku-20240307-v1:0"
        assert sent["messages"] == [
            {"role": "user", "content": [{"text": "hello"}]}
        ]
        assert sent["inferenceConfig"]["temperature"] == 0.2

    @pytest.mark.asyncio
    async def test_complete_client_gets_explicit_read_timeout(self) -> None:
        """Security rule 2: the client must carry an explicit read timeout
        sized to the configured LLM timeout — not boto's default.

        Reproduces the gap where ``_client_kwargs`` returned only
        ``endpoint_url``, so the bedrock-runtime client had NO timeout,
        retry, or pool config and ``LLMConfig.timeout`` was a silent no-op.
        """
        client = _StubBedrockClient(
            converse_response={
                "output": {"message": {"content": [{"text": "ok"}]}},
                "stopReason": "end_turn",
            }
        )
        provider = _stub_provider(
            LLMConfig(
                provider="bedrock",
                model="anthropic.claude-3-haiku-20240307-v1:0",
                timeout=42.0,
            ),
            client,
        )
        await provider.complete("hi")
        _, client_kwargs = provider._session.client_calls[0]
        boto_config = client_kwargs["config"]
        assert boto_config.read_timeout == 42.0
        assert boto_config.connect_timeout == 10
        # retry/pool tuning present too (was dropped entirely before)
        assert boto_config.retries["max_attempts"] == 3

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
        # Streaming final chunk normalized to match the buffered path.
        assert final.finish_reason == "stop"
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

    @pytest.mark.asyncio
    async def test_stream_read_timeout_decoupled_from_total_timeout(
        self,
    ) -> None:
        """Streaming must NOT apply ``LLMConfig.timeout`` as read_timeout.

        Reproduces the semantic bug: ``read_timeout`` is botocore's per-read
        (inter-chunk) timeout, so passing the whole-response ``timeout`` (here
        a tight 5s) would kill the stream on any inter-token pause > 5s. The
        streaming client must default to boto's read timeout (``None``) instead.
        """
        client = _StubBedrockClient(
            stream_events=[{"messageStop": {"stopReason": "end_turn"}}]
        )
        provider = _stub_provider(
            LLMConfig(
                provider="bedrock",
                model="anthropic.claude-3-haiku-20240307-v1:0",
                timeout=5.0,
            ),
            client,
        )
        _ = [c async for c in provider.stream_complete("hi")]
        _, client_kwargs = provider._session.client_calls[0]
        # Decoupled: the tight 5s total budget is not the inter-chunk timeout;
        # the client falls back to botocore's default read timeout instead.
        assert client_kwargs["config"].read_timeout != 5.0

    @pytest.mark.asyncio
    async def test_stream_read_timeout_from_options(self) -> None:
        """``options["stream_read_timeout"]`` sets the inter-chunk timeout."""
        client = _StubBedrockClient(
            stream_events=[{"messageStop": {"stopReason": "end_turn"}}]
        )
        provider = _stub_provider(
            LLMConfig(
                provider="bedrock",
                model="anthropic.claude-3-haiku-20240307-v1:0",
                timeout=5.0,
                options={"stream_read_timeout": 120},
            ),
            client,
        )
        _ = [c async for c in provider.stream_complete("hi")]
        _, client_kwargs = provider._session.client_calls[0]
        assert client_kwargs["config"].read_timeout == 120.0


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
    async def test_titan_concurrency_is_bounded(self) -> None:
        """A large Titan batch must not fan out unbounded invoke_model calls.

        Reproduces the throttling / pool-exhaustion hazard: ``embed`` gathered
        one ``invoke_model`` per text with no bound. With
        ``embed_max_concurrency=2`` no more than 2 calls may be in flight.
        """
        import asyncio

        class _ConcurrencyTrackingClient:
            def __init__(self) -> None:
                self.in_flight = 0
                self.max_in_flight = 0

            async def __aenter__(self) -> Self:
                return self

            async def __aexit__(self, *exc: object) -> None:
                return None

            async def invoke_model(self, **kwargs: Any) -> dict[str, Any]:
                self.in_flight += 1
                self.max_in_flight = max(self.max_in_flight, self.in_flight)
                # Yield twice so unbounded peers pile up before any releases.
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                self.in_flight -= 1
                return {"body": _StubBody({"embedding": [0.0]})}

        client = _ConcurrencyTrackingClient()
        provider = _stub_provider(
            LLMConfig(
                provider="bedrock",
                model="amazon.titan-embed-text-v2:0",
                options={"embed_max_concurrency": 2},
            ),
            client,  # type: ignore[arg-type]
        )
        await provider.embed(["a", "b", "c", "d", "e", "f"])
        assert client.max_in_flight <= 2
        assert client.max_in_flight >= 1

    @pytest.mark.asyncio
    async def test_titan_normalize_from_options(self) -> None:
        """``normalize`` is read from options (default ``True``).

        Reproduces the hardcoded ``normalize=True`` — a consumer could not
        request un-normalized Titan embeddings.
        """
        client = _StubBedrockClient(invoke_payloads=[{"embedding": [0.1]}])
        provider = _stub_provider(
            LLMConfig(
                provider="bedrock",
                model="amazon.titan-embed-text-v2:0",
                options={"normalize": False},
            ),
            client,
        )
        await provider.embed("hello")
        body = json.loads(client.invoke_calls[0]["body"])
        assert body["normalize"] is False

    @pytest.mark.asyncio
    async def test_titan_normalize_string_false_is_false(self) -> None:
        """A string ``"False"`` option must disable normalization.

        Reproduces the ``bool("False") is True`` footgun: a raw ``bool()``
        coercion of the string would silently re-enable normalization.
        """
        client = _StubBedrockClient(invoke_payloads=[{"embedding": [0.1]}])
        provider = _stub_provider(
            LLMConfig(
                provider="bedrock",
                model="amazon.titan-embed-text-v2:0",
                options={"normalize": "False"},
            ),
            client,
        )
        await provider.embed("hello")
        body = json.loads(client.invoke_calls[0]["body"])
        assert body["normalize"] is False

    @pytest.mark.asyncio
    async def test_embed_max_concurrency_invalid_raises_config_error(
        self,
    ) -> None:
        """A non-numeric ``embed_max_concurrency`` raises ConfigurationError.

        Reproduces the unguarded ``int(override)`` which raised a bare
        ``ValueError`` with no option context.
        """
        provider = _stub_provider(
            LLMConfig(
                provider="bedrock",
                model="amazon.titan-embed-text-v2:0",
                options={"embed_max_concurrency": "auto"},
            ),
            _StubBedrockClient(invoke_payloads=[{"embedding": [0.1]}]),
        )
        with pytest.raises(ConfigurationError, match="embed_max_concurrency"):
            await provider.embed("hello")

    @pytest.mark.asyncio
    async def test_cohere_input_type_from_options(self) -> None:
        """``input_type`` is read from options.

        Reproduces the hardcoded ``search_document`` — query-time embeddings
        were mislabeled, skewing Cohere retrieval scoring.
        """
        client = _StubBedrockClient(invoke_payloads=[{"embeddings": [[0.1]]}])
        provider = _stub_provider(
            LLMConfig(
                provider="bedrock",
                model="cohere.embed-english-v3",
                options={"input_type": "search_query"},
            ),
            client,
        )
        await provider.embed("q")
        body = json.loads(client.invoke_calls[0]["body"])
        assert body["input_type"] == "search_query"

    @pytest.mark.asyncio
    async def test_cohere_input_type_defaults_to_search_document(self) -> None:
        client = _StubBedrockClient(invoke_payloads=[{"embeddings": [[0.1]]}])
        provider = _stub_provider(
            LLMConfig(provider="bedrock", model="cohere.embed-english-v3"),
            client,
        )
        await provider.embed("doc")
        body = json.loads(client.invoke_calls[0]["body"])
        assert body["input_type"] == "search_document"

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

    @pytest.mark.asyncio
    async def test_truncated_tool_call_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A truncated tool-call turn on this path must fire the warning.

        Reproduce-first: FAILS when function_call() returns ``parsed`` directly
        (flag survives but the shared warn hook never runs); passes once it
        routes through ``_analyze_response`` like ``complete()``.
        """
        client = _StubBedrockClient(
            converse_response={
                "output": {
                    "message": {
                        "content": [
                            {
                                "toolUse": {
                                    "toolUseId": "u1",
                                    "name": "submit",
                                    "input": {"q": "x"},
                                }
                            }
                        ]
                    }
                },
                "stopReason": "max_tokens",
            }
        )
        provider = _stub_provider(
            LLMConfig(
                provider="bedrock",
                model="anthropic.claude-3-haiku-20240307-v1:0",
            ),
            client,
        )
        with caplog.at_level(logging.WARNING, logger="dataknobs_llm.llm.base"):
            with pytest.warns(DeprecationWarning):
                response = await provider.function_call(
                    [LLMMessage(role="user", content="hi")],
                    [{"name": "submit", "description": "d", "parameters": {}}],
                )
        assert response.truncated is True
        assert response.finish_reason == "length"
        assert response.function_call == {"name": "submit", "arguments": {"q": "x"}}
        assert any(
            "mid tool-call" in r.getMessage() and r.levelno == logging.WARNING
            for r in caplog.records
        )


class TestObservability:
    """The provider logs sanitized per-call diagnostics (no credentials)."""

    @pytest.mark.asyncio
    async def test_complete_emits_debug_log(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        client = _StubBedrockClient(
            converse_response={
                "output": {"message": {"content": [{"text": "ok"}]}},
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
            ),
            client,
        )
        with caplog.at_level(
            logging.DEBUG, logger="dataknobs_llm.llm.providers.bedrock"
        ):
            await provider.complete("hi")
        assert any(
            "converse complete" in r.getMessage() for r in caplog.records
        )

    @pytest.mark.asyncio
    async def test_embed_emits_debug_log(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        client = _StubBedrockClient(invoke_payloads=[{"embedding": [0.1]}])
        provider = _stub_provider(
            LLMConfig(
                provider="bedrock", model="amazon.titan-embed-text-v2:0"
            ),
            client,
        )
        with caplog.at_level(
            logging.DEBUG, logger="dataknobs_llm.llm.providers.bedrock"
        ):
            await provider.embed("hello")
        assert any("embed complete" in r.getMessage() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_stream_emits_debug_log_with_latency(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        client = _StubBedrockClient(
            stream_events=[
                {"contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": {"text": "hi"},
                }},
                {"messageStop": {"stopReason": "end_turn"}},
            ]
        )
        provider = _stub_provider(
            LLMConfig(
                provider="bedrock",
                model="anthropic.claude-3-haiku-20240307-v1:0",
            ),
            client,
        )
        with caplog.at_level(
            logging.DEBUG, logger="dataknobs_llm.llm.providers.bedrock"
        ):
            _ = [c async for c in provider.stream_complete("hi")]
        done = [
            r for r in caplog.records if "converse_stream done" in r.getMessage()
        ]
        assert done and "latency_ms" in done[0].getMessage()


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
