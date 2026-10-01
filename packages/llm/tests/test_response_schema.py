"""A JSON Schema a reply must satisfy: ``LLMConfig.response_schema``.

``response_format="json"`` asks for *some* JSON. A caller that needs one shape
had no way to say so, and JSON mode alone lets a model answer in any shape:
measured on Ollama with ``qwen2.5:7b``, a table asked for as
``{"columns", "rows": [{"cells"}]}`` came back with ``"rows"`` repeated once
per table row (which ``json.loads`` reads as the last row alone), or with its
cells as bare strings. The same prompts with the shape sent as a schema came
back in the shape.

A stated schema is honoured or refused by name, never ignored, by the rule
``embedding_overflow`` set: Ollama sends it as ``format``, OpenAI as a
``json_schema`` response format, Anthropic as ``output_config.format``; a
provider that cannot constrain its output refuses it before any request. The
build methods are pure (no network) and are the single choke point each
provider's ``complete`` and ``stream_complete`` share, so they are exercised
directly; the live tests run against a real Ollama server, because the
constraint is the server's, not ours.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any, ClassVar

import pytest

from dataknobs_common.exceptions import OperationError, ValidationError
from dataknobs_common.testing import requires_ollama, requires_ollama_model
from dataknobs_llm import EchoProvider
from dataknobs_llm.llm.base import LLMConfig, LLMMessage, LLMProvider
from dataknobs_llm.llm.providers.anthropic import AnthropicProvider
from dataknobs_llm.llm.providers.base import SyncProviderAdapter
from dataknobs_llm.llm.providers.bedrock import BedrockProvider
from dataknobs_llm.llm.providers.caching import CachingEmbedProvider, MemoryEmbeddingCache
from dataknobs_llm.llm.providers.huggingface import HuggingFaceProvider
from dataknobs_llm.llm.providers.ollama import OllamaProvider
from dataknobs_llm.llm.providers.openai import OpenAIProvider
from dataknobs_llm.testing import CapturingProvider


def _closed(node: Any) -> Any:
    """Close every object in a schema, as OpenAI's strict mode and Anthropic require."""
    if isinstance(node, dict):
        out = {key: _closed(value) for key, value in node.items()}
        if out.get("type") == "object":
            out["additionalProperties"] = False
            out["required"] = sorted(out.get("properties", {}))
        return out
    if isinstance(node, list):
        return [_closed(item) for item in node]
    return node


#: A table: named columns, and rows of cells that each cite something.
TABLE: dict[str, Any] = _closed(
    {
        "type": "object",
        "properties": {
            "columns": {"type": "array", "items": {"type": "string"}, "minItems": 2},
            "rows": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "cells": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "text": {"type": "string"},
                                    "cites": {"type": "array", "items": {"type": "string"}},
                                },
                            },
                        }
                    },
                },
            },
        },
    }
)


def _config(provider: str = "ollama", model: str = "m", **fields: Any) -> LLMConfig:
    return LLMConfig(provider=provider, model=model, **fields)


# ---------------------------------------------------------------------------
# The schema is checked where it is written
# ---------------------------------------------------------------------------


class TestASchemaIsCheckedWhereItIsWritten:
    """A malformed schema fails at the config, not as a 400 on the first call."""

    def test_a_string_is_refused_and_pointed_at_response_format(self) -> None:
        # ``response_schema: json`` is an easy slip beside ``response_format: json``.
        with pytest.raises(ValidationError, match="response_format"):
            _config(response_schema="json")  # type: ignore[arg-type]

    def test_an_empty_schema_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="empty"):
            _config(response_schema={})

    def test_a_schema_that_cannot_be_sent_as_json_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="JSON"):
            _config(response_schema={"type": "object", "default": object()})

    def test_a_model_class_is_refused_with_how_to_pass_its_schema(self) -> None:
        class Reply:
            @classmethod
            def model_json_schema(cls) -> dict[str, Any]:
                return {"type": "object"}

        with pytest.raises(ValidationError, match="model_json_schema"):
            _config(response_schema=Reply)  # type: ignore[arg-type]

    def test_from_dict_checks_it_too(self) -> None:
        with pytest.raises(ValidationError):
            LLMConfig.from_dict({"provider": "ollama", "model": "m", "response_schema": "json"})

    async def test_a_per_call_override_is_checked_too(self) -> None:
        provider = EchoProvider({"provider": "echo", "model": "test"})
        await provider.initialize()
        with pytest.raises(ValidationError):
            await provider.complete("a table", config_overrides={"response_schema": "json"})

    def test_a_valid_schema_survives_a_round_trip(self) -> None:
        config = _config(response_schema=TABLE, response_schema_strict=False)
        again = LLMConfig.from_dict(config.to_dict())
        assert again.response_schema == TABLE
        assert again.response_schema_strict is False


# ---------------------------------------------------------------------------
# Honoured or refused by name
# ---------------------------------------------------------------------------


class TestAStatedSchemaIsHonouredOrRefusedByName:
    """No provider receives a schema and sends an unconstrained request."""

    def test_huggingface_refuses_it_by_name(self) -> None:
        provider = HuggingFaceProvider(
            _config("huggingface", "meta-llama/Llama-3-8b", response_schema=TABLE)
        )
        with pytest.raises(ValidationError, match=r"HuggingFaceProvider.*response_schema"):
            provider._build_hf_parameters(provider.config)

    def test_bedrock_refuses_it_by_name(self) -> None:
        provider = BedrockProvider(
            _config(
                "bedrock",
                "anthropic.claude-sonnet-4-5-20250929-v1:0",
                response_schema=TABLE,
            )
        )
        with pytest.raises(ValidationError, match=r"BedrockProvider.*response_schema"):
            provider._build_converse_request("hi", provider.config, None)

    def test_a_refusing_provider_still_serves_a_request_without_one(self) -> None:
        provider = HuggingFaceProvider(_config("huggingface", "meta-llama/Llama-3-8b"))
        provider._build_hf_parameters(provider.config)

    def test_which_providers_honour_it(self) -> None:
        """Pinned so that adding support, or losing it, is a decision, not drift."""
        honouring = {
            cls.__name__
            for cls in (
                AnthropicProvider,
                BedrockProvider,
                EchoProvider,
                HuggingFaceProvider,
                OllamaProvider,
                OpenAIProvider,
            )
            if cls(_config(model="m")).supports_response_schema()
        }
        assert honouring == {
            "AnthropicProvider",
            "EchoProvider",
            "OllamaProvider",
            "OpenAIProvider",
        }

    @pytest.mark.parametrize(
        "wrap",
        [
            pytest.param(CapturingProvider, id="capturing"),
            pytest.param(lambda p: CachingEmbedProvider(p, MemoryEmbeddingCache()), id="caching"),
            pytest.param(SyncProviderAdapter, id="sync-bridge"),
        ],
    )
    def test_a_wrapper_answers_with_the_wrapped_providers_support(self, wrap: Any) -> None:
        honouring = EchoProvider({"provider": "echo", "model": "test"})
        refusing = HuggingFaceProvider(_config("huggingface", "meta-llama/Llama-3-8b"))
        assert wrap(honouring).supports_response_schema() is True
        assert wrap(refusing).supports_response_schema() is False

    async def test_echo_honours_it_so_a_test_double_does_not_refuse_what_production_sends(
        self,
    ) -> None:
        provider = EchoProvider({"provider": "echo", "model": "test"})
        provider.set_responses(['{"columns": ["a", "b"], "rows": []}'])
        response = await provider.complete("a table", config_overrides={"response_schema": TABLE})
        assert json.loads(response.content)["columns"] == ["a", "b"]


# ---------------------------------------------------------------------------
# What each honouring provider sends
# ---------------------------------------------------------------------------


class TestOpenAI:
    def _provider(self, **fields: Any) -> OpenAIProvider:
        return OpenAIProvider(_config("openai", "gpt-4o", **fields))

    def test_a_schema_is_sent_strict_by_default(self) -> None:
        """Without ``strict`` OpenAI treats a schema as guidance only."""
        provider = self._provider(response_schema=TABLE)
        wire = provider._build_api_kwargs(provider.config)
        assert wire["response_format"] == {
            "type": "json_schema",
            "json_schema": {"name": "reply", "schema": TABLE, "strict": True},
        }

    def test_strict_can_be_turned_off_for_a_schema_strict_mode_rejects(self) -> None:
        provider = self._provider(response_schema=TABLE, response_schema_strict=False)
        wire = provider._build_api_kwargs(provider.config)
        assert wire["response_format"]["json_schema"]["strict"] is False

    def test_json_mode_without_a_schema_is_unchanged(self) -> None:
        provider = self._provider(response_format="json")
        wire = provider._build_api_kwargs(provider.config)
        assert wire["response_format"] == {"type": "json_object"}

    def test_a_schema_wins_over_a_per_call_response_format_and_says_so(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The narrower request wins where both are stated, and not silently."""
        provider = self._provider(response_schema=TABLE)
        with caplog.at_level(logging.WARNING, logger="dataknobs_llm.llm.base"):
            wire = provider._build_api_kwargs(
                provider.config, {"response_format": {"type": "json_object"}}
            )
        assert wire["response_format"]["type"] == "json_schema"
        assert any("response_format" in record.getMessage() for record in caplog.records)

    def test_a_per_call_response_format_without_a_schema_still_rides_through(self) -> None:
        provider = self._provider()
        wire = provider._build_api_kwargs(provider.config, {"response_format": {"type": "text"}})
        assert wire["response_format"] == {"type": "text"}


class TestAnthropic:
    def test_a_schema_is_sent_as_the_output_format(self) -> None:
        provider = AnthropicProvider(_config("anthropic", "claude-opus-5-5", response_schema=TABLE))
        wire = provider._build_api_kwargs(provider.config)
        assert wire["output_config"] == {"format": {"type": "json_schema", "schema": TABLE}}

    def test_no_schema_sends_no_output_format(self) -> None:
        provider = AnthropicProvider(_config("anthropic", "claude-opus-5-5"))
        assert "output_config" not in provider._build_api_kwargs(provider.config)


class TestOllama:
    """``complete`` and ``stream_complete`` build one payload, in one place."""

    def _provider(self, **fields: Any) -> OllamaProvider:
        return OllamaProvider(_config("ollama", "qwen2.5:7b", **fields))

    @pytest.mark.parametrize("stream", [False, True])
    def test_a_schema_is_sent_as_the_format(self, stream: bool) -> None:
        provider = self._provider(response_schema=TABLE)
        payload = provider._build_chat_payload(provider.config, "a table", None, stream=stream)
        assert payload["format"] == TABLE
        assert payload["stream"] is stream

    @pytest.mark.parametrize("stream", [False, True])
    def test_a_schema_wins_over_json_mode(self, stream: bool) -> None:
        provider = self._provider(response_format="json", response_schema=TABLE)
        payload = provider._build_chat_payload(provider.config, "a table", None, stream=stream)
        assert payload["format"] == TABLE

    def test_json_mode_is_sent_as_json_and_neither_sends_no_format(self) -> None:
        json_mode = self._provider(response_format="json")
        plain = self._provider()
        assert json_mode._build_chat_payload(json_mode.config, "x", None, stream=False)[
            "format"
        ] == ("json")
        assert "format" not in plain._build_chat_payload(plain.config, "x", None, stream=False)

    def test_the_system_prompt_and_messages_are_assembled_once_for_both(self) -> None:
        provider = self._provider(system_prompt="Be brief.")
        buffered = provider._build_chat_payload(provider.config, "hi", None, stream=False)
        streamed = provider._build_chat_payload(provider.config, "hi", None, stream=True)
        assert {**buffered, "stream": True} == streamed
        assert [m["role"] for m in buffered["messages"]] == ["system", "user"]

    def test_a_format_with_tools_warns_that_no_tool_call_will_come(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Measured on Ollama 0.33.2 with ``qwen2.5:7b``: with a ``format`` set,
        asked to use a tool, the model wrote JSON text and made no tool call.
        Sent without the format, it made the call.
        """

        class Weather:
            name = "get_weather"
            description = "Get the weather for a city"
            schema: ClassVar[dict[str, Any]] = {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            }

        provider = self._provider(response_schema=TABLE)
        with caplog.at_level(logging.WARNING, logger="dataknobs_llm.llm.providers.ollama"):
            payload = provider._build_chat_payload(
                provider.config, "weather?", [Weather()], stream=False
            )
        assert payload["tools"] and payload["format"] == TABLE
        assert any("tool call" in record.getMessage() for record in caplog.records)

    def test_no_warning_without_a_format(self, caplog: pytest.LogCaptureFixture) -> None:
        class Weather:
            name = "get_weather"
            description = "Get the weather"
            schema: ClassVar[dict[str, Any]] = {"type": "object", "properties": {}}

        provider = self._provider()
        with caplog.at_level(logging.WARNING, logger="dataknobs_llm.llm.providers.ollama"):
            provider._build_chat_payload(provider.config, "weather?", [Weather()], stream=False)
        assert not caplog.records


def test_every_shipped_provider_declares_whether_it_honours_a_schema() -> None:
    """The default is to refuse, so a new provider cannot ignore one by omission."""
    assert LLMProvider._response_schema_supported is False


# ---------------------------------------------------------------------------
# Live: the constraint is the server's
# ---------------------------------------------------------------------------

_MODEL = "gemma3:1b"


@pytest.fixture
async def ollama() -> AsyncIterator[OllamaProvider]:
    llm = OllamaProvider(LLMConfig(provider="ollama", model=_MODEL, temperature=0.0))
    await llm.initialize()
    try:
        yield llm
    finally:
        await llm.close()


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [key for key, _ in pairs]
    assert len(keys) == len(set(keys)), f"a key repeats: {keys}"
    return dict(pairs)


def _prompt() -> list[LLMMessage]:
    rows = "\n".join(
        f"[r{n}] {text}"
        for n, text in enumerate(
            (
                "Multi-factor sign-in makes a stolen password less useful.",
                "Alerting on unusual sign-ins surfaces theft quickly.",
                "A revocation playbook shortens the window after a theft.",
            ),
            start=1,
        )
    )
    prompt = (
        "Compare these controls as a table, one table row per control, each cell"
        f" citing the row id it comes from.\n{rows}"
    )
    return [LLMMessage(role="user", content=prompt)]


def _assert_a_table(content: str) -> None:
    table = json.loads(content, object_pairs_hook=_unique)
    assert isinstance(table["columns"], list) and len(table["columns"]) >= 2
    assert table["rows"] and all(
        isinstance(cell, dict) and isinstance(cell["text"], str) and isinstance(cell["cites"], list)
        for row in table["rows"]
        for cell in row["cells"]
    )


@requires_ollama
@requires_ollama_model(_MODEL)
async def test_ollama_constrains_a_reply_to_the_schema(ollama: OllamaProvider) -> None:
    response = await ollama.complete(_prompt(), config_overrides={"response_schema": TABLE})
    _assert_a_table(response.content)


@requires_ollama
@requires_ollama_model(_MODEL)
async def test_ollama_constrains_a_streamed_reply_to_the_schema(ollama: OllamaProvider) -> None:
    chunks = [
        chunk.delta
        async for chunk in ollama.stream_complete(
            _prompt(), config_overrides={"response_schema": TABLE}
        )
    ]
    _assert_a_table("".join(chunks))


@requires_ollama
async def test_an_ollama_error_does_not_log_the_conversation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A non-200 logs what identifies the request, never the caller's text."""
    secret = "the caller's private words"
    llm = OllamaProvider(LLMConfig(provider="ollama", model="no-such-model:0"))
    await llm.initialize()
    try:
        with caplog.at_level(logging.DEBUG, logger="dataknobs_llm"), pytest.raises(OperationError):
            await llm.complete(secret, config_overrides={"response_schema": TABLE})
    finally:
        await llm.close()
    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert "no-such-model:0" in logged
    assert secret not in logged
    assert "cites" not in logged
