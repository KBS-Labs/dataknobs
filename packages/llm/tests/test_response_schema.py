"""A JSON Schema a reply must satisfy: ``LLMConfig.response_schema``.

``response_format="json"`` asks for *some* JSON. A caller that needs one shape
had no way to say so, and JSON mode alone lets a model answer in any shape:
measured on Ollama with ``qwen2.5:7b``, a table asked for as
``{"columns", "rows": [{"cells"}]}`` came back with ``"rows"`` repeated once
per table row (which ``json.loads`` reads as the last row alone), or with its
cells as bare strings. The same prompts with the shape sent as a schema came
back in the shape.

A provider that can constrain output to a schema sends it (Ollama as
``format``, OpenAI as a ``json_schema`` response format); the live test
below runs against a real Ollama server, because the constraint is the
server's, not ours.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

from dataknobs_common.testing import requires_ollama, requires_ollama_model
from dataknobs_llm import EchoProvider
from dataknobs_llm.llm.base import LLMConfig, LLMMessage
from dataknobs_llm.llm.providers.ollama import OllamaAdapter, OllamaProvider
from dataknobs_llm.llm.providers.openai import OpenAIAdapter

#: A table: named columns, and rows of cells that each cite something.
TABLE: dict[str, Any] = {
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
                            "required": ["text", "cites"],
                        },
                    }
                },
                "required": ["cells"],
            },
        },
    },
    "required": ["columns", "rows"],
}


def _config(**fields: Any) -> LLMConfig:
    return LLMConfig(provider="ollama", model="m", **fields)


def test_ollama_sends_a_schema_as_its_format_and_json_mode_as_json() -> None:
    adapter = OllamaAdapter()
    assert adapter.adapt_format(_config(response_schema=TABLE)) == TABLE
    assert adapter.adapt_format(_config(response_format="json")) == "json"
    assert adapter.adapt_format(_config()) is None
    # The schema is the narrower request, so it wins where both are set.
    both = _config(response_format="json", response_schema=TABLE)
    assert adapter.adapt_format(both) == TABLE


def test_openai_sends_a_schema_as_a_json_schema_response_format() -> None:
    adapter = OpenAIAdapter()
    wire = adapter.adapt_config(LLMConfig(provider="openai", model="m", response_schema=TABLE))
    assert wire["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "reply", "schema": TABLE},
    }
    plain = adapter.adapt_config(LLMConfig(provider="openai", model="m", response_format="json"))
    assert plain["response_format"] == {"type": "json_object"}


async def test_a_schema_is_a_per_call_override() -> None:
    """Callers ask for a shape per call, as they ask for JSON per call."""
    provider = EchoProvider({"provider": "echo", "model": "test"})
    await provider.initialize()
    runtime = provider._get_runtime_config({"response_schema": TABLE})
    assert runtime.response_schema == TABLE
    assert provider.config.response_schema is None
    await provider.complete("a table", config_overrides={"response_schema": TABLE})
    assert provider.get_last_call()["config_overrides"] == {"response_schema": TABLE}


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


@requires_ollama
@requires_ollama_model(_MODEL)
async def test_ollama_constrains_a_reply_to_the_schema(ollama: OllamaProvider) -> None:
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
    response = await ollama.complete(
        [LLMMessage(role="user", content=prompt)],
        config_overrides={"response_schema": TABLE},
    )
    table = json.loads(response.content, object_pairs_hook=_unique)
    assert isinstance(table["columns"], list) and len(table["columns"]) >= 2
    assert table["rows"] and all(
        isinstance(cell, dict) and isinstance(cell["text"], str) and isinstance(cell["cites"], list)
        for row in table["rows"]
        for cell in row["cells"]
    )
