"""Bedrock provider-boundary stubs, shared by the Bedrock test modules.

Bedrock is a paid external API with no faithful local emulator. This thin
stub sits exactly at the ``session.client("bedrock-runtime")`` boundary
and returns canned payloads, so complete()/stream_complete()/embed() run
through their REAL code paths (request build, adapt, response parse). All
methods are ``async``/async-context-manager so the stub cannot mask a
missing ``await`` (the guardrail from testing-practices.md).

Underscore-prefixed so pytest does not collect it. It lives in its own module
because three test modules need these stubs, and they reached them by importing
``test_bedrock_provider`` — which resolves only while pytest is inserting each
collected file's directory onto ``sys.path``, and which imports an
already-collected module a second time under a second name, running its
module-level code twice.
"""

from __future__ import annotations

import json
from typing import Any, Self

from dataknobs_llm.llm.base import LLMConfig
from dataknobs_llm.llm.providers.bedrock import BedrockProvider

__all__ = [
    "_StubBedrockClient",
    "_StubBody",
    "_StubSession",
    "_stub_provider",
]


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
