"""Tests for OpenAI vendor-error translation (S4).

The provider previously called ``chat.completions.create`` / ``embeddings.create``
with no try/except, so any API error surfaced as a raw ``openai.*`` exception and
a consumer could only catch it by coupling to the SDK. The fix translates OpenAI
errors into ``dataknobs_common`` exceptions (``ValidationError`` /
``RateLimitError`` / ``OperationError``), preserving the original on
``__cause__``; non-OpenAI errors propagate unchanged.

These tests construct **real** ``openai`` SDK error objects (the package is a
dev dependency for exactly this reason — no fakes for the real dependency's
exception classes) and drive the real ``OpenAIProvider`` via a minimal raising
client stub. They FAIL against HEAD (raw errors propagate) and pass after the
fix — a raw ``openai.RateLimitError`` is not a dataknobs ``RateLimitError``, so
``pytest.raises(RateLimitError)`` fails on HEAD.
"""

from __future__ import annotations

import types
from typing import Any

import httpx
import openai
import pytest

from dataknobs_common.exceptions import (
    OperationError,
    RateLimitError,
    ValidationError,
)
from dataknobs_llm.llm.base import LLMConfig
from dataknobs_llm.llm.providers.openai import OpenAIProvider


# ---------------------------------------------------------------------------
# Real openai error builders + a raising client stub
# ---------------------------------------------------------------------------


def _request() -> httpx.Request:
    return httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


def _status_error(
    cls: type[openai.APIStatusError],
    status: int,
    message: str,
    headers: dict[str, str] | None = None,
) -> openai.APIStatusError:
    resp = httpx.Response(
        status,
        request=_request(),
        headers=headers or {},
        json={"error": {"type": "error", "message": message}},
    )
    return cls(message, response=resp, body=None)


class _RaisingCall:
    """A ``.create`` endpoint that raises the next scripted outcome."""

    def __init__(self, outcomes: list[Exception | None]) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> object:
        self.calls.append(dict(kwargs))
        outcome = self._outcomes.pop(0) if self._outcomes else None
        if outcome is not None:
            raise outcome
        raise AssertionError("stub exhausted — test expected an error outcome")


class _RaisingClient:
    """Minimal ``openai.AsyncOpenAI`` stand-in with scripted outcomes.

    Exposes ``.chat.completions.create`` and ``.embeddings.create`` with the
    same shape the provider calls.
    """

    def __init__(
        self,
        completion_outcomes: list[Exception | None] | None = None,
        embed_outcomes: list[Exception | None] | None = None,
    ) -> None:
        self._completions = _RaisingCall(completion_outcomes or [])
        self.chat = types.SimpleNamespace(completions=self._completions)
        self.embeddings = _RaisingCall(embed_outcomes or [])


def _provider(client: Any, **config_kwargs: Any) -> OpenAIProvider:
    provider = OpenAIProvider(
        LLMConfig(provider="openai", model="gpt-4", **config_kwargs)
    )
    provider._client = client
    provider._is_initialized = True
    return provider


class TestVendorErrorTranslation:
    """Raw OpenAI errors become catchable dataknobs exceptions."""

    async def test_400_becomes_validation_error(self) -> None:
        client = _RaisingClient(
            [_status_error(openai.BadRequestError, 400, "malformed request")]
        )
        provider = _provider(client)
        with pytest.raises(ValidationError) as excinfo:
            await provider.complete("hi")
        assert isinstance(excinfo.value.__cause__, openai.BadRequestError)

    async def test_429_becomes_rate_limit_error_with_retry_after(self) -> None:
        client = _RaisingClient(
            [
                _status_error(
                    openai.RateLimitError,
                    429,
                    "slow down",
                    headers={"retry-after": "3"},
                )
            ]
        )
        provider = _provider(client)
        with pytest.raises(RateLimitError) as excinfo:
            await provider.complete("hi")
        assert excinfo.value.retry_after == 3.0
        assert isinstance(excinfo.value.__cause__, openai.RateLimitError)

    async def test_401_becomes_operation_error(self) -> None:
        client = _RaisingClient(
            [_status_error(openai.AuthenticationError, 401, "bad key")]
        )
        provider = _provider(client)
        with pytest.raises(OperationError):
            await provider.complete("hi")

    async def test_403_becomes_operation_error(self) -> None:
        client = _RaisingClient(
            [_status_error(openai.PermissionDeniedError, 403, "denied")]
        )
        provider = _provider(client)
        with pytest.raises(OperationError):
            await provider.complete("hi")

    async def test_timeout_becomes_operation_error(self) -> None:
        """A timeout carries no ``status_code`` → OperationError."""
        client = _RaisingClient([openai.APITimeoutError(request=_request())])
        provider = _provider(client)
        with pytest.raises(OperationError) as excinfo:
            await provider.complete("hi")
        assert isinstance(excinfo.value.__cause__, openai.APITimeoutError)

    async def test_connection_error_becomes_operation_error(self) -> None:
        client = _RaisingClient(
            [openai.APIConnectionError(request=_request())]
        )
        provider = _provider(client)
        with pytest.raises(OperationError):
            await provider.complete("hi")

    async def test_embed_error_is_translated(self) -> None:
        client = _RaisingClient(
            embed_outcomes=[
                _status_error(
                    openai.RateLimitError,
                    429,
                    "slow down",
                    headers={"retry-after": "5"},
                )
            ]
        )
        provider = _provider(client)
        with pytest.raises(RateLimitError) as excinfo:
            await provider.embed("some text")
        assert excinfo.value.retry_after == 5.0

    async def test_stream_error_is_translated(self) -> None:
        client = _RaisingClient(
            [_status_error(openai.BadRequestError, 400, "bad stream")]
        )
        provider = _provider(client)
        with pytest.raises(ValidationError):
            async for _ in provider.stream_complete("hi"):
                pass

    async def test_non_openai_error_propagates_unchanged(self) -> None:
        """A bug in our own code is never masked as an API error."""
        client = _RaisingClient([ValueError("internal bug")])
        provider = _provider(client)
        with pytest.raises(ValueError):
            await provider.complete("hi")
