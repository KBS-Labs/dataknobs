"""Tests for Anthropic vendor-error translation (S4) and the 400-retry net.

Two concerns, both at the ``messages.create``/``messages.stream`` boundary:

* **S4 — vendor-error translation.** The provider previously called
  ``messages.create`` with no try/except, so any API error surfaced as a raw
  ``anthropic.*`` exception and a consumer could only catch it by coupling to
  the SDK. The fix translates Anthropic errors into ``dataknobs_common``
  exceptions (``ValidationError`` / ``RateLimitError`` / ``OperationError``),
  preserving the original on ``__cause__``; non-Anthropic errors propagate
  unchanged.

* **400-retry safety net.** For a model family the static constraint table
  doesn't know yet, an "unsupported sampling param" 400 is recovered once: the
  named param is dropped, the request retried, and the discovery memoized so
  subsequent requests to that model drop it up front.

These tests construct **real** ``anthropic`` SDK error objects (the package is
a dev dependency for exactly this reason — no fakes for the real dependency's
exception classes) and drive the real ``AnthropicProvider`` wiring via a
minimal raising client stub (a sanctioned SDK stand-in). They FAIL against HEAD
(raw errors propagate; no retry) and pass after the fix.
"""

from __future__ import annotations

import logging
from typing import Any

import anthropic
import httpx
import pytest

from dataknobs_common.exceptions import (
    OperationError,
    RateLimitError,
    ValidationError,
)
from dataknobs_llm.llm.base import LLMConfig
from dataknobs_llm.llm.providers import anthropic as anthropic_module
from dataknobs_llm.llm.providers.anthropic import AnthropicProvider

from test_anthropic_param_handling import make_anthropic_response


# ---------------------------------------------------------------------------
# Real anthropic error builders + a raising client stub
# ---------------------------------------------------------------------------


def _request() -> httpx.Request:
    return httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def _status_error(
    cls: type[anthropic.APIStatusError],
    status: int,
    message: str,
    headers: dict[str, str] | None = None,
) -> anthropic.APIStatusError:
    resp = httpx.Response(
        status,
        request=_request(),
        headers=headers or {},
        json={"error": {"type": "error", "message": message}},
    )
    return cls(message, response=resp, body=None)


class _RaisingClient:
    """Minimal ``anthropic.AsyncAnthropic`` stand-in with scripted outcomes.

    Each ``messages.create`` call pops the next scripted outcome: an exception
    is raised, ``None`` yields a canned success. Records a copy of every call's
    kwargs so tests can assert what reached the API on each attempt.
    """

    def __init__(self, outcomes: list[Exception | None]) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []
        self.messages = self

    async def create(self, **kwargs: Any) -> object:
        self.calls.append(dict(kwargs))
        outcome = self._outcomes.pop(0) if self._outcomes else None
        if outcome is not None:
            raise outcome
        return make_anthropic_response([{"type": "text", "text": "ok"}])


class _StreamCtx:
    """Async-context-manager stand-in for ``messages.stream(...)``.

    Raises ``error`` on entry when set (a rejected-param 400 fails on stream
    entry, before any chunk); otherwise yields no deltas and returns a canned
    final message.
    """

    def __init__(self, error: Exception | None) -> None:
        self._error = error

    async def __aenter__(self) -> "_StreamCtx":
        if self._error is not None:
            raise self._error
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    def __aiter__(self) -> "_StreamCtx":
        return self

    async def __anext__(self) -> object:
        raise StopAsyncIteration

    async def get_final_message(self) -> object:
        return make_anthropic_response([{"type": "text", "text": "ok"}])


class _RaisingStreamClient:
    """Like ``_RaisingClient`` but for the ``messages.stream`` path."""

    def __init__(self, outcomes: list[Exception | None]) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []
        self.messages = self

    def stream(self, **kwargs: Any) -> _StreamCtx:
        self.calls.append(dict(kwargs))
        outcome = self._outcomes.pop(0) if self._outcomes else None
        return _StreamCtx(outcome)


def _provider(
    model: str, client: Any, **config_kwargs: Any
) -> AnthropicProvider:
    provider = AnthropicProvider(
        LLMConfig(provider="anthropic", model=model, **config_kwargs)
    )
    provider._client = client
    provider._is_initialized = True
    return provider


@pytest.fixture(autouse=True)
def _clear_discovered_cache() -> Any:
    """Isolate the process-level discovered-rejected-params cache per test."""
    anthropic_module._DISCOVERED_REJECTED_PARAMS.clear()
    yield
    anthropic_module._DISCOVERED_REJECTED_PARAMS.clear()


# ---------------------------------------------------------------------------
# S4 — vendor-error translation
# ---------------------------------------------------------------------------


class TestVendorErrorTranslation:
    """Raw Anthropic errors become catchable dataknobs exceptions."""

    async def test_400_becomes_validation_error(self) -> None:
        client = _RaisingClient(
            [_status_error(anthropic.BadRequestError, 400, "malformed request")]
        )
        provider = _provider("claude-3-haiku", client)
        with pytest.raises(ValidationError) as excinfo:
            await provider.complete("hi")
        assert isinstance(excinfo.value.__cause__, anthropic.BadRequestError)

    async def test_429_becomes_rate_limit_error_with_retry_after(self) -> None:
        client = _RaisingClient(
            [
                _status_error(
                    anthropic.RateLimitError,
                    429,
                    "slow down",
                    headers={"retry-after": "3"},
                )
            ]
        )
        provider = _provider("claude-3-haiku", client)
        with pytest.raises(RateLimitError) as excinfo:
            await provider.complete("hi")
        assert excinfo.value.retry_after == 3.0
        assert isinstance(excinfo.value.__cause__, anthropic.RateLimitError)

    async def test_401_becomes_operation_error(self) -> None:
        client = _RaisingClient(
            [_status_error(anthropic.AuthenticationError, 401, "bad key")]
        )
        provider = _provider("claude-3-haiku", client)
        with pytest.raises(OperationError):
            await provider.complete("hi")

    async def test_connection_error_becomes_operation_error(self) -> None:
        client = _RaisingClient(
            [anthropic.APIConnectionError(request=_request())]
        )
        provider = _provider("claude-3-haiku", client)
        with pytest.raises(OperationError):
            await provider.complete("hi")

    async def test_non_anthropic_error_propagates_unchanged(self) -> None:
        """A bug in our own code is never masked as an API error."""
        client = _RaisingClient([ValueError("internal bug")])
        provider = _provider("claude-3-haiku", client)
        with pytest.raises(ValueError):
            await provider.complete("hi")


# ---------------------------------------------------------------------------
# 400-retry safety net + memoization
# ---------------------------------------------------------------------------


class TestRejectedParamRetry:
    """An unexpected sampling-param 400 is recovered once and memoized."""

    async def test_unexpected_param_dropped_and_retried(self, caplog) -> None:
        # A model NOT in the static table, so recovery is the only path.
        client = _RaisingClient(
            [
                _status_error(
                    anthropic.BadRequestError,
                    400,
                    "temperature: Extra inputs are not permitted",
                ),
                None,  # retry succeeds
            ]
        )
        provider = _provider("claude-6-future", client, temperature=0.3)
        with caplog.at_level(logging.WARNING):
            result = await provider.complete("hi")
        assert result.content == "ok"
        # Two calls: first carried temperature, retry dropped it.
        assert len(client.calls) == 2
        assert "temperature" in client.calls[0]
        assert "temperature" not in client.calls[1]
        assert any("temperature" in rec.message for rec in caplog.records)

    async def test_discovery_memoized_across_calls(self) -> None:
        """After discovery, the param is dropped up front (no second 400)."""
        client = _RaisingClient(
            [
                _status_error(
                    anthropic.BadRequestError,
                    400,
                    "temperature: Extra inputs are not permitted",
                ),
                None,
            ]
        )
        provider = _provider("claude-6-future", client, temperature=0.3)
        await provider.complete("hi")

        # New provider, same model: memoized discovery drops temperature up
        # front, so there is exactly one call and no 400.
        client2 = _RaisingClient([None])
        provider2 = _provider("claude-6-future", client2, temperature=0.3)
        await provider2.complete("hi")
        assert len(client2.calls) == 1
        assert "temperature" not in client2.calls[0]

    async def test_ambiguous_400_not_retried(self) -> None:
        """A 400 naming no forwarded sampling param is translated, not retried."""
        client = _RaisingClient(
            [_status_error(anthropic.BadRequestError, 400, "messages: too long")]
        )
        provider = _provider("claude-6-future", client, temperature=0.3)
        with pytest.raises(ValidationError):
            await provider.complete("hi")
        assert len(client.calls) == 1  # no retry

    async def test_retry_failure_is_translated(self) -> None:
        """If the retry also fails, the second error is translated (S4)."""
        client = _RaisingClient(
            [
                _status_error(
                    anthropic.BadRequestError,
                    400,
                    "temperature: Extra inputs are not permitted",
                ),
                _status_error(anthropic.BadRequestError, 400, "still bad"),
            ]
        )
        provider = _provider("claude-6-future", client, temperature=0.3)
        with pytest.raises(ValidationError):
            await provider.complete("hi")
        assert len(client.calls) == 2


class TestStreamErrorHandling:
    """The streaming path shares S4 translation and the 400-retry net."""

    async def test_stream_400_translated(self) -> None:
        client = _RaisingStreamClient(
            [_status_error(anthropic.BadRequestError, 400, "messages: bad")]
        )
        provider = _provider("claude-6-future", client, temperature=0.3)
        with pytest.raises(ValidationError):
            async for _ in provider.stream_complete("hi"):
                pass
        assert len(client.calls) == 1  # no retry (ambiguous 400)

    async def test_stream_rejected_param_retried_on_entry(self) -> None:
        client = _RaisingStreamClient(
            [
                _status_error(
                    anthropic.BadRequestError,
                    400,
                    "temperature: Extra inputs are not permitted",
                ),
                None,  # retry stream succeeds
            ]
        )
        provider = _provider("claude-6-future", client, temperature=0.3)
        chunks = [c async for c in provider.stream_complete("hi")]
        assert len(client.calls) == 2
        assert "temperature" in client.calls[0]
        assert "temperature" not in client.calls[1]
        assert any(c.is_final for c in chunks)

    async def test_stream_non_anthropic_error_propagates(self) -> None:
        client = _RaisingStreamClient([ValueError("internal bug")])
        provider = _provider("claude-6-future", client)
        with pytest.raises(ValueError):
            async for _ in provider.stream_complete("hi"):
                pass
