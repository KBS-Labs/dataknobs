"""Tests for Ollama vendor-error translation (S4).

Ollama has no vendor SDK — the provider speaks the Ollama HTTP API over
``aiohttp`` and calls ``raise_for_status()``, so any API error previously
surfaced as a raw ``aiohttp.ClientResponseError`` (or a connection/timeout
error) that a consumer could only catch by coupling to ``aiohttp``. The fix
translates these into ``dataknobs_common`` exceptions (``ValidationError`` /
``RateLimitError`` / ``OperationError``), preserving the original on
``__cause__``; non-transport errors — crucially the domain-specific
``ToolsNotSupportedError`` — propagate unchanged.

These tests construct **real** ``aiohttp.ClientResponseError`` objects and drive
the real ``OllamaProvider`` via a minimal raising session stub. They FAIL
against HEAD (raw errors propagate) and pass after the fix.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from dataknobs_common.exceptions import (
    OperationError,
    RateLimitError,
    ValidationError,
)
from dataknobs_llm.exceptions import ToolsNotSupportedError
from dataknobs_llm.llm.base import LLMConfig, LLMMessage
from dataknobs_llm.llm.providers.ollama import OllamaProvider

from _aiohttp_error_stub import (
    FakeResponse,
    FakeSession,
    make_client_response_error,
)


def _provider(session: Any, **config_kwargs: Any) -> OllamaProvider:
    provider = OllamaProvider(
        LLMConfig(provider="ollama", model="llama3.2", **config_kwargs)
    )
    provider._session = session
    provider._is_initialized = True
    return provider


class TestVendorErrorTranslation:
    """Raw aiohttp transport errors become catchable dataknobs exceptions."""

    async def test_400_becomes_validation_error(self) -> None:
        err = make_client_response_error(400, "malformed request")
        session = FakeSession(
            [FakeSession.responding(FakeResponse(400, raise_exc=err))]
        )
        provider = _provider(session)
        with pytest.raises(ValidationError) as excinfo:
            await provider.complete("hi")
        assert excinfo.value.__cause__ is err

    async def test_429_becomes_rate_limit_error_with_retry_after(self) -> None:
        err = make_client_response_error(
            429, "slow down", headers={"retry-after": "4"}
        )
        session = FakeSession(
            [FakeSession.responding(FakeResponse(429, raise_exc=err))]
        )
        provider = _provider(session)
        with pytest.raises(RateLimitError) as excinfo:
            await provider.complete("hi")
        assert excinfo.value.retry_after == 4.0
        assert excinfo.value.__cause__ is err

    async def test_401_becomes_operation_error(self) -> None:
        err = make_client_response_error(401, "unauthorized")
        session = FakeSession(
            [FakeSession.responding(FakeResponse(401, raise_exc=err))]
        )
        provider = _provider(session)
        with pytest.raises(OperationError):
            await provider.complete("hi")

    async def test_connection_error_becomes_operation_error(self) -> None:
        """A connection error carries no status → OperationError."""
        import aiohttp

        exc = aiohttp.ClientConnectionError("cannot connect")
        session = FakeSession([FakeSession.failing(exc)])
        provider = _provider(session)
        with pytest.raises(OperationError) as excinfo:
            await provider.complete("hi")
        assert excinfo.value.__cause__ is exc

    async def test_timeout_becomes_operation_error(self) -> None:
        exc = asyncio.TimeoutError()
        session = FakeSession([FakeSession.failing(exc)])
        provider = _provider(session)
        with pytest.raises(OperationError):
            await provider.complete("hi")

    async def test_embed_error_is_translated(self) -> None:
        err = make_client_response_error(429, "slow down")
        session = FakeSession(
            [FakeSession.responding(FakeResponse(429, raise_exc=err))]
        )
        provider = _provider(session)
        with pytest.raises(RateLimitError):
            await provider.embed("some text")

    async def test_stream_error_is_translated(self) -> None:
        err = make_client_response_error(400, "bad stream")
        session = FakeSession(
            [FakeSession.responding(FakeResponse(400, raise_exc=err))]
        )
        provider = _provider(session)
        with pytest.raises(ValidationError):
            async for _ in provider.stream_complete("hi"):
                pass


class TestFunctionCallErrorSurfacing:
    """Deprecated ``function_call()`` surfaces transport errors directly.

    Previously the native-tools call was wrapped in a broad ``except`` that
    logged "native tools failed" and re-issued the whole request prompt-based —
    so a 429 / timeout triggered a *second*, wasteful full request instead of
    surfacing ``RateLimitError``. The fallback now fires only for the genuine
    "does not support tools" signal.
    """

    _FUNCTIONS = [{"name": "f", "description": "d", "parameters": {}}]

    async def test_rate_limit_surfaces_without_second_request(self) -> None:
        err = make_client_response_error(
            429, "slow down", headers={"retry-after": "4"}
        )
        session = FakeSession(
            [FakeSession.responding(FakeResponse(429, raise_exc=err))]
        )
        provider = _provider(session)
        with pytest.warns(DeprecationWarning):
            with pytest.raises(RateLimitError) as excinfo:
                await provider.function_call(
                    [LLMMessage(role="user", content="hi")], self._FUNCTIONS
                )
        assert excinfo.value.retry_after == 4.0
        assert excinfo.value.__cause__ is err
        # A throttle must NOT trigger a second, prompt-based request.
        assert len(session.calls) == 1

    async def test_tools_not_supported_still_falls_back_to_prompt(self) -> None:
        """The genuine "does not support tools" 400 still falls back."""
        native = FakeSession.responding(
            FakeResponse(400, text="this model does not support tools")
        )
        fallback = FakeSession.responding(
            FakeResponse(
                200,
                json_data={
                    "message": {"content": '{"function": "f", "arguments": {}}'},
                    "done": True,
                },
            )
        )
        session = FakeSession([native, fallback])
        provider = _provider(session)
        with pytest.warns(DeprecationWarning):
            result = await provider.function_call(
                [LLMMessage(role="user", content="hi")], self._FUNCTIONS
            )
        # Native attempt + prompt-based fallback = two requests.
        assert len(session.calls) == 2
        assert result.function_call is not None
        assert result.function_call["name"] == "f"


class TestDomainErrorPreserved:
    """The ``ToolsNotSupportedError`` domain branch wins over translation."""

    async def test_tools_not_supported_beats_validation_error(self) -> None:
        """A 400 "does not support tools" body must stay ToolsNotSupportedError.

        This 400 is raised by the provider's own domain branch *before*
        ``raise_for_status()``. The S4 translator sits after that branch and its
        gate (aiohttp errors only) does not match ``ToolsNotSupportedError``, so
        the better domain error must survive — not get flattened to a generic
        ``ValidationError``.
        """
        response = FakeResponse(
            400, text="this model does not support tools"
        )
        session = FakeSession([FakeSession.responding(response)])
        provider = _provider(session)
        with pytest.raises(ToolsNotSupportedError):
            await provider.complete("hi")
