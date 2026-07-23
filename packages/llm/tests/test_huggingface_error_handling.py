"""Tests for HuggingFace vendor-error translation (S4).

The HuggingFace Inference API is spoken over ``aiohttp`` (no vendor SDK) with a
``raise_for_status()`` call, so any API error previously surfaced as a raw
``aiohttp.ClientResponseError`` (or connection/timeout) that a consumer could
only catch by coupling to ``aiohttp``. The fix translates these into
``dataknobs_common`` exceptions, preserving the original on ``__cause__``;
non-transport errors propagate unchanged.

These tests construct **real** ``aiohttp.ClientResponseError`` objects and drive
the real ``HuggingFaceProvider`` via a minimal raising session stub. They FAIL
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
from dataknobs_llm.llm.base import LLMConfig
from dataknobs_llm.llm.providers.huggingface import HuggingFaceProvider

from _aiohttp_error_stub import (
    FakeResponse,
    FakeSession,
    make_client_response_error,
)


def _provider(session: Any, **config_kwargs: Any) -> HuggingFaceProvider:
    provider = HuggingFaceProvider(
        LLMConfig(provider="huggingface", model="gpt2", **config_kwargs)
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
            429, "slow down", headers={"retry-after": "7"}
        )
        session = FakeSession(
            [FakeSession.responding(FakeResponse(429, raise_exc=err))]
        )
        provider = _provider(session)
        with pytest.raises(RateLimitError) as excinfo:
            await provider.complete("hi")
        assert excinfo.value.retry_after == 7.0
        assert excinfo.value.__cause__ is err

    async def test_403_becomes_operation_error(self) -> None:
        err = make_client_response_error(403, "forbidden")
        session = FakeSession(
            [FakeSession.responding(FakeResponse(403, raise_exc=err))]
        )
        provider = _provider(session)
        with pytest.raises(OperationError):
            await provider.complete("hi")

    async def test_connection_error_becomes_operation_error(self) -> None:
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


class TestDomainErrorPreserved:
    """The ``ToolsNotSupportedError`` domain error is untouched by S4."""

    async def test_tools_raise_tools_not_supported(self) -> None:
        """HuggingFace rejects tools *before* any HTTP call — no translation.

        The provider raises ``ToolsNotSupportedError`` at the top of
        ``complete()`` when tools are passed, so it never reaches the aiohttp
        choke point. The session is a no-op stub to prove the error fires ahead
        of any request.
        """
        session = FakeSession([])
        provider = _provider(session)
        with pytest.raises(ToolsNotSupportedError):
            await provider.complete("hi", tools=[object()])
        assert session.calls == []  # never reached the HTTP layer
