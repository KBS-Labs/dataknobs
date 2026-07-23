"""Tests for Amazon Bedrock vendor-error translation (S4).

The provider called ``converse`` / ``converse_stream`` / ``invoke_model`` with
no error translation, so any API error surfaced as a raw
``botocore.exceptions.ClientError`` / ``BotoCoreError`` that a consumer could
only catch by coupling to ``botocore``. The fix translates these into
``dataknobs_common`` exceptions, preserving the original on ``__cause__``;
non-botocore errors propagate unchanged. Bedrock's status lives *nested* in the
``ClientError.response`` dict, and throttling is signalled by an error *code*
(``ThrottlingException`` / ``TooManyRequestsException``) even when the HTTP
status is ambiguous — both are covered here.

The boundary stub sits at ``session.client("bedrock-runtime")`` (Bedrock has no
faithful local emulator; a boundary stub is the sanctioned last resort). The
error objects themselves are **real** ``botocore`` exceptions — no fakes for the
real dependency's error classes. These FAIL against HEAD (raw errors propagate)
and pass after the fix.
"""

from __future__ import annotations

from typing import Any, Self

import pytest
from botocore.exceptions import (
    ClientError,
    EndpointConnectionError,
)

from dataknobs_common.exceptions import (
    OperationError,
    RateLimitError,
    ValidationError,
)
from dataknobs_llm.llm.base import LLMConfig
from dataknobs_llm.llm.providers.bedrock import BedrockProvider


def _client_error(code: str, http_status: int | None, op: str = "Converse") -> ClientError:
    metadata: dict[str, Any] = {}
    if http_status is not None:
        metadata["HTTPStatusCode"] = http_status
    return ClientError(
        {"Error": {"Code": code, "Message": code}, "ResponseMetadata": metadata},
        op,
    )


class _RaisingBedrockClient:
    """Async bedrock-runtime client stub that raises a scripted error."""

    def __init__(self, error: Exception) -> None:
        self._error = error

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def converse(self, **kwargs: Any) -> dict[str, Any]:
        raise self._error

    async def converse_stream(self, **kwargs: Any) -> dict[str, Any]:
        raise self._error

    async def invoke_model(self, **kwargs: Any) -> dict[str, Any]:
        raise self._error


class _RaisingEventStream:
    """An async iterator that raises *exc* mid-stream (after create succeeds)."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def __aiter__(self) -> _RaisingEventStream:
        return self

    async def __anext__(self) -> object:
        raise self._exc


class _StreamingBedrockClient:
    """``converse_stream`` *returns* a response whose event stream raises.

    Models the streaming gap: the ``converse_stream`` create succeeds, then a
    throttle / connection drop surfaces while iterating ``response["stream"]``.
    """

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def converse_stream(self, **kwargs: Any) -> dict[str, Any]:
        return {"stream": _RaisingEventStream(self._exc)}


class _StubSession:
    """aioboto3.Session stub returning a fixed bedrock-runtime client."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def client(self, service: str, **kwargs: Any) -> Any:
        return self._client


def _provider(error: Exception, *, model: str = "anthropic.claude-3-haiku") -> BedrockProvider:
    provider = BedrockProvider(LLMConfig(provider="bedrock", model=model))
    provider._session = _StubSession(_RaisingBedrockClient(error))
    provider._is_initialized = True
    return provider


def _streaming_provider(
    error: Exception, *, model: str = "anthropic.claude-3-haiku"
) -> BedrockProvider:
    provider = BedrockProvider(LLMConfig(provider="bedrock", model=model))
    provider._session = _StubSession(_StreamingBedrockClient(error))
    provider._is_initialized = True
    return provider


class TestVendorErrorTranslation:
    """Raw botocore errors become catchable dataknobs exceptions."""

    async def test_nested_400_becomes_validation_error(self) -> None:
        err = _client_error("ValidationException", 400)
        provider = _provider(err)
        with pytest.raises(ValidationError) as excinfo:
            await provider.complete("hi")
        assert excinfo.value.__cause__ is err

    async def test_throttling_becomes_rate_limit_error(self) -> None:
        err = _client_error("ThrottlingException", 429)
        provider = _provider(err)
        with pytest.raises(RateLimitError) as excinfo:
            await provider.complete("hi")
        assert excinfo.value.__cause__ is err

    async def test_throttling_code_without_http_status_still_429(self) -> None:
        """The throttling *code* maps to 429 even when HTTP status is absent."""
        err = _client_error("TooManyRequestsException", None)
        provider = _provider(err)
        with pytest.raises(RateLimitError):
            await provider.complete("hi")

    async def test_nested_403_becomes_operation_error(self) -> None:
        err = _client_error("AccessDeniedException", 403)
        provider = _provider(err)
        with pytest.raises(OperationError):
            await provider.complete("hi")

    async def test_botocore_error_without_status_becomes_operation_error(
        self,
    ) -> None:
        err = EndpointConnectionError(
            endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com"
        )
        provider = _provider(err)
        with pytest.raises(OperationError) as excinfo:
            await provider.complete("hi")
        assert excinfo.value.__cause__ is err

    async def test_stream_error_is_translated(self) -> None:
        err = _client_error("ThrottlingException", 429)
        provider = _provider(err)
        with pytest.raises(RateLimitError):
            async for _ in provider.stream_complete("hi"):
                pass

    async def test_mid_stream_error_is_translated(self) -> None:
        """A throttle raised *during* stream iteration (not at create) is
        translated. Before the streaming wrapper this raw botocore
        ``ClientError`` leaked straight to the consumer.
        """
        err = _client_error("ThrottlingException", 429)
        provider = _streaming_provider(err)
        with pytest.raises(RateLimitError) as excinfo:
            async for _ in provider.stream_complete("hi"):
                pass
        assert excinfo.value.__cause__ is err

    async def test_embed_error_is_translated(self) -> None:
        err = _client_error("ThrottlingException", 429)
        provider = _provider(err, model="amazon.titan-embed-text-v2:0")
        with pytest.raises(RateLimitError):
            await provider.embed("some text")

    async def test_non_botocore_error_propagates_unchanged(self) -> None:
        """A bug in our own code is never masked as an API error."""
        provider = _provider(ValueError("internal bug"))
        with pytest.raises(ValueError):
            await provider.complete("hi")
