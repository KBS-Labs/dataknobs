"""Tests for the distinct context-window-overflow exception.

A context-window overflow is a 400 / invalid-request error, so before this
change it surfaced as the generic ``ValidationError`` — indistinguishable from
a rejected sampling parameter or a malformed request. Consumers that want to
*react* to overflow specifically (compact history and retry, switch to a
larger-context model, surface a distinct message) had nothing narrower to catch.

The fix specializes the shared status dispatch so an overflow 400 raises
``ContextLengthExceededError`` — a ``ValidationError`` subclass, so every
existing ``except ValidationError`` keeps matching (purely additive). Detection
is a machine ``code`` (OpenAI) or a conservative message marker. Every provider
folds the vendor error *body* into the dispatched message: the SDK providers via
``str(exc)``, and the two aiohttp providers (Ollama, HuggingFace) via
``raise_for_status_with_body`` — aiohttp's own ``raise_for_status`` carries only
the reason phrase (``"Bad Request"``) and drops the body where the overflow
wording lives, so those two route through the shared helper to preserve it.

These build **real** vendor SDK error objects (openai / botocore are dev deps
for exactly this reason — no fakes for the real dependency's error classes), and
the aiohttp providers are driven through a **real** ``aiohttp.ClientResponseError``
via the sanctioned raising session stub. Each FAILS against the unfixed base:
the raised type is the generic ``ValidationError``, not a
``ContextLengthExceededError``, so ``pytest.raises(ContextLengthExceededError)``
fails until the specialization (and, for Ollama/HuggingFace, the body-preserving
helper) lands.

Discriminating assertions:
- ``test_openai_context_length_via_code_only`` — an **opaque** message with no
  marker, overflow identified by the machine ``code`` alone, isolating the
  ``code`` channel end-to-end (the marker-bearing openai test would pass even if
  the code were ignored).
- ``test_ollama_non_overflow_400_stays_validation_error`` /
  ``test_huggingface`` overflow cases — prove the aiohttp body-folding both
  fires on overflow wording *and* stays narrow on an unrelated 400.
"""

from __future__ import annotations

import types
from typing import Any, Self

import httpx
import openai
import pytest
from botocore.exceptions import ClientError

from dataknobs_common.exceptions import OperationError, ValidationError
from dataknobs_llm.exceptions import ContextLengthExceededError
from dataknobs_llm.llm.base import LLMConfig, LLMProvider, ModelCapability
from dataknobs_llm.llm.providers.bedrock import BedrockProvider
from dataknobs_llm.llm.providers.huggingface import HuggingFaceProvider
from dataknobs_llm.llm.providers.ollama import OllamaProvider
from dataknobs_llm.llm.providers.openai import OpenAIProvider

from _aiohttp_error_stub import (
    FakeResponse,
    FakeSession,
    make_client_response_error,
)


# ---------------------------------------------------------------------------
# openai: real BadRequestError + a raising client stub
# ---------------------------------------------------------------------------


def _request() -> httpx.Request:
    return httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


def _openai_bad_request(message: str, code: str | None) -> openai.BadRequestError:
    body: dict[str, Any] = {"message": message, "type": "invalid_request_error"}
    if code is not None:
        body["code"] = code
    resp = httpx.Response(
        400,
        request=_request(),
        json={"error": body},
    )
    return openai.BadRequestError(message, response=resp, body=body)


class _RaisingCall:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def create(self, **kwargs: Any) -> object:
        raise self._exc


class _RaisingOpenAIClient:
    def __init__(self, exc: Exception) -> None:
        self.chat = types.SimpleNamespace(completions=_RaisingCall(exc))
        self.embeddings = _RaisingCall(exc)


def _openai_provider(exc: Exception) -> OpenAIProvider:
    provider = OpenAIProvider(LLMConfig(provider="openai", model="gpt-4"))
    provider._client = _RaisingOpenAIClient(exc)
    provider._is_initialized = True
    return provider


# ---------------------------------------------------------------------------
# bedrock: real botocore ClientError + a raising session stub
# ---------------------------------------------------------------------------


def _client_error(code: str, message: str, http_status: int) -> ClientError:
    return ClientError(
        {
            "Error": {"Code": code, "Message": message},
            "ResponseMetadata": {"HTTPStatusCode": http_status},
        },
        "Converse",
    )


class _RaisingBedrockClient:
    def __init__(self, error: Exception) -> None:
        self._error = error

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def converse(self, **kwargs: Any) -> dict[str, Any]:
        raise self._error


class _BedrockSession:
    def __init__(self, error: Exception) -> None:
        self._error = error

    def client(self, *args: Any, **kwargs: Any) -> _RaisingBedrockClient:
        return _RaisingBedrockClient(self._error)


def _bedrock_provider(exc: Exception) -> BedrockProvider:
    provider = BedrockProvider(LLMConfig(provider="bedrock", model="anthropic.claude-3-sonnet"))
    provider._session = _BedrockSession(exc)
    provider._is_initialized = True
    return provider


# ---------------------------------------------------------------------------
# ollama / huggingface: real aiohttp ClientResponseError + a raising session stub
#
# aiohttp's raise_for_status() carries only the reason phrase; the overflow
# wording lives in the response *body* (FakeResponse.text). The provider routes
# through raise_for_status_with_body, which folds that body into the error's
# message so the shared markers fire.
# ---------------------------------------------------------------------------


def _ollama_provider(session: Any) -> OllamaProvider:
    provider = OllamaProvider(LLMConfig(provider="ollama", model="llama3.2"))
    provider._session = session
    provider._is_initialized = True
    return provider


def _hf_provider(session: Any) -> HuggingFaceProvider:
    provider = HuggingFaceProvider(LLMConfig(provider="huggingface", model="gpt2", api_key="test"))
    provider._session = session
    provider._is_initialized = True
    return provider


def _error_response(status: int, reason: str, body: str) -> FakeResponse:
    """A *status* response whose vendor wording lives only in the *body*."""
    return FakeResponse(status, text=body, raise_exc=make_client_response_error(status, reason))


#: Ollama's measured overflow body, on ``/api/embeddings`` (as a 500) and on
#: ``/api/embed`` with ``truncate: false`` (as a 400). Ollama 0.33.2.
_OLLAMA_OVERFLOW_BODY = '{"error":"the input length exceeds the context length"}'

#: Ollama's measured answer when a completion model is asked to embed: also a
#: 500 on ``/api/embeddings``, and not an overflow.
_OLLAMA_NO_EMBEDDINGS_BODY = (
    '{"error":"This server does not support embeddings. Start it with `--embeddings`"}'
)


def _bad_request_response(body: str) -> FakeResponse:
    """A 400 whose vendor wording lives in the *body*, not the reason phrase.

    ``raise_exc`` is the real ``aiohttp.ClientResponseError`` that
    ``raise_for_status()`` raises — its bare reason phrase (``"Bad Request"``)
    carries no marker, so overflow is detected only once the body is folded in
    (and a non-overflow body stays a plain ``ValidationError``).
    """
    return FakeResponse(400, text=body, raise_exc=make_client_response_error(400, "Bad Request"))


# ---------------------------------------------------------------------------
# base predicate: a minimal concrete provider
# ---------------------------------------------------------------------------


class _BaseProvider(LLMProvider):
    def initialize(self) -> None:  # pragma: no cover - stub
        pass

    def close(self) -> None:  # pragma: no cover - stub
        pass

    async def validate_model(self) -> bool:  # pragma: no cover - stub
        return True

    def _detect_capabilities(self) -> list[ModelCapability]:  # pragma: no cover
        return []


def _base_provider() -> _BaseProvider:
    return _BaseProvider(LLMConfig(provider="test", model="test-model"))


# ---------------------------------------------------------------------------
# Provider-level: real overflow errors map to ContextLengthExceededError
# ---------------------------------------------------------------------------


class TestContextLengthTranslation:
    """A context-window overflow 400 becomes ``ContextLengthExceededError``."""

    async def test_openai_context_length_400_via_code(self) -> None:
        """OpenAI carries a machine ``code`` on the body."""
        exc = _openai_bad_request(
            "This model's maximum context length is 8192 tokens. "
            "However, you requested 9000 tokens.",
            code="context_length_exceeded",
        )
        provider = _openai_provider(exc)
        with pytest.raises(ContextLengthExceededError) as excinfo:
            await provider.complete("hi")
        # Backward compatibility: still a ValidationError.
        assert isinstance(excinfo.value, ValidationError)
        # Original SDK error preserved on __cause__.
        assert isinstance(excinfo.value.__cause__, openai.BadRequestError)

    async def test_openai_context_length_via_code_only(self) -> None:
        """OpenAI overflow with an **opaque** message — only the ``code`` fires.

        Isolates the machine-``code`` channel end-to-end: the message carries no
        marker, so this passes only if ``getattr(exc, "code", None)`` actually
        reaches the predicate. The marker-bearing openai test above would pass
        even if the code were dropped, so it does not cover this path.
        """
        exc = _openai_bad_request(
            "The request could not be processed.",
            code="context_length_exceeded",
        )
        provider = _openai_provider(exc)
        with pytest.raises(ContextLengthExceededError) as excinfo:
            await provider.complete("hi")
        assert isinstance(excinfo.value.__cause__, openai.BadRequestError)

    async def test_anthropic_context_length_400_via_marker(self) -> None:
        """Anthropic carries no code — the message marker fires."""
        provider = _base_provider()
        err = provider._dataknobs_error_for_status(
            400,
            "Anthropic API error: prompt is too long: 215334 tokens > 200000 maximum",
        )
        assert isinstance(err, ContextLengthExceededError)

    async def test_bedrock_context_length_400_via_marker(self) -> None:
        """Bedrock folds the message into ``str(exc)`` — the marker fires."""
        exc = _client_error(
            "ValidationException",
            "Input is too long for requested model.",
            http_status=400,
        )
        provider = _bedrock_provider(exc)
        with pytest.raises(ContextLengthExceededError) as excinfo:
            await provider.complete("hi")
        assert isinstance(excinfo.value, ValidationError)
        assert isinstance(excinfo.value.__cause__, ClientError)

    async def test_non_context_400_stays_validation_error(self) -> None:
        """A non-overflow 400 stays the generic ``ValidationError``.

        Pins that the specialization is narrow — a rejected sampling parameter
        or a plain malformed request must NOT be classified as overflow.
        """
        exc = _openai_bad_request(
            "Invalid value for 'temperature': must be <= 2.0",
            code="invalid_value",
        )
        provider = _openai_provider(exc)
        with pytest.raises(ValidationError) as excinfo:
            await provider.complete("hi")
        assert not isinstance(excinfo.value, ContextLengthExceededError)


# ---------------------------------------------------------------------------
# aiohttp providers: overflow lives in the body, folded by the shared helper
# ---------------------------------------------------------------------------


class TestAiohttpProviderContextLength:
    """Ollama / HuggingFace overflow is detected from the response **body**.

    aiohttp's ``raise_for_status()`` keeps only the reason phrase, so these FAIL
    against the unfixed base (the body — and its marker — is discarded, leaving
    a plain ``ValidationError``) and pass once ``raise_for_status_with_body``
    preserves it.
    """

    async def test_ollama_context_length_via_body_marker(self) -> None:
        """Ollama's embedding endpoint reports an overflow as a **500**.

        The body and the status are the ones Ollama sends (measured, 0.33.2). An
        earlier version of this test pinned a 400 with wording Ollama never
        uses, and passed while the real path raised ``OperationError``;
        ``test_ollama_context_overflow.py`` runs the same case against a live
        server.
        """
        response = _error_response(500, "Internal Server Error", _OLLAMA_OVERFLOW_BODY)
        provider = _ollama_provider(FakeSession([FakeSession.responding(response)]))
        with pytest.raises(ContextLengthExceededError) as excinfo:
            await provider.embed("an over-long text")
        # Backward compatibility + original error preserved on __cause__.
        assert isinstance(excinfo.value, ValidationError)
        assert excinfo.value.__cause__ is response._raise_exc

    async def test_ollama_context_length_as_a_400(self) -> None:
        """``/api/embed`` with ``truncate: false`` sends the same words as a 400."""
        response = _bad_request_response(_OLLAMA_OVERFLOW_BODY)
        provider = _ollama_provider(FakeSession([FakeSession.responding(response)]))
        with pytest.raises(ContextLengthExceededError):
            await provider.embed("an over-long text")

    async def test_ollama_500_without_a_marker_stays_operation_error(self) -> None:
        """A 500 alone is not an overflow: the marker decides.

        Ollama answers a completion model asked to embed with a 500 too, and
        that is the server's condition, not the caller's input.
        """
        response = _error_response(500, "Internal Server Error", _OLLAMA_NO_EMBEDDINGS_BODY)
        provider = _ollama_provider(FakeSession([FakeSession.responding(response)]))
        with pytest.raises(OperationError) as excinfo:
            await provider.embed("text")
        assert not isinstance(excinfo.value, ValidationError)

    async def test_the_500_is_ollamas_alone(self) -> None:
        """HuggingFace declares no 500, so the same response stays an OperationError.

        Pins that the widened status set is a declaration of one provider, not a
        change to the shared default.
        """
        response = _error_response(500, "Internal Server Error", _OLLAMA_OVERFLOW_BODY)
        provider = _hf_provider(FakeSession([FakeSession.responding(response)]))
        with pytest.raises(OperationError) as excinfo:
            await provider.complete("hi")
        assert not isinstance(excinfo.value, ValidationError)

    async def test_ollama_non_overflow_400_stays_validation_error(self) -> None:
        """A 400 whose body has no overflow marker stays a plain ValidationError.

        Pins that folding the body did not widen the net — an unrelated Ollama
        400 (a rejected option) must not be misclassified as overflow.
        """
        response = _bad_request_response('{"error":"invalid options: temperature must be <= 2"}')
        provider = _ollama_provider(FakeSession([FakeSession.responding(response)]))
        with pytest.raises(ValidationError) as excinfo:
            await provider.complete("hi")
        assert not isinstance(excinfo.value, ContextLengthExceededError)

    async def test_huggingface_context_length_via_body_marker(self) -> None:
        response = _bad_request_response('{"error":"Input validation error: prompt is too long"}')
        provider = _hf_provider(FakeSession([FakeSession.responding(response)]))
        with pytest.raises(ContextLengthExceededError) as excinfo:
            await provider.complete("hi")
        assert isinstance(excinfo.value, ValidationError)
        assert excinfo.value.__cause__ is response._raise_exc


# ---------------------------------------------------------------------------
# Base predicate: pinned once, independent of any SDK
# ---------------------------------------------------------------------------


class TestExports:
    """The consumer-facing type is reachable and backward-compatible."""

    def test_top_level_and_module_exports_are_the_same_type(self) -> None:
        import dataknobs_llm

        assert dataknobs_llm.ContextLengthExceededError is (ContextLengthExceededError)

    def test_is_a_validation_error_subclass(self) -> None:
        assert issubclass(ContextLengthExceededError, ValidationError)


class TestIsContextLengthError:
    """``_is_context_length_error`` — status-gated code/marker detection."""

    @pytest.mark.parametrize(
        "message",
        [
            "context_length_exceeded",
            "This model's maximum context length is 8192 tokens",
            "prompt is too long: 215334 tokens > 200000 maximum",
            "Input is too long for requested model",
            "too many input tokens",
            "the context window is exceeded",
            "the input length exceeds the context length",
        ],
    )
    def test_fires_on_400_with_marker(self, message: str) -> None:
        assert LLMProvider._is_context_length_error(400, message) is True

    def test_fires_on_400_with_openai_code(self) -> None:
        assert (
            LLMProvider._is_context_length_error(
                400, "some opaque message", code="context_length_exceeded"
            )
            is True
        )

    def test_not_fired_on_400_unrelated_message(self) -> None:
        assert LLMProvider._is_context_length_error(400, "invalid temperature") is False

    @pytest.mark.parametrize("status", [429, 401, 500, None])
    def test_status_gate_first(self, status: int | None) -> None:
        """Only a 400 qualifies — a marker in a 429/401/500 message never fires."""
        assert LLMProvider._is_context_length_error(status, "prompt is too long") is False

    def test_a_declared_status_admits_the_marker(self) -> None:
        """A provider that declares 500 has its 500s read for a marker."""
        overflow = "the input length exceeds the context length"
        assert (
            LLMProvider._is_context_length_error(500, overflow, statuses=frozenset({400, 500}))
            is True
        )
        assert LLMProvider._is_context_length_error(500, overflow) is False

    def test_a_declared_status_still_needs_the_marker(self) -> None:
        assert (
            LLMProvider._is_context_length_error(
                500, "internal error", statuses=frozenset({400, 500})
            )
            is False
        )


class TestStatusDispatchContextLength:
    """``_dataknobs_error_for_status`` routes overflow before the generic 400."""

    def test_overflow_marker_becomes_context_length_error(self) -> None:
        provider = _base_provider()
        err = provider._dataknobs_error_for_status(400, "prompt is too long")
        assert isinstance(err, ContextLengthExceededError)

    def test_overflow_code_becomes_context_length_error(self) -> None:
        provider = _base_provider()
        err = provider._dataknobs_error_for_status(400, "opaque", code="context_length_exceeded")
        assert isinstance(err, ContextLengthExceededError)

    def test_plain_400_stays_validation_error(self) -> None:
        provider = _base_provider()
        err = provider._dataknobs_error_for_status(400, "bad request")
        assert type(err) is ValidationError

    def test_a_base_provider_500_with_a_marker_stays_operation_error(self) -> None:
        """The shared default admits only a 400: a 5xx is the server's failure."""
        provider = _base_provider()
        err = provider._dataknobs_error_for_status(
            500, "the input length exceeds the context length"
        )
        assert type(err) is OperationError

    def test_429_with_marker_stays_rate_limit_error(self) -> None:
        """The 429 branch wins even if the message happens to carry a marker."""
        from dataknobs_common.exceptions import RateLimitError

        provider = _base_provider()
        err = provider._dataknobs_error_for_status(429, "prompt is too long", retry_after=1.0)
        assert type(err) is RateLimitError
