"""Unit tests for the shared vendor-error dispatch on ``LLMProvider``.

The status→dataknobs-type policy lives once on the base class
(``_dataknobs_error_for_status``), and every provider's ``_translate_api_error``
does only the SDK-specific gate + extraction before deferring here. These tests
pin that shared policy — and the ``retry-after`` header parser
(``_retry_after_from_headers``) — independently of any provider, so a change to
the policy is caught in one focused place.
"""

from __future__ import annotations

from typing import List

import pytest

from dataknobs_common.exceptions import (
    OperationError,
    RateLimitError,
    ValidationError,
)
from dataknobs_llm.llm.base import LLMConfig, LLMProvider, ModelCapability


class _BaseProvider(LLMProvider):
    """Minimal concrete ``LLMProvider`` to exercise the base helpers.

    The hoisted helpers do not touch the client or any provider state, so the
    abstract methods are stubbed — construction alone is enough to call them.
    """

    def initialize(self) -> None:  # pragma: no cover - stub
        pass

    def close(self) -> None:  # pragma: no cover - stub
        pass

    async def validate_model(self) -> bool:  # pragma: no cover - stub
        return True

    def _detect_capabilities(self) -> List[ModelCapability]:  # pragma: no cover
        return []


def _provider() -> _BaseProvider:
    return _BaseProvider(LLMConfig(provider="test", model="test-model"))


class TestStatusDispatch:
    """``_dataknobs_error_for_status`` maps a status to a dataknobs type."""

    def test_429_is_rate_limit_error_with_retry_after(self) -> None:
        provider = _provider()
        err = provider._dataknobs_error_for_status(
            429, "too fast", retry_after=2.5
        )
        assert isinstance(err, RateLimitError)
        assert err.retry_after == 2.5

    def test_429_without_retry_after(self) -> None:
        provider = _provider()
        err = provider._dataknobs_error_for_status(429, "too fast")
        assert isinstance(err, RateLimitError)
        assert err.retry_after is None

    def test_400_is_validation_error(self) -> None:
        provider = _provider()
        err = provider._dataknobs_error_for_status(400, "bad request")
        assert isinstance(err, ValidationError)

    def test_401_is_operation_error(self) -> None:
        provider = _provider()
        err = provider._dataknobs_error_for_status(401, "unauthorized")
        assert isinstance(err, OperationError)
        # A 401 must NOT be a RateLimitError/ValidationError (both subclass
        # OperationError, so assert the concrete type).
        assert type(err) is OperationError

    def test_403_is_operation_error(self) -> None:
        provider = _provider()
        err = provider._dataknobs_error_for_status(403, "forbidden")
        assert type(err) is OperationError

    def test_500_is_operation_error(self) -> None:
        provider = _provider()
        err = provider._dataknobs_error_for_status(500, "server error")
        assert type(err) is OperationError

    def test_none_status_is_operation_error(self) -> None:
        """A connection error / timeout carries no status → OperationError."""
        provider = _provider()
        err = provider._dataknobs_error_for_status(None, "connection reset")
        assert type(err) is OperationError

    def test_message_is_preserved(self) -> None:
        provider = _provider()
        err = provider._dataknobs_error_for_status(400, "the exact message")
        assert "the exact message" in str(err)


class TestRetryAfterFromHeaders:
    """``_retry_after_from_headers`` parses a ``retry-after`` header."""

    def test_present_numeric(self) -> None:
        assert LLMProvider._retry_after_from_headers({"retry-after": "3"}) == 3.0

    def test_present_float(self) -> None:
        headers = {"retry-after": "1.5"}
        assert LLMProvider._retry_after_from_headers(headers) == 1.5

    def test_absent_key(self) -> None:
        assert LLMProvider._retry_after_from_headers({}) is None

    def test_none_headers(self) -> None:
        assert LLMProvider._retry_after_from_headers(None) is None

    def test_unparseable_value(self) -> None:
        headers = {"retry-after": "not-a-number"}
        assert LLMProvider._retry_after_from_headers(headers) is None

    def test_empty_value(self) -> None:
        assert LLMProvider._retry_after_from_headers({"retry-after": ""}) is None

    def test_mapping_without_get(self) -> None:
        """A header object with no ``.get`` yields ``None`` (not a crash)."""
        assert LLMProvider._retry_after_from_headers(object()) is None


class TestRetryAfterHttpDate:
    """``_retry_after_from_headers`` also parses the RFC 7231 HTTP-date form.

    Previously only the numeric-seconds form was parsed; an HTTP-date value
    (equally valid per RFC 7231) silently yielded ``None``. These fail against
    the numeric-only parser and pass once the date form is handled.
    """

    def test_future_http_date_returns_positive_seconds(self) -> None:
        headers = {"retry-after": "Wed, 21 Oct 2099 07:28:00 GMT"}
        result = LLMProvider._retry_after_from_headers(headers)
        assert result is not None
        assert result > 0

    def test_past_http_date_floored_to_zero(self) -> None:
        headers = {"retry-after": "Wed, 21 Oct 1999 07:28:00 GMT"}
        assert LLMProvider._retry_after_from_headers(headers) == 0.0

    def test_garbage_that_is_neither_seconds_nor_date_is_none(self) -> None:
        headers = {"retry-after": "not-a-number-or-date"}
        assert LLMProvider._retry_after_from_headers(headers) is None


# ---------------------------------------------------------------------------
# Shared choke-point helpers — pinned once at the base, independent of any SDK
# ---------------------------------------------------------------------------


class _TranslatingProvider(_BaseProvider):
    """A base provider whose extractor translates a sentinel vendor error.

    Stands in for a real provider so the *shared* ``_raise_translated`` /
    ``_iter_translated`` helpers can be exercised without any SDK: a
    ``_VendorError`` translates to a 429 ``RateLimitError`` exactly as a
    provider's ``_translate_api_error`` would; anything else returns ``None``
    (the passthrough contract).
    """

    class _VendorError(Exception):
        pass

    def _translate_api_error(self, exc: Exception) -> Exception | None:
        if isinstance(exc, self._VendorError):
            return self._dataknobs_error_for_status(
                429, f"translated: {exc}", retry_after=1.0
            )
        return None


def _translating_provider() -> _TranslatingProvider:
    return _TranslatingProvider(LLMConfig(provider="test", model="test-model"))


async def _yield_then_raise(items: list, exc: Exception):
    for item in items:
        yield item
    raise exc


async def _yield_all(items: list):
    for item in items:
        yield item


class TestTranslateApiErrorDefault:
    """The base extractor default performs no translation (passthrough)."""

    def test_base_default_returns_none(self) -> None:
        assert _provider()._translate_api_error(ValueError("x")) is None


class TestRaiseTranslated:
    """``_raise_translated`` translates a vendor error, else re-raises as-is."""

    def test_translates_vendor_error_preserving_cause(self) -> None:
        provider = _translating_provider()
        original = provider._VendorError("throttled")
        with pytest.raises(RateLimitError) as excinfo:
            try:
                raise original
            except Exception as exc:
                provider._raise_translated(exc)
        assert excinfo.value.retry_after == 1.0
        assert excinfo.value.__cause__ is original

    def test_passes_through_non_vendor_error_unchanged(self) -> None:
        provider = _translating_provider()
        original = ValueError("our own bug")
        with pytest.raises(ValueError) as excinfo:
            try:
                raise original
            except Exception as exc:
                provider._raise_translated(exc)
        assert excinfo.value is original


class TestIterTranslated:
    """``_iter_translated`` closes the streaming half of the choke point.

    A vendor error raised *during* iteration — not just at stream creation —
    is translated, so a consumer never sees a raw vendor error from the
    streaming path.
    """

    async def test_yields_all_when_no_error(self) -> None:
        provider = _translating_provider()
        got = [x async for x in provider._iter_translated(_yield_all([1, 2, 3]))]
        assert got == [1, 2, 3]

    async def test_translates_mid_iteration_vendor_error(self) -> None:
        provider = _translating_provider()
        original = provider._VendorError("mid-stream throttle")
        got: list[int] = []
        with pytest.raises(RateLimitError) as excinfo:
            async for x in provider._iter_translated(
                _yield_then_raise([1, 2], original)
            ):
                got.append(x)
        # Chunks before the error still reached the consumer.
        assert got == [1, 2]
        assert excinfo.value.retry_after == 1.0
        assert excinfo.value.__cause__ is original

    async def test_passes_through_non_vendor_mid_iteration_error(self) -> None:
        provider = _translating_provider()
        original = ValueError("our own bug")
        with pytest.raises(ValueError) as excinfo:
            async for _ in provider._iter_translated(
                _yield_then_raise([], original)
            ):
                pass
        assert excinfo.value is original

    async def test_consumer_body_error_is_not_mistranslated(self) -> None:
        """An error from the *consumer's* loop body is never translated — even a
        vendor-shaped one — because it does not arise inside the wrapper.
        """
        provider = _translating_provider()
        with pytest.raises(provider._VendorError):
            async for _ in provider._iter_translated(_yield_all([1, 2, 3])):
                raise provider._VendorError("raised by consumer, not the stream")
