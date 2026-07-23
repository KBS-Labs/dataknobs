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
