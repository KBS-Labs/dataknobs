"""Tests for the service-aware LocalStack skip guard.

Bug: ``requires_localstack`` / ``is_localstack_available`` probe only
edge-port TCP reachability, so a LocalStack container started with a
restricted ``SERVICES`` list (e.g. ``s3`` without ``sqs``) is "available",
the SQS suite *runs*, and every case fails with
``Service 'sqs' is not enabled``. A partially-configured LocalStack should
make a service-specific suite *skip*, not *fail*.

Fix: an optional ``service=`` on ``is_localstack_available`` (and the
``requires_localstack_service(service)`` marker) that additionally verifies
the service reports ``running``/``available`` at ``/_localstack/health``.

These tests drive the real guard code with the health endpoint and the TCP
probe stubbed (a sanctioned stand-in — no live LocalStack, and the fail-soft
contract must hold whether or not one is running). They fail against HEAD,
which has no ``service`` awareness.
"""

from __future__ import annotations

import json
from typing import Any
from urllib.error import URLError

import pytest

from dataknobs_common.testing import is_localstack_available
from dataknobs_common.testing._core import _localstack_service_enabled

ENDPOINT = "http://localhost:4566"


# ---------------------------------------------------------------------------
# Stubs for the health endpoint and the TCP probe
# ---------------------------------------------------------------------------


class _FakeResponse:
    """Context-manager stand-in for a ``urlopen`` response."""

    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def read(self) -> bytes:
        return self._body


def _health_payload(**services: str) -> bytes:
    return json.dumps({"services": services}).encode("utf-8")


def _patch_urlopen(monkeypatch: pytest.MonkeyPatch, body: bytes) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda url, timeout=None: _FakeResponse(body),
    )


def _patch_urlopen_raises(
    monkeypatch: pytest.MonkeyPatch, exc: Exception
) -> None:
    def _raise(url: str, timeout: float | None = None) -> Any:
        raise exc

    monkeypatch.setattr("urllib.request.urlopen", _raise)


class _FakeSock:
    """Stand-in socket whose ``connect_ex`` returns a scripted result."""

    def __init__(self, connect_result: int) -> None:
        self._connect_result = connect_result

    def settimeout(self, timeout: float) -> None:
        return None

    def connect_ex(self, address: tuple[str, int]) -> int:
        return self._connect_result

    def close(self) -> None:
        return None


def _patch_socket(monkeypatch: pytest.MonkeyPatch, connect_result: int) -> None:
    monkeypatch.setattr(
        "socket.socket",
        lambda *args, **kwargs: _FakeSock(connect_result),
    )


# ---------------------------------------------------------------------------
# _localstack_service_enabled — health-endpoint parsing
# ---------------------------------------------------------------------------


class TestServiceEnabledProbe:
    """The health-endpoint probe reports a service ready only when it is."""

    def test_running_is_ready(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_urlopen(monkeypatch, _health_payload(sqs="running", s3="running"))
        assert _localstack_service_enabled(ENDPOINT, "sqs") is True

    def test_available_is_ready(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_urlopen(monkeypatch, _health_payload(sqs="available"))
        assert _localstack_service_enabled(ENDPOINT, "sqs") is True

    def test_disabled_is_not_ready(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_urlopen(monkeypatch, _health_payload(s3="running", sqs="disabled"))
        assert _localstack_service_enabled(ENDPOINT, "sqs") is False

    def test_missing_service_is_not_ready(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_urlopen(monkeypatch, _health_payload(s3="running"))
        assert _localstack_service_enabled(ENDPOINT, "sqs") is False

    def test_unreachable_health_is_not_ready(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_urlopen_raises(monkeypatch, URLError("no route to host"))
        assert _localstack_service_enabled(ENDPOINT, "sqs") is False

    def test_malformed_body_is_not_ready(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_urlopen(monkeypatch, b"not json")
        assert _localstack_service_enabled(ENDPOINT, "sqs") is False

    def test_non_dict_payload_is_not_ready(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_urlopen(monkeypatch, b"[1, 2, 3]")
        assert _localstack_service_enabled(ENDPOINT, "sqs") is False


# ---------------------------------------------------------------------------
# is_localstack_available(service=...) — TCP probe + service gate
# ---------------------------------------------------------------------------


class TestIsAvailableServiceGate:
    """The service gate composes the TCP probe with the health probe."""

    def test_closed_port_skips_without_touching_health(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Health would say ready, but the port is closed -> not available.
        _patch_socket(monkeypatch, connect_result=1)
        _patch_urlopen(monkeypatch, _health_payload(sqs="running"))
        assert is_localstack_available(service="sqs") is False

    def test_open_port_service_enabled_is_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_socket(monkeypatch, connect_result=0)
        _patch_urlopen(monkeypatch, _health_payload(sqs="running"))
        assert is_localstack_available(service="sqs") is True

    def test_open_port_service_disabled_is_not_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_socket(monkeypatch, connect_result=0)
        _patch_urlopen(monkeypatch, _health_payload(s3="running", sqs="disabled"))
        assert is_localstack_available(service="sqs") is False

    def test_no_service_arg_is_backcompat_tcp_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Without a service, the health endpoint is never consulted.
        _patch_socket(monkeypatch, connect_result=0)

        def _fail(url: str, timeout: float | None = None) -> Any:
            raise AssertionError("health endpoint must not be queried")

        monkeypatch.setattr("urllib.request.urlopen", _fail)
        assert is_localstack_available() is True
