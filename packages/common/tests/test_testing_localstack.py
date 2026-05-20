"""Tests for LocalStack endpoint resolution and availability probe.

Deterministic unit tests for ``get_localstack_endpoint`` and the
shared-resolver delegation contract on ``is_localstack_available``.
No LocalStack service required — ``/.dockerenv`` and env-var lookups
are fenced via ``monkeypatch`` so the suite runs anywhere.

Test idiom mirrors ``test_testing_postgres.py`` and
``test_testing_elasticsearch.py``: a ``_clear_*_env`` helper and a
``_fence_dockerenv`` helper that patches ``os.path.exists`` on the
exact module under test.
"""

from __future__ import annotations

import os
from typing import Any

from dataknobs_common.testing import (
    get_localstack_endpoint,
    is_localstack_available,
)


def _clear_localstack_env(monkeypatch: Any) -> None:
    for name in (
        "DOCKER_CONTAINER",
        "LOCALSTACK_ENDPOINT",
        "LOCALSTACK_HOST",
        "LOCALSTACK_PORT",
        "AWS_ENDPOINT_URL",
    ):
        monkeypatch.delenv(name, raising=False)


def _fence_dockerenv(monkeypatch: Any, present: bool) -> None:
    real_exists = os.path.exists
    monkeypatch.setattr(
        "dataknobs_common.testing._core.os.path.exists",
        lambda p: (present if p == "/.dockerenv" else real_exists(p)),
    )


# -- get_localstack_endpoint -----------------------------------------------


def test_default_endpoint_on_host(monkeypatch: Any) -> None:
    """No env vars, not in Docker → ``http://localhost:4566``."""
    _clear_localstack_env(monkeypatch)
    _fence_dockerenv(monkeypatch, present=False)
    assert get_localstack_endpoint() == "http://localhost:4566"


def test_default_endpoint_in_docker_dockerenv_file(monkeypatch: Any) -> None:
    """``/.dockerenv`` present → ``http://localstack:4566``."""
    _clear_localstack_env(monkeypatch)
    _fence_dockerenv(monkeypatch, present=True)
    assert get_localstack_endpoint() == "http://localstack:4566"


def test_default_endpoint_in_docker_env_var(monkeypatch: Any) -> None:
    """``DOCKER_CONTAINER=1`` (no ``/.dockerenv``) → ``http://localstack:4566``."""
    _clear_localstack_env(monkeypatch)
    _fence_dockerenv(monkeypatch, present=False)
    monkeypatch.setenv("DOCKER_CONTAINER", "1")
    assert get_localstack_endpoint() == "http://localstack:4566"


def test_localstack_endpoint_env_wins(monkeypatch: Any) -> None:
    """``LOCALSTACK_ENDPOINT`` wins, even inside Docker."""
    _clear_localstack_env(monkeypatch)
    _fence_dockerenv(monkeypatch, present=True)
    monkeypatch.setenv("LOCALSTACK_ENDPOINT", "http://other:5000")
    assert get_localstack_endpoint() == "http://other:5000"


def test_aws_endpoint_url_fallback(monkeypatch: Any) -> None:
    """``LOCALSTACK_ENDPOINT`` unset, ``AWS_ENDPOINT_URL`` honored."""
    _clear_localstack_env(monkeypatch)
    _fence_dockerenv(monkeypatch, present=False)
    monkeypatch.setenv("AWS_ENDPOINT_URL", "http://aws:5001")
    assert get_localstack_endpoint() == "http://aws:5001"


def test_localstack_endpoint_takes_priority_over_aws_endpoint_url(
    monkeypatch: Any,
) -> None:
    """Both set → ``LOCALSTACK_ENDPOINT`` wins (precedence guard)."""
    _clear_localstack_env(monkeypatch)
    _fence_dockerenv(monkeypatch, present=False)
    monkeypatch.setenv("LOCALSTACK_ENDPOINT", "http://primary:5000")
    monkeypatch.setenv("AWS_ENDPOINT_URL", "http://fallback:5001")
    assert get_localstack_endpoint() == "http://primary:5000"


def test_localstack_host_port_env_vars(monkeypatch: Any) -> None:
    """Lower-priority arm: ``LOCALSTACK_HOST``/``LOCALSTACK_PORT``."""
    _clear_localstack_env(monkeypatch)
    _fence_dockerenv(monkeypatch, present=False)
    monkeypatch.setenv("LOCALSTACK_HOST", "h")
    monkeypatch.setenv("LOCALSTACK_PORT", "9999")
    assert get_localstack_endpoint() == "http://h:9999"


def test_explicit_host_port_override(monkeypatch: Any) -> None:
    """Explicit args ignore every env var."""
    _clear_localstack_env(monkeypatch)
    _fence_dockerenv(monkeypatch, present=True)
    monkeypatch.setenv("LOCALSTACK_ENDPOINT", "http://envwins:9999")
    assert get_localstack_endpoint("foo", 1234) == "http://foo:1234"


def test_explicit_host_only_resolves_port_from_env(monkeypatch: Any) -> None:
    """Independent overrides: explicit host, env-driven port."""
    _clear_localstack_env(monkeypatch)
    _fence_dockerenv(monkeypatch, present=False)
    monkeypatch.setenv("LOCALSTACK_PORT", "9999")
    assert get_localstack_endpoint("foo") == "http://foo:9999"


def test_scheme_less_localstack_endpoint_is_normalized(monkeypatch: Any) -> None:
    """A scheme-less env value falls through to defaults (urlparse fails)."""
    _clear_localstack_env(monkeypatch)
    _fence_dockerenv(monkeypatch, present=False)
    # urlparse("host:4566") returns scheme="host", hostname=None — the
    # helper treats that as "no usable host" and falls through to defaults.
    monkeypatch.setenv("LOCALSTACK_ENDPOINT", "host:4566")
    assert get_localstack_endpoint() == "http://localhost:4566"


def test_https_localstack_endpoint_preserves_scheme(monkeypatch: Any) -> None:
    """``https://`` in the env value is honored, not coerced to http."""
    _clear_localstack_env(monkeypatch)
    _fence_dockerenv(monkeypatch, present=False)
    monkeypatch.setenv("LOCALSTACK_ENDPOINT", "https://secure:5000")
    assert get_localstack_endpoint() == "https://secure:5000"


# -- is_localstack_available delegation ------------------------------------


def test_is_localstack_available_uses_shared_resolver(monkeypatch: Any) -> None:
    """The probe targets ``(host, port)`` returned by the shared resolver.

    Cross-helper consistency guard: when the helper resolves to
    ``probe-target:9999``, the probe must hit exactly that pair. This
    is the test that catches future drift between helper and probe.
    """
    _clear_localstack_env(monkeypatch)
    _fence_dockerenv(monkeypatch, present=False)
    monkeypatch.setenv("LOCALSTACK_ENDPOINT", "http://probe-target:9999")

    recorded: dict[str, tuple[str, int]] = {}

    import socket as _socket

    real_socket = _socket.socket

    class _RecordingSocket:
        def __init__(self, *_a: Any, **_kw: Any) -> None:
            pass

        def settimeout(self, _seconds: float) -> None:
            pass

        def connect_ex(self, addr: tuple[str, int]) -> int:
            recorded["addr"] = addr
            return 0

        def close(self) -> None:
            pass

    monkeypatch.setattr(_socket, "socket", _RecordingSocket)
    try:
        assert is_localstack_available() is True
    finally:
        monkeypatch.setattr(_socket, "socket", real_socket)

    assert recorded["addr"] == ("probe-target", 9999)


def test_is_localstack_available_in_docker_no_env_var_probes_localstack_host(
    monkeypatch: Any,
) -> None:
    """In-Docker default arm: probe targets ``localstack:4566``.

    Explicit guard for the intended behaviour change documented in
    the impl plan G1: when ``/.dockerenv`` exists and no env vars
    are set, the probe targets the Docker-network hostname.
    """
    _clear_localstack_env(monkeypatch)
    _fence_dockerenv(monkeypatch, present=True)

    recorded: dict[str, tuple[str, int]] = {}

    import socket as _socket

    real_socket = _socket.socket

    class _RecordingSocket:
        def __init__(self, *_a: Any, **_kw: Any) -> None:
            pass

        def settimeout(self, _seconds: float) -> None:
            pass

        def connect_ex(self, addr: tuple[str, int]) -> int:
            recorded["addr"] = addr
            return 0

        def close(self) -> None:
            pass

    monkeypatch.setattr(_socket, "socket", _RecordingSocket)
    try:
        is_localstack_available()
    finally:
        monkeypatch.setattr(_socket, "socket", real_socket)

    assert recorded["addr"] == ("localstack", 4566)


def test_is_localstack_available_returns_bool() -> None:
    """Type contract: the probe returns ``bool`` (not ``int``)."""
    result = is_localstack_available()
    assert isinstance(result, bool)
