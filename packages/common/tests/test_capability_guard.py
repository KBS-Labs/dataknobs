"""Tests for require_capability and CapabilityNotSupportedError."""

from __future__ import annotations

from typing import ClassVar

import pytest

from dataknobs_common.capabilities import (
    Capability,
    CapabilityMixin,
    CapabilityNotSupportedError,
    require_capability,
)


class _Backend(CapabilityMixin):
    SUPPORTED_CAPABILITIES: ClassVar[frozenset[Capability]] = frozenset({
        Capability.STREAMING_READS,
    })


def test_require_capability_supported_returns_none() -> None:
    backend = _Backend()
    assert require_capability(backend, Capability.STREAMING_READS) is None


def test_require_capability_unsupported_raises() -> None:
    backend = _Backend()
    with pytest.raises(CapabilityNotSupportedError) as exc_info:
        require_capability(backend, Capability.SNAPSHOT_ISOLATION)
    assert exc_info.value.capability == Capability.SNAPSHOT_ISOLATION
    assert exc_info.value.host is backend


def test_require_capability_raw_string() -> None:
    """Raw-string capabilities work for consumer-defined features."""

    class _CustomBackend(CapabilityMixin):
        SUPPORTED_CAPABILITIES: ClassVar[frozenset] = frozenset({"custom_x"})  # type: ignore[assignment]

    backend = _CustomBackend()
    require_capability(backend, "custom_x")  # no raise
    with pytest.raises(CapabilityNotSupportedError):
        require_capability(backend, "custom_y")


def test_require_capability_object_without_supports_raises() -> None:
    """An object that doesn't implement the protocol fails the guard."""

    class _NotAContract:
        pass

    with pytest.raises(CapabilityNotSupportedError):
        require_capability(_NotAContract(), Capability.STREAMING_READS)


def test_error_message_includes_capability_value() -> None:
    backend = _Backend()
    with pytest.raises(CapabilityNotSupportedError) as exc_info:
        require_capability(backend, Capability.SNAPSHOT_ISOLATION)
    assert "snapshot_isolation" in str(exc_info.value)


def test_error_message_includes_host_class_name() -> None:
    backend = _Backend()
    with pytest.raises(CapabilityNotSupportedError) as exc_info:
        require_capability(backend, Capability.SNAPSHOT_ISOLATION)
    assert "_Backend" in str(exc_info.value)
