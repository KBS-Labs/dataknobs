"""Tests for the capability guards and CapabilityNotSupportedError.

``supports_capability`` and ``require_capability`` are the ask and the
act, and the cells below hold them to the same reading of every host --
including one that does not implement the contract at all, which the
ask must answer rather than crash on.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from dataknobs_common.capabilities import (
    Capability,
    CapabilityMixin,
    CapabilityNotSupportedError,
    require_capability,
    supports_capability,
)
from dataknobs_common.exceptions import DataknobsError, OperationError


class _Backend(CapabilityMixin):
    SUPPORTED_CAPABILITIES: ClassVar[frozenset[Capability]] = frozenset(
        {
            Capability.STREAMING_READS,
        }
    )


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
        SUPPORTED_CAPABILITIES: ClassVar[frozenset[str]] = frozenset({"custom_x"})

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


def test_error_is_dataknobs_error_hierarchy_member() -> None:
    """The error must extend DataknobsError so unified catch-all handlers see it."""
    backend = _Backend()
    with pytest.raises(DataknobsError) as exc_info:
        require_capability(backend, Capability.SNAPSHOT_ISOLATION)
    assert isinstance(exc_info.value, OperationError)
    assert isinstance(exc_info.value, CapabilityNotSupportedError)


def test_error_context_carries_capability_and_host() -> None:
    """The context dict is populated for structured logging."""
    backend = _Backend()
    with pytest.raises(CapabilityNotSupportedError) as exc_info:
        require_capability(backend, Capability.SNAPSHOT_ISOLATION)
    assert exc_info.value.context == {
        "capability": "snapshot_isolation",
        "host": "_Backend",
    }


# ---------------------------------------------------------------------------
# ``supports_capability`` — the non-raising half.
# ---------------------------------------------------------------------------


def test_supports_capability_answers_the_contract() -> None:
    backend = _Backend()
    assert supports_capability(backend, Capability.STREAMING_READS) is True
    assert supports_capability(backend, Capability.SNAPSHOT_ISOLATION) is False


def test_supports_capability_raw_string() -> None:
    """Consumer-defined vocabulary works here too."""

    class _CustomBackend(CapabilityMixin):
        SUPPORTED_CAPABILITIES: ClassVar[frozenset[str]] = frozenset({"custom_x"})

    backend = _CustomBackend()
    assert supports_capability(backend, "custom_x") is True
    assert supports_capability(backend, "custom_y") is False


def test_supports_capability_on_a_non_contract_host_is_false_not_an_error() -> None:
    """A duck-typed object answers ``False`` rather than ``AttributeError``.

    This is the case the helper exists for. A caller holding an
    attribute typed ``Any`` -- a store a consumer handed in, a plugin
    loaded by name -- cannot know its object speaks the contract, and
    ``host.supports(...)`` would raise on one that does not. The
    capability question has an answer for such an object, and the
    answer is no.
    """

    class _NotAContract:
        pass

    assert supports_capability(_NotAContract(), Capability.STREAMING_READS) is False


def test_the_ask_and_the_act_agree_on_every_host() -> None:
    """Whatever ``supports_capability`` denies, ``require_capability`` raises on.

    They are two readings of one question, so a host they disagree about
    would let a caller check successfully and then be refused -- which is
    exactly the failure the pair exists to prevent. ``require_capability``
    is implemented in terms of the ask, and this cell is what holds that
    to be true rather than merely currently so.
    """

    class _NotAContract:
        pass

    hosts = [_Backend(), _NotAContract()]
    caps = [Capability.STREAMING_READS, Capability.SNAPSHOT_ISOLATION, "custom_x"]

    for host in hosts:
        for cap in caps:
            if supports_capability(host, cap):
                assert require_capability(host, cap) is None
            else:
                with pytest.raises(CapabilityNotSupportedError):
                    require_capability(host, cap)
