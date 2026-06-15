"""Protocol-conformance tests for CapabilityContract."""

from __future__ import annotations

from typing import ClassVar

from dataknobs_common.capabilities import (
    Capability,
    CapabilityContract,
    CapabilityMixin,
)


class _MinimalConforming(CapabilityMixin):
    SUPPORTED_CAPABILITIES: ClassVar[frozenset[Capability]] = frozenset({
        Capability.STREAMING_READS,
    })


class _MissingMethods:
    pass


def test_protocol_accepts_mixin_subclass() -> None:
    assert isinstance(_MinimalConforming(), CapabilityContract)


def test_protocol_rejects_missing_methods() -> None:
    assert not isinstance(_MissingMethods(), CapabilityContract)


def test_supported_capabilities_classmethod_invocable_on_class() -> None:
    """Classmethod query works without instantiation."""
    assert (
        Capability.STREAMING_READS in _MinimalConforming.supported_capabilities()
    )
