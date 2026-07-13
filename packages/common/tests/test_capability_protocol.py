"""Protocol-conformance tests for CapabilityContract."""

from __future__ import annotations

from typing import ClassVar

import pytest

from dataknobs_common.capabilities import (
    CAPABILITY_FAMILIES,
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


def test_conditional_write_member_and_family() -> None:
    """The conditional-write capability is a stable consistency-family member."""
    assert Capability.CONDITIONAL_WRITE.value == "conditional_write"
    assert Capability.CONDITIONAL_WRITE in CAPABILITY_FAMILIES["consistency"]


def test_transactional_metadata_not_a_member() -> None:
    """The conditional-write contract has one name; the metadata-flavored
    synonym is not part of the enum."""
    with pytest.raises(AttributeError):
        _ = Capability.TRANSACTIONAL_METADATA  # type: ignore[attr-defined]
    assert not any(
        cap.value == "transactional_metadata"
        for members in CAPABILITY_FAMILIES.values()
        for cap in members
    )
