"""Protocol-conformance tests for StateBridge."""

from __future__ import annotations

from dataknobs_bots.reasoning.state_bridge import (
    BiDirectionalBridge,
    InboxOnlyBridge,
    PeekBridge,
    StateBridge,
    SubscribingBridge,
    SubsetBridge,
)
from dataknobs_common.callbacks import CallbackRegistry


def test_inbox_only_conforms_to_protocol() -> None:
    assert isinstance(InboxOnlyBridge(), StateBridge)


def test_peek_conforms_to_protocol() -> None:
    assert isinstance(PeekBridge(), StateBridge)


def test_bidirectional_conforms_to_protocol() -> None:
    assert isinstance(BiDirectionalBridge(), StateBridge)


def test_subset_conforms_to_protocol() -> None:
    assert isinstance(SubsetBridge(project=lambda x: x), StateBridge)


def test_subscribing_conforms_to_protocol() -> None:
    bridge = SubscribingBridge(InboxOnlyBridge(), registry=CallbackRegistry())
    assert isinstance(bridge, StateBridge)


def test_protocol_is_runtime_checkable() -> None:
    class CustomBridge:
        def read_inbox(self, host, key):
            return None

        def write_outbox(self, host, key, value):
            pass

    assert isinstance(CustomBridge(), StateBridge)


def test_protocol_rejects_non_conformer() -> None:
    class NotABridge:
        def read(self, host, key):  # wrong method name
            return None

    assert not isinstance(NotABridge(), StateBridge)
