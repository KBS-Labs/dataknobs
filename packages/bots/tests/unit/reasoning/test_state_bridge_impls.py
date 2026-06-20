"""Behavior tests for StateBridge reference implementations."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from dataknobs_bots.reasoning.state_bridge import (
    BiDirectionalBridge,
    InboxOnlyBridge,
    PeekBridge,
    SubscribingBridge,
    SubsetBridge,
)
from dataknobs_common.callbacks import (
    CallbackRegistry,
    CapturingCallbackRegistry,
)
from dataknobs_common.scope import WhitelistProjector


def _host(metadata: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(metadata=metadata if metadata is not None else {})


# --- InboxOnlyBridge -------------------------------------------------- #


def test_inbox_only_read_pops_key() -> None:
    bridge: InboxOnlyBridge = InboxOnlyBridge()
    host = _host({"k": "v"})
    assert bridge.read_inbox(host, "k") == "v"
    assert "k" not in host.metadata


def test_inbox_only_read_missing_returns_none() -> None:
    bridge: InboxOnlyBridge = InboxOnlyBridge()
    host = _host({})
    assert bridge.read_inbox(host, "absent") is None


def test_inbox_only_write_raises() -> None:
    bridge: InboxOnlyBridge = InboxOnlyBridge()
    host = _host({})
    with pytest.raises(NotImplementedError, match="read-only"):
        bridge.write_outbox(host, "k", "v")


def test_inbox_only_host_without_metadata_raises() -> None:
    bridge: InboxOnlyBridge = InboxOnlyBridge()
    with pytest.raises(AttributeError, match="metadata"):
        bridge.read_inbox(SimpleNamespace(), "k")


# --- PeekBridge ------------------------------------------------------- #


def test_peek_read_does_not_consume_key() -> None:
    bridge: PeekBridge = PeekBridge()
    host = _host({"k": "v"})
    assert bridge.read_inbox(host, "k") == "v"
    # Peek — key survives the read.
    assert host.metadata["k"] == "v"


def test_peek_read_missing_returns_none() -> None:
    bridge: PeekBridge = PeekBridge()
    host = _host({})
    assert bridge.read_inbox(host, "absent") is None


def test_peek_repeated_read_returns_same_value() -> None:
    bridge: PeekBridge = PeekBridge()
    host = _host({"k": "v"})
    assert bridge.read_inbox(host, "k") == "v"
    assert bridge.read_inbox(host, "k") == "v"


def test_peek_write_raises() -> None:
    bridge: PeekBridge = PeekBridge()
    host = _host({})
    with pytest.raises(NotImplementedError, match="read-only"):
        bridge.write_outbox(host, "k", "v")


# --- BiDirectionalBridge ---------------------------------------------- #


def test_bidirectional_read_pops_key() -> None:
    bridge: BiDirectionalBridge = BiDirectionalBridge()
    host = _host({"k": "v"})
    assert bridge.read_inbox(host, "k") == "v"
    assert "k" not in host.metadata


def test_bidirectional_write_assigns_when_no_merge_fn() -> None:
    bridge: BiDirectionalBridge = BiDirectionalBridge()
    host = _host({})
    bridge.write_outbox(host, "k", "v")
    assert host.metadata["k"] == "v"


def test_bidirectional_write_overwrites_existing_when_no_merge_fn() -> None:
    bridge: BiDirectionalBridge = BiDirectionalBridge()
    host = _host({"k": "old"})
    bridge.write_outbox(host, "k", "new")
    assert host.metadata["k"] == "new"


def test_bidirectional_write_merges_when_merge_fn_provided() -> None:
    bridge: BiDirectionalBridge = BiDirectionalBridge(
        merge_fn=lambda existing, new: existing.update(new),
    )
    host = _host({"k": {"a": 1}})
    bridge.write_outbox(host, "k", {"b": 2})
    assert host.metadata["k"] == {"a": 1, "b": 2}


def test_bidirectional_write_assigns_when_no_existing_value() -> None:
    """A configured merge_fn still assigns when the key is absent."""
    bridge: BiDirectionalBridge = BiDirectionalBridge(
        merge_fn=lambda existing, new: existing.update(new),
    )
    host = _host({})
    bridge.write_outbox(host, "k", {"x": 1})
    assert host.metadata["k"] == {"x": 1}


def test_bidirectional_write_assigns_when_existing_not_dict() -> None:
    """If the existing value isn't a dict, merge_fn can't apply — fall
    back to assignment.
    """
    bridge: BiDirectionalBridge = BiDirectionalBridge(
        merge_fn=lambda existing, new: existing.update(new),
    )
    host = _host({"k": "scalar"})
    bridge.write_outbox(host, "k", {"x": 1})
    assert host.metadata["k"] == {"x": 1}


# --- SubsetBridge ----------------------------------------------------- #


def test_subset_write_applies_projection() -> None:
    bridge: SubsetBridge = SubsetBridge(
        project=lambda src: {"name": src["name"]},
    )
    host = _host({})
    bridge.write_outbox(host, "subset", {"name": "Alice", "secret": "x"})
    assert host.metadata["subset"] == {"name": "Alice"}


def test_subset_read_pops_key() -> None:
    bridge: SubsetBridge = SubsetBridge(project=lambda x: x)
    host = _host({"k": "v"})
    assert bridge.read_inbox(host, "k") == "v"
    assert "k" not in host.metadata


def test_subset_accepts_scope_projector() -> None:
    """A scope projector (duck-typed on ``.project``) drops straight into
    a SubsetBridge — the documented Pattern D / Pattern E interop.
    """
    projector = WhitelistProjector(
        {"name": "ignored-source"},  # captured source is ignored by write
        frozenset({"name"}),
    )
    bridge: SubsetBridge = SubsetBridge(project=projector)
    host = _host({})
    # WhitelistProjector captures its source at construction, so the write
    # value is irrelevant to the projection; it still projects the declared
    # key from the captured source.
    bridge.write_outbox(host, "subset", {"anything": True})
    assert host.metadata["subset"] == {"name": "ignored-source"}


def test_subset_callable_and_projector_produce_same_write() -> None:
    """A bare callable and an equivalent scope projector write the same
    projected outbox value.
    """
    source = {"name": "Bob", "secret": "x"}

    callable_bridge: SubsetBridge = SubsetBridge(
        project=lambda src: {"name": src["name"]},
    )
    host_a = _host({})
    callable_bridge.write_outbox(host_a, "out", source)

    projector_bridge: SubsetBridge = SubsetBridge(
        project=WhitelistProjector(source, frozenset({"name"})),
    )
    host_b = _host({})
    projector_bridge.write_outbox(host_b, "out", source)

    assert host_a.metadata["out"] == host_b.metadata["out"] == {"name": "Bob"}


# --- SubscribingBridge ------------------------------------------------ #


def test_subscribing_fires_on_read() -> None:
    registry = CapturingCallbackRegistry()
    bridge = SubscribingBridge(InboxOnlyBridge(), registry=registry)
    host = _host({"k": "v"})
    assert bridge.read_inbox(host, "k") == "v"
    assert registry.captured == [
        ("state_bridge:read", {"host": host, "key": "k", "popped": "v"}),
    ]


def test_subscribing_fires_on_write() -> None:
    registry = CapturingCallbackRegistry()
    bridge = SubscribingBridge(BiDirectionalBridge(), registry=registry)
    host = _host({})
    bridge.write_outbox(host, "k", "v")
    assert registry.captured == [
        ("state_bridge:write", {"host": host, "key": "k", "value": "v"}),
    ]


def test_subscribing_delegates_to_inner_behavior() -> None:
    registry = CapturingCallbackRegistry()
    bridge = SubscribingBridge(InboxOnlyBridge(), registry=registry)
    host = _host({"k": "v"})
    bridge.read_inbox(host, "k")
    # Inner pop semantics preserved.
    assert "k" not in host.metadata


def test_subscribing_uses_custom_topics() -> None:
    registry = CapturingCallbackRegistry()
    bridge = SubscribingBridge(
        InboxOnlyBridge(),
        registry=registry,
        read_topic="custom:read",
        write_topic="custom:write",
    )
    host = _host({"k": "v"})
    bridge.read_inbox(host, "k")
    assert registry.captured[0][0] == "custom:read"


def test_subscribing_observability_with_real_registry() -> None:
    """End-to-end: registry callbacks observe bridge activity."""
    registry: CallbackRegistry = CallbackRegistry()
    seen_reads: list[dict] = []
    registry.register("state_bridge:read", seen_reads.append)
    bridge = SubscribingBridge(InboxOnlyBridge(), registry=registry)
    host = _host({"k1": "v1", "k2": "v2"})
    bridge.read_inbox(host, "k1")
    bridge.read_inbox(host, "k2")
    assert len(seen_reads) == 2
    assert seen_reads[0]["key"] == "k1"
    assert seen_reads[1]["key"] == "k2"
