"""Substrate-specific tests for :class:`LifecycleHooks`.

These tests pin guarantees that emerge from the
:class:`~dataknobs_common.callbacks.CallbackRegistry` substrate:
the :attr:`registry` accessor, the
:class:`~dataknobs_common.capabilities.Capability.CALLBACK_REGISTRY`
capability advertisement, and the consumer-facing escape hatches
(pluggable ordering, priority-tagged callbacks, EventBus fan-out).
The pre-substrate public surface (registration / triggering / clear
/ from_config) is covered by ``test_lifecycle_hooks.py``.
"""
from __future__ import annotations

from typing import Any

import pytest

from dataknobs_bots.reasoning.lifecycle import LifecycleHooks
from dataknobs_common.callbacks import (
    CallbackRegistry,
    PriorityOrdering,
)
from dataknobs_common.capabilities import Capability
from dataknobs_common.events import Event, EventType, InMemoryEventBus


def _evt(stage: str = "s", **extra: Any) -> dict[str, Any]:
    return {"stage": stage, "phase": "start", "reason": "normal", **extra}


# ---------------------------------------------------------------------------
# Capability advertisement
# ---------------------------------------------------------------------------


def test_lifecycle_hooks_supports_callback_registry_capability() -> None:
    """``LifecycleHooks`` advertises :attr:`Capability.CALLBACK_REGISTRY`
    so consumers can probe via :meth:`CapabilityContract.supports` before
    reaching through the :attr:`registry` accessor.
    """
    hooks = LifecycleHooks()
    assert hooks.supports(Capability.CALLBACK_REGISTRY) is True


def test_supported_capabilities_classvar_declared() -> None:
    """The class-level capability set is the registration surface that
    :class:`~dataknobs_common.capabilities.CapabilityMixin` reads."""
    assert Capability.CALLBACK_REGISTRY in LifecycleHooks.SUPPORTED_CAPABILITIES


# ---------------------------------------------------------------------------
# Registry accessor — exposes the underlying CallbackRegistry
# ---------------------------------------------------------------------------


def test_registry_accessor_returns_callback_registry() -> None:
    hooks = LifecycleHooks()
    assert isinstance(hooks.registry, CallbackRegistry)


def test_registry_accessor_returns_same_instance_on_repeat() -> None:
    """The accessor is stable — a consumer caching the reference can
    rely on it across calls."""
    hooks = LifecycleHooks()
    assert hooks.registry is hooks.registry


# ---------------------------------------------------------------------------
# Pluggable ordering via the escape hatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_consumer_can_swap_ordering_via_registry() -> None:
    """A consumer reaching through ``hooks.registry`` can swap in
    :class:`PriorityOrdering`. Subsequent ``trigger_turn_start`` calls
    honor the new ordering."""
    hooks = LifecycleHooks()
    seen: list[str] = []
    hooks.registry.set_ordering(PriorityOrdering())

    # Register via the registry directly so we can pass priority.
    hooks.registry.register(
        "turn_start", lambda _: seen.append("default"),
    )
    hooks.registry.register(
        "turn_start", lambda _: seen.append("low"), priority=-10,
    )
    hooks.registry.register(
        "turn_start", lambda _: seen.append("high"), priority=10,
    )
    await hooks.trigger_turn_start(_evt())
    assert seen == ["low", "default", "high"]


# ---------------------------------------------------------------------------
# EventBus fan-out via the escape hatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_consumer_can_fanout_lifecycle_events_to_bus() -> None:
    """``hooks.registry.also_publish_to(bus, topic_prefix="wizard:")``
    composes the lifecycle surface with an EventBus — every trigger
    additionally publishes the event payload to the bus under the
    prefixed topic. Cross-replica observability without changing the
    wizard surface."""
    hooks = LifecycleHooks()
    bus = InMemoryEventBus()
    await bus.connect()
    received: list[Event] = []

    async def handler(event: Event) -> None:
        received.append(event)

    await bus.subscribe("wizard:turn_start", handler)
    hooks.registry.also_publish_to(bus, topic_prefix="wizard:")

    # Also register a local callback to confirm both paths run.
    local_seen: list[dict[str, Any]] = []
    hooks.on_turn_start(lambda e: local_seen.append(e))

    payload = _evt(stage="greet")
    await hooks.trigger_turn_start(payload)

    assert local_seen == [payload]
    assert len(received) == 1
    assert received[0].topic == "wizard:turn_start"
    assert received[0].type is EventType.CUSTOM
    assert received[0].payload is payload


# ---------------------------------------------------------------------------
# clear() preserves registry instance identity
# ---------------------------------------------------------------------------


def test_clear_preserves_registry_instance_identity() -> None:
    """The :class:`WizardHooks.lifecycle` invariant — "the lifecycle
    instance identity is preserved (drained in place)" — also holds
    for the underlying registry. A consumer that customized the
    registry (e.g. installed a custom ordering or a fan-out target)
    keeps those customizations across :meth:`clear`.
    """
    hooks = LifecycleHooks()
    registry_before = hooks.registry
    bus = InMemoryEventBus()
    hooks.registry.also_publish_to(bus, topic_prefix="wizard:")
    hooks.on_turn_start(lambda _: None)
    hooks.on_turn_end(lambda _: None)

    hooks.clear()

    assert hooks.registry is registry_before
    assert hooks.turn_start_count == 0
    assert hooks.turn_end_count == 0
    # Fan-out target survives clear (drain-in-place) — the registry
    # advertises configured fan-out via supports_event_bus_emission.
    assert hooks.registry.supports_event_bus_emission() is True
