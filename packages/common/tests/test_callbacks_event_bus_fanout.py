"""Behavior tests for :meth:`CallbackRegistry.also_publish_to`.

Exercises the fan-out adapter against :class:`InMemoryEventBus` so the
composition path runs through the real event-bus surface (no mocks).
"""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_common.callbacks import CallbackRegistry
from dataknobs_common.events import Event, EventType, InMemoryEventBus


class _CapturingBus:
    """Minimal duck-typed :class:`EventBus` for the sync ``fire()``
    coverage.

    ``InMemoryEventBus`` binds an :class:`asyncio.Lock` to the loop
    where it is first used, which makes it awkward to exercise the
    "no running loop" branch of :meth:`CallbackRegistry._publish_to_buses`
    (which spins a fresh loop via :func:`asyncio.run`). This shim
    captures every ``publish`` call synchronously without holding
    loop-bound state. Only the ``publish`` surface is required —
    :class:`CallbackRegistry` never calls ``connect`` / ``subscribe``
    on a configured fan-out target.
    """

    def __init__(self) -> None:
        self.published: list[tuple[str, Event]] = []

    async def publish(self, topic: str, event: Event) -> None:
        self.published.append((topic, event))


@pytest.mark.asyncio
async def test_fanout_publishes_to_bus_under_prefix() -> None:
    registry: CallbackRegistry = CallbackRegistry()
    bus = InMemoryEventBus()
    await bus.connect()
    received: list[Event] = []

    async def handler(event: Event) -> None:
        received.append(event)

    await bus.subscribe("wizard:turn_start", handler)
    registry.also_publish_to(bus, topic_prefix="wizard:")
    registry.register("turn_start", lambda _: None)
    await registry.fire_async("turn_start", {"stage": "greet"})

    assert len(received) == 1
    assert received[0].topic == "wizard:turn_start"
    assert received[0].payload == {"stage": "greet"}
    assert received[0].type is EventType.CUSTOM


@pytest.mark.asyncio
async def test_fanout_publishes_without_prefix_when_unset() -> None:
    registry: CallbackRegistry = CallbackRegistry()
    bus = InMemoryEventBus()
    await bus.connect()
    received: list[Event] = []

    async def handler(event: Event) -> None:
        received.append(event)

    await bus.subscribe("ingest:domain:end", handler)
    registry.also_publish_to(bus)  # default topic_prefix=""
    await registry.fire_async("ingest:domain:end", {"tenant_id": "acme"})

    assert len(received) == 1
    assert received[0].topic == "ingest:domain:end"
    assert received[0].payload == {"tenant_id": "acme"}


@pytest.mark.asyncio
async def test_fanout_runs_local_callbacks_too() -> None:
    registry: CallbackRegistry = CallbackRegistry()
    bus = InMemoryEventBus()
    await bus.connect()
    seen_locally: list[dict] = []
    registry.register("t", lambda payload: seen_locally.append(payload))
    registry.also_publish_to(bus, topic_prefix="prefix:")
    await registry.fire_async("t", {"k": 1})

    assert seen_locally == [{"k": 1}]


@pytest.mark.asyncio
async def test_fanout_multiple_buses_each_receive_payload() -> None:
    registry: CallbackRegistry = CallbackRegistry()
    bus_a = InMemoryEventBus()
    bus_b = InMemoryEventBus()
    await bus_a.connect()
    await bus_b.connect()
    received_a: list[Event] = []
    received_b: list[Event] = []

    async def handler_a(event: Event) -> None:
        received_a.append(event)

    async def handler_b(event: Event) -> None:
        received_b.append(event)

    await bus_a.subscribe("a:t", handler_a)
    await bus_b.subscribe("b:t", handler_b)
    registry.also_publish_to(bus_a, topic_prefix="a:")
    registry.also_publish_to(bus_b, topic_prefix="b:")
    await registry.fire_async("t", {"k": 1})

    assert len(received_a) == 1
    assert received_a[0].topic == "a:t"
    assert len(received_b) == 1
    assert received_b[0].topic == "b:t"


@pytest.mark.asyncio
async def test_fanout_without_subscribers_is_silent() -> None:
    registry: CallbackRegistry = CallbackRegistry()
    bus = InMemoryEventBus()
    await bus.connect()
    registry.also_publish_to(bus, topic_prefix="x:")
    # No subscriber on "x:t" — fan-out should still publish without raising.
    await registry.fire_async("t", {"k": 1})


@pytest.mark.asyncio
async def test_fanout_local_callback_error_does_not_block_bus_publish() -> None:
    # The fan-out publishes BEFORE local callbacks run, so a failing
    # local callback (under the default LOG_AND_CONTINUE policy) does
    # not suppress the bus delivery.
    registry: CallbackRegistry = CallbackRegistry()
    bus = InMemoryEventBus()
    await bus.connect()
    received: list[Event] = []

    async def handler(event: Event) -> None:
        received.append(event)

    await bus.subscribe("p:t", handler)
    registry.also_publish_to(bus, topic_prefix="p:")

    def boom(_: dict) -> None:
        raise RuntimeError("local callback failed")

    registry.register("t", boom)
    await registry.fire_async("t", {"k": 1})

    assert len(received) == 1
    assert received[0].topic == "p:t"


def test_supports_event_bus_emission_advertises_configuration() -> None:
    registry: CallbackRegistry = CallbackRegistry()
    bus = InMemoryEventBus()
    assert registry.supports_event_bus_emission() is False
    registry.also_publish_to(bus, topic_prefix="x:")
    assert registry.supports_event_bus_emission() is True


@pytest.mark.asyncio
async def test_fanout_event_payload_is_passed_by_reference() -> None:
    # Documented behavior: the registry forwards the payload to the bus
    # by reference (no defensive copy). Consumer-side mutation between
    # fire_async() and handler invocation is the consumer's concern.
    registry: CallbackRegistry = CallbackRegistry()
    bus = InMemoryEventBus()
    await bus.connect()
    received: list[Event] = []

    async def handler(event: Event) -> None:
        received.append(event)

    await bus.subscribe("t", handler)
    registry.also_publish_to(bus)
    payload = {"k": 1}
    await registry.fire_async("t", payload)
    assert received[0].payload is payload


# ---------------------------------------------------------------------------
# Sync fire() + EventBus fan-out
# ---------------------------------------------------------------------------


def test_sync_fire_with_bus_no_running_loop_drives_publish_to_completion() -> None:
    """Sync ``fire()`` with fan-out configured and no running loop runs
    the bus publishes synchronously via :func:`asyncio.run`. The publish
    completes before ``fire()`` returns — no fire-and-forget tasks, no
    silent exception loss.
    """
    registry: CallbackRegistry = CallbackRegistry()
    bus = _CapturingBus()
    registry.also_publish_to(bus, topic_prefix="x:")
    local_seen: list[dict[str, Any]] = []
    registry.register("t", lambda payload: local_seen.append(payload))

    registry.fire("t", {"k": 1})

    assert local_seen == [{"k": 1}]
    assert len(bus.published) == 1
    full_topic, event = bus.published[0]
    assert full_topic == "x:t"
    assert event.topic == "x:t"
    assert event.payload == {"k": 1}
    assert event.type is EventType.CUSTOM


def test_sync_fire_with_bus_no_running_loop_multiple_targets() -> None:
    """Multiple fan-out targets all receive the publish under their
    respective prefixes when ``fire()`` runs without a loop. The
    :func:`asyncio.run` path delegates to ``_publish_to_buses_async``
    which gathers all targets — identical semantics to ``fire_async``.
    """
    registry: CallbackRegistry = CallbackRegistry()
    bus_a = _CapturingBus()
    bus_b = _CapturingBus()
    registry.also_publish_to(bus_a, topic_prefix="a:")
    registry.also_publish_to(bus_b, topic_prefix="b:")

    registry.fire("t", {"k": 1})

    assert [t for t, _ in bus_a.published] == ["a:t"]
    assert [t for t, _ in bus_b.published] == ["b:t"]


def test_sync_fire_without_bus_no_running_loop_does_not_invoke_run() -> None:
    """The ``_publish_to_buses`` early-return guard means a registry
    without configured fan-out never touches :func:`asyncio.run` — sync
    callbacks fire inline regardless of loop state.
    """
    registry: CallbackRegistry = CallbackRegistry()
    seen: list[dict[str, Any]] = []
    registry.register("t", lambda payload: seen.append(payload))

    registry.fire("t", {"k": 1})

    assert seen == [{"k": 1}]


@pytest.mark.asyncio
async def test_sync_fire_with_bus_inside_running_loop_raises_typeerror() -> None:
    """Calling ``fire()`` from inside a running event loop with fan-out
    configured raises ``TypeError`` — the bus publish would otherwise be
    scheduled as a fire-and-forget task subject to garbage collection
    and silent exception loss. Consumers must use ``fire_async`` instead.
    """
    registry: CallbackRegistry = CallbackRegistry()
    bus = _CapturingBus()
    registry.also_publish_to(bus, topic_prefix="x:")

    with pytest.raises(TypeError, match="fire_async"):
        registry.fire("t", {"k": 1})

    # Guard fired before any publish — the bus must not have been touched.
    assert bus.published == []


@pytest.mark.asyncio
async def test_sync_fire_inside_running_loop_without_bus_does_not_raise() -> None:
    """The running-loop guard is scoped to fan-out — a fan-out-free
    registry's ``fire()`` is safe to call from inside a running loop
    because there's no async work to schedule.
    """
    registry: CallbackRegistry = CallbackRegistry()
    seen: list[dict[str, Any]] = []
    registry.register("t", lambda payload: seen.append(payload))

    registry.fire("t", {"k": 1})

    assert seen == [{"k": 1}]
