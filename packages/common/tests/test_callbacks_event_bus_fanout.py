"""Behavior tests for :meth:`CallbackRegistry.also_publish_to`.

Exercises the fan-out adapter against :class:`InMemoryEventBus` so the
composition path runs through the real event-bus surface (no mocks).
"""

from __future__ import annotations

import pytest

from dataknobs_common.callbacks import CallbackRegistry
from dataknobs_common.events import Event, EventType, InMemoryEventBus


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
