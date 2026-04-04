"""Integration tests for RedisEventBus.

These tests require a running Redis instance.  The host and port are read
from ``REDIS_HOST`` / ``REDIS_PORT`` environment variables, falling back
to ``localhost:6379``.

Skipped when:
- TEST_REDIS env var is "false" (explicit opt-out), OR
- Redis is not reachable

Run via:
    uv run pytest packages/common/tests/test_redis_events.py
Or with full integration test runner:
    bin/test.sh common
"""

from __future__ import annotations

import asyncio
import os

import pytest

from dataknobs_common.events import Event, EventType
from dataknobs_common.events.redis import RedisEventBus
from dataknobs_common.testing import is_redis_available

TEST_REDIS = os.getenv("TEST_REDIS", "").lower() != "false"

skip_redis = pytest.mark.skipif(
    not TEST_REDIS or not is_redis_available(),
    reason="Redis integration tests skipped. Set TEST_REDIS=true and ensure Redis is running.",
)


@skip_redis
class TestRedisEventBusIntegration:
    """Integration tests exercising real Redis pub/sub."""

    @pytest.mark.asyncio
    async def test_publish_subscribe_roundtrip(self):
        """Subscribe, publish, verify handler fires with correct event."""
        bus = RedisEventBus()
        await bus.connect()
        try:
            received: list[Event] = []

            async def handler(event: Event) -> None:
                received.append(event)

            await bus.subscribe("test:roundtrip", handler)

            event = Event(
                type=EventType.CREATED,
                topic="test:roundtrip",
                payload={"key": "value"},
            )
            await bus.publish("test:roundtrip", event)

            for _ in range(50):
                if received:
                    break
                await asyncio.sleep(0.05)

            assert len(received) == 1
            assert received[0].type == EventType.CREATED
            assert received[0].payload == {"key": "value"}
        finally:
            await bus.close()

    @pytest.mark.asyncio
    async def test_unsubscribe_stops_delivery(self):
        """After unsubscribing, events are no longer delivered."""
        bus = RedisEventBus()
        await bus.connect()
        try:
            received: list[Event] = []

            async def handler(event: Event) -> None:
                received.append(event)

            sub = await bus.subscribe("test:unsub", handler)
            await sub.cancel()

            await bus.publish(
                "test:unsub",
                Event(type=EventType.CREATED, topic="test:unsub", payload={}),
            )
            await asyncio.sleep(0.5)

            assert len(received) == 0
        finally:
            await bus.close()

    @pytest.mark.asyncio
    async def test_pattern_subscription(self):
        """Pattern subscription receives events matching the pattern."""
        bus = RedisEventBus()
        await bus.connect()
        try:
            received: list[Event] = []

            async def handler(event: Event) -> None:
                received.append(event)

            await bus.subscribe("test:pattern", handler, pattern="test:*")

            await bus.publish(
                "test:pattern",
                Event(
                    type=EventType.CREATED,
                    topic="test:pattern",
                    payload={"matched": True},
                ),
            )

            for _ in range(50):
                if received:
                    break
                await asyncio.sleep(0.05)

            assert len(received) == 1
            assert received[0].payload == {"matched": True}
        finally:
            await bus.close()

    @pytest.mark.asyncio
    async def test_multiple_subscribers_same_topic(self):
        """Two handlers on the same topic both receive the event."""
        bus = RedisEventBus()
        await bus.connect()
        try:
            received_a: list[Event] = []
            received_b: list[Event] = []

            async def handler_a(event: Event) -> None:
                received_a.append(event)

            async def handler_b(event: Event) -> None:
                received_b.append(event)

            await bus.subscribe("test:multi", handler_a)
            await bus.subscribe("test:multi", handler_b)

            await bus.publish(
                "test:multi",
                Event(type=EventType.CREATED, topic="test:multi", payload={"n": 1}),
            )

            for _ in range(50):
                if received_a and received_b:
                    break
                await asyncio.sleep(0.05)

            assert len(received_a) == 1
            assert len(received_b) == 1
        finally:
            await bus.close()

    @pytest.mark.asyncio
    async def test_close_then_publish_raises(self):
        """After close(), publish raises RuntimeError."""
        bus = RedisEventBus()
        await bus.connect()
        await bus.close()

        with pytest.raises(RuntimeError, match="not connected"):
            await bus.publish(
                "test:closed",
                Event(type=EventType.CREATED, topic="test:closed", payload={}),
            )
