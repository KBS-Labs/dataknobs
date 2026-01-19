"""Tests for the event bus abstraction."""

import asyncio
from datetime import datetime, timezone

import pytest

from dataknobs_common.events import (
    Event,
    EventBus,
    EventType,
    InMemoryEventBus,
    Subscription,
    create_event_bus,
)


class TestEventType:
    """Tests for EventType enum."""

    def test_event_types_exist(self):
        """Test that all expected event types exist."""
        assert EventType.CREATED.value == "created"
        assert EventType.UPDATED.value == "updated"
        assert EventType.DELETED.value == "deleted"
        assert EventType.ACTIVATED.value == "activated"
        assert EventType.DEACTIVATED.value == "deactivated"
        assert EventType.ERROR.value == "error"
        assert EventType.CUSTOM.value == "custom"


class TestEvent:
    """Tests for Event dataclass."""

    def test_create_event(self):
        """Test creating an event."""
        event = Event(
            type=EventType.CREATED,
            topic="test:topic",
            payload={"key": "value"},
        )

        assert event.type == EventType.CREATED
        assert event.topic == "test:topic"
        assert event.payload == {"key": "value"}
        assert event.event_id is not None
        assert isinstance(event.timestamp, datetime)

    def test_event_defaults(self):
        """Test event default values."""
        event = Event(type=EventType.CREATED, topic="test")

        assert event.payload == {}
        assert event.metadata == {}
        assert event.source is None
        assert event.correlation_id is None

    def test_event_to_dict(self):
        """Test serializing event to dict."""
        event = Event(
            type=EventType.UPDATED,
            topic="registry:bots",
            payload={"bot_id": "test-bot"},
            source="test-source",
        )

        data = event.to_dict()

        assert data["type"] == "updated"
        assert data["topic"] == "registry:bots"
        assert data["payload"] == {"bot_id": "test-bot"}
        assert data["source"] == "test-source"
        assert "timestamp" in data
        assert "event_id" in data

    def test_event_from_dict(self):
        """Test deserializing event from dict."""
        data = {
            "type": "created",
            "topic": "test:topic",
            "payload": {"data": "value"},
            "timestamp": "2024-01-15T10:30:00+00:00",
            "event_id": "test-id-123",
            "source": "test",
        }

        event = Event.from_dict(data)

        assert event.type == EventType.CREATED
        assert event.topic == "test:topic"
        assert event.payload == {"data": "value"}
        assert event.event_id == "test-id-123"
        assert event.source == "test"

    def test_event_roundtrip(self):
        """Test serialization roundtrip."""
        original = Event(
            type=EventType.DELETED,
            topic="registry:users",
            payload={"user_id": "123"},
            source="api",
            correlation_id="corr-456",
            metadata={"extra": "data"},
        )

        data = original.to_dict()
        restored = Event.from_dict(data)

        assert restored.type == original.type
        assert restored.topic == original.topic
        assert restored.payload == original.payload
        assert restored.source == original.source
        assert restored.correlation_id == original.correlation_id
        assert restored.metadata == original.metadata

    def test_event_with_correlation(self):
        """Test creating event with correlation ID."""
        event = Event(type=EventType.CREATED, topic="test")
        correlated = event.with_correlation("correlation-id-123")

        assert correlated.correlation_id == "correlation-id-123"
        assert correlated.event_id == event.event_id
        assert correlated.type == event.type


class TestSubscription:
    """Tests for Subscription dataclass."""

    def test_create_subscription(self):
        """Test creating a subscription."""
        sub = Subscription(
            subscription_id="sub-123",
            topic="test:topic",
            handler=lambda e: None,
        )

        assert sub.subscription_id == "sub-123"
        assert sub.topic == "test:topic"
        assert sub.pattern is None

    def test_subscription_repr(self):
        """Test subscription string representation."""
        sub = Subscription(
            subscription_id="sub-123",
            topic="test:topic",
            handler=lambda e: None,
            pattern="test:*",
        )

        repr_str = repr(sub)
        assert "sub-123" in repr_str
        assert "test:topic" in repr_str
        assert "test:*" in repr_str


class TestInMemoryEventBus:
    """Tests for InMemoryEventBus."""

    @pytest.fixture
    async def bus(self):
        """Create a fresh event bus for each test."""
        bus = InMemoryEventBus()
        await bus.connect()
        yield bus
        await bus.close()

    @pytest.mark.asyncio
    async def test_connect(self, bus):
        """Test bus connection."""
        assert bus._connected is True

    @pytest.mark.asyncio
    async def test_close(self):
        """Test bus cleanup on close."""
        bus = InMemoryEventBus()
        await bus.connect()

        async def handler(event):
            pass

        await bus.subscribe("test", handler)
        assert bus.subscription_count == 1

        await bus.close()
        assert bus.subscription_count == 0
        assert bus._connected is False

    @pytest.mark.asyncio
    async def test_publish_subscribe(self, bus):
        """Test basic pub/sub."""
        received_events = []

        async def handler(event):
            received_events.append(event)

        await bus.subscribe("test:topic", handler)

        event = Event(
            type=EventType.CREATED,
            topic="test:topic",
            payload={"message": "hello"},
        )
        await bus.publish("test:topic", event)

        assert len(received_events) == 1
        assert received_events[0].payload == {"message": "hello"}

    @pytest.mark.asyncio
    async def test_multiple_subscribers(self, bus):
        """Test multiple subscribers receive events."""
        received1 = []
        received2 = []

        async def handler1(event):
            received1.append(event)

        async def handler2(event):
            received2.append(event)

        await bus.subscribe("test", handler1)
        await bus.subscribe("test", handler2)

        await bus.publish("test", Event(type=EventType.CREATED, topic="test"))

        assert len(received1) == 1
        assert len(received2) == 1

    @pytest.mark.asyncio
    async def test_pattern_subscription(self, bus):
        """Test wildcard pattern matching."""
        received = []

        async def handler(event):
            received.append(event)

        await bus.subscribe("registry:*", handler, pattern="registry:*")

        await bus.publish(
            "registry:bots",
            Event(type=EventType.CREATED, topic="registry:bots"),
        )
        await bus.publish(
            "registry:users",
            Event(type=EventType.UPDATED, topic="registry:users"),
        )
        await bus.publish(
            "other:topic",
            Event(type=EventType.DELETED, topic="other:topic"),
        )

        assert len(received) == 2
        topics = [e.topic for e in received]
        assert "registry:bots" in topics
        assert "registry:users" in topics
        assert "other:topic" not in topics

    @pytest.mark.asyncio
    async def test_cancel_subscription(self, bus):
        """Test canceling a subscription."""
        received = []

        async def handler(event):
            received.append(event)

        sub = await bus.subscribe("test", handler)
        assert bus.subscription_count == 1

        await bus.publish("test", Event(type=EventType.CREATED, topic="test"))
        assert len(received) == 1

        await sub.cancel()
        assert bus.subscription_count == 0

        await bus.publish("test", Event(type=EventType.CREATED, topic="test"))
        assert len(received) == 1  # No new events

    @pytest.mark.asyncio
    async def test_no_subscribers(self, bus):
        """Test publishing with no subscribers doesn't error."""
        event = Event(type=EventType.CREATED, topic="no-subscribers")
        await bus.publish("no-subscribers", event)  # Should not raise

    @pytest.mark.asyncio
    async def test_handler_exception(self, bus):
        """Test that handler exceptions don't break the bus."""
        received = []

        async def bad_handler(event):
            raise ValueError("Handler error")

        async def good_handler(event):
            received.append(event)

        await bus.subscribe("test", bad_handler)
        await bus.subscribe("test", good_handler)

        await bus.publish("test", Event(type=EventType.CREATED, topic="test"))

        # Good handler should still receive the event
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_sync_handler(self, bus):
        """Test that sync handlers work too."""
        received = []

        def sync_handler(event):
            received.append(event)

        await bus.subscribe("test", sync_handler)
        await bus.publish("test", Event(type=EventType.CREATED, topic="test"))

        assert len(received) == 1


class TestCreateEventBus:
    """Tests for create_event_bus factory."""

    @pytest.mark.asyncio
    async def test_create_memory_bus(self):
        """Test creating memory event bus."""
        bus = create_event_bus({"backend": "memory"})
        assert isinstance(bus, InMemoryEventBus)

    @pytest.mark.asyncio
    async def test_create_default_bus(self):
        """Test default is memory backend."""
        bus = create_event_bus({})
        assert isinstance(bus, InMemoryEventBus)

    @pytest.mark.asyncio
    async def test_unknown_backend_raises(self):
        """Test unknown backend raises ValueError."""
        with pytest.raises(ValueError) as excinfo:
            create_event_bus({"backend": "unknown"})

        assert "unknown" in str(excinfo.value).lower()
        assert "memory" in str(excinfo.value)


class TestEventBusProtocol:
    """Tests that implementations satisfy EventBus protocol."""

    @pytest.mark.asyncio
    async def test_inmemory_satisfies_protocol(self):
        """Test InMemoryEventBus satisfies EventBus protocol."""
        bus = InMemoryEventBus()

        # Check all required methods exist
        assert hasattr(bus, "connect")
        assert hasattr(bus, "close")
        assert hasattr(bus, "publish")
        assert hasattr(bus, "subscribe")

        # Check it's recognized as EventBus
        assert isinstance(bus, EventBus)


class TestEventBusIntegration:
    """Integration tests for event bus."""

    @pytest.mark.asyncio
    async def test_registry_events_workflow(self):
        """Test a realistic registry events workflow."""
        bus = create_event_bus({"backend": "memory"})
        await bus.connect()

        # Track events for audit
        audit_log = []

        async def audit_handler(event):
            audit_log.append({
                "type": event.type.value,
                "topic": event.topic,
                "timestamp": event.timestamp,
            })

        # Cache invalidation
        cache = {}

        async def cache_invalidator(event):
            if event.type in (EventType.UPDATED, EventType.DELETED):
                bot_id = event.payload.get("bot_id")
                if bot_id and bot_id in cache:
                    del cache[bot_id]

        # Subscribe to all registry events
        await bus.subscribe("registry:*", audit_handler, pattern="registry:*")
        await bus.subscribe("registry:bots", cache_invalidator)

        # Simulate bot lifecycle
        cache["bot-1"] = {"name": "Test Bot"}

        # Create event
        await bus.publish(
            "registry:bots",
            Event(
                type=EventType.CREATED,
                topic="registry:bots",
                payload={"bot_id": "bot-1"},
            ),
        )

        # Update event - should invalidate cache
        await bus.publish(
            "registry:bots",
            Event(
                type=EventType.UPDATED,
                topic="registry:bots",
                payload={"bot_id": "bot-1"},
            ),
        )

        assert "bot-1" not in cache  # Cache was invalidated
        assert len(audit_log) == 2  # Both events logged

        await bus.close()

    @pytest.mark.asyncio
    async def test_correlation_tracking(self):
        """Test tracking related events with correlation IDs."""
        bus = create_event_bus({"backend": "memory"})
        await bus.connect()

        events_by_correlation = {}

        async def tracking_handler(event):
            if event.correlation_id:
                if event.correlation_id not in events_by_correlation:
                    events_by_correlation[event.correlation_id] = []
                events_by_correlation[event.correlation_id].append(event)

        await bus.subscribe("workflow:*", tracking_handler, pattern="workflow:*")

        # Simulate a workflow with correlated events
        correlation_id = "workflow-123"

        await bus.publish(
            "workflow:started",
            Event(
                type=EventType.CREATED,
                topic="workflow:started",
                correlation_id=correlation_id,
                payload={"step": "start"},
            ),
        )

        await bus.publish(
            "workflow:step",
            Event(
                type=EventType.UPDATED,
                topic="workflow:step",
                correlation_id=correlation_id,
                payload={"step": "processing"},
            ),
        )

        await bus.publish(
            "workflow:completed",
            Event(
                type=EventType.UPDATED,
                topic="workflow:completed",
                correlation_id=correlation_id,
                payload={"step": "done"},
            ),
        )

        assert correlation_id in events_by_correlation
        assert len(events_by_correlation[correlation_id]) == 3

        await bus.close()
