# Event Bus System

The Event Bus provides a unified pub/sub event system for distributed applications in the DataKnobs ecosystem. It enables decoupled communication between components through asynchronous event publishing and subscription.

## Overview

The event bus abstraction allows you to:

- **Publish events** to named topics
- **Subscribe to topics** with async handlers
- **Switch backends** without changing application code
- **Scale from single-process** (in-memory) **to distributed** (Redis/PostgreSQL)

## Installation

The event bus is included in `dataknobs-common`:

```bash
pip install dataknobs-common
```

For production backends:

```bash
# PostgreSQL (uses LISTEN/NOTIFY)
pip install dataknobs-common[postgres]

# Redis (uses pub/sub)
pip install dataknobs-common[redis]
```

## Quick Start

```python
import asyncio
from dataknobs_common.events import create_event_bus, Event, EventType

async def main():
    # Create event bus from config
    bus = create_event_bus({"backend": "memory"})
    await bus.connect()

    # Subscribe to events
    async def on_event(event: Event) -> None:
        print(f"Received {event.type.value} on {event.topic}")
        print(f"Payload: {event.payload}")

    subscription = await bus.subscribe("my-topic", on_event)

    # Publish an event
    await bus.publish("my-topic", Event(
        type=EventType.CREATED,
        topic="my-topic",
        payload={"message": "Hello, World!"}
    ))

    # Give time for async delivery
    await asyncio.sleep(0.1)

    # Cleanup
    await subscription.cancel()
    await bus.close()

if __name__ == "__main__":
    asyncio.run(main())
```

## Core Concepts

### Events

An `Event` is an immutable message with:

| Field | Type | Description |
|-------|------|-------------|
| `type` | `EventType` | Category of the event |
| `topic` | `str` | Topic/channel the event is published to |
| `payload` | `dict` | Event data (JSON-serializable) |
| `timestamp` | `datetime` | When the event was created (auto-generated) |
| `event_id` | `str` | Unique event identifier (auto-generated) |
| `source` | `str \| None` | Optional identifier for the event source |
| `correlation_id` | `str \| None` | Optional ID to correlate related events |
| `metadata` | `dict` | Additional metadata for the event |

### Event Types

```python
from dataknobs_common.events import EventType

EventType.CREATED      # New resource created
EventType.UPDATED      # Resource modified
EventType.DELETED      # Resource removed
EventType.ACTIVATED    # Resource activated/enabled
EventType.DEACTIVATED  # Resource deactivated/disabled
EventType.ERROR        # Error occurred
EventType.CUSTOM       # Custom event type
```

### Subscriptions

A `Subscription` represents an active subscription and provides:

```python
subscription = await bus.subscribe("topic", handler)

# Cancel when done
await subscription.cancel()
```

Subscription attributes:

| Field | Type | Description |
|-------|------|-------------|
| `subscription_id` | `str` | Unique identifier for this subscription |
| `topic` | `str` | The topic this subscription is for |
| `pattern` | `str \| None` | Optional wildcard pattern |
| `created_at` | `datetime` | When the subscription was created |

## Backend Selection

Choose your backend based on deployment needs:

### In-Memory (Development/Testing)

Single-process, no external dependencies. Events are delivered synchronously within the same Python process.

```python
bus = create_event_bus({"backend": "memory"})
```

**Use when:**

- Unit testing
- Local development
- Single-process applications

### PostgreSQL (Production)

Uses PostgreSQL's LISTEN/NOTIFY for real-time event delivery. Works with local PostgreSQL and AWS RDS.

```python
bus = create_event_bus({
    "backend": "postgres",
    "connection_string": "postgresql://user:pass@host:5432/database"
})
```

**Use when:**

- You already have PostgreSQL
- Moderate event volume
- Don't want additional infrastructure

**Note:** Import directly for type hints:

```python
from dataknobs_common.events.postgres import PostgresEventBus
```

### Redis (Scaled Production)

Uses Redis pub/sub for high-throughput event delivery. Works with local Redis and AWS ElastiCache.

```python
bus = create_event_bus({
    "backend": "redis",
    "host": "localhost",
    "port": 6379,
    "ssl": False
})
```

For AWS ElastiCache:

```python
bus = create_event_bus({
    "backend": "redis",
    "host": "my-cluster.cache.amazonaws.com",
    "port": 6379,
    "ssl": True
})
```

**Use when:**

- High event volume
- Multiple application instances
- Need for horizontal scaling

**Note:** Import directly for type hints:

```python
from dataknobs_common.events.redis import RedisEventBus
```

## Usage Patterns

### Multiple Subscribers

Multiple handlers can subscribe to the same topic:

```python
async def logger(event: Event) -> None:
    print(f"LOG: {event.type.value} - {event.payload}")

async def metrics(event: Event) -> None:
    # Record metrics
    pass

async def notifier(event: Event) -> None:
    # Send notifications
    pass

await bus.subscribe("registry:bots", logger)
await bus.subscribe("registry:bots", metrics)
await bus.subscribe("registry:bots", notifier)
```

### Topic Conventions

Use colon-separated hierarchical topics:

```python
"registry:bots"           # Bot registry events
"registry:bots:config"    # Bot configuration changes
"knowledge:my-domain"     # Knowledge base events
"knowledge:ingestion"     # Ingestion status events
```

### Error Handling in Handlers

Handlers should handle their own exceptions:

```python
async def safe_handler(event: Event) -> None:
    try:
        await process_event(event)
    except Exception as e:
        logger.error(f"Failed to process event {event.id}: {e}")
        # Optionally publish error event
        await bus.publish("errors", Event(
            type=EventType.ERROR,
            topic="errors",
            payload={"original_event_id": event.id, "error": str(e)}
        ))
```

### Graceful Shutdown

Always clean up subscriptions and close the bus:

```python
import signal

subscriptions = []
bus = create_event_bus(config)

async def shutdown():
    for sub in subscriptions:
        await sub.cancel()
    await bus.close()

# Handle shutdown signals
loop = asyncio.get_event_loop()
for sig in (signal.SIGTERM, signal.SIGINT):
    loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown()))
```

## Integration with Registry

The event bus integrates with the bot registry for cache invalidation:

```python
from dataknobs_common.events import create_event_bus, Event, EventType
from dataknobs_bots.registry import CachingRegistryManager

# Event bus publishes when configs change
async def on_config_change(bot_id: str, config: dict):
    await bus.publish("registry:bots", Event(
        type=EventType.UPDATED,
        topic="registry:bots",
        payload={"bot_id": bot_id, "config": config}
    ))

# CachingRegistryManager subscribes for invalidation
manager = MyBotManager(
    backend=registry_backend,
    event_bus=bus,
)
await manager.initialize()  # Auto-subscribes to events
```

## API Reference

### EventBus Protocol

```python
class EventBus(Protocol):
    async def connect(self) -> None:
        """Connect to the event bus."""

    async def close(self) -> None:
        """Close the connection."""

    async def publish(self, topic: str, event: Event) -> None:
        """Publish an event to a topic."""

    async def subscribe(
        self,
        topic: str,
        handler: Callable[[Event], Any],
        pattern: str | None = None,
    ) -> Subscription:
        """Subscribe to events on a topic.

        Args:
            topic: The topic to subscribe to
            handler: Function to call with each event
            pattern: Optional wildcard pattern (uses fnmatch syntax)
        """
```

### Factory Function

```python
def create_event_bus(config: dict) -> EventBus:
    """Create an event bus from configuration.

    Args:
        config: Configuration dict with:
            - backend: "memory", "postgres", or "redis"
            - Additional backend-specific options

    Returns:
        EventBus implementation
    """
```

## Configuration Reference

### Memory Backend

```python
{"backend": "memory"}
```

### PostgreSQL Backend

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `backend` | str | Yes | `"postgres"` |
| `connection_string` | str | Yes | PostgreSQL connection URL |
| `channel_prefix` | str | No | Prefix for NOTIFY channels (default: `"events"`) |

### Redis Backend

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `backend` | str | Yes | `"redis"` |
| `host` | str | Yes | Redis host |
| `port` | int | No | Redis port (default: 6379) |
| `ssl` | bool | No | Enable TLS (default: False) |
| `password` | str | No | Redis password |
| `db` | int | No | Redis database number (default: 0) |

## Testing

For testing, use the in-memory backend:

```python
import pytest
from dataknobs_common.events import create_event_bus, Event, EventType

@pytest.fixture
async def event_bus():
    bus = create_event_bus({"backend": "memory"})
    await bus.connect()
    yield bus
    await bus.close()

async def test_publish_subscribe(event_bus):
    received = []

    async def handler(event: Event) -> None:
        received.append(event)

    await event_bus.subscribe("test", handler)
    await event_bus.publish("test", Event(
        type=EventType.CREATED,
        topic="test",
        payload={"key": "value"}
    ))

    assert len(received) == 1
    assert received[0].payload["key"] == "value"
```

## Module Exports

```python
from dataknobs_common.events import (
    # Protocol
    EventBus,
    # Factory
    create_event_bus,
    # Types
    Event,
    EventType,
    Subscription,
    # In-memory implementation
    InMemoryEventBus,
)

# Production backends (import directly)
from dataknobs_common.events.postgres import PostgresEventBus
from dataknobs_common.events.redis import RedisEventBus
```
