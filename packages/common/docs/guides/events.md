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
pip install 'dataknobs-common[postgres]'

# Redis (uses pub/sub)
pip install 'dataknobs-common[redis]'

# AWS SQS (cloud-native, at-least-once)
pip install 'dataknobs-common[sqs]'
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

Accepts any input shape supported by the shared
[Postgres connection config normalizer](postgres-config.md):
`connection_string`, individual `host`/`port`/`database`/`user`/`password`
keys, `DATABASE_URL`, or `POSTGRES_*` env vars.

```python
# Connection string
bus = create_event_bus({
    "backend": "postgres",
    "connection_string": "postgresql://user:pass@host:5432/database"
})

# Individual keys
bus = create_event_bus({
    "backend": "postgres",
    "host": "host",
    "port": 5432,
    "database": "database",
    "user": "user",
    "password": "pass",
})

# Environment variables (POSTGRES_* or DATABASE_URL)
bus = create_event_bus({"backend": "postgres"})
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

### AWS SQS (Cloud-native, at-least-once)

`SqsEventBus` is backed by a single AWS SQS queue. Every topic shares
the queue; the topic travels in a message attribute (default `"topic"`)
and each subscriber long-polls and filters by exact match. Delivery is
**at-least-once** — a handler that raises is *not* acked, so the message
is redelivered after the queue's visibility timeout. **Handlers must be
idempotent** (the ingest trigger path already is).

Requires the optional `[sqs]` extra (`aioboto3`):

```bash
pip install 'dataknobs-common[sqs]'
```

```python
bus = create_event_bus({
    "backend": "sqs",
    "queue_url": "https://sqs.us-east-1.amazonaws.com/123456789012/events",
    # All optional below — omit to defer to boto's default chains:
    "region": "us-east-1",
    "endpoint_url": "http://localhost:4566",   # LocalStack / VPC endpoint
    "wait_time_seconds": 20,                    # ReceiveMessage long-poll
    "visibility_timeout": 60,                   # at-least-once retry window
    "topic_attribute": "topic",                 # routing attribute name
    "require_topic_attribute": True,            # default — see "Single-topic
                                                # bridge mode" below
    # "aws_access_key_id" / "aws_secret_access_key" optional —
    # default credential chain is used when omitted.
})
```

A `queue_url` ending in `.fifo` is treated as a FIFO queue:
`MessageGroupId` is the topic (per-topic ordering) and
`MessageDeduplicationId` is the event id.

**Use when:**

- AWS-native deployment; no broker to operate
- Durable triggers that must survive subscriber restarts
- Downstream of an SNS→SQS fan-out (works unchanged)

**Limitations:**

- At-least-once only — handlers must be idempotent
- Single-queue topic-attribute routing (queue-per-topic is out of
  scope); a topic with no subscriber recirculates until retention
  expires
- Wildcard `pattern` subscriptions are unsupported and raise
  `NotImplementedError` (loud rather than silent mis-routing)

#### Single-topic bridge mode

AWS-native event sources that bridge into SQS — `EventBridge → SQS`
targets, S3 → SQS bucket notifications, raw SNS → SQS delivery —
cannot set arbitrary SQS message attributes. A message produced by
such a source arrives without a `topic` attribute, and under the
default routing model it gets released back to the queue and
recirculates forever.

Set `require_topic_attribute=False` to dispatch attribute-less
messages to every active subscription on the bus instead. This mode
assumes the queue is **dedicated to a single topic** (your CDK or
infrastructure wires it that way).

```python
bus = create_event_bus({
    "backend": "sqs",
    "queue_url": "https://sqs.us-east-1.amazonaws.com/123/knowledge-trigger",
    "region": "us-east-1",
    "require_topic_attribute": False,   # accept attribute-less messages
})
await bus.subscribe("knowledge:trigger", handler)
```

Behaviour matrix:

| Topic attribute | `require_topic_attribute=True` (default) | `require_topic_attribute=False` |
|---|---|---|
| Present + matches sub | Dispatch to that sub | Dispatch to that sub |
| Present + mismatched  | Release back to queue (other sub may pick it up) | Release back to queue |
| Absent                | Release back to queue                            | Fan out to every active sub |

When the body is valid JSON but not `Event.to_dict()`-shaped (e.g. a
raw EventBridge envelope), it is delivered as a synthesised
`Event(type=EventType.CUSTOM, topic=<receiving poll task's topic>,
payload=<decoded body>)` event with one WARNING log per synthesis.
When the body is not valid JSON, it is discarded as poison (same as
the default mode).

**Note:** Import directly for type hints (lazy — does not pull
`aioboto3` until first access):

```python
from dataknobs_common.events import SqsEventBus
```

### Custom Backends (Plugin Registry)

`create_event_bus()` resolves the `backend` key through the
`event_bus_backends` registry. You can register your own `EventBus`
implementation and select it by name — no fork of DataKnobs required:

```python
from dataknobs_common.events import (
    event_bus_backends,
    create_event_bus,
)


def _make_kafka_bus(config: dict) -> "EventBus":
    from my_pkg.kafka_bus import KafkaEventBus
    return KafkaEventBus(brokers=config["brokers"])


# Register once at startup (e.g. in your package __init__).
event_bus_backends.register("kafka", _make_kafka_bus)

# Now selectable like any built-in backend.
bus = create_event_bus({"backend": "kafka", "brokers": "broker:9092"})
```

A backend factory is any `Callable[[dict], EventBus]`. Registering a key
that already exists raises `OperationError` unless you pass
`allow_overwrite=True`. The built-in `memory`, `postgres`, `redis`, and
`sqs` backends are registered automatically and are unchanged. Passing
`allow_overwrite=True` for one of those built-in names *will* replace
the built-in backend process-wide — this is supported but strongly
discouraged; prefer a distinct backend name. Selecting an unregistered
backend raises `ValueError` listing every registered backend (including
consumer-registered ones).

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

### Connection Resilience

The poll- and push-based backends recover from transient backend
failures automatically; no consumer code is required.

- **SQS and Redis** run their listener loops under a shared supervised
  loop. A transient failure is logged and retried after an
  exponential back-off **with jitter** that **escalates** under
  sustained failure (capped) and **resets** to the base delay after a
  clean iteration. This avoids a thundering herd — listeners across
  replicas do not all wake on the same 1-second boundary and re-hammer
  a degraded backend in lockstep.
- **Redis** additionally re-establishes its pub/sub connection on
  connection loss (rebuilding it and re-subscribing every active
  channel and pattern) rather than retrying a dead connection.
- **PostgreSQL** delivery is push-based via a dedicated `LISTEN`
  connection. A supervised watchdog probes that connection's liveness
  and, if it drops, re-opens it and re-registers every active channel,
  so delivery resumes instead of stopping silently.

A listener never gives up: it backs off and keeps retrying until the
backend recovers or the bus is closed. Handlers must still be
idempotent — SQS delivery is at-least-once, and a reconnect can replay
the redelivery window.

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

    Backends are resolved through the ``event_bus_backends`` registry,
    so custom backends registered by consumers are selectable too.

    Args:
        config: Configuration dict with:
            - backend: a registered backend name ("memory", "postgres",
              "redis", or any consumer-registered key)
            - Additional backend-specific options

    Returns:
        EventBus implementation

    Raises:
        ValueError: If the backend is not registered (message lists all
            registered backends).
    """
```

### Plugin Registry

```python
event_bus_backends: Registry[EventBusFactory]
# EventBusFactory = Callable[[dict[str, Any]], EventBus]

event_bus_backends.register("name", factory)   # add a backend
event_bus_backends.list_keys()                  # registered backend names
```

See [Custom Backends](#custom-backends-plugin-registry) for usage.

## Configuration Reference

### Memory Backend

```python
{"backend": "memory"}
```

### PostgreSQL Backend

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `backend` | str | Yes | `"postgres"` |
| `connection_string` | str | No† | PostgreSQL connection URL |
| `host` / `port` / `database` / `user` / `password` | various | No† | Individual connection keys |
| `channel_prefix` | str | No | Prefix for NOTIFY channels (default: `"events"`) |

† At least one postgres connection form must be resolvable: a
`connection_string`, individual keys, `DATABASE_URL` env var, or
`POSTGRES_*` env vars. See the
[Postgres connection config reference](postgres-config.md) for the
full precedence rules.

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
