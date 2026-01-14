"""Event bus abstraction for distributed event handling.

This module provides a unified event bus interface supporting multiple backends
for publish-subscribe messaging. Choose the appropriate backend based on your
deployment scenario:

Backends:
- InMemoryEventBus: Single process, no external dependencies
- PostgresEventBus: Uses LISTEN/NOTIFY, works on local and AWS RDS
- RedisEventBus: Uses pub/sub, works with local Redis and ElastiCache

Example:
    ```python
    from dataknobs_common.events import create_event_bus, Event, EventType

    # Create event bus from configuration
    bus = create_event_bus({"backend": "memory"})
    await bus.connect()

    # Subscribe to events
    async def on_bot_created(event: Event) -> None:
        print(f"Bot created: {event.payload['bot_id']}")

    subscription = await bus.subscribe("registry:bots", on_bot_created)

    # Publish an event
    await bus.publish("registry:bots", Event(
        type=EventType.CREATED,
        topic="registry:bots",
        payload={"bot_id": "my-bot", "config": {...}}
    ))

    # Cleanup
    await subscription.cancel()
    await bus.close()
    ```

Configuration Examples:
    ```python
    # In-memory (development/testing)
    config = {"backend": "memory"}

    # PostgreSQL (production, no extra infra)
    config = {
        "backend": "postgres",
        "connection_string": "postgresql://user:pass@host/db"
    }

    # Redis (production, scaled deployment)
    config = {
        "backend": "redis",
        "host": "elasticache.amazonaws.com",
        "port": 6379,
        "ssl": True
    }
    ```
"""

from __future__ import annotations

from .bus import EventBus, create_event_bus
from .memory import InMemoryEventBus
from .types import Event, EventType, Subscription

__all__ = [
    # Protocol
    "EventBus",
    # Factory
    "create_event_bus",
    # Types
    "Event",
    "EventType",
    "Subscription",
    # Implementations
    "InMemoryEventBus",
]

# Note: PostgresEventBus and RedisEventBus are not exported by default
# to avoid requiring their dependencies. Import them directly:
#
#   from dataknobs_common.events.postgres import PostgresEventBus
#   from dataknobs_common.events.redis import RedisEventBus
