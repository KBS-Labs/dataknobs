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

from typing import TYPE_CHECKING

from .bus import EventBus, create_event_bus, create_event_bus_async
from .config import (
    EventBusConfig,
    MemoryEventBusConfig,
    PostgresEventBusConfig,
    RedisEventBusConfig,
    SqsEventBusConfig,
)
from .memory import InMemoryEventBus
from .registry import EventBusFactory, event_bus_backends
from .types import Event, EventType, Subscription

if TYPE_CHECKING:
    from .sqs import SqsEventBus

__all__ = [
    # Protocol
    "EventBus",
    # Factory
    "create_event_bus",
    "create_event_bus_async",
    # Plugin registry
    "event_bus_backends",
    "EventBusFactory",
    # Types
    "Event",
    "EventType",
    "Subscription",
    # Structured config dataclasses
    "EventBusConfig",
    "MemoryEventBusConfig",
    "PostgresEventBusConfig",
    "RedisEventBusConfig",
    "SqsEventBusConfig",
    # Implementations
    "InMemoryEventBus",
    "SqsEventBus",
]


def __getattr__(name: str) -> object:
    """Lazily expose ``SqsEventBus`` (PEP 562).

    Importing it eagerly would pull the optional ``aioboto3`` dependency
    at ``dataknobs_common.events`` import time, breaking the
    ``dependencies = []`` base install. This defers the import to first
    attribute access. The registry's ``"sqs"`` factory lazy-imports
    independently, so ``create_event_bus({"backend": "sqs"})`` works even
    if this top-level symbol is never touched.
    """
    if name == "SqsEventBus":
        from .sqs import SqsEventBus

        return SqsEventBus
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Note: PostgresEventBus and RedisEventBus are not exported by default
# to avoid requiring their dependencies. Import them directly:
#
#   from dataknobs_common.events.postgres import PostgresEventBus
#   from dataknobs_common.events.redis import RedisEventBus
