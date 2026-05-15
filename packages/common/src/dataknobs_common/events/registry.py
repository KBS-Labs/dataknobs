"""Registry-extensible factory for event bus backends.

``create_event_bus()`` resolves the ``backend`` config key through this
registry instead of a sealed ``if/elif`` chain. Out-of-tree consumers can
add a custom :class:`~dataknobs_common.events.bus.EventBus` backend without
forking DataKnobs::

    from dataknobs_common.events import event_bus_backends, create_event_bus

    def _make_kafka_bus(config):
        from my_pkg.kafka_bus import KafkaEventBus
        return KafkaEventBus(brokers=config["brokers"])

    event_bus_backends.register("kafka", _make_kafka_bus)
    bus = create_event_bus({"backend": "kafka", "brokers": "..."})

This mirrors the data-layer factories (``async_backends``,
``VectorStoreFactory``) and the sibling lock registry
(``dataknobs_common.locks.lock_backends``); the three stay structurally
consistent.

Each built-in wrapper imports its concrete backend *lazily* (inside the
factory call) so importing this module never pulls optional backend
dependencies (asyncpg, redis) at module load time, preserving the
``dependencies = []`` base install.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from dataknobs_common.registry import Registry

from .bus import EventBus

EventBusFactory = Callable[[dict[str, Any]], EventBus]
"""A backend factory: maps a config dict to an :class:`EventBus`."""

event_bus_backends: Registry[EventBusFactory] = Registry(
    name="event_bus_backends"
)
"""Registry of named :data:`EventBusFactory` callables.

Register a custom backend with
``event_bus_backends.register("name", factory)`` and select it via
``create_event_bus({"backend": "name", ...})``.
"""


def _create_memory_bus(config: dict[str, Any]) -> EventBus:
    from .memory import InMemoryEventBus

    return InMemoryEventBus()


def _create_postgres_bus(config: dict[str, Any]) -> EventBus:
    from .postgres import PostgresEventBus

    # Pass the full config through so the bus can accept any input shape
    # supported by ``normalize_postgres_connection_config`` (connection_string,
    # individual host/port/... keys, DATABASE_URL env var, or POSTGRES_* env
    # vars).
    return PostgresEventBus(
        config=config,
        channel_prefix=config.get("channel_prefix", "events"),
    )


def _create_redis_bus(config: dict[str, Any]) -> EventBus:
    from .redis import RedisEventBus

    return RedisEventBus(
        host=config.get("host", "localhost"),
        port=config.get("port", 6379),
        password=config.get("password"),
        ssl=config.get("ssl", False),
        channel_prefix=config.get("channel_prefix", "events"),
    )


event_bus_backends.register("memory", _create_memory_bus)
event_bus_backends.register("postgres", _create_postgres_bus)
event_bus_backends.register("redis", _create_redis_bus)
# Note: the "sqs" backend (SqsEventBus) is registered in Phase 2 alongside
# events/sqs.py and the optional [sqs] dependency. Until then an unknown
# "sqs" backend resolves to the same ValueError as any other unknown
# backend, keeping this change behaviour-identical to the prior if/elif.
