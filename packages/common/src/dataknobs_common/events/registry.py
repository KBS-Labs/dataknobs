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
dependencies (asyncpg, redis, aioboto3) at module load time, preserving
the ``dependencies = []`` base install.
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

    return InMemoryEventBus.from_config(config)


def _create_postgres_bus(config: dict[str, Any]) -> EventBus:
    # Delegating to ``from_config`` routes the config dict through the
    # typed ``PostgresEventBusConfig``, so every backend kwarg is a
    # dataclass field rather than a per-factory allowlist entry — and
    # the dataclass internally normalizes via
    # ``normalize_postgres_connection_config`` (supports
    # ``connection_string``, individual host/port/... keys, DATABASE_URL,
    # POSTGRES_* env-var fallbacks).
    from .postgres import PostgresEventBus

    return PostgresEventBus.from_config(config)


def _create_redis_bus(config: dict[str, Any]) -> EventBus:
    from .redis import RedisEventBus

    return RedisEventBus.from_config(config)


def _create_sqs_bus(config: dict[str, Any]) -> EventBus:
    # Lazy: aioboto3 (optional [sqs] extra) is imported only here, so a
    # consumer that never selects "sqs" needs no AWS dependency. The
    # config dict flows through ``SqsEventBusConfig.from_dict``, so a
    # missing ``queue_url`` surfaces as the dataclass's ``ValueError``
    # — same error class as the legacy direct-kwarg path.
    from .sqs import SqsEventBus

    return SqsEventBus.from_config(config)


event_bus_backends.register("memory", _create_memory_bus)
event_bus_backends.register("postgres", _create_postgres_bus)
event_bus_backends.register("redis", _create_redis_bus)
event_bus_backends.register("sqs", _create_sqs_bus)
