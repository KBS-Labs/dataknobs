"""Registry-extensible factory for event bus backends.

``create_event_bus()`` resolves the ``backend`` config key through this
registry. Out-of-tree consumers can add a custom
:class:`~dataknobs_common.events.bus.EventBus` backend without forking
DataKnobs::

    from dataknobs_common.events import event_bus_backends, create_event_bus

    def _make_kafka_bus(config):
        from my_pkg.kafka_bus import KafkaEventBus
        return KafkaEventBus(brokers=config["brokers"])

    event_bus_backends.register("kafka", _make_kafka_bus)
    bus = create_event_bus({"backend": "kafka", "brokers": "..."})

The registry is a :class:`~dataknobs_common.registry.PluginRegistry` —
the shared config-driven factory abstraction also used by
``lock_backends`` and the bots-side ``memory_backends`` /
``knowledge_base_backends`` / ``source_backends``. Resolution of the
``backend`` discriminator, the not-found error shape ("Unknown event bus
backend: <name>. Available backends: …"), the ``ValueError`` exception
class, and the lazy-init flow live in :class:`PluginRegistry`; this
module declares the per-domain knobs (kind label, validate_type, default
backend) and the four built-in backend factories.

Each built-in wrapper imports its concrete backend *lazily* (inside the
factory call) so importing this module never pulls optional backend
dependencies (asyncpg, redis, aioboto3) at module load time, preserving
the ``dependencies = []`` base install.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from dataknobs_common.registry import PluginRegistry

from .bus import EventBus

EventBusFactory = Callable[[dict[str, Any]], EventBus]
"""A backend factory: maps a config dict to an :class:`EventBus`.

Preserved as a public typealias for out-of-tree consumers that annotate
their factory closures. The registry holds factories of this shape; no
behavioural difference from the underlying ``Callable``.
"""

event_bus_backends: PluginRegistry[EventBus] = PluginRegistry(
    name="event_bus_backends",
    validate_type=EventBus,
    config_key="backend",
    config_key_default="memory",
    not_found_kind="event bus backend",
    not_found_exception=ValueError,
)
"""Registry of named :data:`EventBusFactory` callables.

Register a custom backend with
``event_bus_backends.register("name", factory)`` and select it via
``create_event_bus({"backend": "name", ...})``. The registry conforms
to :class:`~dataknobs_common.registry.BackendRegistry` for ``isinstance``
checks.
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
    # missing ``queue_url`` raises ``ValueError`` in
    # ``SqsEventBusConfig.__post_init__``; the ``PluginRegistry.create()``
    # wrapper re-raises it as ``OperationError`` (with the original
    # ``ValueError`` preserved on ``__cause__``).
    from .sqs import SqsEventBus

    return SqsEventBus.from_config(config)


event_bus_backends.register("memory", _create_memory_bus)
event_bus_backends.register("postgres", _create_postgres_bus)
event_bus_backends.register("redis", _create_redis_bus)
event_bus_backends.register("sqs", _create_sqs_bus)
