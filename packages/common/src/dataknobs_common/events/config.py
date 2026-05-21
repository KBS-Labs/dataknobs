"""Structured configuration dataclasses for event bus backends.

Mirrors the dataknobs structured-config idiom used by ``LLMConfig``,
``RateLimiterConfig``, ``RetryConfig``, ``StreamConfig``, and
``VectorConfig`` — every backend kwarg is a typed dataclass field, and
``<Backend>EventBusConfig.from_dict(config)`` is the single source of
truth for translating a config-dict to typed construction. The registry
factories at :mod:`dataknobs_common.events.registry` collapse to
one-line wrappers over ``<EventBus>.from_config(config)``, so adding a
new ctor knob is a dataclass-field addition (consumed wholesale by
``from_dict``) rather than a per-factory allowlist edit. Drift between
ctor surface and config-driven entry point becomes structurally
impossible.

The dataclasses are ``frozen=True`` so ``bus.config`` is a safe
read-only window onto the construction parameters — runtime mutation is
intentionally unsupported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast


@dataclass(frozen=True)
class EventBusConfig:
    """Base class for every backend's typed configuration dataclass.

    Empty today — reserved for shared knobs (e.g., cross-backend
    observability hooks) without forcing a per-backend re-export.
    """


@dataclass(frozen=True)
class MemoryEventBusConfig(EventBusConfig):
    """Configuration for :class:`InMemoryEventBus`.

    The in-memory bus has no construction parameters today; the
    dataclass exists for structural symmetry with the other backends so
    every event bus exposes the same ``config``/``from_config`` surface.
    """

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> MemoryEventBusConfig:
        """Build a config from a dict, ignoring backend-routing keys.

        The ``"backend"`` key in the input dict is consumed upstream by
        :func:`create_event_bus`; this classmethod tolerates its
        presence so callers can pass the same dict through.
        """
        del config  # unused — no fields today
        return cls()


@dataclass(frozen=True)
class RedisEventBusConfig(EventBusConfig):
    """Configuration for :class:`RedisEventBus`.

    Attributes:
        host: Redis host.
        port: Redis port.
        password: Optional Redis password.
        ssl: Whether to use SSL/TLS (for ElastiCache).
        channel_prefix: Prefix applied to every Redis channel name.
    """

    host: str = "localhost"
    port: int = 6379
    password: str | None = None
    ssl: bool = False
    channel_prefix: str = "events"

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> RedisEventBusConfig:
        return cls(
            host=config.get("host", "localhost"),
            port=config.get("port", 6379),
            password=config.get("password"),
            ssl=config.get("ssl", False),
            channel_prefix=config.get("channel_prefix", "events"),
        )


@dataclass(frozen=True)
class PostgresEventBusConfig(EventBusConfig):
    """Configuration for :class:`PostgresEventBus`.

    Construction always normalizes through
    :func:`normalize_postgres_connection_config`, so ``from_dict``
    accepts every input shape that function supports
    (``connection_string``, individual host/port/... keys,
    ``DATABASE_URL``, ``POSTGRES_*`` env-var fallbacks). The
    ``connection_string`` field holds the resolved string after
    normalization — direct construction without normalization is
    supported but unusual; ``from_dict`` is the recommended path.

    Attributes:
        connection_string: Resolved PostgreSQL DSN.
        channel_prefix: Prefix applied to every LISTEN/NOTIFY channel.
    """

    connection_string: str
    channel_prefix: str = "events"

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> PostgresEventBusConfig:
        from dataknobs_common.postgres_config import (
            normalize_postgres_connection_config,
        )

        # ``require=True`` raises ConfigurationError when nothing
        # resolvable — surface the same error class today's bus does.
        normalized = cast(
            "dict[str, Any]",
            normalize_postgres_connection_config(dict(config), require=True),
        )
        return cls(
            connection_string=normalized["connection_string"],
            channel_prefix=config.get("channel_prefix", "events"),
        )


@dataclass(frozen=True)
class SqsEventBusConfig(EventBusConfig):
    """Configuration for :class:`SqsEventBus`.

    Mirrors :meth:`SqsEventBus.__init__` exactly — every ctor kwarg is
    a field here. The dataclass is the single source of truth for the
    backend's configurable surface; the registry factory and the
    typed-construction path both build through it.

    Attributes:
        queue_url: Full SQS queue URL (required, non-empty). A URL
            ending in ``.fifo`` is treated as a FIFO queue.
        region: AWS region. ``None`` defers to boto's default chain.
        endpoint_url: Override the SQS endpoint (LocalStack, VPC
            endpoint). ``None`` uses the public endpoint for ``region``.
        wait_time_seconds: ``ReceiveMessage`` long-poll wait (0-20).
        visibility_timeout: At-least-once retry window (seconds).
        topic_attribute: Message-attribute name carrying the topic.
        require_topic_attribute: ``True`` (default) releases
            attribute-less messages back to the queue; ``False``
            dispatches them to the bus's single subscription
            (single-topic bridge mode for AWS-native event sources).
        aws_access_key_id: Explicit access key (paired with
            ``aws_secret_access_key``); ``None`` defers to boto's
            default credential chain.
        aws_secret_access_key: Explicit secret key.
    """

    queue_url: str
    region: str | None = None
    endpoint_url: str | None = None
    wait_time_seconds: int = 20
    visibility_timeout: int = 60
    topic_attribute: str = "topic"
    require_topic_attribute: bool = True
    # AWS credential fields use ``repr=False`` so they are omitted from
    # the auto-generated ``__repr__`` — the new ``bus.config`` accessor
    # makes it much easier to accidentally log the full config (pytest
    # failure output, debug logging, exception formatting) than the
    # legacy kwarg-only construction shape, and an IAM access/secret key
    # pair must not appear in those streams.
    aws_access_key_id: str | None = field(default=None, repr=False)
    aws_secret_access_key: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.queue_url:
            raise ValueError(
                "SqsEventBusConfig requires a non-empty queue_url"
            )

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> SqsEventBusConfig:
        return cls(
            queue_url=config.get("queue_url", ""),
            region=config.get("region"),
            endpoint_url=config.get("endpoint_url"),
            wait_time_seconds=config.get("wait_time_seconds", 20),
            visibility_timeout=config.get("visibility_timeout", 60),
            topic_attribute=config.get("topic_attribute", "topic"),
            require_topic_attribute=config.get(
                "require_topic_attribute", True
            ),
            aws_access_key_id=config.get("aws_access_key_id"),
            aws_secret_access_key=config.get("aws_secret_access_key"),
        )


__all__ = [
    "EventBusConfig",
    "MemoryEventBusConfig",
    "PostgresEventBusConfig",
    "RedisEventBusConfig",
    "SqsEventBusConfig",
]
