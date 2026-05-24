"""Structured configuration dataclasses for event bus backends.

Every backend ctor knob is a typed dataclass field; the
auto-derived :meth:`StructuredConfig.from_dict
<dataknobs_common.structured_config.StructuredConfig.from_dict>`
classmethod is the single source of truth for translating a config-dict
to typed construction. The registry factories at
:mod:`dataknobs_common.events.registry` collapse to one-line wrappers
over ``<EventBus>.from_config(config)``, so adding a new ctor knob is a
dataclass-field addition (consumed wholesale by ``from_dict``) rather
than a per-factory allowlist edit. Drift between ctor surface and
config-driven entry point is structurally impossible.

The dataclasses are ``frozen=True`` so ``bus.config`` is a safe
read-only window onto the construction parameters — runtime mutation is
intentionally unsupported.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, ClassVar, cast

from dataknobs_common.structured_config import StructuredConfig


@dataclass(frozen=True)
class EventBusConfig(StructuredConfig):
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
    The backend-routing key ``"backend"`` (and any other unrecognised
    keys) pass through cleanly via the inherited ``from_dict``.
    """


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

    # Redacted from ``repr`` by the StructuredConfig base.
    _SENSITIVE_FIELDS: ClassVar[frozenset[str]] = frozenset({"password"})


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

    The ``channel_prefix`` is sanitized in ``__post_init__`` (stripped to
    ``[a-zA-Z0-9_]``): it is interpolated directly into LISTEN/UNLISTEN
    statements, which cannot use parameterized queries, so the typed
    config is the single boundary that guarantees a SQL-safe prefix for
    every construction path (typed, dict, ``from_config``, legacy
    positional). A prefix that is empty after sanitization raises
    ``ValueError``.

    Attributes:
        connection_string: Resolved PostgreSQL DSN. Embeds the password;
            redacted from ``repr``.
        channel_prefix: Prefix applied to every LISTEN/NOTIFY channel.
            Sanitized to ``[a-zA-Z0-9_]`` at construction.
    """

    connection_string: str
    channel_prefix: str = "events"

    # The DSN embeds the password; redacted from ``repr`` by the base.
    _SENSITIVE_FIELDS: ClassVar[frozenset[str]] = frozenset({"connection_string"})

    def __post_init__(self) -> None:
        safe_prefix = re.sub(r"[^a-zA-Z0-9_]", "", self.channel_prefix)
        if not safe_prefix:
            raise ValueError(
                f"channel_prefix {self.channel_prefix!r} is empty after "
                "sanitization"
            )
        if safe_prefix != self.channel_prefix:
            # Frozen dataclass — bypass the immutability guard to store
            # the sanitized value computed from the caller's input.
            object.__setattr__(self, "channel_prefix", safe_prefix)

    @classmethod
    def _normalize_dict(cls, raw: dict[str, Any]) -> dict[str, Any]:
        from dataknobs_common.postgres_config import (
            normalize_postgres_connection_config,
        )

        # ``require=True`` raises ConfigurationError when nothing
        # resolvable — surface the same error class today's bus does.
        normalized = cast(
            "dict[str, Any]",
            normalize_postgres_connection_config(raw, require=True),
        )
        # ``normalize_postgres_connection_config`` returns a superset
        # of canonical keys; project only what this dataclass cares
        # about, and preserve ``channel_prefix`` from the raw dict if
        # present (the normalizer doesn't touch it).
        out: dict[str, Any] = {
            "connection_string": normalized["connection_string"],
        }
        if "channel_prefix" in raw:
            out["channel_prefix"] = raw["channel_prefix"]
        return out


@dataclass(frozen=True)
class SqsEventBusConfig(EventBusConfig):
    """Configuration for :class:`SqsEventBus`.

    Mirrors :meth:`SqsEventBus.__init__` exactly — every ctor kwarg is
    a field here. The dataclass is the single source of truth for the
    backend's configurable surface; the registry factory and the
    typed-construction path both build through it.

    ``queue_url`` defaults to ``""`` rather than being required so that
    ``SqsEventBusConfig.from_dict({})`` triggers the ``__post_init__``
    validator (the historical ``ValueError`` contract) rather than
    raising ``TypeError`` for a missing required argument.

    Attributes:
        queue_url: Full SQS queue URL (required non-empty post-init).
            A URL ending in ``.fifo`` is treated as a FIFO queue.
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
            default credential chain. Redacted from ``repr``.
        aws_secret_access_key: Explicit secret key. Redacted from ``repr``.
    """

    queue_url: str = ""
    region: str | None = None
    endpoint_url: str | None = None
    wait_time_seconds: int = 20
    visibility_timeout: int = 60
    topic_attribute: str = "topic"
    require_topic_attribute: bool = True
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None

    # The ``bus.config`` accessor makes it far easier to accidentally log
    # the full config (pytest failure output, debug logging, exception
    # formatting) than the legacy kwarg-only construction shape, and an
    # IAM access/secret key pair must never appear in those streams. The
    # StructuredConfig base masks both as ``'***'`` in ``repr``.
    _SENSITIVE_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"aws_access_key_id", "aws_secret_access_key"}
    )

    def __post_init__(self) -> None:
        if not self.queue_url:
            raise ValueError(
                "SqsEventBusConfig requires a non-empty queue_url"
            )


__all__ = [
    "EventBusConfig",
    "MemoryEventBusConfig",
    "PostgresEventBusConfig",
    "RedisEventBusConfig",
    "SqsEventBusConfig",
]
