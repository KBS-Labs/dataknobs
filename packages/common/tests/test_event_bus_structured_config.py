"""Construction-path tests for the structured-config event bus refactor.

The original bug was a registry-factory allowlist drift: the SQS
factory enumerated 8 ctor kwargs out of 9, silently dropping
``require_topic_attribute``. The fix restructured every event bus's
construction surface around a typed
:class:`<Backend>EventBusConfig` dataclass; the four
``_create_*_bus`` registry factories collapse to one-line wrappers
over ``<EventBus>.from_config(config)``.

These tests pin the new contract: typed-config, ``from_config``, and
loose-kwarg paths all behave identically; mixing typed config with
loose kwargs raises ``TypeError``; ``create_event_bus`` threads every
config-dict key through to the bus. No service is required — they
exercise construction only.

The behavioural arm for ``require_topic_attribute=False`` (poll-loop
dispatch of attribute-less messages) is covered by
``test_sqs_events.py`` against real LocalStack.
"""

from __future__ import annotations

import pytest

from dataknobs_common.events import (
    EventBusConfig,
    InMemoryEventBus,
    MemoryEventBusConfig,
    PostgresEventBusConfig,
    RedisEventBusConfig,
    SqsEventBusConfig,
    create_event_bus,
)
from dataknobs_common.events.redis import RedisEventBus
from dataknobs_common.events.sqs import SqsEventBus

# ---------------------------------------------------------------------------
# Registry factory threads every kwarg through to the bus
# ---------------------------------------------------------------------------


class TestCreateSqsBusThreadsKnobs:
    """``create_event_bus({"backend": "sqs", ...})`` exposes every ctor knob.

    Regression family: a config dict key the bus ctor accepts MUST
    land on the constructed bus, not be silently dropped by a factory
    allowlist.
    """

    def _base_config(self, **overrides: object) -> dict[str, object]:
        config: dict[str, object] = {
            "backend": "sqs",
            "queue_url": (
                "https://sqs.us-east-1.amazonaws.com/000000000000/q"
            ),
            "endpoint_url": "http://127.0.0.1:65535",
        }
        config.update(overrides)
        return config

    def test_factory_forwards_require_topic_attribute_false(self) -> None:
        """``require_topic_attribute=False`` reaches the bus.

        Before the refactor the factory's explicit allowlist dropped
        this kwarg and the bus silently received the constructor
        default ``True``. After the structured-config refactor the
        factory consumes the dict wholesale via
        ``SqsEventBusConfig.from_dict`` so every key is forwarded.
        """
        bus = create_event_bus(
            self._base_config(require_topic_attribute=False)
        )
        assert isinstance(bus, SqsEventBus)
        assert bus.require_topic_attribute is False
        assert bus.config.require_topic_attribute is False

    def test_factory_defaults_require_topic_attribute_true(self) -> None:
        """Omitted key → factory uses the dataclass default ``True``.

        Backwards-compat guard: existing consumers who never asked for
        bridge mode must continue to get the safety default.
        """
        bus = create_event_bus(self._base_config())
        assert isinstance(bus, SqsEventBus)
        assert bus.require_topic_attribute is True

    def test_factory_threads_every_documented_knob(self) -> None:
        """All nine ``SqsEventBus`` ctor kwargs reach the bus.

        Catches a future allowlist regression for any of the other
        documented knobs — the same allowlist-drops-a-knob drift mode.
        """
        config = self._base_config(
            region="us-west-2",
            wait_time_seconds=5,
            visibility_timeout=120,
            topic_attribute="custom_topic",
            require_topic_attribute=False,
            aws_access_key_id="ak",
            aws_secret_access_key="sk",
        )
        bus = create_event_bus(config)
        assert isinstance(bus, SqsEventBus)
        c = bus.config
        assert c.queue_url == config["queue_url"]
        assert c.region == "us-west-2"
        assert c.endpoint_url == config["endpoint_url"]
        assert c.wait_time_seconds == 5
        assert c.visibility_timeout == 120
        assert c.topic_attribute == "custom_topic"
        assert c.require_topic_attribute is False
        assert c.aws_access_key_id == "ak"
        assert c.aws_secret_access_key == "sk"


# ---------------------------------------------------------------------------
# Typed config / from_config / backward-compat kwarg construction
# ---------------------------------------------------------------------------


class TestSqsEventBusConstructionShapes:
    """``SqsEventBus`` supports three construction shapes; all agree."""

    def test_typed_config_construction(self) -> None:
        cfg = SqsEventBusConfig(
            queue_url="https://sqs/q",
            region="us-east-1",
            require_topic_attribute=False,
        )
        bus = SqsEventBus(cfg)
        assert bus.config is cfg
        assert bus.require_topic_attribute is False

    def test_from_config_dict(self) -> None:
        bus = SqsEventBus.from_config(
            {
                "queue_url": "https://sqs/q",
                "require_topic_attribute": False,
            }
        )
        assert bus.require_topic_attribute is False

    def test_from_config_typed(self) -> None:
        cfg = SqsEventBusConfig(queue_url="https://sqs/q")
        bus = SqsEventBus.from_config(cfg)
        assert bus.config is cfg

    def test_kwarg_construction_back_compat(self) -> None:
        """Legacy kwarg call shape — ``SqsEventBus(queue_url=...)`` — still works."""
        bus = SqsEventBus(
            queue_url="https://sqs/q",
            region="us-east-1",
            require_topic_attribute=False,
        )
        assert bus.config.queue_url == "https://sqs/q"
        assert bus.config.region == "us-east-1"
        assert bus.require_topic_attribute is False

    def test_kwarg_construction_defaults_unchanged(self) -> None:
        """Omitted kwargs match the dataclass defaults (== legacy ctor defaults)."""
        bus = SqsEventBus(queue_url="https://sqs/q")
        assert bus.config.wait_time_seconds == 20
        assert bus.config.visibility_timeout == 60
        assert bus.config.topic_attribute == "topic"
        assert bus.require_topic_attribute is True
        assert bus.config.region is None

    def test_mixing_typed_config_and_kwargs_raises(self) -> None:
        """Ambiguity is surfaced loudly, not silently resolved."""
        cfg = SqsEventBusConfig(queue_url="https://sqs/q")
        with pytest.raises(TypeError, match="cannot mix"):
            SqsEventBus(cfg, region="us-east-1")  # type: ignore[call-overload]

    def test_empty_queue_url_raises_valueerror(self) -> None:
        """The historical validation error remains a ``ValueError``."""
        with pytest.raises(ValueError, match="queue_url"):
            SqsEventBus(queue_url="")
        with pytest.raises(ValueError, match="queue_url"):
            SqsEventBus.from_config({})


class TestRedisEventBusConstructionShapes:
    """Redis bus: same structured-config contract as SQS."""

    def test_default_construction(self) -> None:
        bus = RedisEventBus()
        assert bus.config.host == "localhost"
        assert bus.config.port == 6379

    def test_typed_config(self) -> None:
        cfg = RedisEventBusConfig(host="redis", port=6380)
        bus = RedisEventBus(cfg)
        assert bus.config.host == "redis"
        assert bus.config.port == 6380

    def test_from_config_dict(self) -> None:
        bus = RedisEventBus.from_config({"host": "redis", "port": 6380})
        assert bus.config.host == "redis"

    def test_kwarg_back_compat(self) -> None:
        bus = RedisEventBus(host="redis", port=6380, ssl=True)
        assert bus.config.host == "redis"
        assert bus.config.ssl is True

    def test_mixing_typed_config_and_kwargs_raises(self) -> None:
        cfg = RedisEventBusConfig()
        with pytest.raises(TypeError, match="cannot mix"):
            RedisEventBus(cfg, host="redis")  # type: ignore[call-overload]

    def test_factory_threads_every_knob(self) -> None:
        bus = create_event_bus(
            {
                "backend": "redis",
                "host": "elastic.amazonaws.com",
                "port": 6380,
                "password": "secret",
                "ssl": True,
                "channel_prefix": "myapp",
            }
        )
        assert isinstance(bus, RedisEventBus)
        c = bus.config
        assert c.host == "elastic.amazonaws.com"
        assert c.port == 6380
        assert c.password == "secret"
        assert c.ssl is True
        assert c.channel_prefix == "myapp"


class TestInMemoryEventBusConstructionShapes:
    """In-memory bus: structural symmetry with the other backends."""

    def test_default_construction(self) -> None:
        bus = InMemoryEventBus()
        assert isinstance(bus.config, MemoryEventBusConfig)

    def test_typed_config(self) -> None:
        cfg = MemoryEventBusConfig()
        bus = InMemoryEventBus(cfg)
        assert bus.config is cfg

    def test_from_config_dict(self) -> None:
        bus = InMemoryEventBus.from_config({"backend": "memory"})
        assert isinstance(bus.config, MemoryEventBusConfig)

    def test_factory_path(self) -> None:
        bus = create_event_bus({"backend": "memory"})
        assert isinstance(bus, InMemoryEventBus)
        assert isinstance(bus.config, MemoryEventBusConfig)


class TestPostgresEventBusConstructionShapes:
    """Postgres bus: three construction shapes, all routed through dataclass."""

    DSN = "postgresql://u:p@h:5432/db"

    def test_positional_connection_string(self) -> None:
        """Legacy positional call form ``PostgresEventBus(dsn)`` still works."""
        from dataknobs_common.events.postgres import PostgresEventBus

        bus = PostgresEventBus(self.DSN)
        assert bus.config.connection_string == self.DSN
        assert bus.config.channel_prefix == "events"

    def test_keyword_connection_string(self) -> None:
        from dataknobs_common.events.postgres import PostgresEventBus

        bus = PostgresEventBus(
            connection_string=self.DSN, channel_prefix="test_"
        )
        assert bus.config.connection_string == self.DSN
        assert bus.config.channel_prefix == "test_"

    def test_dict_config(self) -> None:
        from dataknobs_common.events.postgres import PostgresEventBus

        bus = PostgresEventBus(
            config={"connection_string": self.DSN, "channel_prefix": "x"}
        )
        assert bus.config.connection_string == self.DSN
        assert bus.config.channel_prefix == "x"

    def test_typed_config(self) -> None:
        from dataknobs_common.events.postgres import PostgresEventBus

        cfg = PostgresEventBusConfig(
            connection_string=self.DSN, channel_prefix="t"
        )
        bus = PostgresEventBus(config=cfg)
        assert bus.config is cfg

    def test_from_config_dict(self) -> None:
        from dataknobs_common.events.postgres import PostgresEventBus

        bus = PostgresEventBus.from_config(
            {"connection_string": self.DSN, "channel_prefix": "fc"}
        )
        assert bus.config.connection_string == self.DSN
        assert bus.config.channel_prefix == "fc"

    def test_mixing_typed_config_with_positional_raises(self) -> None:
        from dataknobs_common.events.postgres import PostgresEventBus

        cfg = PostgresEventBusConfig(connection_string=self.DSN)
        with pytest.raises(TypeError, match="cannot mix"):
            PostgresEventBus(self.DSN, config=cfg)


# ---------------------------------------------------------------------------
# Config dataclass behaviour
# ---------------------------------------------------------------------------


class TestSqsEventBusConfigDataclass:
    """Structural guarantees of the :class:`SqsEventBusConfig` dataclass."""

    def test_frozen_dataclass_rejects_mutation(self) -> None:
        import dataclasses

        cfg = SqsEventBusConfig(queue_url="https://sqs/q")
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.queue_url = "https://other/q"  # type: ignore[misc]

    def test_post_init_rejects_empty_queue_url(self) -> None:
        with pytest.raises(ValueError, match="queue_url"):
            SqsEventBusConfig(queue_url="")

    def test_from_dict_ignores_unknown_keys(self) -> None:
        """Backend-routing keys (``"backend"``) in the same dict are tolerated."""
        cfg = SqsEventBusConfig.from_dict(
            {"backend": "sqs", "queue_url": "https://sqs/q"}
        )
        assert cfg.queue_url == "https://sqs/q"

    def test_base_class_export(self) -> None:
        """``EventBusConfig`` is the base; every backend's config extends it."""
        assert issubclass(SqsEventBusConfig, EventBusConfig)
        assert issubclass(RedisEventBusConfig, EventBusConfig)
        assert issubclass(MemoryEventBusConfig, EventBusConfig)
        assert issubclass(PostgresEventBusConfig, EventBusConfig)
