"""Tests for PostgresEventBus.

Unit tests cover channel name sanitization and SQL construction.
Integration tests (gated by TEST_POSTGRES env var) cover actual pub/sub
with a real PostgreSQL instance.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import pytest

from dataknobs_common.events import Event, EventType, PostgresEventBusConfig
from dataknobs_common.events.postgres import PostgresEventBus
from dataknobs_common.testing import is_postgres_available


class TestTopicToChannel:
    """Tests for _topic_to_channel sanitization."""

    def _make_bus(self) -> PostgresEventBus:
        """Create a bus instance for testing pure Python methods."""
        return PostgresEventBus(connection_string="postgresql://unused")

    def test_basic_topic(self):
        """Test standard topic conversion."""
        bus = self._make_bus()
        assert bus._topic_to_channel("registry:bots") == "events_registry_bots"

    def test_dots_replaced(self):
        """Test dots are replaced with underscores."""
        bus = self._make_bus()
        assert bus._topic_to_channel("com.example.topic") == "events_com_example_topic"

    def test_dashes_replaced(self):
        """Test dashes are replaced with underscores."""
        bus = self._make_bus()
        assert bus._topic_to_channel("my-topic-name") == "events_my_topic_name"

    def test_custom_prefix(self):
        """Test custom channel prefix."""
        bus = PostgresEventBus(
            connection_string="postgresql://unused",
            channel_prefix="myapp",
        )
        assert bus._topic_to_channel("test") == "myapp_test"

    def test_sql_injection_stripped(self):
        """Bug: special characters in topics could inject SQL into LISTEN/UNLISTEN.

        Characters like quotes, semicolons, and spaces must be stripped to
        prevent SQL injection in LISTEN/UNLISTEN statements which cannot use
        parameterized queries.
        """
        bus = self._make_bus()
        # Injection attempt: topic with SQL payload
        channel = bus._topic_to_channel("foo'; DROP TABLE users --")
        # Spaces, quotes, semicolons are stripped (not replaced with _)
        assert channel == "events_fooDROPTABLEusers__"
        # The key assertion: no SQL-injectable characters remain
        assert "'" not in channel
        assert ";" not in channel

    def test_spaces_stripped(self):
        """Test spaces are removed from channel names."""
        bus = self._make_bus()
        channel = bus._topic_to_channel("my topic")
        assert " " not in channel
        assert channel == "events_mytopic"

    def test_quotes_stripped(self):
        """Test quotes are removed from channel names."""
        bus = self._make_bus()
        channel = bus._topic_to_channel("it's a \"test\"")
        assert "'" not in channel
        assert '"' not in channel

    def test_empty_after_sanitization_raises(self):
        """Test that a topic producing an empty channel name raises ValueError."""
        bus = self._make_bus()
        with pytest.raises(ValueError, match="empty channel name"):
            bus._topic_to_channel("!@#$%^&*()")

    def test_unicode_stripped(self):
        """Test unicode characters are stripped."""
        bus = self._make_bus()
        channel = bus._topic_to_channel("événement")
        # Only ASCII alphanumeric + underscore survive
        assert channel == "events_vnement"


class TestChannelPrefixSanitization:
    """Tests for channel_prefix sanitization in __init__.

    Bug: channel_prefix was interpolated directly into LISTEN/UNLISTEN
    SQL statements without sanitization. While topics were sanitized by
    _topic_to_channel(), a malicious channel_prefix could inject SQL.
    """

    def test_clean_prefix_accepted(self):
        """Test a normal prefix is accepted unchanged."""
        bus = PostgresEventBus(
            connection_string="postgresql://unused",
            channel_prefix="myapp",
        )
        assert bus.config.channel_prefix == "myapp"

    def test_prefix_with_underscores(self):
        """Test prefix with underscores is accepted."""
        bus = PostgresEventBus(
            connection_string="postgresql://unused",
            channel_prefix="my_app_events",
        )
        assert bus.config.channel_prefix == "my_app_events"

    def test_sql_injection_in_prefix_stripped(self):
        """Bug: SQL injection via channel_prefix was not prevented.

        Characters like quotes, semicolons, and spaces must be stripped
        to prevent injection in LISTEN/UNLISTEN statements.
        """
        bus = PostgresEventBus(
            connection_string="postgresql://unused",
            channel_prefix="foo'; DROP TABLE users --",
        )
        assert "'" not in bus.config.channel_prefix
        assert ";" not in bus.config.channel_prefix
        assert " " not in bus.config.channel_prefix

    def test_empty_prefix_after_sanitization_raises(self):
        """Test that a prefix producing an empty result raises ValueError."""
        with pytest.raises(ValueError, match="empty after sanitization"):
            PostgresEventBus(
                connection_string="postgresql://unused",
                channel_prefix="!@#$%^&*()",
            )

    def test_directly_constructed_config_is_sanitized(self):
        """A directly-built typed config sanitizes its own prefix.

        Sanitization lives on ``PostgresEventBusConfig.__post_init__``,
        so every construction path — not just the bus ctor — produces a
        SQL-safe prefix. This guards the typed-config path that bypasses
        ``PostgresEventBus.__init__`` (e.g. ``from_config(typed_cfg)``).
        """
        cfg = PostgresEventBusConfig(
            connection_string="postgresql://unused",
            channel_prefix="foo'; DROP TABLE users --",
        )
        assert cfg.channel_prefix == "fooDROPTABLEusers"
        bus = PostgresEventBus.from_config(cfg)
        assert bus.config.channel_prefix == "fooDROPTABLEusers"

    def test_directly_constructed_config_empty_prefix_raises(self):
        """Empty-after-sanitization raises at config construction."""
        with pytest.raises(ValueError, match="empty after sanitization"):
            PostgresEventBusConfig(
                connection_string="postgresql://unused",
                channel_prefix="!@#$%^&*()",
            )


class TestConfigShapeSupport:
    """Tests for the expanded config input shapes on __init__.

    The bus now accepts the same unified shape as the other dataknobs
    postgres constructs (connection_string, individual keys, env-var
    fallbacks) via the shared ``normalize_postgres_connection_config``
    helper.
    """

    _POSTGRES_ENV_KEYS = (
        "DATABASE_URL",
        "POSTGRES_HOST",
        "POSTGRES_PORT",
        "POSTGRES_DB",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
    )

    @pytest.fixture(autouse=True)
    def _clear_env(self, monkeypatch):
        for key in self._POSTGRES_ENV_KEYS:
            monkeypatch.delenv(key, raising=False)
        # Disable ``.env`` / ``.project_vars`` loading — the normalizer
        # would otherwise read workspace dotenv files and shadow the
        # "nothing configured" assertions below.
        monkeypatch.setattr(
            "dataknobs_common.postgres_config._load_dotenv_fallbacks",
            lambda start_path=None: {},
        )

    def test_accepts_positional_connection_string(self):
        bus = PostgresEventBus("postgresql://u:p@h/db")
        assert bus.config.connection_string == "postgresql://u:p@h/db"

    def test_accepts_individual_keys_via_config(self):
        bus = PostgresEventBus(
            config={
                "host": "h",
                "port": 5433,
                "database": "db",
                "user": "u",
                "password": "p",
            }
        )
        assert (
            bus.config.connection_string
            == "postgresql://u:p@h:5433/db"
        )

    def test_accepts_database_url_env_fallback(self, monkeypatch):
        monkeypatch.setenv(
            "DATABASE_URL", "postgresql://u:p@env-h/env-db"
        )
        bus = PostgresEventBus(config={})
        assert (
            bus.config.connection_string == "postgresql://u:p@env-h/env-db"
        )

    def test_accepts_postgres_env_vars(self, monkeypatch):
        monkeypatch.setenv("POSTGRES_HOST", "env-h")
        monkeypatch.setenv("POSTGRES_DB", "env-db")
        monkeypatch.setenv("POSTGRES_USER", "env-u")
        monkeypatch.setenv("POSTGRES_PASSWORD", "env-p")
        bus = PostgresEventBus(config={})
        assert "env-h" in bus.config.connection_string
        assert "env-db" in bus.config.connection_string

    def test_raises_when_nothing_configured(self):
        from dataknobs_common.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError):
            PostgresEventBus(config={})

    def test_factory_routes_full_config_without_connection_string(self):
        """create_event_bus passes full config dict; individual keys work."""
        from dataknobs_common.events import create_event_bus

        bus = create_event_bus(
            {
                "backend": "postgres",
                "host": "h",
                "database": "db",
                "user": "u",
                "password": "p",
                "channel_prefix": "myapp",
            }
        )
        assert bus.config.connection_string == "postgresql://u:p@h:5432/db"
        assert bus.config.channel_prefix == "myapp"


class TestPublishSqlConstruction:
    """Tests verifying publish uses pg_notify with parameterized queries.

    Bug: The original code used `NOTIFY {channel}, $1` which doesn't support
    parameterized queries in PostgreSQL. The $1 was sent as a literal string
    payload, causing json.loads("$1") to fail in the notification handler.
    Events were silently lost (caught as a warning in _notification_handler).
    """

    @pytest.mark.asyncio
    async def test_publish_uses_pg_notify_with_parameters(self):
        """Verify publish calls pg_notify($1, $2) not NOTIFY channel, $1.

        This is the core reproduction of the bug: the old code passed the
        payload as a $1 parameter to NOTIFY which doesn't support it.
        pg_notify() is a regular SQL function that supports parameterized queries.
        """
        bus = PostgresEventBus(connection_string="postgresql://unused")
        bus._connected = True

        # Record what SQL gets executed
        executed_calls: list[tuple[str, tuple[Any, ...]]] = []

        class FakePublishConn:
            async def execute(self, query: str, *args: Any) -> None:
                executed_calls.append((query, args))

            def __bool__(self) -> bool:
                return True

        bus._conn = FakePublishConn()

        event = Event(
            type=EventType.CREATED,
            topic="test:topic",
            payload={"key": "value"},
        )
        await bus.publish("test:topic", event)

        assert len(executed_calls) == 1
        query, args = executed_calls[0]

        # Must use pg_notify function, not NOTIFY statement
        assert "pg_notify" in query, (
            f"Expected pg_notify() function call, got: {query}"
        )
        assert "$1" in query and "$2" in query, (
            f"Expected parameterized query with $1, $2, got: {query}"
        )

        # Channel and payload must be passed as parameters, not interpolated
        assert args[0] == "events_test_topic"  # channel
        payload_dict = json.loads(args[1])
        assert payload_dict["type"] == "created"
        assert payload_dict["topic"] == "test:topic"
        assert payload_dict["payload"] == {"key": "value"}


class TestListenerRegistration:
    """Tests verifying per-channel listener registration.

    Bug: The original code used add_listener("*", handler) in connect(),
    assuming "*" acts as a wildcard. In asyncpg, add_listener registers
    for a specific channel name — "*" only matches a literal "*" channel.
    Notifications on real channels never reached the handler.
    """

    @pytest.mark.asyncio
    async def test_subscribe_registers_listener_per_channel(self):
        """Verify subscribe registers an asyncpg listener for the specific channel."""
        bus = PostgresEventBus(connection_string="postgresql://unused")
        bus._connected = True

        registered_listeners: list[tuple[str, Any]] = []
        listen_calls: list[str] = []

        class FakeListenConn:
            async def execute(self, query: str, *args: Any) -> None:
                listen_calls.append(query)

            async def add_listener(self, channel: str, callback: Any) -> None:
                registered_listeners.append((channel, callback))

            def __bool__(self) -> bool:
                return True

        bus._listen_conn = FakeListenConn()

        async def handler(event: Event) -> None:
            pass

        await bus.subscribe("registry:bots", handler)

        # Should have issued LISTEN for the channel
        assert any("LISTEN" in call for call in listen_calls)

        # Should have registered a listener for the specific channel
        assert len(registered_listeners) == 1
        channel, callback = registered_listeners[0]
        assert channel == "events_registry_bots"
        assert callback == bus._notification_handler

    @pytest.mark.asyncio
    async def test_connect_does_not_register_wildcard_listener(self):
        """Verify connect() no longer registers a wildcard '*' listener.

        The old code did add_listener("*", handler) which doesn't work
        as a catch-all in asyncpg.
        """
        bus = PostgresEventBus(connection_string="postgresql://unused")

        registered_listeners: list[tuple[str, Any]] = []

        class _FakeListenConn:
            """Hand-rolled stand-in for an asyncpg connection.

            asyncpg cannot run in a unit test, so this fake stubs the
            connection methods ``connect()``/``close()`` touch. The
            methods are ``async`` to match asyncpg's real coroutine
            interface — a sync stub would silently mask a missing
            ``await`` in the code under test.
            """

            async def add_listener(self, ch: str, cb: Any) -> None:
                registered_listeners.append((ch, cb))

            async def remove_listener(self, ch: str, cb: Any) -> None:
                pass

            async def close(self) -> None:
                pass

        async def fake_connect(dsn: str) -> Any:
            return _FakeListenConn()

        # Patch asyncpg import
        import types

        fake_asyncpg = types.ModuleType("asyncpg")
        fake_asyncpg.connect = fake_connect  # type: ignore[attr-defined]

        import sys

        original = sys.modules.get("asyncpg")
        sys.modules["asyncpg"] = fake_asyncpg
        try:
            await bus.connect()

            # No wildcard listeners should be registered during connect
            wildcard_listeners = [
                (ch, cb) for ch, cb in registered_listeners if ch == "*"
            ]
            assert wildcard_listeners == [], (
                "connect() should not register a wildcard '*' listener"
            )
        finally:
            if original is not None:
                sys.modules["asyncpg"] = original
            else:
                del sys.modules["asyncpg"]
            # Clean up connection state
            bus._connected = False
            bus._conn = None
            bus._listen_conn = None


class TestNotificationHandlerParsesPayload:
    """Test that the notification handler correctly parses real JSON payloads.

    This verifies the end-to-end data flow: if pg_notify sends the real payload
    (not "$1"), the handler should parse it successfully.
    """

    @pytest.mark.asyncio
    async def test_handler_parses_valid_json_payload(self):
        """Verify notification handler correctly deserializes event JSON."""
        bus = PostgresEventBus(connection_string="postgresql://unused")
        bus._connected = True

        # Set up a channel mapping
        bus._channel_topics["events_test"] = "test"

        dispatched_events: list[Event] = []

        async def tracking_dispatch(topic: str, event: Event) -> None:
            dispatched_events.append(event)

        bus._dispatch_event = tracking_dispatch  # type: ignore[method-assign]

        event = Event(
            type=EventType.CREATED,
            topic="test",
            payload={"message": "hello"},
        )
        payload_json = json.dumps(event.to_dict())

        # Simulate receiving the notification
        bus._notification_handler(None, 0, "events_test", payload_json)

        # Give the asyncio task a chance to run
        await asyncio.sleep(0.01)

        assert len(dispatched_events) == 1
        assert dispatched_events[0].type == EventType.CREATED
        assert dispatched_events[0].payload == {"message": "hello"}

    @pytest.mark.asyncio
    async def test_handler_rejects_literal_dollar_one(self):
        """Reproduce the original bug: if payload is literal "$1", parsing fails.

        This is what happened with the old NOTIFY {channel}, $1 syntax —
        the payload "$1" was sent literally instead of the actual JSON.
        """
        bus = PostgresEventBus(connection_string="postgresql://unused")
        bus._connected = True
        bus._channel_topics["events_test"] = "test"

        dispatched_events: list[Event] = []

        async def tracking_dispatch(topic: str, event: Event) -> None:
            dispatched_events.append(event)

        bus._dispatch_event = tracking_dispatch  # type: ignore[method-assign]

        # This is what the old code sent: the literal string "$1"
        bus._notification_handler(None, 0, "events_test", "$1")

        await asyncio.sleep(0.01)

        # The handler should NOT dispatch — "$1" is not valid JSON
        assert len(dispatched_events) == 0


# ---------------------------------------------------------------------------
# Integration tests — require a real PostgreSQL instance
# ---------------------------------------------------------------------------

# Construct DSN from individual env vars (matches bin/run-integration-tests.sh)
PG_DSN = "postgresql://{}:{}@{}:{}/{}".format(
    os.getenv("POSTGRES_USER", "postgres"),
    os.getenv("POSTGRES_PASSWORD", "postgres"),
    os.getenv("POSTGRES_HOST", "localhost"),
    os.getenv("POSTGRES_PORT", "5432"),
    os.getenv("POSTGRES_DB", "dataknobs"),
)

TEST_POSTGRES = os.getenv("TEST_POSTGRES", "").lower() != "false"

skip_postgres = pytest.mark.skipif(
    not TEST_POSTGRES or not is_postgres_available(),
    reason="PostgreSQL integration tests skipped. Set TEST_POSTGRES=true and ensure Postgres is running.",
)


@skip_postgres
class TestPostgresEventBusIntegration:
    """Integration tests exercising real PostgreSQL LISTEN/NOTIFY.

    These require a running Postgres instance. Run via:
        TEST_POSTGRES=true uv run pytest tests/test_postgres_events.py -k Integration
    Or via the full integration test runner:
        bin/test.sh common
    """

    @pytest.mark.asyncio
    async def test_publish_subscribe_roundtrip(self):
        """Subscribe, publish, verify handler fires with correct event."""
        bus = PostgresEventBus(connection_string=PG_DSN)
        await bus.connect()
        try:
            received: list[Event] = []

            async def handler(event: Event) -> None:
                received.append(event)

            await bus.subscribe("test:roundtrip", handler)

            event = Event(
                type=EventType.CREATED,
                topic="test:roundtrip",
                payload={"key": "value"},
            )
            await bus.publish("test:roundtrip", event)

            # Wait for async notification delivery
            for _ in range(50):
                if received:
                    break
                await asyncio.sleep(0.05)

            assert len(received) == 1
            assert received[0].type == EventType.CREATED
            assert received[0].payload == {"key": "value"}
        finally:
            await bus.close()

    @pytest.mark.asyncio
    async def test_unsubscribe_stops_delivery(self):
        """After unsubscribing, events are no longer delivered."""
        bus = PostgresEventBus(connection_string=PG_DSN)
        await bus.connect()
        try:
            received: list[Event] = []

            async def handler(event: Event) -> None:
                received.append(event)

            sub = await bus.subscribe("test:unsub", handler)
            await sub.cancel()

            await bus.publish(
                "test:unsub",
                Event(type=EventType.CREATED, topic="test:unsub", payload={}),
            )
            await asyncio.sleep(0.5)

            assert len(received) == 0
        finally:
            await bus.close()

    @pytest.mark.asyncio
    async def test_multiple_subscribers_same_topic(self):
        """Two handlers on the same topic both receive the event."""
        bus = PostgresEventBus(connection_string=PG_DSN)
        await bus.connect()
        try:
            received_a: list[Event] = []
            received_b: list[Event] = []

            async def handler_a(event: Event) -> None:
                received_a.append(event)

            async def handler_b(event: Event) -> None:
                received_b.append(event)

            await bus.subscribe("test:multi", handler_a)
            await bus.subscribe("test:multi", handler_b)

            await bus.publish(
                "test:multi",
                Event(type=EventType.CREATED, topic="test:multi", payload={"n": 1}),
            )

            for _ in range(50):
                if received_a and received_b:
                    break
                await asyncio.sleep(0.05)

            assert len(received_a) == 1
            assert len(received_b) == 1
        finally:
            await bus.close()

    @pytest.mark.asyncio
    async def test_listen_connection_reconnects_after_drop(self):
        """A dropped LISTEN connection must be re-established (P2).

        Reproduce-first against a real server (no fakes — a
        sync/async-mismatched fake would hide exactly the missing-await
        class this path touches): before this fix nothing detected or
        recovered a dropped ``_listen_conn``; the callback simply stopped
        firing and the bus silently stopped delivering events. Now the
        supervised watchdog re-opens the connection and re-registers
        every active channel, so delivery resumes.
        """
        bus = PostgresEventBus(connection_string=PG_DSN)
        await bus.connect()
        try:
            # The push-based bus has a supervisory watchdog (public
            # health signal — no reach into the private _listen_task).
            assert bus.is_listening

            received: list[Event] = []

            async def handler(event: Event) -> None:
                received.append(event)

            await bus.subscribe("test:reconnect", handler)

            # Forcibly drop the dedicated LISTEN connection.
            original = bus._listen_conn
            assert original is not None
            await original.close()
            assert original.is_closed()

            # The watchdog detects the dead connection (next liveness
            # poll), re-opens it, and re-registers the channel. Swap of
            # self._listen_conn happens only after re-LISTEN succeeds.
            # Observing the physical connection-object swap is the
            # reproduce mechanism itself — there is no public signal for
            # "the connection instance changed", so the internal access
            # here is deliberate, not incidental coupling. The behavior
            # is independently re-proven below by delivery resuming.
            # 25s bound = generous headroom over the worst real-server
            # case: up to one _LISTEN_POLL_INTERVAL (2s) before the drop
            # is detected + up to _LISTEN_RECONNECT_TIMEOUT (10s) for the
            # rebuild. Not a tight assertion — just keeps a wedged
            # reconnect from hanging the suite.
            deadline = asyncio.get_event_loop().time() + 25.0
            while asyncio.get_event_loop().time() < deadline:
                conn = bus._listen_conn
                if conn is not None and conn is not original and not conn.is_closed():
                    break
                await asyncio.sleep(0.2)
            assert (
                bus._listen_conn is not None
                and bus._listen_conn is not original
                and not bus._listen_conn.is_closed()
            ), "LISTEN connection was not re-established after the drop"

            # Delivery resumes on the rebuilt + re-registered connection.
            await bus.publish(
                "test:reconnect",
                Event(
                    type=EventType.UPDATED,
                    topic="test:reconnect",
                    payload={"resumed": True},
                ),
            )
            for _ in range(100):
                if received:
                    break
                await asyncio.sleep(0.05)

            assert len(received) == 1, "delivery did not resume after reconnect"
            assert received[0].payload == {"resumed": True}
        finally:
            await bus.close()

    @pytest.mark.asyncio
    async def test_close_then_publish_raises(self):
        """After close(), publish raises RuntimeError."""
        bus = PostgresEventBus(connection_string=PG_DSN)
        await bus.connect()
        await bus.close()

        with pytest.raises(RuntimeError, match="not connected"):
            await bus.publish(
                "test:closed",
                Event(type=EventType.CREATED, topic="test:closed", payload={}),
            )
