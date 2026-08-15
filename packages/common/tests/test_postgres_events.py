"""Tests for PostgresEventBus.

Unit tests cover channel name sanitization and SQL construction.
Integration tests (gated by TEST_POSTGRES env var) cover actual pub/sub
with a real PostgreSQL instance.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

import pytest

from dataknobs_common.events import Event, EventType, PostgresEventBusConfig, Subscription
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
        channel = bus._topic_to_channel('it\'s a "test"')
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
        assert bus.config.connection_string == "postgresql://u:p@h:5433/db"

    def test_accepts_database_url_env_fallback(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@env-h/env-db")
        bus = PostgresEventBus(config={})
        assert bus.config.connection_string == "postgresql://u:p@env-h/env-db"

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
        assert "pg_notify" in query, f"Expected pg_notify() function call, got: {query}"
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
            wildcard_listeners = [(ch, cb) for ch, cb in registered_listeners if ch == "*"]
            assert wildcard_listeners == [], "connect() should not register a wildcard '*' listener"
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


class TestDispatchTasksAreRetainedAndDrained:
    """Bug: a dispatched event could silently never happen.

    ``_notification_handler`` fired ``asyncio.create_task(...)`` and threw the
    handle away. asyncio keeps only a *weak* reference to a bare task, so a
    dispatch could be garbage-collected mid-flight, and ``close()`` — which
    knew nothing about these tasks — tore the bus down out from under any that
    were still running. An event the bus accepted and reported dispatching
    could reach no handler at all.

    These drive the real ``_notification_handler``, ``_dispatch_event`` and
    ``close()``. No connection is stubbed because this path needs none: every
    connection attribute is None-guarded, so the only setup is the private
    state ``connect()``/``subscribe()`` would have written. The end-to-end
    path over a real server is covered by the integration test of the same
    name below.
    """

    def _armed_bus(self, handler: Any) -> PostgresEventBus:
        """A bus in the state connect() + subscribe() would have left it."""
        bus = PostgresEventBus(connection_string="postgresql://unused")
        bus._connected = True
        bus._channel_topics["events_test"] = "test"
        bus._topic_channels["test"] = "events_test"
        bus._subscriptions["sub-1"] = Subscription(
            subscription_id="sub-1",
            topic="test",
            handler=handler,
            pattern=None,
            _cancel_callback=bus._unsubscribe,
        )
        return bus

    @staticmethod
    def _payload() -> str:
        return json.dumps(Event(type=EventType.CREATED, topic="test", payload={"n": 1}).to_dict())

    @pytest.mark.asyncio
    async def test_close_waits_for_an_in_flight_dispatch(self):
        """close() must not drop a dispatch that is already running."""
        completed: list[Event] = []

        async def slow_handler(event: Event) -> None:
            await asyncio.sleep(0.05)
            completed.append(event)

        bus = self._armed_bus(slow_handler)
        bus._notification_handler(None, 0, "events_test", self._payload())

        # Timed, because the two ways to break this arrangement fail
        # differently: clearing _subscriptions before the drain leaves
        # `completed` empty and fails the assertion, but moving the drain
        # inside the lock deadlocks _dispatch_event against it and would
        # otherwise hang the suite with nothing reporting why.
        await asyncio.wait_for(bus.close(), timeout=5.0)

        assert completed, "close() returned while a dispatch was still in flight"

    @pytest.mark.asyncio
    async def test_the_dispatch_task_is_retained_while_it_runs(self):
        """Structural: the handle is held, so the loop's weak reference is not the only one.

        This asserts the mechanism that makes mid-flight collection
        impossible, not the collection itself -- a garbage-collection race is
        not deterministically reproducible, which is why the test above
        targets the half that is.
        """
        started = asyncio.Event()

        async def blocking_handler(event: Event) -> None:
            started.set()
            await asyncio.sleep(0.05)

        bus = self._armed_bus(blocking_handler)
        bus._notification_handler(None, 0, "events_test", self._payload())

        await asyncio.wait_for(started.wait(), timeout=1.0)
        assert bus._dispatch_tasks, "the dispatch task is referenced by nothing but the loop"

        await bus.close()
        assert not bus._dispatch_tasks, "finished tasks are not discarded from the set"


class TestCloseIsSafeWhileItsOwnTeardownIsObservable:
    """Splitting close() into two locked sections around the drain is what
    makes the drain possible, and it introduced two hazards of its own.

    Both are about the window between the sections -- a window that did not
    exist when close() held the lock from end to end, and whose length is
    bounded only by the slowest subscriber handler.

    Every assertion here carries a timeout. The regressions these guard
    against do not raise; they *hang*, and an unbounded hang in a suite is a
    worse failure than a red test because nothing reports it.
    """

    def _armed_bus(self, handler: Any) -> PostgresEventBus:
        """A bus in the state connect() + subscribe() would have left it."""
        bus = PostgresEventBus(connection_string="postgresql://unused")
        bus._connected = True
        bus._channel_topics["events_test"] = "test"
        bus._topic_channels["test"] = "events_test"
        bus._subscriptions["sub-1"] = Subscription(
            subscription_id="sub-1",
            topic="test",
            handler=handler,
            pattern=None,
            _cancel_callback=bus._unsubscribe,
        )
        return bus

    @staticmethod
    def _payload() -> str:
        return json.dumps(Event(type=EventType.CREATED, topic="test", payload={"n": 1}).to_dict())

    @pytest.mark.asyncio
    async def test_a_handler_may_close_the_bus_it_is_running_under(self):
        """close() must not wait on the dispatch task that is calling it.

        Bug: the drain awaited every task in ``_dispatch_tasks``, and a
        handler runs *inside* one of them. A subscriber that tears the bus
        down -- a shutdown topic, or a test closing from a handler -- put
        close() into ``gather`` on the task awaiting that same gather.

        asyncio raises ``Task cannot await on itself`` only for a direct
        ``await task``; routed through ``gather`` it is an unbounded hang
        with no exception and no log. Hence the timeout: unfixed, this test
        does not fail on an assertion, it stops.
        """
        closed = asyncio.Event()

        async def closing_handler(event: Event) -> None:
            await bus.close()
            closed.set()

        bus = self._armed_bus(closing_handler)
        bus._notification_handler(None, 0, "events_test", self._payload())

        await asyncio.wait_for(closed.wait(), timeout=2.0)
        assert not bus._connected, "close() returned without finishing its teardown"

    @pytest.mark.asyncio
    async def test_the_bus_reports_itself_closed_before_it_begins_draining(self):
        """The drain window must not look like an open bus.

        Bug: ``_connected`` was cleared only in the final locked section,
        after the drain. ``subscribe()`` reads ``_connected`` and
        ``_listen_conn`` at its first line, before taking the lock, so
        throughout the drain it saw an open bus, acquired the lock in the
        gap, ran LISTEN against a connection that section was about to
        close, and wrote into the very dicts it then cleared -- handing back
        a live-looking Subscription that would never deliver anything.
        Before the split the same call blocked on the lock for the whole of
        close() and then failed loudly.

        Both sibling backends clear their stop flag as the first statement
        inside the lock (``SqsEventBus`` ``_running``, ``RedisEventBus``
        ``_running``); this one borrowed the task-retention shape from
        ``sqs.py`` without its guard shape.

        The ``_connected`` assertion is the reproduction -- it is the flag
        ``subscribe()`` actually reads. The ``subscribe()`` call below
        corroborates the consumer-visible half but cannot stand alone: this
        bus has no listen connection, so unfixed code refuses it for the
        second half of the same guard.
        """
        in_handler = asyncio.Event()
        release = asyncio.Event()

        async def slow_handler(event: Event) -> None:
            in_handler.set()
            await release.wait()

        bus = self._armed_bus(slow_handler)
        bus._notification_handler(None, 0, "events_test", self._payload())
        await asyncio.wait_for(in_handler.wait(), timeout=1.0)

        closing = asyncio.create_task(bus.close())
        try:
            # The first locked section holds no awaits for a bus with no
            # listen task and no listen connection, so one scheduling turn
            # puts close() in the drain, where it stays until the handler is
            # released.
            await asyncio.sleep(0.01)
            assert not closing.done(), "close() finished without waiting for the dispatch"

            assert not bus._connected, "the bus reports itself open while it is tearing down"
            with pytest.raises(RuntimeError):
                await bus.subscribe("other", slow_handler)
        finally:
            release.set()
            await asyncio.wait_for(closing, timeout=2.0)

    @pytest.mark.asyncio
    async def test_a_dispatch_created_during_the_drain_is_also_awaited(self):
        """The drain runs to a fixpoint, not from a single snapshot.

        Bug: ``in_flight`` was taken once. asyncpg delivers notifications
        through ``loop.call_soon``, so a callback scheduled before
        ``remove_listener`` completed can run *after* that snapshot -- and
        the task it creates is then never awaited, blocks on the lock the
        final section holds, and dispatches against subscriptions that
        section has cleared. That is the lost event this whole arrangement
        exists to prevent, reached by a later route.

        The second notification here stands in for that late callback; it is
        delivered through the real ``_notification_handler``, which is the
        same entry point asyncpg uses.
        """
        calls = 0
        first_release = asyncio.Event()
        second_started = asyncio.Event()
        second_release = asyncio.Event()
        completed: list[Event] = []

        async def handler(event: Event) -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                await first_release.wait()
            else:
                second_started.set()
                await second_release.wait()
            completed.append(event)

        bus = self._armed_bus(handler)
        bus._notification_handler(None, 0, "events_test", self._payload())

        closing = asyncio.create_task(bus.close())
        try:
            await asyncio.sleep(0.01)
            assert not closing.done(), "close() finished without waiting for the first dispatch"

            # The late delivery, arriving while close() is in the drain.
            bus._notification_handler(None, 0, "events_test", self._payload())
            await asyncio.wait_for(second_started.wait(), timeout=1.0)

            first_release.set()
            await asyncio.sleep(0.02)
            assert not closing.done(), (
                "close() returned while a dispatch created during the drain was still running"
            )
        finally:
            first_release.set()
            second_release.set()
            await asyncio.wait_for(closing, timeout=2.0)

        assert len(completed) == 2, "the late dispatch was abandoned rather than drained"

    @pytest.mark.asyncio
    async def test_a_dispatch_failing_outside_its_handler_is_logged(self, caplog):
        """The drain must not be where an exception goes to disappear.

        ``gather(..., return_exceptions=True)``'s results were discarded with
        a comment claiming they were "already logged per handler". That holds
        only for exceptions raised *by a handler*, which ``_dispatch_event``
        catches itself. Anything ``_dispatch_event`` raises on its own --
        acquiring the lock, matching a malformed pattern -- reached nobody,
        which is worse than before the drain existed: an un-retrieved task
        exception at least produced asyncio's "Task exception was never
        retrieved" on the console.

        ``_dispatch_event`` is replaced rather than provoked because the
        property under test belongs to the drain, not to any particular way
        of reaching it -- the same substitution this file already uses to
        pin the payload-parsing path.
        """

        async def exploding_dispatch(topic: str, event: Event) -> None:
            raise RuntimeError("dispatch machinery failed")

        bus = self._armed_bus(lambda event: None)
        bus._dispatch_event = exploding_dispatch  # type: ignore[method-assign]
        bus._notification_handler(None, 0, "events_test", self._payload())

        with caplog.at_level(logging.ERROR, logger="dataknobs_common.events.postgres"):
            await asyncio.wait_for(bus.close(), timeout=5.0)

        assert "dispatch machinery failed" in caplog.text, (
            "an exception raised by _dispatch_event itself was swallowed by the drain"
        )


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
    async def test_close_waits_for_an_in_flight_dispatch(self):
        """close() must not drop a dispatch already running, over a real server.

        The unit test of the same name drives ``_notification_handler``
        directly; this drives the whole path -- real LISTEN/NOTIFY, real
        asyncpg callback, real handler -- so the drain is proven where the
        notification actually originates rather than where it is simulated.
        """
        bus = PostgresEventBus(connection_string=PG_DSN)
        await bus.connect()
        closed = False
        try:
            started = asyncio.Event()
            completed: list[Event] = []

            async def slow_handler(event: Event) -> None:
                started.set()
                await asyncio.sleep(0.2)
                completed.append(event)

            await bus.subscribe("test:drain", slow_handler)
            await bus.publish(
                "test:drain",
                Event(type=EventType.CREATED, topic="test:drain", payload={"n": 1}),
            )

            # Close only once the handler is demonstrably mid-flight; closing
            # before delivery would prove nothing about the drain.
            await asyncio.wait_for(started.wait(), timeout=10.0)
            assert not completed, "handler finished before close() was called"

            await bus.close()
            closed = True

            assert completed, "close() returned while a dispatch was still in flight"
        finally:
            if not closed:
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
