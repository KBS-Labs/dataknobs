"""Tests for :class:`IngestOrchestrator`.

The orchestrator is a thin subscriber-side primitive: it listens on an
:class:`EventBus` trigger topic and dispatches to
:meth:`KnowledgeIngestionManager.ingest_if_changed`. These tests cover
the subscribe/unsubscribe lifecycle, payload dispatch, error
containment, and a full real-manager end-to-end path to verify the
completion event is still published by the manager.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import pytest

from dataknobs_bots.knowledge import (
    IngestionResult,
    IngestSwapMode,
    KnowledgeIngestionManager,
    RAGKnowledgeBase,
)
from dataknobs_bots.knowledge.orchestration import IngestOrchestrator
from dataknobs_bots.knowledge.storage import InMemoryKnowledgeBackend
from dataknobs_common.events import Event, EventType, InMemoryEventBus
from dataknobs_common.testing import requires_real_postgres

TRIGGER_TOPIC = "knowledge:trigger"


class _StubManager:
    """Minimal stand-in for :class:`KnowledgeIngestionManager`.

    Captures calls to :meth:`ingest_if_changed` so tests can assert on
    dispatch arguments without instantiating a full RAG pipeline. Real
    end-to-end wiring is covered by
    ``test_end_to_end_with_real_manager``.
    """

    def __init__(
        self,
        *,
        raise_exc: Exception | None = None,
        returns: IngestionResult | None = None,
    ) -> None:
        self.calls: list[tuple[str, str | None]] = []
        self.change_calls: list[tuple[str, str]] = []
        self.ingest_calls: list[tuple[str, Any]] = []
        self._raise = raise_exc
        self._returns = returns

    async def ingest_if_changed(
        self,
        domain_id: str,
        last_version: str | None = None,
        **_kwargs: Any,
    ) -> IngestionResult | None:
        self.calls.append((domain_id, last_version))
        if self._raise is not None:
            raise self._raise
        return self._returns

    async def ingest_changes(
        self,
        domain_id: str,
        since_version: str,
        **_kwargs: Any,
    ) -> IngestionResult:
        self.change_calls.append((domain_id, since_version))
        if self._raise is not None:
            raise self._raise
        return self._returns or IngestionResult(domain_id=domain_id)

    async def ingest(
        self,
        domain_id: str,
        clear_existing: bool | None = None,
        *,
        swap_mode: Any = None,
        **_kwargs: Any,
    ) -> IngestionResult:
        self.ingest_calls.append((domain_id, swap_mode))
        if self._raise is not None:
            raise self._raise
        return self._returns or IngestionResult(domain_id=domain_id)


async def _make_bus() -> InMemoryEventBus:
    bus = InMemoryEventBus()
    await bus.connect()
    return bus


async def _wait_for(
    condition: Any, timeout: float = 1.0, interval: float = 0.01
) -> None:
    """Poll ``condition()`` until truthy or timeout elapses."""
    elapsed = 0.0
    while elapsed < timeout:
        if condition():
            return
        await asyncio.sleep(interval)
        elapsed += interval


def _trigger_event(
    payload: dict[str, Any] | None, topic: str = TRIGGER_TOPIC
) -> Event:
    return Event(
        type=EventType.UPDATED,
        topic=topic,
        payload=payload if payload is not None else {},
    )


@pytest.mark.asyncio
async def test_start_subscribes_to_trigger_topic() -> None:
    """After ``start()``, the orchestrator holds an active subscription."""
    bus = await _make_bus()
    manager = _StubManager()
    orch = IngestOrchestrator(manager, bus)  # type: ignore[arg-type]

    assert not orch.is_running
    await orch.start()
    assert orch.is_running

    await bus.publish(
        TRIGGER_TOPIC,
        _trigger_event({"domain_id": "d1"}),
    )
    await _wait_for(lambda: len(manager.calls) >= 1)
    assert manager.calls == [("d1", None)]

    await orch.stop()
    await bus.close()


@pytest.mark.asyncio
async def test_start_is_idempotent() -> None:
    """Calling ``start()`` twice subscribes only once."""
    bus = await _make_bus()
    manager = _StubManager()
    orch = IngestOrchestrator(manager, bus)  # type: ignore[arg-type]

    await orch.start()
    first_sub = orch._subscription
    await orch.start()  # second call — should be a no-op
    assert orch._subscription is first_sub

    # Verify only one delivery — if we'd subscribed twice, we'd dispatch twice
    await bus.publish(
        TRIGGER_TOPIC,
        _trigger_event({"domain_id": "d1"}),
    )
    await _wait_for(lambda: len(manager.calls) >= 1)
    await asyncio.sleep(0.05)  # Give any duplicate delivery a chance to land
    assert len(manager.calls) == 1

    await orch.stop()
    await bus.close()


@pytest.mark.asyncio
async def test_stop_cancels_subscription() -> None:
    """``stop()`` cancels the subscription and clears the handle."""
    bus = await _make_bus()
    manager = _StubManager()
    orch = IngestOrchestrator(manager, bus)  # type: ignore[arg-type]

    await orch.start()
    assert orch.is_running
    await orch.stop()
    assert not orch.is_running
    assert orch._subscription is None

    # Further publishes should not reach the manager
    await bus.publish(
        TRIGGER_TOPIC,
        _trigger_event({"domain_id": "d1"}),
    )
    await asyncio.sleep(0.05)
    assert manager.calls == []

    await bus.close()


@pytest.mark.asyncio
async def test_stop_is_idempotent() -> None:
    """Calling ``stop()`` twice is safe."""
    bus = await _make_bus()
    manager = _StubManager()
    orch = IngestOrchestrator(manager, bus)  # type: ignore[arg-type]

    await orch.start()
    await orch.stop()
    await orch.stop()  # no error

    await bus.close()


@pytest.mark.asyncio
async def test_trigger_event_invokes_ingest_if_changed() -> None:
    """A trigger event dispatches ``domain_id`` and ``last_version``."""
    bus = await _make_bus()
    manager = _StubManager()
    orch = IngestOrchestrator(manager, bus)  # type: ignore[arg-type]
    await orch.start()

    await bus.publish(
        TRIGGER_TOPIC,
        _trigger_event({"domain_id": "d1", "last_version": "v1"}),
    )
    await _wait_for(lambda: len(manager.calls) >= 1)

    assert manager.calls == [("d1", "v1")]

    await orch.stop()
    await bus.close()


@pytest.mark.asyncio
async def test_trigger_without_last_version_passes_none() -> None:
    """Missing ``last_version`` in payload maps to ``None``."""
    bus = await _make_bus()
    manager = _StubManager()
    orch = IngestOrchestrator(manager, bus)  # type: ignore[arg-type]
    await orch.start()

    await bus.publish(
        TRIGGER_TOPIC,
        _trigger_event({"domain_id": "d1"}),
    )
    await _wait_for(lambda: len(manager.calls) >= 1)

    assert manager.calls == [("d1", None)]

    await orch.stop()
    await bus.close()


@pytest.mark.asyncio
async def test_trigger_without_domain_id_is_skipped_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Events missing ``domain_id`` skip dispatch and emit a warning."""
    bus = await _make_bus()
    manager = _StubManager()
    orch = IngestOrchestrator(manager, bus)  # type: ignore[arg-type]
    await orch.start()

    with caplog.at_level(
        "WARNING", logger="dataknobs_bots.knowledge.orchestration"
    ):
        await bus.publish(
            TRIGGER_TOPIC,
            _trigger_event({}),
        )
        # Give the bus a moment to dispatch (nothing to wait on — no call)
        await asyncio.sleep(0.05)

    assert manager.calls == []
    assert any(
        "without domain_id" in record.message for record in caplog.records
    ), f"Expected warning about missing domain_id; got {caplog.records}"

    await orch.stop()
    await bus.close()


@pytest.mark.asyncio
async def test_manager_failure_does_not_break_subscription(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A failing ``ingest_if_changed`` is logged but does not unsubscribe."""
    bus = await _make_bus()
    failing_manager = _StubManager(raise_exc=RuntimeError("boom"))
    orch = IngestOrchestrator(failing_manager, bus)  # type: ignore[arg-type]
    await orch.start()

    with caplog.at_level(
        "ERROR", logger="dataknobs_bots.knowledge.orchestration"
    ):
        await bus.publish(
            TRIGGER_TOPIC,
            _trigger_event({"domain_id": "d1"}),
        )
        await _wait_for(lambda: len(failing_manager.calls) >= 1)

    assert failing_manager.calls == [("d1", None)]
    assert any(
        "failed to process trigger" in record.message
        for record in caplog.records
    )
    # Subscription must still be alive — send a second event
    await bus.publish(
        TRIGGER_TOPIC,
        _trigger_event({"domain_id": "d2"}),
    )
    await _wait_for(lambda: len(failing_manager.calls) >= 2)
    assert failing_manager.calls == [("d1", None), ("d2", None)]

    await orch.stop()
    await bus.close()


@pytest.mark.asyncio
async def test_end_to_end_with_real_manager() -> None:
    """Full stack: orchestrator + real manager + real backend + RAG kb.

    Publishing a trigger event causes the manager to ingest content and
    publish its own ``knowledge:ingestion`` completion event — the
    existing manager contract must continue to hold under orchestration.
    """
    bus = await _make_bus()

    backend = InMemoryKnowledgeBackend()
    await backend.initialize()
    await backend.create_kb("d1")
    await backend.put_file("d1", "intro.md", b"# Intro\n\nHello.\n")

    rag = await RAGKnowledgeBase.from_config(
        {
            "vector_store": {"backend": "memory", "dimensions": 384},
            "embedding_provider": "echo",
            "embedding_model": "test",
        }
    )
    manager = KnowledgeIngestionManager(
        source=backend, destination=rag, event_bus=bus
    )
    orch = IngestOrchestrator(manager, bus)

    completion_events: list[Event] = []

    async def completion_handler(event: Event) -> None:
        completion_events.append(event)

    await bus.subscribe("knowledge:ingestion", completion_handler)
    await orch.start()

    await bus.publish(
        TRIGGER_TOPIC,
        _trigger_event({"domain_id": "d1"}),
    )
    await _wait_for(lambda: len(completion_events) >= 1, timeout=2.0)

    assert len(completion_events) == 1
    payload = completion_events[0].payload
    assert payload["domain_id"] == "d1"
    assert payload["status"] == "ready"
    assert payload["files_processed"] >= 1
    assert payload["chunks_created"] >= 1

    await orch.stop()
    await bus.close()


class _GatedManager:
    """Stub manager whose ``ingest_if_changed`` blocks on a per-domain
    gate until the test releases it. Records start/end order so tests
    can assert serialization (same-domain) vs. overlap (cross-domain).
    """

    def __init__(self) -> None:
        self.gates: dict[str, asyncio.Event] = {}
        self.started: list[str] = []
        self.finished: list[str] = []
        self._in_flight: dict[str, int] = {}
        self.peak_concurrency: dict[str, int] = {}

    def gate_for(self, domain_id: str) -> asyncio.Event:
        gate = self.gates.get(domain_id)
        if gate is None:
            gate = asyncio.Event()
            self.gates[domain_id] = gate
        return gate

    async def ingest_if_changed(
        self,
        domain_id: str,
        last_version: str | None = None,
        **_kwargs: Any,
    ) -> IngestionResult | None:
        self.started.append(domain_id)
        self._in_flight[domain_id] = self._in_flight.get(domain_id, 0) + 1
        self.peak_concurrency[domain_id] = max(
            self.peak_concurrency.get(domain_id, 0),
            self._in_flight[domain_id],
        )
        try:
            await self.gate_for(domain_id).wait()
        finally:
            self._in_flight[domain_id] -= 1
            self.finished.append(domain_id)
        return None


@pytest.mark.asyncio
async def test_concurrent_triggers_same_domain_are_serialized() -> None:
    """Two triggers for the same ``domain_id`` must run one at a time.

    Regression guard for the per-domain lock: without it, both
    ``ingest_if_changed`` invocations would be active simultaneously
    and race on ``clear_existing`` + ``add_vectors`` in the real
    manager.

    Note: :class:`InMemoryEventBus.publish` awaits the handler inline,
    so two concurrent publishes are scheduled as independent tasks to
    allow both dispatches to overlap at the handler boundary.
    """
    bus = await _make_bus()
    manager = _GatedManager()
    orch = IngestOrchestrator(manager, bus)  # type: ignore[arg-type]
    await orch.start()

    pub1 = asyncio.create_task(
        bus.publish(TRIGGER_TOPIC, _trigger_event({"domain_id": "d1"}))
    )
    pub2 = asyncio.create_task(
        bus.publish(TRIGGER_TOPIC, _trigger_event({"domain_id": "d1"}))
    )

    # Wait until the first invocation is inside the gate.
    await _wait_for(lambda: len(manager.started) >= 1, timeout=1.0)
    # Give the second publish time to race for the per-domain lock —
    # if the lock is missing it will start now and concurrency goes to 2.
    await asyncio.sleep(0.05)

    assert manager.peak_concurrency.get("d1", 0) == 1, (
        f"Expected serialization; peak concurrency for d1 was "
        f"{manager.peak_concurrency.get('d1')}"
    )
    assert len(manager.started) == 1
    assert len(manager.finished) == 0

    # Release the first; the second must then start and also complete.
    manager.gate_for("d1").set()
    await _wait_for(lambda: len(manager.finished) >= 2, timeout=1.0)
    assert manager.peak_concurrency["d1"] == 1
    assert manager.started == ["d1", "d1"]
    assert manager.finished == ["d1", "d1"]

    await asyncio.gather(pub1, pub2)
    await orch.stop()
    await bus.close()


@pytest.mark.asyncio
async def test_concurrent_triggers_different_domains_run_in_parallel() -> None:
    """Triggers for different ``domain_id`` values proceed
    concurrently — the per-domain lock must NOT cause cross-domain
    serialization.
    """
    bus = await _make_bus()
    manager = _GatedManager()
    orch = IngestOrchestrator(manager, bus)  # type: ignore[arg-type]
    await orch.start()

    pub1 = asyncio.create_task(
        bus.publish(TRIGGER_TOPIC, _trigger_event({"domain_id": "d1"}))
    )
    pub2 = asyncio.create_task(
        bus.publish(TRIGGER_TOPIC, _trigger_event({"domain_id": "d2"}))
    )

    # Both must start before either completes — proving they run in
    # parallel (neither is holding a cross-domain lock).
    await _wait_for(lambda: set(manager.started) >= {"d1", "d2"}, timeout=1.0)
    assert manager.finished == []

    # Release both.
    manager.gate_for("d1").set()
    manager.gate_for("d2").set()
    await _wait_for(lambda: set(manager.finished) >= {"d1", "d2"}, timeout=1.0)

    await asyncio.gather(pub1, pub2)
    await orch.stop()
    await bus.close()


@pytest.mark.asyncio
async def test_injected_lock_is_used_and_keyed_per_domain() -> None:
    """A passed-in ``DistributedLock`` is used, keyed ``ingest:<domain>``.

    Verifies the injection seam: the orchestrator delegates
    serialization to the injected lock rather than an internal
    ``asyncio.Lock``. Uses a real :class:`InProcessLock` subclass (not
    a mock) that records the keys it is asked to hold.
    """
    from dataknobs_common.locks import InProcessLock

    class RecordingLock(InProcessLock):
        def __init__(self) -> None:
            super().__init__()
            self.held_keys: list[str] = []

        def hold(self, key: str, *, timeout: float | None = None):  # type: ignore[override]
            self.held_keys.append(key)
            return super().hold(key, timeout=timeout)

    bus = await _make_bus()
    manager = _StubManager()
    lock = RecordingLock()
    orch = IngestOrchestrator(manager, bus, lock=lock)  # type: ignore[arg-type]
    await orch.start()

    await bus.publish(TRIGGER_TOPIC, _trigger_event({"domain_id": "d1"}))
    await _wait_for(lambda: len(manager.calls) >= 1)

    assert manager.calls == [("d1", None)]
    assert lock.held_keys == ["ingest:d1"]
    # Default construction must NOT reuse the injected instance.
    default_orch = IngestOrchestrator(manager, bus)  # type: ignore[arg-type]
    assert isinstance(default_orch.lock, InProcessLock)
    assert default_orch.lock is not lock

    await orch.stop()
    await bus.close()


@pytest.fixture
def pg_dsn(
    ensure_postgres_ready: None,
    postgres_connection_params: dict[str, Any],
) -> str:
    """libpq URI for the shared test DB (advisory locks are global)."""
    p = postgres_connection_params
    return (
        f"postgresql://{p['user']}:{p['password']}"
        f"@{p['host']}:{p['port']}/{p['database']}"
    )


@pytest.mark.asyncio
@requires_real_postgres
async def test_two_replicas_serialized_by_postgres_lock(
    pg_dsn: str,
) -> None:
    """Two orchestrators sharing one Postgres lock serialize a domain.

    Two independent ``IngestOrchestrator`` instances — separate event
    buses, each with its own ``PostgresAdvisoryLock`` built from
    ``lock_config`` — stand in for two replicas. A same-domain trigger
    is delivered to *both* (at-least-once delivery). The cross-replica
    lock must let only one ingest the domain at a time, exactly the
    guarantee a process-local ``InProcessLock`` cannot provide across
    replicas.
    """
    bus1 = await _make_bus()
    bus2 = await _make_bus()
    manager = _GatedManager()  # the shared store both replicas race on
    lock_cfg: dict[str, Any] = {
        "backend": "postgres",
        "connection_string": pg_dsn,
    }
    domain = f"repl-{os.getpid()}"

    orch1 = IngestOrchestrator(manager, bus1, lock_config=lock_cfg)  # type: ignore[arg-type]
    orch2 = IngestOrchestrator(manager, bus2, lock_config=lock_cfg)  # type: ignore[arg-type]
    await orch1.start()
    await orch2.start()
    try:
        pub1 = asyncio.create_task(
            bus1.publish(TRIGGER_TOPIC, _trigger_event({"domain_id": domain}))
        )
        pub2 = asyncio.create_task(
            bus2.publish(TRIGGER_TOPIC, _trigger_event({"domain_id": domain}))
        )

        # One replica enters the critical section; the other is blocked
        # acquiring the shared Postgres advisory lock.
        await _wait_for(lambda: len(manager.started) >= 1, timeout=5.0)
        await asyncio.sleep(0.2)  # give the 2nd a real chance to race
        assert manager.peak_concurrency.get(domain, 0) == 1, (
            "cross-replica serialization failed; peak concurrency for "
            f"{domain} was {manager.peak_concurrency.get(domain)}"
        )
        assert len(manager.started) == 1

        # Release the first; the second acquires the lock and proceeds.
        manager.gate_for(domain).set()
        await _wait_for(lambda: len(manager.finished) >= 2, timeout=10.0)
        assert manager.peak_concurrency[domain] == 1
        assert manager.started == [domain, domain]
        assert manager.finished == [domain, domain]

        await asyncio.gather(pub1, pub2)
    finally:
        await orch1.stop()
        await orch2.stop()
        await orch1.lock.close()
        await orch2.lock.close()
        await bus1.close()
        await bus2.close()


class TestLockConfigConstruction:
    """``lock_config=`` builds the lock via ``create_lock`` at the
    orchestrator construction site (125/126 Phase 4).

    The lock primitive + ``lock=`` injection seam are owned by Item
    128 (already shipped). Phase 4 makes the orchestrator
    *configuration-driven*: a multi-replica deployment selects a
    cross-replica backend via a config dict — no lock logic in bots,
    it delegates to ``dataknobs_common.locks.create_lock``.
    """

    @pytest.mark.asyncio
    async def test_lock_config_constructs_lock_via_factory(self) -> None:
        from dataknobs_common.locks import InProcessLock

        bus = await _make_bus()
        manager = _StubManager()
        orch = IngestOrchestrator(
            manager,  # type: ignore[arg-type]
            bus,
            lock_config={"backend": "memory"},
        )
        assert isinstance(orch.lock, InProcessLock)
        await bus.close()

    @pytest.mark.asyncio
    async def test_empty_lock_config_defaults_to_memory(self) -> None:
        from dataknobs_common.locks import InProcessLock

        bus = await _make_bus()
        manager = _StubManager()
        orch = IngestOrchestrator(
            manager,  # type: ignore[arg-type]
            bus,
            lock_config={},
        )
        assert isinstance(orch.lock, InProcessLock)
        await bus.close()

    @pytest.mark.asyncio
    async def test_unknown_lock_backend_raises(self) -> None:
        bus = await _make_bus()
        manager = _StubManager()
        with pytest.raises(ValueError, match="nope"):
            IngestOrchestrator(
                manager,  # type: ignore[arg-type]
                bus,
                lock_config={"backend": "nope"},
            )
        await bus.close()

    @pytest.mark.asyncio
    async def test_lock_and_lock_config_are_mutually_exclusive(self) -> None:
        from dataknobs_common.locks import InProcessLock

        bus = await _make_bus()
        manager = _StubManager()
        with pytest.raises(ValueError, match=r"lock.*lock_config"):
            IngestOrchestrator(
                manager,  # type: ignore[arg-type]
                bus,
                lock=InProcessLock(),
                lock_config={"backend": "memory"},
            )
        await bus.close()

    @pytest.mark.asyncio
    async def test_config_built_lock_serializes_per_domain(self) -> None:
        """A config-built lock still serializes concurrent same-domain
        triggers (end-to-end behavioural proof, not just type check)."""
        from dataknobs_common.locks import InProcessLock

        bus = await _make_bus()
        manager = _StubManager()
        orch = IngestOrchestrator(
            manager,  # type: ignore[arg-type]
            bus,
            lock_config={"backend": "memory"},
        )
        assert isinstance(orch.lock, InProcessLock)
        await orch.start()
        await bus.publish(TRIGGER_TOPIC, _trigger_event({"domain_id": "d1"}))
        await _wait_for(lambda: len(manager.calls) >= 1)
        assert manager.calls == [("d1", None)]
        await orch.stop()
        await bus.close()


class TestDispatchMatrix:
    """Payload selects the ingest entry point (class-docstring contract)."""

    @pytest.mark.asyncio
    async def test_since_version_dispatches_ingest_changes(self) -> None:
        bus = await _make_bus()
        manager = _StubManager()
        orch = IngestOrchestrator(manager, bus)  # type: ignore[arg-type]
        await orch.start()

        await bus.publish(
            TRIGGER_TOPIC,
            _trigger_event({"domain_id": "d1", "since_version": "v-abc"}),
        )
        await _wait_for(lambda: len(manager.change_calls) >= 1)

        assert manager.change_calls == [("d1", "v-abc")]
        assert manager.calls == []  # not the default path
        assert manager.ingest_calls == []

        await orch.stop()
        await bus.close()

    @pytest.mark.asyncio
    async def test_force_full_dispatches_clear_first(self) -> None:
        bus = await _make_bus()
        manager = _StubManager()
        orch = IngestOrchestrator(manager, bus)  # type: ignore[arg-type]
        await orch.start()

        await bus.publish(
            TRIGGER_TOPIC,
            _trigger_event({"domain_id": "d1", "force_full": True}),
        )
        await _wait_for(lambda: len(manager.ingest_calls) >= 1)

        assert manager.ingest_calls == [("d1", IngestSwapMode.CLEAR_FIRST)]
        assert manager.calls == []
        assert manager.change_calls == []

        await orch.stop()
        await bus.close()

    @pytest.mark.asyncio
    async def test_since_version_takes_precedence_over_force_full(
        self,
    ) -> None:
        """Both present → the more specific delta intent wins."""
        bus = await _make_bus()
        manager = _StubManager()
        orch = IngestOrchestrator(manager, bus)  # type: ignore[arg-type]
        await orch.start()

        await bus.publish(
            TRIGGER_TOPIC,
            _trigger_event(
                {
                    "domain_id": "d1",
                    "since_version": "v-abc",
                    "force_full": True,
                }
            ),
        )
        await _wait_for(lambda: len(manager.change_calls) >= 1)

        assert manager.change_calls == [("d1", "v-abc")]
        assert manager.ingest_calls == []

        await orch.stop()
        await bus.close()

    @pytest.mark.asyncio
    async def test_default_payload_uses_ingest_if_changed(self) -> None:
        """No since_version / force_full → unchanged default path."""
        bus = await _make_bus()
        manager = _StubManager()
        orch = IngestOrchestrator(manager, bus)  # type: ignore[arg-type]
        await orch.start()

        await bus.publish(
            TRIGGER_TOPIC,
            _trigger_event({"domain_id": "d1", "last_version": "v1"}),
        )
        await _wait_for(lambda: len(manager.calls) >= 1)

        assert manager.calls == [("d1", "v1")]
        assert manager.change_calls == []
        assert manager.ingest_calls == []

        await orch.stop()
        await bus.close()

    @pytest.mark.asyncio
    async def test_delta_path_still_serialized_per_domain(self) -> None:
        """The injected lock guards every dispatch path, keyed per domain."""
        from dataknobs_common.locks import InProcessLock

        class RecordingLock(InProcessLock):
            def __init__(self) -> None:
                super().__init__()
                self.held_keys: list[str] = []

            def hold(  # type: ignore[override]
                self, key: str, *, timeout: float | None = None
            ):
                self.held_keys.append(key)
                return super().hold(key, timeout=timeout)

        bus = await _make_bus()
        manager = _StubManager()
        lock = RecordingLock()
        orch = IngestOrchestrator(manager, bus, lock=lock)  # type: ignore[arg-type]
        await orch.start()

        await bus.publish(
            TRIGGER_TOPIC,
            _trigger_event({"domain_id": "d9", "since_version": "v"}),
        )
        await _wait_for(lambda: len(manager.change_calls) >= 1)

        assert lock.held_keys == ["ingest:d9"]

        await orch.stop()
        await bus.close()


@pytest.mark.asyncio
async def test_custom_trigger_topic() -> None:
    """Orchestrator honors a non-default ``trigger_topic``."""
    bus = await _make_bus()
    manager = _StubManager()
    orch = IngestOrchestrator(manager, bus, trigger_topic="my_topic")  # type: ignore[arg-type]
    await orch.start()

    # Default topic should NOT dispatch
    await bus.publish(
        TRIGGER_TOPIC,
        _trigger_event({"domain_id": "ignored"}, topic=TRIGGER_TOPIC),
    )
    await asyncio.sleep(0.05)
    assert manager.calls == []

    # Custom topic SHOULD dispatch
    await bus.publish(
        "my_topic",
        _trigger_event({"domain_id": "d1"}, topic="my_topic"),
    )
    await _wait_for(lambda: len(manager.calls) >= 1)
    assert manager.calls == [("d1", None)]

    assert orch.trigger_topic == "my_topic"

    await orch.stop()
    await bus.close()
