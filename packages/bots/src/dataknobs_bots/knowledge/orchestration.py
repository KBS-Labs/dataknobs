"""Subscriber-side orchestration for knowledge ingestion triggers.

Thin primitive that subscribes to a trigger topic on an
:class:`~dataknobs_common.events.EventBus` and dispatches payloads to
:meth:`KnowledgeIngestionManager.ingest_if_changed`.

Consumer-side trigger adapters (S3 event notifications → EventBus, SQS
messages → EventBus, cron → EventBus) are deployment-specific and
remain the consumer's responsibility. This class is the generic
receive side.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from dataknobs_common.locks import InProcessLock, create_lock

from .ingestion import IngestSwapMode

if TYPE_CHECKING:
    from dataknobs_common.events import Event, EventBus, Subscription
    from dataknobs_common.locks import DistributedLock

    from .ingestion import KnowledgeIngestionManager

logger = logging.getLogger(__name__)


class IngestOrchestrator:
    """Subscribe to a trigger topic and dispatch an ingest for a domain.

    Expected trigger-event payload shape::

        {
            "domain_id": str,             # required
            "since_version": str | None,  # optional
            "force_full": bool | None,    # optional
            "last_version": str | None,   # optional
        }

    Dispatch is decided by the payload (checked in this order, so a
    ``since_version`` present alongside ``force_full`` takes the
    delta path — the more specific intent wins):

    1. ``since_version`` present (truthy) →
       :meth:`KnowledgeIngestionManager.ingest_changes` — a per-file
       delta re-ingest of only what changed since that canonical
       snapshot id.
    2. ``force_full`` truthy →
       :meth:`KnowledgeIngestionManager.ingest` with
       :attr:`~dataknobs_bots.knowledge.ingestion.IngestSwapMode.CLEAR_FIRST`
       — an unconditional full re-ingest.
    3. otherwise →
       :meth:`KnowledgeIngestionManager.ingest_if_changed` with
       ``last_version`` (the default; skips when nothing changed).

    The orchestrator is stateless across restarts — it does not persist
    last-seen versions. Consumers needing version persistence either:

    1. include ``last_version`` in every trigger event (trigger-adapter
       responsibility), or
    2. wire a status store into the
       :class:`KnowledgeIngestionManager` so the current version is
       sourced from there.

    Ingests are serialized **per domain** through the injected
    :class:`~dataknobs_common.locks.DistributedLock` (keyed
    ``f"ingest:{domain_id}"``). Concurrent triggers for the same
    ``domain_id`` (common under at-least-once delivery from
    SQS/S3/cron) are queued one-at-a-time so they don't race on
    ``clear_existing`` + ``add_vectors``; different domains still
    ingest in parallel.

    The *scope* of that serialization is exactly the scope of the
    configured lock. With the default :class:`InProcessLock` it is
    **process-local** — sufficient for single-replica deployments and
    behaviour-identical to prior releases. **Multi-replica deployments
    MUST configure a cross-replica lock**, either by passing a
    pre-built one (``lock=create_lock({"backend": "postgres", ...})``)
    or, configuration-driven, by passing the factory config directly
    (``lock_config={"backend": "postgres", ...}``); otherwise two
    replicas can ingest the same domain concurrently and race on the
    vector store, which a process-local lock cannot prevent.

    Example:
        ```python
        from dataknobs_common.events import InMemoryEventBus, Event, EventType
        from dataknobs_bots.knowledge import (
            KnowledgeIngestionManager,
            IngestOrchestrator,
        )

        bus = InMemoryEventBus()
        await bus.connect()

        manager = KnowledgeIngestionManager(source=backend, destination=rag)
        orchestrator = IngestOrchestrator(manager, bus)
        await orchestrator.start()

        await bus.publish(
            "knowledge:trigger",
            Event(
                type=EventType.UPDATED,
                topic="knowledge:trigger",
                payload={"domain_id": "my-domain"},
            ),
        )
        # ...
        await orchestrator.stop()
        ```
    """

    def __init__(
        self,
        ingestion_manager: KnowledgeIngestionManager,
        event_bus: EventBus,
        trigger_topic: str = "knowledge:trigger",
        lock: DistributedLock | None = None,
        lock_config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the orchestrator.

        Args:
            ingestion_manager: Manager dispatched to per trigger event
                (``ingest_changes`` / ``ingest`` / ``ingest_if_changed``
                depending on the payload — see the class docstring)
            event_bus: Bus to subscribe on
            trigger_topic: Topic name to subscribe to (default
                ``"knowledge:trigger"``)
            lock: A pre-built
                :class:`~dataknobs_common.locks.DistributedLock`
                backing per-domain serialization. Mutually exclusive
                with ``lock_config``.
            lock_config: A :func:`~dataknobs_common.locks.create_lock`
                config dict (e.g.
                ``{"backend": "postgres", "connection_string": ...}``)
                — the configuration-driven alternative to ``lock`` so a
                multi-replica deployment selects a cross-replica
                backend without writing code. Resolved through the
                shared ``dataknobs_common.locks`` factory (no lock
                logic lives here). Mutually exclusive with ``lock``.

                When neither is given the default is a process-local
                :class:`~dataknobs_common.locks.InProcessLock` —
                behaviour-identical to prior releases and correct for
                single-replica deployments. **Multi-replica
                deployments MUST supply a cross-replica lock** via one
                of these (otherwise two replicas can ingest the same
                domain concurrently — a process-local lock cannot
                prevent that).

        Raises:
            ValueError: If both ``lock`` and ``lock_config`` are
                supplied, or ``lock_config`` names an unknown backend.
        """
        if lock is not None and lock_config is not None:
            raise ValueError(
                "IngestOrchestrator: pass either lock= (a pre-built "
                "DistributedLock) or lock_config= (a create_lock "
                "config dict), not both."
            )
        self._manager = ingestion_manager
        self._event_bus = event_bus
        self._topic = trigger_topic
        self._subscription: Subscription | None = None
        if lock is not None:
            self._lock: DistributedLock = lock
        elif lock_config is not None:
            self._lock = create_lock(lock_config)
        else:
            self._lock = InProcessLock()

    @property
    def trigger_topic(self) -> str:
        """Topic this orchestrator subscribes to."""
        return self._topic

    @property
    def is_running(self) -> bool:
        """``True`` after :meth:`start` and before :meth:`stop`."""
        return self._subscription is not None

    async def start(self) -> None:
        """Subscribe to the trigger topic. Idempotent."""
        if self._subscription is not None:
            return
        self._subscription = await self._event_bus.subscribe(
            self._topic, self._handle_trigger
        )
        logger.info("IngestOrchestrator subscribed to %s", self._topic)

    async def stop(self) -> None:
        """Cancel the subscription. Idempotent."""
        if self._subscription is None:
            return
        await self._subscription.cancel()
        self._subscription = None
        logger.info("IngestOrchestrator unsubscribed from %s", self._topic)

    async def _handle_trigger(self, event: Event) -> None:
        """Dispatch a trigger event to the appropriate ingest entry point.

        ``since_version`` → ``ingest_changes`` (per-file delta);
        ``force_full`` → ``ingest(swap_mode=CLEAR_FIRST)`` (full
        re-ingest); otherwise ``ingest_if_changed(last_version)`` (the
        default skip-if-unchanged path) — see the class docstring for
        the precedence.

        Concurrent triggers for the same ``domain_id`` are serialized
        through the injected lock (keyed ``f"ingest:{domain_id}"``)
        regardless of which path is taken; different domains proceed in
        parallel. Errors are logged but not re-raised — the EventBus
        dispatcher continues serving subsequent events to other
        handlers.
        """
        payload: dict[str, Any] = event.payload or {}
        domain_id = payload.get("domain_id")
        if not domain_id:
            logger.warning(
                "IngestOrchestrator received trigger without domain_id; "
                "skipping (event_id=%s)",
                event.event_id,
            )
            return
        since_version = payload.get("since_version")
        force_full = payload.get("force_full")
        last_version = payload.get("last_version")
        # No timeout: queue-and-wait, preserving the prior
        # ``async with asyncio.Lock()`` semantics exactly. timeout=None
        # always acquires per the DistributedLock contract, but the
        # body is guarded on ``acquired`` so a future timed/best-effort
        # lock can never run the critical section unheld.
        async with self._lock.hold(f"ingest:{domain_id}") as acquired:
            if not acquired:
                logger.warning(
                    "IngestOrchestrator could not acquire lock for "
                    "domain=%s; skipping trigger",
                    domain_id,
                )
                return
            try:
                if since_version:
                    result = await self._manager.ingest_changes(
                        domain_id, since_version
                    )
                elif force_full:
                    result = await self._manager.ingest(
                        domain_id,
                        swap_mode=IngestSwapMode.CLEAR_FIRST,
                    )
                else:
                    result = await self._manager.ingest_if_changed(
                        domain_id, last_version=last_version
                    )
                    # Only the default path can yield None
                    # (ingest_changes / ingest always return a result).
                    if result is None:
                        logger.debug(
                            "No changes for domain=%s since version=%s",
                            domain_id,
                            last_version,
                        )
                if result is not None:
                    logger.info(
                        "Ingest complete for domain=%s (chunks=%d)",
                        domain_id,
                        result.chunks_created,
                    )
            except Exception:
                logger.exception(
                    "IngestOrchestrator failed to process trigger for domain=%s",
                    domain_id,
                )


__all__ = ["IngestOrchestrator"]
