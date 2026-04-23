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

import asyncio
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dataknobs_common.events import Event, EventBus, Subscription

    from .ingestion import KnowledgeIngestionManager

logger = logging.getLogger(__name__)


class IngestOrchestrator:
    """Subscribe to a trigger topic and dispatch to ``ingest_if_changed``.

    Expected trigger-event payload shape::

        {
            "domain_id": str,             # required
            "last_version": str | None,   # optional
        }

    The orchestrator is stateless across restarts — it does not persist
    last-seen versions. Consumers needing version persistence either:

    1. include ``last_version`` in every trigger event (trigger-adapter
       responsibility), or
    2. wire a status store into the
       :class:`KnowledgeIngestionManager` so the current version is
       sourced from there.

    Within a single orchestrator instance, ingests are serialized
    **per domain** via an :class:`asyncio.Lock`. Concurrent triggers
    for the same ``domain_id`` (common under at-least-once delivery
    from SQS/S3/cron) are queued one-at-a-time so they don't race on
    ``clear_existing`` + ``add_vectors``. Different domains still
    ingest in parallel.

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
    ) -> None:
        """Initialize the orchestrator.

        Args:
            ingestion_manager: Manager whose ``ingest_if_changed`` is
                invoked for each trigger event
            event_bus: Bus to subscribe on
            trigger_topic: Topic name to subscribe to (default
                ``"knowledge:trigger"``)
        """
        self._manager = ingestion_manager
        self._event_bus = event_bus
        self._topic = trigger_topic
        self._subscription: Subscription | None = None
        self._domain_locks: dict[str, asyncio.Lock] = {}

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

    def _lock_for(self, domain_id: str) -> asyncio.Lock:
        """Return (creating if needed) the lock for a domain."""
        lock = self._domain_locks.get(domain_id)
        if lock is None:
            lock = asyncio.Lock()
            self._domain_locks[domain_id] = lock
        return lock

    async def _handle_trigger(self, event: Event) -> None:
        """Dispatch a trigger event to ``ingest_if_changed``.

        Concurrent triggers for the same ``domain_id`` are serialized
        via a per-domain lock; different domains proceed in parallel.
        Errors are logged but not re-raised — the EventBus dispatcher
        continues serving subsequent events to other handlers.
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
        last_version = payload.get("last_version")
        lock = self._lock_for(domain_id)
        async with lock:
            try:
                result = await self._manager.ingest_if_changed(
                    domain_id, last_version=last_version
                )
                if result is None:
                    logger.debug(
                        "No changes for domain=%s since version=%s",
                        domain_id,
                        last_version,
                    )
                else:
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
