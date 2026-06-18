"""Tests for ``KnowledgeIngestionManager`` lifecycle-event publication.

Exercises the in-process ``lifecycle_callbacks`` registry, the
``ingest:domain:start`` / ``ingest:domain:end`` fires, the tenant-bound
payload shape, and the automatic ``also_publish_to`` cross-replica
fan-out when the manager is constructed with an event bus. Uses real
constructs throughout (``InMemoryKnowledgeBackend`` source, RAG KB over a
memory vector store + echo embedder, ``InMemoryEventBus``).
"""

from __future__ import annotations

import pytest

from dataknobs_bots.knowledge import (
    INGEST_DOMAIN_END,
    INGEST_DOMAIN_START,
    KnowledgeIngestionManager,
    RAGKnowledgeBase,
    TenantFilteredCallback,
)
from dataknobs_bots.knowledge.storage import InMemoryKnowledgeBackend
from dataknobs_common.events import InMemoryEventBus


async def _make_source(domain_id: str = "d1") -> InMemoryKnowledgeBackend:
    backend = InMemoryKnowledgeBackend()
    await backend.initialize()
    await backend.create_kb(domain_id)
    await backend.put_file(domain_id, "intro.md", b"# Intro\n\nHello.\n")
    return backend


async def _make_rag() -> RAGKnowledgeBase:
    return await RAGKnowledgeBase.from_config(
        {
            "vector_store": {"backend": "memory", "dimensions": 384},
            "embedding_provider": "echo",
            "embedding_model": "test",
        }
    )


@pytest.mark.asyncio
async def test_end_payload_carries_tenant_id_when_bound() -> None:
    source = await _make_source()
    rag = await _make_rag()
    manager = KnowledgeIngestionManager(
        source=source, destination=rag, tenant_id="acme"
    )

    captured: list[dict] = []
    manager.lifecycle_callbacks.register(INGEST_DOMAIN_END, captured.append)

    await manager.ingest("d1")

    assert len(captured) == 1
    payload = captured[0]
    assert payload["tenant_id"] == "acme"
    assert payload["domain_id"] == "d1"
    assert payload["status"] == "completed"
    assert payload["files_processed"] >= 1
    assert "completed_at" in payload


@pytest.mark.asyncio
async def test_end_payload_omits_tenant_id_when_unbound() -> None:
    source = await _make_source()
    rag = await _make_rag()
    manager = KnowledgeIngestionManager(source=source, destination=rag)

    captured: list[dict] = []
    manager.lifecycle_callbacks.register(INGEST_DOMAIN_END, captured.append)

    await manager.ingest("d1")

    assert len(captured) == 1
    assert "tenant_id" not in captured[0]


@pytest.mark.asyncio
async def test_start_fires_before_end() -> None:
    source = await _make_source()
    rag = await _make_rag()
    manager = KnowledgeIngestionManager(
        source=source, destination=rag, tenant_id="acme"
    )

    order: list[str] = []
    manager.lifecycle_callbacks.register(
        INGEST_DOMAIN_START, lambda ev: order.append("start")
    )
    manager.lifecycle_callbacks.register(
        INGEST_DOMAIN_END, lambda ev: order.append("end")
    )
    start_payloads: list[dict] = []
    manager.lifecycle_callbacks.register(
        INGEST_DOMAIN_START, start_payloads.append
    )

    await manager.ingest("d1")

    assert order == ["start", "end"]
    assert start_payloads[0]["domain_id"] == "d1"
    assert start_payloads[0]["tenant_id"] == "acme"
    assert "started_at" in start_payloads[0]
    # The start payload is lean — no completion stats.
    assert "files_processed" not in start_payloads[0]


@pytest.mark.asyncio
async def test_lifecycle_callbacks_stable_identity() -> None:
    source = await _make_source()
    rag = await _make_rag()
    manager = KnowledgeIngestionManager(source=source, destination=rag)

    assert manager.lifecycle_callbacks is manager.lifecycle_callbacks

    fired: list[dict] = []
    manager.lifecycle_callbacks.register(INGEST_DOMAIN_END, fired.append)
    await manager.ingest("d1")
    assert len(fired) == 1


@pytest.mark.asyncio
async def test_event_bus_auto_composes_fan_out() -> None:
    """An event_bus-bound manager fans out to the bus AND fires the
    in-process callback — both observers see the end event."""
    source = await _make_source()
    rag = await _make_rag()
    bus = InMemoryEventBus()
    await bus.connect()

    bus_events: list = []

    async def bus_handler(event) -> None:
        bus_events.append(event)

    await bus.subscribe(INGEST_DOMAIN_END, bus_handler)

    manager = KnowledgeIngestionManager(
        source=source, destination=rag, event_bus=bus, tenant_id="acme"
    )
    local_events: list[dict] = []
    manager.lifecycle_callbacks.register(INGEST_DOMAIN_END, local_events.append)

    await manager.ingest("d1")

    assert len(local_events) == 1
    assert len(bus_events) == 1
    assert bus_events[0].payload["tenant_id"] == "acme"
    assert bus_events[0].payload["status"] == "completed"
    assert manager.lifecycle_callbacks.supports_event_bus_emission()

    await bus.close()


@pytest.mark.asyncio
async def test_tenant_filtered_callback_on_lifecycle_registry() -> None:
    source = await _make_source()
    rag = await _make_rag()
    manager = KnowledgeIngestionManager(
        source=source, destination=rag, tenant_id="acme"
    )

    acme_hits: list[dict] = []
    other_hits: list[dict] = []
    manager.lifecycle_callbacks.register(
        INGEST_DOMAIN_END,
        TenantFilteredCallback(acme_hits.append, tenant_id="acme"),
    )
    manager.lifecycle_callbacks.register(
        INGEST_DOMAIN_END,
        TenantFilteredCallback(other_hits.append, tenant_id="umbrella"),
    )

    await manager.ingest("d1")

    assert len(acme_hits) == 1
    assert len(other_hits) == 0
