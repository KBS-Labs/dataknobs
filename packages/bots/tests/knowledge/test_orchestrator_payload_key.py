"""Tests for ``IngestOrchestrator`` optional ``payload["key"]`` filtering.

When a trigger carries the originating backend ``key`` (S3 → EventBridge,
filesystem inotify), the orchestrator classifies it and skips
non-``CONTENT`` keys so the DK-managed ``_metadata.json`` / ``_snapshots/``
writes the ingest itself performs do not re-trigger ingestion. Absent
``key`` proceeds unchanged (back-compat).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from dataknobs_bots.knowledge.orchestration import IngestOrchestrator
from dataknobs_bots.knowledge.storage import InMemoryKnowledgeBackend
from dataknobs_common.events import Event, EventType, InMemoryEventBus

TRIGGER_TOPIC = "knowledge:trigger"


class _KeyStubManager:
    """Stub manager exposing a real backend ``source`` for key
    classification plus a recording ``ingest_if_changed``."""

    def __init__(self, backend: InMemoryKnowledgeBackend) -> None:
        self._source = backend
        self.calls: list[str] = []

    @property
    def source(self) -> InMemoryKnowledgeBackend:
        return self._source

    async def ingest_if_changed(
        self, domain_id: str, last_version: str | None = None, **_: Any
    ) -> None:
        self.calls.append(domain_id)
        return None


async def _make_bus() -> InMemoryEventBus:
    bus = InMemoryEventBus()
    await bus.connect()
    return bus


def _trigger(payload: dict[str, Any]) -> Event:
    return Event(type=EventType.UPDATED, topic=TRIGGER_TOPIC, payload=payload)


async def _wait(condition, timeout: float = 1.0) -> None:
    elapsed = 0.0
    while elapsed < timeout:
        if condition():
            return
        await asyncio.sleep(0.01)
        elapsed += 0.01


@pytest.mark.asyncio
async def test_content_key_proceeds() -> None:
    bus = await _make_bus()
    backend = InMemoryKnowledgeBackend()
    manager = _KeyStubManager(backend)
    orch = IngestOrchestrator(manager, bus)  # type: ignore[arg-type]
    await orch.start()

    await bus.publish(
        TRIGGER_TOPIC,
        _trigger({"domain_id": "d1", "key": "d1/content/foo.md"}),
    )
    await _wait(lambda: manager.calls == ["d1"])
    assert manager.calls == ["d1"]

    await orch.stop()
    await bus.close()


@pytest.mark.asyncio
async def test_metadata_and_snapshot_keys_skip(
    caplog: pytest.LogCaptureFixture,
) -> None:
    bus = await _make_bus()
    backend = InMemoryKnowledgeBackend()
    manager = _KeyStubManager(backend)
    orch = IngestOrchestrator(manager, bus)  # type: ignore[arg-type]
    await orch.start()

    with caplog.at_level("INFO"):
        await bus.publish(
            TRIGGER_TOPIC,
            _trigger({"domain_id": "d1", "key": "d1/_metadata.json"}),
        )
        await bus.publish(
            TRIGGER_TOPIC,
            _trigger({"domain_id": "d1", "key": "d1/_snapshots/v1.json"}),
        )
        # Give the dispatcher a beat; nothing should be ingested.
        await asyncio.sleep(0.05)

    assert manager.calls == []
    assert "skipping ingest" in caplog.text


@pytest.mark.asyncio
async def test_content_key_with_resolver_resolves_manager_once() -> None:
    """A content-key trigger under a ``manager_resolver`` resolves the
    per-tenant manager exactly once.

    Regression guard: the key-classification step and the dispatch step
    must share one resolved manager. Before the fix the resolver ran
    twice per content trigger (once unserialized, before the lock, to
    classify; once inside the lock to dispatch) — doubling a
    potentially expensive per-tenant resolution.
    """
    bus = await _make_bus()
    backend = InMemoryKnowledgeBackend()
    manager = _KeyStubManager(backend)
    resolve_calls: list[tuple[str | None, str]] = []

    async def resolver(*, tenant_id: str | None, domain_id: str) -> Any:
        resolve_calls.append((tenant_id, domain_id))
        return manager

    orch = IngestOrchestrator(None, bus, manager_resolver=resolver)
    await orch.start()

    await bus.publish(
        TRIGGER_TOPIC,
        _trigger(
            {
                "domain_id": "d1",
                "tenant_id": "acme",
                "key": "d1/content/foo.md",
            }
        ),
    )
    await _wait(lambda: manager.calls == ["d1"])
    assert manager.calls == ["d1"]
    # Exactly one resolution despite key-classification + dispatch.
    assert resolve_calls == [("acme", "d1")]

    await orch.stop()
    await bus.close()


@pytest.mark.asyncio
async def test_absent_key_is_back_compat() -> None:
    bus = await _make_bus()
    backend = InMemoryKnowledgeBackend()
    manager = _KeyStubManager(backend)
    orch = IngestOrchestrator(manager, bus)  # type: ignore[arg-type]
    await orch.start()

    await bus.publish(TRIGGER_TOPIC, _trigger({"domain_id": "d1"}))
    await _wait(lambda: manager.calls == ["d1"])
    assert manager.calls == ["d1"]

    await orch.stop()
    await bus.close()
