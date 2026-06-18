"""Tests for the ``subscribe_to_changes`` composition convenience.

The default-kinds resolution (``{CONTENT}``), per-kind pattern
derivation, the multi-kind guard, and the ``changes_subscription``
async-context-manager teardown are exercised against a
``FileKnowledgeBackend`` (which produces a real fnmatch-shaped pattern)
plus an ``InMemoryEventBus``.
"""

from __future__ import annotations

import pytest

from dataknobs_bots.knowledge.storage import (
    FileKnowledgeBackend,
    KnowledgeKeyKind,
)
from dataknobs_common.events import Event, EventType, InMemoryEventBus


async def _backend(tmp_path) -> FileKnowledgeBackend:
    backend = FileKnowledgeBackend(str(tmp_path / "kb"))
    await backend.initialize()
    await backend.create_kb("d1")
    return backend


async def _bus() -> InMemoryEventBus:
    bus = InMemoryEventBus()
    await bus.connect()
    return bus


async def _noop(_event: Event) -> None:
    return None


@pytest.mark.asyncio
async def test_default_kinds_is_content(tmp_path) -> None:
    backend = await _backend(tmp_path)
    bus = await _bus()
    sub = await backend.subscribe_to_changes(bus, handler=_noop)
    assert sub.pattern == backend.key_pattern(KnowledgeKeyKind.CONTENT)
    await bus.close()


@pytest.mark.asyncio
async def test_explicit_metadata_kind_pattern(tmp_path) -> None:
    backend = await _backend(tmp_path)
    bus = await _bus()
    sub = await backend.subscribe_to_changes(
        bus, kinds={KnowledgeKeyKind.METADATA}, handler=_noop
    )
    assert sub.pattern == backend.key_pattern(KnowledgeKeyKind.METADATA)
    await bus.close()


@pytest.mark.asyncio
async def test_multi_kind_raises_with_guidance(tmp_path) -> None:
    backend = await _backend(tmp_path)
    bus = await _bus()
    with pytest.raises(ValueError, match="once per kind"):
        await backend.subscribe_to_changes(
            bus,
            kinds={KnowledgeKeyKind.CONTENT, KnowledgeKeyKind.METADATA},
            handler=_noop,
        )
    await bus.close()


@pytest.mark.asyncio
async def test_changes_subscription_context_manager_teardown(
    tmp_path,
) -> None:
    backend = await _backend(tmp_path)
    bus = await _bus()
    base = str(tmp_path / "kb")
    content_topic = f"{base}/d1/content/intro.md"

    received: list[str] = []

    async def handler(event: Event) -> None:
        received.append(event.topic)

    def _publish() -> Event:
        return Event(
            type=EventType.CUSTOM, topic=content_topic, payload={}
        )

    async with backend.changes_subscription(
        bus, domain_id="d1", handler=handler
    ):
        await bus.publish(content_topic, _publish())

    # After the context exits the subscription is cancelled.
    await bus.publish(content_topic, _publish())

    assert received == [content_topic]  # exactly one delivery (inside block)
    await bus.close()
