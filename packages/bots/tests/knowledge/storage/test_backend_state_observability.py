"""Cross-backend conformance: state writes fire observability events.

Every in-tree backend fires ``ingest:metadata:write`` on a metadata
state write and ``ingest:snapshot:write`` on a snapshot state write,
through the shared ``_fire_state_write`` helper on
``KnowledgeResourceBackendMixin``. The helper is zero-overhead when no
callbacks were ever registered (the registry is not constructed).
"""

from __future__ import annotations

import pytest

from dataknobs_bots.knowledge import (
    INGEST_METADATA_WRITE,
    INGEST_SNAPSHOT_WRITE,
)
from dataknobs_bots.knowledge.storage import (
    FileKnowledgeBackend,
    InMemoryKnowledgeBackend,
    KnowledgeKeyKind,
    S3KnowledgeBackend,
)
from dataknobs_common.testing import requires_localstack


async def _build(kind: str, tmp_path) -> object:
    if kind == "memory":
        backend: object = InMemoryKnowledgeBackend()
    else:
        backend = FileKnowledgeBackend(str(tmp_path / "kb"))
    await backend.initialize()
    await backend.create_kb("d1")
    return backend


@pytest.mark.parametrize("kind", ["memory", "file"])
@pytest.mark.asyncio
async def test_metadata_write_fires(kind: str, tmp_path) -> None:
    backend = await _build(kind, tmp_path)
    events: list[dict] = []
    backend.state_write_callbacks.register(
        INGEST_METADATA_WRITE, events.append
    )

    # set_ingestion_status is a metadata state write on every backend.
    await backend.set_ingestion_status("d1", "ready")

    assert len(events) >= 1
    ev = events[-1]
    assert ev["domain_id"] == "d1"
    assert ev["kind"] is KnowledgeKeyKind.METADATA
    assert isinstance(ev["byte_size"], int)
    assert isinstance(ev["key"], str) and ev["key"]


@pytest.mark.parametrize("kind", ["memory", "file"])
@pytest.mark.asyncio
async def test_snapshot_write_fires(kind: str, tmp_path) -> None:
    backend = await _build(kind, tmp_path)
    events: list[dict] = []
    backend.state_write_callbacks.register(
        INGEST_SNAPSHOT_WRITE, events.append
    )

    # put_file records a content snapshot on every backend.
    await backend.put_file("d1", "intro.md", b"# Intro\n")

    assert len(events) >= 1
    ev = events[-1]
    assert ev["domain_id"] == "d1"
    assert ev["kind"] is KnowledgeKeyKind.SNAPSHOT
    assert isinstance(ev["byte_size"], int)


@pytest.mark.parametrize("kind", ["memory", "file"])
@pytest.mark.asyncio
async def test_zero_overhead_when_no_callbacks(kind: str, tmp_path) -> None:
    """No callback ever registered ⇒ the registry is never constructed."""
    backend = await _build(kind, tmp_path)

    await backend.put_file("d1", "intro.md", b"# Intro\n")
    await backend.set_ingestion_status("d1", "ready")

    assert getattr(backend, "_state_write_callbacks", None) is None


@pytest.mark.parametrize("kind", ["memory", "file"])
@pytest.mark.asyncio
async def test_state_write_callbacks_stable_identity(
    kind: str, tmp_path
) -> None:
    backend = await _build(kind, tmp_path)
    assert backend.state_write_callbacks is backend.state_write_callbacks


# ---------------------------------------------------------------------------
# S3 backend — the third in-tree backend, advertised the same
# BACKEND_STATE_OBSERVABILITY capability, exercised here against real
# LocalStack so its metadata/snapshot fires are not advertised-but-unproven.
# (moto's mock_aws is incompatible with the aioboto3 transport.) Start
# LocalStack with ``bin/dk up``; these skip when it is unavailable.
# ---------------------------------------------------------------------------


async def _s3_backend(cfg: dict) -> S3KnowledgeBackend:
    """Initialize an S3 backend on the LocalStack bucket and seed a KB."""
    backend = S3KnowledgeBackend.from_config(cfg)
    await backend.initialize()
    await backend.create_kb("d1")
    return backend


@pytest.mark.integration
@pytest.mark.s3
@requires_localstack
async def test_s3_metadata_write_fires(s3_kb_config) -> None:
    backend = await _s3_backend(s3_kb_config)
    try:
        events: list[dict] = []
        backend.state_write_callbacks.register(
            INGEST_METADATA_WRITE, events.append
        )

        await backend.set_ingestion_status("d1", "ready")

        assert len(events) >= 1
        ev = events[-1]
        assert ev["domain_id"] == "d1"
        assert ev["kind"] is KnowledgeKeyKind.METADATA
        assert isinstance(ev["byte_size"], int)
        assert isinstance(ev["key"], str) and ev["key"]
    finally:
        await backend.close()


@pytest.mark.integration
@pytest.mark.s3
@requires_localstack
async def test_s3_snapshot_write_fires(s3_kb_config) -> None:
    backend = await _s3_backend(s3_kb_config)
    try:
        events: list[dict] = []
        backend.state_write_callbacks.register(
            INGEST_SNAPSHOT_WRITE, events.append
        )

        await backend.put_file("d1", "intro.md", b"# Intro\n")

        assert len(events) >= 1
        ev = events[-1]
        assert ev["domain_id"] == "d1"
        assert ev["kind"] is KnowledgeKeyKind.SNAPSHOT
        assert isinstance(ev["byte_size"], int)
    finally:
        await backend.close()


@pytest.mark.integration
@pytest.mark.s3
@requires_localstack
async def test_s3_zero_overhead_when_no_callbacks(s3_kb_config) -> None:
    """No callback ever registered ⇒ the registry is never constructed."""
    backend = await _s3_backend(s3_kb_config)
    try:
        await backend.put_file("d1", "intro.md", b"# Intro\n")
        await backend.set_ingestion_status("d1", "ready")

        assert getattr(backend, "_state_write_callbacks", None) is None
    finally:
        await backend.close()
