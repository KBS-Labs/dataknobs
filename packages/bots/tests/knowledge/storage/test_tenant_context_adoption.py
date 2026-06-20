"""Behavior of the optional tenant-context parameter on the backends.

These exercise the additive ``ctx: TenantContext`` parameter through the
public state surface (``set_ingestion_status`` / ``get_info`` /
``get_checksum`` / ``list_changes_since``) plus the snapshot store, on
the in-memory and file backends (in-process / ``tmp_path`` — no
services). They assert:

* ``ctx=None`` and a no-prefix ``SingleTenantContext`` are observably
  identical (single-tenant behavior unchanged).
* Two ``BoundTenantContext`` scopes over the same ``domain_id`` keep
  independent ingest state (status + snapshot lineage).
* Content reads are unaffected by tenant scoping.
* Every backend advertises ``TENANT_SCOPED_STATE`` + ``SNAPSHOT_ISOLATION``
  and does NOT advertise ``TENANT_SCOPED_LOCKS``.
"""

from __future__ import annotations

import pytest

from dataknobs_bots.knowledge.storage import (
    FileKnowledgeBackend,
    InMemoryKnowledgeBackend,
    InvalidVersionError,
)
from dataknobs_common.capabilities import Capability
from dataknobs_common.tenancy import BoundTenantContext, SingleTenantContext


async def _build(kind: str, tmp_path) -> object:
    """Build + initialize a backend of ``kind`` (``"memory"`` / ``"file"``)."""
    if kind == "memory":
        backend: object = InMemoryKnowledgeBackend()
    else:
        backend = FileKnowledgeBackend(str(tmp_path / "kb"))
    await backend.initialize()
    return backend


@pytest.mark.parametrize("kind", ["memory", "file"])
async def test_none_and_single_tenant_are_identical(kind, tmp_path) -> None:
    """``ctx=None`` and ``SingleTenantContext`` observe the same state."""
    b = await _build(kind, tmp_path)

    await b.create_kb("kb")
    await b.set_ingestion_status("kb", "ready")
    via_none = await b.get_info("kb")
    via_single = await b.get_info("kb", ctx=SingleTenantContext("kb"))
    assert via_none is not None and via_single is not None
    assert via_none.ingestion_status == via_single.ingestion_status

    # Writing through SingleTenantContext also lands on the shared store.
    await b.set_ingestion_status(
        "kb", "ingesting", ctx=SingleTenantContext("kb")
    )
    refreshed = await b.get_info("kb")
    again = await b.get_info("kb", ctx=SingleTenantContext("kb"))
    assert refreshed.ingestion_status == again.ingestion_status

    await b.close()


@pytest.mark.parametrize("kind", ["memory", "file"])
async def test_two_tenants_isolated_ingestion_status(kind, tmp_path) -> None:
    """Same ``domain_id``, different tenants → independent ingest state."""
    b = await _build(kind, tmp_path)

    await b.create_kb("shared_kb")
    alpha = BoundTenantContext("alpha", "shared_kb")
    beta = BoundTenantContext("beta", "shared_kb")

    await b.set_ingestion_status("shared_kb", "ready", ctx=alpha)
    await b.set_ingestion_status("shared_kb", "ingesting", ctx=beta)

    info_a = await b.get_info("shared_kb", ctx=alpha)
    info_b = await b.get_info("shared_kb", ctx=beta)
    assert info_a is not None and info_b is not None
    assert info_a.ingestion_status != info_b.ingestion_status

    await b.close()


@pytest.mark.parametrize("kind", ["memory", "file"])
async def test_tenant_state_requires_existing_domain(kind, tmp_path) -> None:
    """KB existence stays keyed by ``domain_id`` even with a tenant ctx."""
    b = await _build(kind, tmp_path)
    with pytest.raises(ValueError, match="does not exist"):
        await b.set_ingestion_status(
            "missing", "ready", ctx=BoundTenantContext("alpha", "missing")
        )
    await b.close()


@pytest.mark.parametrize("kind", ["memory", "file"])
async def test_content_is_shared_across_tenants(kind, tmp_path) -> None:
    """Content (files) stays domain-keyed; tenants read the same bytes."""
    b = await _build(kind, tmp_path)

    await b.create_kb("kb")
    await b.put_file("kb", "a.md", b"hello")

    # get_checksum is a content identity — identical across tenants.
    base = await b.get_checksum("kb")
    alpha = await b.get_checksum("kb", ctx=BoundTenantContext("alpha", "kb"))
    assert base == alpha
    assert await b.get_file("kb", "a.md") == b"hello"

    await b.close()


@pytest.mark.parametrize("kind", ["memory", "file"])
async def test_capability_advertisement(kind, tmp_path) -> None:
    b = await _build(kind, tmp_path)
    assert b.supports(Capability.TENANT_SCOPED_STATE)
    assert b.supports(Capability.SNAPSHOT_ISOLATION)
    # No backend locking in this layer — must not advertise it.
    assert not b.supports(Capability.TENANT_SCOPED_LOCKS)
    assert not b.supports(Capability.TRANSACTIONAL_METADATA)
    # The mixin's base capabilities survive the union (not replaced).
    assert b.supports(Capability.KEY_PATTERN_FILTERING)
    await b.close()


async def test_memory_snapshot_store_is_tenant_isolated(tmp_path) -> None:
    """A tenant-recorded snapshot is invisible to other tenants (memory)."""
    b = InMemoryKnowledgeBackend()
    await b.initialize()
    await b.create_kb("kb")
    await b.put_file("kb", "a.md", b"hello")
    version = await b.get_checksum("kb")
    expected = {f.path: f.checksum for f in await b.list_files("kb")}

    alpha = BoundTenantContext("alpha", "kb")
    await b._record_snapshot("kb", ctx=alpha)

    # The tenant that recorded it resolves the snapshot.
    assert await b._load_snapshot("kb", version, ctx=alpha) == expected
    # A different tenant does not.
    with pytest.raises(InvalidVersionError):
        await b._load_snapshot(
            "kb", version, ctx=BoundTenantContext("beta", "kb")
        )
    # The single-tenant store (content mutation) resolves it too.
    assert await b._load_snapshot("kb", version) == expected

    await b.close()


async def test_file_snapshot_store_is_tenant_isolated(tmp_path) -> None:
    """A tenant-recorded snapshot is invisible to other tenants (file)."""
    b = FileKnowledgeBackend(str(tmp_path / "kb"))
    await b.initialize()
    await b.create_kb("kb")
    await b.put_file("kb", "a.md", b"hello")
    version = await b.get_checksum("kb")
    files = {f.path: f.to_dict() for f in await b.list_files("kb")}
    expected = {f.path: f.checksum for f in await b.list_files("kb")}

    alpha = BoundTenantContext("alpha", "kb")
    await b._record_snapshot("kb", files, ctx=alpha)

    assert await b._load_snapshot("kb", version, ctx=alpha) == expected
    with pytest.raises(InvalidVersionError):
        await b._load_snapshot(
            "kb", version, ctx=BoundTenantContext("beta", "kb")
        )

    await b.close()
