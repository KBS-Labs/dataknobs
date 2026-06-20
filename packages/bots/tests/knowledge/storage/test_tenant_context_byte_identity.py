"""Byte-identity guarantee for the optional tenant-context parameter.

The knowledge backends accept an optional ``ctx: TenantContext`` on their
state-touching methods. Two properties are load-bearing:

1. ``ctx=None`` (every pre-existing call site) AND a context whose
   ``state_key_prefix()`` is empty (``SingleTenantContext``) produce
   exactly the pre-tenancy state keys/paths — so adopting the parameter
   changes nothing for single-tenant consumers.
2. A ``BoundTenantContext`` isolates state under
   ``tenants/{tenant_id}/_state/`` — so two tenants of the same
   ``domain_id`` never collide on ingest state.

These exercise the pure key/path helpers directly (no I/O), so they run
without initializing a backend or touching the network.
"""

from __future__ import annotations

from pathlib import Path

from dataknobs_bots.knowledge.storage.file import FileKnowledgeBackend
from dataknobs_bots.knowledge.storage.memory import InMemoryKnowledgeBackend
from dataknobs_bots.knowledge.storage.s3 import S3KnowledgeBackend
from dataknobs_common.tenancy import BoundTenantContext, SingleTenantContext


# --- File backend: on-disk state paths ---


def test_file_metadata_path_none_matches_pre_adoption(tmp_path: Path) -> None:
    b = FileKnowledgeBackend(base_path=str(tmp_path))
    pre = tmp_path / "my_kb" / "_metadata.json"
    assert b._metadata_path("my_kb") == pre
    assert b._metadata_path("my_kb", ctx=None) == pre
    assert b._metadata_path("my_kb", ctx=SingleTenantContext("my_kb")) == pre


def test_file_snapshot_paths_none_match_pre_adoption(tmp_path: Path) -> None:
    b = FileKnowledgeBackend(base_path=str(tmp_path))
    assert b._snapshots_path("my_kb") == tmp_path / "my_kb" / "_snapshots"
    assert (
        b._snapshot_file("my_kb", "abc123")
        == tmp_path / "my_kb" / "_snapshots" / "abc123.json"
    )
    # SingleTenantContext contributes no prefix → identical paths.
    ctx = SingleTenantContext("my_kb")
    assert b._snapshots_path("my_kb", ctx) == tmp_path / "my_kb" / "_snapshots"
    assert (
        b._snapshot_file("my_kb", "abc123", ctx)
        == tmp_path / "my_kb" / "_snapshots" / "abc123.json"
    )


def test_file_content_paths_never_tenant_scoped(tmp_path: Path) -> None:
    """Content stays keyed by ``domain_id`` (no ctx on content helpers)."""
    b = FileKnowledgeBackend(base_path=str(tmp_path))
    assert b._content_path("my_kb") == tmp_path / "my_kb" / "content"
    assert (
        b._file_path("my_kb", "intro.md")
        == tmp_path / "my_kb" / "content" / "intro.md"
    )


def test_file_metadata_path_bound_is_isolated(tmp_path: Path) -> None:
    b = FileKnowledgeBackend(base_path=str(tmp_path))
    ctx = BoundTenantContext("acme", "my_kb")
    assert (
        b._metadata_path("my_kb", ctx=ctx)
        == tmp_path / "tenants" / "acme" / "_state" / "my_kb" / "_metadata.json"
    )
    assert (
        b._snapshot_file("my_kb", "v1", ctx)
        == tmp_path
        / "tenants"
        / "acme"
        / "_state"
        / "my_kb"
        / "_snapshots"
        / "v1.json"
    )


def test_file_two_tenants_distinct_state_paths(tmp_path: Path) -> None:
    b = FileKnowledgeBackend(base_path=str(tmp_path))
    alpha = b._metadata_path("kb", ctx=BoundTenantContext("alpha", "kb"))
    beta = b._metadata_path("kb", ctx=BoundTenantContext("beta", "kb"))
    assert alpha != beta


# --- S3 backend: object keys ---


def test_s3_metadata_key_none_matches_pre_adoption() -> None:
    b = S3KnowledgeBackend(bucket="b", prefix="kbs/")
    assert b._metadata_key("my_kb") == "kbs/my_kb/_metadata.json"
    assert b._metadata_key("my_kb", ctx=None) == "kbs/my_kb/_metadata.json"
    assert (
        b._metadata_key("my_kb", ctx=SingleTenantContext("my_kb"))
        == "kbs/my_kb/_metadata.json"
    )


def test_s3_snapshot_key_none_matches_pre_adoption() -> None:
    b = S3KnowledgeBackend(bucket="b", prefix="kbs/")
    pre = "kbs/my_kb/_snapshots/v1.json"
    assert b._snapshot_key("my_kb", "v1") == pre
    assert b._snapshot_key("my_kb", "v1", SingleTenantContext("my_kb")) == pre


def test_s3_content_key_never_tenant_scoped() -> None:
    """Content keys stay keyed by ``domain_id`` (``_s3_key`` has no ctx)."""
    b = S3KnowledgeBackend(bucket="b", prefix="kbs/")
    assert b._s3_key("my_kb", "intro.md") == "kbs/my_kb/content/intro.md"


def test_s3_metadata_key_bound_is_isolated() -> None:
    b = S3KnowledgeBackend(bucket="b", prefix="kbs/")
    ctx = BoundTenantContext("acme", "my_kb")
    assert (
        b._metadata_key("my_kb", ctx=ctx)
        == "kbs/tenants/acme/_state/my_kb/_metadata.json"
    )
    assert (
        b._snapshot_key("my_kb", "v1", ctx)
        == "kbs/tenants/acme/_state/my_kb/_snapshots/v1.json"
    )


def test_s3_two_tenants_distinct_keys() -> None:
    b = S3KnowledgeBackend(bucket="b", prefix="kbs/")
    alpha = b._metadata_key("kb", ctx=BoundTenantContext("alpha", "kb"))
    beta = b._metadata_key("kb", ctx=BoundTenantContext("beta", "kb"))
    assert alpha != beta


# --- Memory backend: overlay-key derivation ---


def test_memory_overlay_key_none_for_single_tenant() -> None:
    """The empty-prefix case routes to the domain-keyed store directly."""
    b = InMemoryKnowledgeBackend()
    assert b._info_overlay_key("kb", None) is None
    assert b._info_overlay_key("kb", SingleTenantContext("kb")) is None


def test_memory_overlay_key_bound_is_isolated() -> None:
    b = InMemoryKnowledgeBackend()
    assert (
        b._info_overlay_key("kb", BoundTenantContext("acme", "kb"))
        == "tenants/acme/_state/kb"
    )
    alpha = b._info_overlay_key("kb", BoundTenantContext("alpha", "kb"))
    beta = b._info_overlay_key("kb", BoundTenantContext("beta", "kb"))
    assert alpha != beta
