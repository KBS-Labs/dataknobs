"""Format-stability contract tests for TenantContext reference impls.

THESE TESTS ARE LOAD-BEARING. They pin the canonical lock-key and
state-key-prefix formats so they cannot drift once backends adopt the context
surface and start depending on them. Two guarantees are genuinely
backwards-compatible with pre-context single-tenant code:

- ``SingleTenantContext.state_key_prefix() == ""`` — the empty prefix keeps
  existing on-disk state-file paths readable; any drift here (e.g. ``"_state/"``)
  silently orphans those files.
- ``BoundTenantContext.lock_key(op) == "{op}:{tenant}:{domain}"`` — the
  tenant-scoped lock shape adopting backends route bound-tenant locks through.

The ``SingleTenantContext.lock_key`` rows pin the *canonical* single-tenant
format. (There is no single pre-context lock-key format to be byte-identical
to — the only existing lock key is a per-domain ingest lock with its own
shape; reconciling specific call sites to the context surface, with their own
per-backend identity tests, is the adoption step that follows this substrate.)

The operation names enumerated below are representative; the parametrization
asserts the format is stable across operation names, not that this is the
exhaustive operation set.
"""

from __future__ import annotations

import pytest

from dataknobs_common.tenancy import (
    BoundTenantContext,
    SharedCorpusTenantContext,
    SingleTenantContext,
)

# Representative operation names used by the knowledge layer. Used to assert
# the lock-key format is stable across operation names (not an exhaustive or
# authoritative operation set — backends own their own operation vocabularies).
SHIMMED_OPERATIONS = [
    "set_ingestion_status",
    "get_info",
    "has_changes_since",
    "list_changes_since",
    "get_checksum",
    "save_metadata",
    "record_snapshot",
    "put_file",
    "delete_file",
    "ingest_changes",
]


@pytest.mark.parametrize("operation", SHIMMED_OPERATIONS)
def test_single_tenant_lock_key_canonical_format(
    operation: str,
) -> None:
    """SingleTenantContext.lock_key MUST equal the canonical
    ``"{operation}:{domain_id}"`` for every operation name. Pinned so the
    format cannot drift once backends route single-tenant locks through it."""
    ctx = SingleTenantContext("my_kb")
    expected = f"{operation}:my_kb"
    assert ctx.lock_key(operation) == expected, (
        f"Format drift: SingleTenantContext('my_kb').lock_key({operation!r}) "
        f"= {ctx.lock_key(operation)!r}, expected {expected!r}."
    )


def test_single_tenant_state_key_prefix_is_empty() -> None:
    """SingleTenantContext.state_key_prefix() MUST return '' exactly.
    Anything else makes pre-context state files unreadable."""
    prefix = SingleTenantContext("any_domain").state_key_prefix()
    assert prefix == "", (
        f"Byte-drift: SingleTenantContext.state_key_prefix() returned "
        f"{prefix!r}; expected ''. Pre-context state files would become "
        f"unreadable."
    )


@pytest.mark.parametrize("operation", SHIMMED_OPERATIONS)
def test_bound_tenant_lock_key_format(operation: str) -> None:
    """BoundTenantContext.lock_key MUST equal
    ``"{operation}:{tenant_id}:{domain_id}"`` exactly. Drift here causes
    cross-tenant lock collisions."""
    ctx = BoundTenantContext("acme", "kb")
    expected = f"{operation}:acme:kb"
    assert ctx.lock_key(operation) == expected


def test_bound_tenant_state_key_prefix() -> None:
    """BoundTenantContext.state_key_prefix() MUST equal
    ``"tenants/{tenant_id}/_state/"`` exactly, including trailing slash.
    The trailing slash is part of the concatenation convention."""
    assert (
        BoundTenantContext("acme", "kb").state_key_prefix()
        == "tenants/acme/_state/"
    )


def test_shared_corpus_lock_key_uses_corpus_not_domain() -> None:
    """SharedCorpusTenantContext locks on (tenant, shared_corpus), not
    (tenant, domain). Two per-tenant views over the same shared corpus
    must share lock space within the same tenant."""
    ctx_view_a = SharedCorpusTenantContext(
        tenant_id="acme",
        domain_id="legal_view",
        shared_corpus_id="legal_corpus",
    )
    ctx_view_b = SharedCorpusTenantContext(
        tenant_id="acme",
        domain_id="contracts_view",
        shared_corpus_id="legal_corpus",
    )
    assert (
        ctx_view_a.lock_key("save_metadata")
        == "save_metadata:acme:legal_corpus"
    )
    assert (
        ctx_view_b.lock_key("save_metadata")
        == "save_metadata:acme:legal_corpus"
    )
    # Same lock key — concurrent state writes from two views serialize.


def test_shared_corpus_matches_across_views() -> None:
    """matches() is equivalence on (tenant_id, shared_corpus_id), not on
    domain_id. Different per-tenant views of the same shared corpus match —
    content-keyed caches share entries."""
    a = SharedCorpusTenantContext("acme", "view_a", "corpus")
    b = SharedCorpusTenantContext("acme", "view_b", "corpus")
    assert a.matches(b)


def test_single_tenant_eq_str_only_when_tenant_id_none() -> None:
    """SingleTenantContext.__eq__ accepts str equality for the single-tenant
    migration case. Multi-tenant impls do NOT support str equality."""
    ctx = SingleTenantContext("my_kb")
    assert ctx == "my_kb"
    assert ctx != "other"


def test_bound_tenant_does_not_support_str_eq() -> None:
    """No str-equality shim for multi-tenant impls — consumers MUST migrate
    to .matches()."""
    ctx = BoundTenantContext("acme", "my_kb")
    assert ctx != "my_kb"
    assert ctx != "acme"
