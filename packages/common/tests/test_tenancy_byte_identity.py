"""Byte-identity contract tests for TenantContext reference impls.

THESE TESTS ARE LOAD-BEARING. Outstanding distributed locks acquired before
the tenant-context surface existed are stranded if
``SingleTenantContext.lock_key`` drifts by even one character; per-tenant
snapshot state files become unreadable if ``state_key_prefix`` drifts.

The operation names enumerated below MUST match exactly the operation names
used in the pre-context backend code. If a backend adds a new operation,
this test gains a row; if a backend renames an operation, this test catches
the drift before consumers do.
"""

from __future__ import annotations

import pytest

from dataknobs_common.tenancy import (
    BoundTenantContext,
    SharedCorpusTenantContext,
    SingleTenantContext,
)

# Operation names used by the wizard / knowledge layer. This list MUST be
# kept in sync with backend code. CI guard: any backend code computing a
# lock key for a name not in this list is a new operation and gains a row.
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
def test_single_tenant_lock_key_matches_pre_context_format(
    operation: str,
) -> None:
    """SingleTenantContext.lock_key MUST equal ``"{operation}:{domain_id}"``
    for every operation name."""
    ctx = SingleTenantContext("my_kb")
    expected = f"{operation}:my_kb"
    assert ctx.lock_key(operation) == expected, (
        f"Byte-drift: SingleTenantContext('my_kb').lock_key({operation!r}) "
        f"= {ctx.lock_key(operation)!r}, expected {expected!r}. "
        f"Outstanding distributed locks would be stranded."
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
