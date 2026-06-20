"""Tests for the config / environment TenantContext factories."""

from __future__ import annotations

import pytest

from dataknobs_common.tenancy import (
    BoundTenantContext,
    PrefixedTenantContext,
    SharedCorpusTenantContext,
    SingleTenantContext,
    create_tenant_context,
    tenant_context_from_env,
)


# --- create_tenant_context: inference --------------------------------- #


def test_create_infers_single_from_domain_only() -> None:
    ctx = create_tenant_context({"domain_id": "kb"})
    assert ctx == SingleTenantContext("kb")


def test_create_infers_bound_from_tenant_id() -> None:
    ctx = create_tenant_context({"domain_id": "kb", "tenant_id": "acme"})
    assert ctx == BoundTenantContext("acme", "kb")


def test_create_infers_prefixed_from_pattern() -> None:
    ctx = create_tenant_context(
        {
            "domain_id": "kb",
            "tenant_id": "acme",
            "prefix_pattern": "cust_{tenant_id}/",
        }
    )
    assert ctx == PrefixedTenantContext("acme", "kb", "cust_{tenant_id}/")


def test_create_infers_shared_corpus_from_corpus_id() -> None:
    ctx = create_tenant_context(
        {
            "domain_id": "view",
            "tenant_id": "acme",
            "shared_corpus_id": "corpus",
        }
    )
    assert ctx == SharedCorpusTenantContext("acme", "view", "corpus")


# --- create_tenant_context: explicit kind ----------------------------- #


def test_create_explicit_kind_overrides_inference() -> None:
    # tenant_id present would infer "bound", but explicit "single" wins.
    ctx = create_tenant_context(
        {"kind": "single", "domain_id": "kb", "tenant_id": "ignored"}
    )
    assert ctx == SingleTenantContext("kb")


def test_create_ignores_extra_keys() -> None:
    ctx = create_tenant_context(
        {"domain_id": "kb", "tenant_id": "acme", "unrelated": "x"}
    )
    assert ctx == BoundTenantContext("acme", "kb")


# --- create_tenant_context: errors ------------------------------------ #


def test_create_requires_domain_id() -> None:
    with pytest.raises(ValueError, match="domain_id"):
        create_tenant_context({"tenant_id": "acme"})


def test_create_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="Unknown tenant-context kind"):
        create_tenant_context({"kind": "bogus", "domain_id": "kb"})


def test_create_bound_requires_tenant_id() -> None:
    with pytest.raises(ValueError, match="tenant_id"):
        create_tenant_context({"kind": "bound", "domain_id": "kb"})


def test_create_prefixed_requires_pattern() -> None:
    with pytest.raises(ValueError, match="prefix_pattern"):
        create_tenant_context(
            {"kind": "prefixed", "domain_id": "kb", "tenant_id": "acme"}
        )


def test_create_shared_corpus_requires_corpus_id() -> None:
    with pytest.raises(ValueError, match="shared_corpus_id"):
        create_tenant_context(
            {"kind": "shared_corpus", "domain_id": "kb", "tenant_id": "acme"}
        )


def test_create_rejects_ambiguous_inference() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        create_tenant_context(
            {
                "domain_id": "kb",
                "tenant_id": "acme",
                "prefix_pattern": "p/",
                "shared_corpus_id": "corpus",
            }
        )


# --- tenant_context_from_env ------------------------------------------ #


def test_env_single_from_domain_only() -> None:
    ctx = tenant_context_from_env(environ={"DOMAIN_ID": "kb"})
    assert ctx == SingleTenantContext("kb")


def test_env_bound_from_tenant_id() -> None:
    ctx = tenant_context_from_env(
        environ={"DOMAIN_ID": "kb", "TENANT_ID": "acme"}
    )
    assert ctx == BoundTenantContext("acme", "kb")


def test_env_prefixed_from_pattern() -> None:
    ctx = tenant_context_from_env(
        environ={
            "DOMAIN_ID": "kb",
            "TENANT_ID": "acme",
            "TENANT_PREFIX_PATTERN": "cust_{tenant_id}/",
        }
    )
    assert ctx == PrefixedTenantContext("acme", "kb", "cust_{tenant_id}/")


def test_env_shared_corpus() -> None:
    ctx = tenant_context_from_env(
        environ={
            "DOMAIN_ID": "view",
            "TENANT_ID": "acme",
            "TENANT_SHARED_CORPUS_ID": "corpus",
        }
    )
    assert ctx == SharedCorpusTenantContext("acme", "view", "corpus")


def test_env_requires_domain_id() -> None:
    with pytest.raises(ValueError, match="DOMAIN_ID"):
        tenant_context_from_env(environ={"TENANT_ID": "acme"})
