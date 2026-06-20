"""Behavior tests for TenantContext reference implementations."""

from __future__ import annotations

import pytest

from dataknobs_common.tenancy import (
    BoundTenantContext,
    PrefixedTenantContext,
    SharedCorpusTenantContext,
    SingleTenantContext,
)


# --- SingleTenantContext ---------------------------------------------- #


def test_single_tenant_id_is_none() -> None:
    assert SingleTenantContext("k").tenant_id is None


def test_single_matches_same_domain() -> None:
    a = SingleTenantContext("k")
    b = SingleTenantContext("k")
    assert a.matches(b)


def test_single_does_not_match_different_domain() -> None:
    assert not SingleTenantContext("k").matches(SingleTenantContext("j"))


def test_single_does_not_match_bound_tenant() -> None:
    assert not SingleTenantContext("k").matches(
        BoundTenantContext("t", "k"),
    )


def test_single_is_hashable() -> None:
    s = {SingleTenantContext("a"), SingleTenantContext("a")}
    assert len(s) == 1


def test_single_is_frozen() -> None:
    ctx = SingleTenantContext("k")
    with pytest.raises(AttributeError):
        ctx.domain_id = "other"  # type: ignore[misc]


def test_single_matches_is_symmetric_and_reflexive() -> None:
    a = SingleTenantContext("k")
    b = SingleTenantContext("k")
    assert a.matches(a)
    assert a.matches(b) == b.matches(a)


# --- BoundTenantContext ----------------------------------------------- #


def test_bound_tenant_id_returns_tenant() -> None:
    assert BoundTenantContext("acme", "kb").tenant_id == "acme"


def test_bound_matches_same_tenant_and_domain() -> None:
    a = BoundTenantContext("acme", "kb")
    b = BoundTenantContext("acme", "kb")
    assert a.matches(b)


def test_bound_does_not_match_different_tenant() -> None:
    assert not BoundTenantContext("a", "k").matches(
        BoundTenantContext("b", "k"),
    )


def test_bound_does_not_match_different_domain() -> None:
    assert not BoundTenantContext("a", "k").matches(
        BoundTenantContext("a", "j"),
    )


def test_bound_is_hashable() -> None:
    s = {BoundTenantContext("a", "k"), BoundTenantContext("a", "k")}
    assert len(s) == 1


# --- PrefixedTenantContext -------------------------------------------- #


def test_prefixed_lock_key_includes_formatted_prefix() -> None:
    ctx = PrefixedTenantContext(
        tenant_id="acme",
        domain_id="kb",
        prefix_pattern="cust_{tenant_id}/",
    )
    key = ctx.lock_key("save")
    assert "cust_acme/" in key
    assert key.startswith("save:")
    assert key.endswith(":kb")


def test_prefixed_state_prefix_uses_pattern() -> None:
    ctx = PrefixedTenantContext(
        tenant_id="acme",
        domain_id="kb",
        prefix_pattern="cust_{tenant_id}/",
    )
    assert ctx.state_key_prefix() == "cust_acme/"


def test_prefixed_matches_includes_pattern() -> None:
    a = PrefixedTenantContext("t", "k", "p_{tenant_id}/")
    b = PrefixedTenantContext("t", "k", "p_{tenant_id}/")
    c = PrefixedTenantContext("t", "k", "other_{tenant_id}/")
    assert a.matches(b)
    assert not a.matches(c)


# --- SharedCorpusTenantContext ---------------------------------------- #


def test_shared_corpus_state_isolated_by_tenant() -> None:
    a = SharedCorpusTenantContext("alpha", "view", "corpus")
    b = SharedCorpusTenantContext("beta", "view", "corpus")
    # State prefix per tenant — different.
    assert a.state_key_prefix() != b.state_key_prefix()


def test_shared_corpus_lock_keyed_by_tenant_and_corpus() -> None:
    a = SharedCorpusTenantContext("alpha", "view_a", "corpus")
    b = SharedCorpusTenantContext("alpha", "view_b", "corpus")
    # Both views of same corpus + tenant share lock.
    assert a.lock_key("save") == b.lock_key("save")


def test_shared_corpus_matches_across_views_same_tenant() -> None:
    a = SharedCorpusTenantContext("alpha", "view_a", "corpus")
    b = SharedCorpusTenantContext("alpha", "view_b", "corpus")
    assert a.matches(b)


def test_shared_corpus_does_not_match_different_tenant() -> None:
    a = SharedCorpusTenantContext("alpha", "view", "corpus")
    b = SharedCorpusTenantContext("beta", "view", "corpus")
    assert not a.matches(b)
