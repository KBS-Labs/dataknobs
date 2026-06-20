"""Protocol-conformance tests for TenantContext."""

from __future__ import annotations

from dataknobs_common.tenancy import (
    BoundTenantContext,
    PrefixedTenantContext,
    SharedCorpusTenantContext,
    SingleTenantContext,
    TenantContext,
)


def test_single_tenant_conforms() -> None:
    assert isinstance(SingleTenantContext("k"), TenantContext)


def test_bound_tenant_conforms() -> None:
    assert isinstance(BoundTenantContext("t", "k"), TenantContext)


def test_prefixed_conforms() -> None:
    assert isinstance(
        PrefixedTenantContext("t", "k", "tenants/{tenant_id}/"),
        TenantContext,
    )


def test_shared_corpus_conforms() -> None:
    assert isinstance(
        SharedCorpusTenantContext("t", "k", "corpus"),
        TenantContext,
    )


def test_protocol_is_runtime_checkable() -> None:
    class CustomContext:
        @property
        def tenant_id(self) -> str | None:
            return None

        @property
        def domain_id(self) -> str:
            return "x"

        def lock_key(self, operation: str) -> str:
            return ""

        def state_key_prefix(self) -> str:
            return ""

        def matches(self, other: object) -> bool:
            return True

    assert isinstance(CustomContext(), TenantContext)


def test_protocol_rejects_non_conformer() -> None:
    class NotAContext:
        @property
        def tenant_id(self) -> str | None:
            return None

        # missing the other members

    assert not isinstance(NotAContext(), TenantContext)
