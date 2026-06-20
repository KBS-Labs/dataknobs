"""Protocol-conformance tests for ScopeProjector."""

from __future__ import annotations

from dataknobs_common.scope import (
    CachedProjector,
    CallableProjector,
    ChainedProjector,
    IdentityProjector,
    ReadOnlyProjector,
    ScopeProjector,
    WhitelistProjector,
)


def test_identity_conforms_to_protocol() -> None:
    assert isinstance(IdentityProjector(), ScopeProjector)


def test_readonly_conforms_to_protocol() -> None:
    assert isinstance(ReadOnlyProjector({}), ScopeProjector)


def test_whitelist_conforms_to_protocol() -> None:
    assert isinstance(WhitelistProjector({}, frozenset()), ScopeProjector)


def test_chained_conforms_to_protocol() -> None:
    assert isinstance(
        ChainedProjector(IdentityProjector()),
        ScopeProjector,
    )


def test_callable_conforms_to_protocol() -> None:
    assert isinstance(
        CallableProjector(lambda _: {}),
        ScopeProjector,
    )


def test_cached_conforms_to_protocol() -> None:
    assert isinstance(
        CachedProjector(IdentityProjector()),
        ScopeProjector,
    )


def test_protocol_is_runtime_checkable() -> None:
    class CustomProjector:
        def project(self, source):
            return {}

    assert isinstance(CustomProjector(), ScopeProjector)


def test_protocol_rejects_non_conformer() -> None:
    class NotAProjector:
        def transform(self, source):  # wrong method name
            return {}

    assert not isinstance(NotAProjector(), ScopeProjector)
