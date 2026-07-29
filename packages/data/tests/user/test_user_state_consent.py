"""Consent-gated access tests for the per-user state coordinator.

Reproduce-first for the governance layer: a section declaring a
``consent_scope`` refuses reads and writes until the user grants that scope,
and ``snapshot()`` omits an ungranted section rather than raising. Real
constructs only (``AsyncMemoryDatabase`` / ``SyncMemoryDatabase``); the real
CAS / Postgres path stays in ``test_user_state_store_postgres.py``.
"""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_common.exceptions import ConfigurationError, ConsentRequiredError
from dataknobs_data.user import (
    AsyncUserStateStore,
    UserStateStore,
    UserStateStoreConfig,
)

# Two consent-scoped sections sharing one scope (``pii_processing``) prove the
# per-``(user, scope)`` granularity; ``preferences`` / ``alerts`` are ungated.
_SECTIONS = [
    {"name": "preferences", "kind": "document"},
    {
        "name": "pii",
        "kind": "document",
        "consent_scope": "pii_processing",
        "sensitivity": "sensitive",
    },
    {"name": "history", "kind": "collection", "consent_scope": "pii_processing"},
    {"name": "alerts", "kind": "collection"},
]

_SCOPE = "pii_processing"


def _config(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "backend": "memory",
        "namespace": "acme",
        "sections": list(_SECTIONS),
    }
    base.update(overrides)
    return base


async def _make_async(**overrides: Any) -> AsyncUserStateStore:
    return await AsyncUserStateStore.from_config(_config(**overrides))


# --------------------------------------------------------------------- #
# 1. Write gate — refused without a grant, allowed after (async + sync).
# --------------------------------------------------------------------- #


async def test_write_to_consent_scoped_section_refused_without_grant() -> None:
    store = await _make_async()
    try:
        # Document and collection writes are both gated.
        with pytest.raises(ConsentRequiredError):
            await store.put_document("u1", "pii", {"ssn": "secret"})
        with pytest.raises(ConsentRequiredError):
            await store.add_record("u1", "history", {"event": "login"})

        # An ungated section is unaffected.
        await store.put_document("u1", "preferences", {"theme": "dark"})

        # After granting, both gated writes succeed.
        await store.grant_consent("u1", _SCOPE)
        assert await store.has_consent("u1", _SCOPE) is True
        await store.put_document("u1", "pii", {"ssn": "secret"})
        await store.add_record("u1", "history", {"event": "login"})
        assert (await store.get_document("u1", "pii")).get_value("ssn") == "secret"
    finally:
        await store.close()


def test_write_gate_sync() -> None:
    store = UserStateStore.from_config(_config())
    try:
        with pytest.raises(ConsentRequiredError):
            store.put_document("u1", "pii", {"ssn": "secret"})
        with pytest.raises(ConsentRequiredError):
            store.add_record("u1", "history", {"event": "login"})

        store.grant_consent("u1", _SCOPE)
        assert store.has_consent("u1", _SCOPE) is True
        store.put_document("u1", "pii", {"ssn": "secret"})
        assert store.get_document("u1", "pii").get_value("ssn") == "secret"
    finally:
        store.close()


async def test_has_consent_defaults_false() -> None:
    store = await _make_async()
    try:
        assert await store.has_consent("nobody", _SCOPE) is False
    finally:
        await store.close()


# --------------------------------------------------------------------- #
# 2 + 3. Read gate; snapshot omits ungranted; revoke blocks but preserves.
# --------------------------------------------------------------------- #


async def test_read_gate_snapshot_omission_and_revoke() -> None:
    store = await _make_async()
    try:
        await store.grant_consent("u1", _SCOPE)
        await store.put_document("u1", "pii", {"ssn": "secret"})
        await store.add_record("u1", "history", {"event": "login"})
        await store.put_document("u1", "preferences", {"theme": "dark"})

        # Granted: direct reads succeed and snapshot surfaces the section.
        assert (await store.get_document("u1", "pii")).get_value("ssn") == "secret"
        snap = await store.snapshot("u1", include_sensitive=True)
        assert snap["pii"] == {"ssn": "secret"}
        assert snap["history"] == [{"event": "login"}]

        # Revoke → future access blocked, data left in place.
        await store.revoke_consent("u1", _SCOPE)
        assert await store.has_consent("u1", _SCOPE) is False
        with pytest.raises(ConsentRequiredError):
            await store.get_document("u1", "pii")
        with pytest.raises(ConsentRequiredError):
            await store.query("u1", "history")

        # snapshot omits the ungranted sections but keeps the ungated one.
        snap = await store.snapshot("u1", include_sensitive=True)
        assert "pii" not in snap
        assert "history" not in snap
        assert snap["preferences"] == {"theme": "dark"}

        # Re-grant → the untouched data surfaces again (block-only revoke).
        await store.grant_consent("u1", _SCOPE)
        assert (await store.get_document("u1", "pii")).get_value("ssn") == "secret"
        snap = await store.snapshot("u1", include_sensitive=True)
        assert snap["pii"] == {"ssn": "secret"}
    finally:
        await store.close()


async def test_clear_erases_under_revoked_consent() -> None:
    """Erasure is never consent-gated — ``clear()`` removes gated data even
    while the scope is revoked (data minimization must always be possible).
    """
    store = await _make_async()
    try:
        await store.grant_consent("u1", _SCOPE)
        await store.put_document("u1", "pii", {"ssn": "secret"})
        await store.add_record("u1", "history", {"event": "login"})
        await store.revoke_consent("u1", _SCOPE)

        # clear() works despite the revoked scope and removes the gated rows.
        assert await store.clear("u1") >= 2
        await store.grant_consent("u1", _SCOPE)
        assert await store.get_document("u1", "pii") is None
        assert await store.query("u1", "history") == []
    finally:
        await store.close()


# --------------------------------------------------------------------- #
# 4. Per-(user, scope): one grant unlocks every section sharing the scope.
# --------------------------------------------------------------------- #


async def test_shared_scope_unlocks_all_sections() -> None:
    store = await _make_async()
    try:
        await store.grant_consent("u1", _SCOPE)
        # Both sections carry consent_scope == _SCOPE; one grant covers both.
        await store.put_document("u1", "pii", {"ssn": "secret"})
        await store.add_record("u1", "history", {"event": "login"})

        # A different user sharing the same scope name is independent.
        with pytest.raises(ConsentRequiredError):
            await store.put_document("u2", "pii", {"ssn": "other"})
    finally:
        await store.close()


# --------------------------------------------------------------------- #
# 5. Reserved section name ``consent`` is rejected at config-load time.
# --------------------------------------------------------------------- #


def test_reserved_consent_section_name_rejected() -> None:
    with pytest.raises(ConfigurationError):
        UserStateStoreConfig.from_dict(
            {
                "namespace": "acme",
                "sections": [{"name": "consent", "kind": "document"}],
            }
        )


# --------------------------------------------------------------------- #
# Consent helpers with no consent-scoped section → clear config error.
# --------------------------------------------------------------------- #


async def test_consent_helpers_without_consent_section_raise() -> None:
    """A store whose sections declare no ``consent_scope`` has no consent to
    manage — the helpers fail fast with a configuration error rather than
    writing an orphan consent document.
    """
    store = await AsyncUserStateStore.from_config(
        {
            "namespace": "acme",
            "sections": [{"name": "preferences", "kind": "document"}],
        }
    )
    try:
        assert store._consent_enabled() is False
        with pytest.raises(ConfigurationError):
            await store.grant_consent("u1", "anything")
        with pytest.raises(ConfigurationError):
            await store.has_consent("u1", "anything")
    finally:
        await store.close()


# --------------------------------------------------------------------- #
# Method parity — the governance API exists on both variants.
# --------------------------------------------------------------------- #


def test_consent_method_parity() -> None:
    import inspect

    for name in ("grant_consent", "revoke_consent", "has_consent"):
        assert hasattr(AsyncUserStateStore, name)
        assert hasattr(UserStateStore, name)
        assert (
            inspect.signature(getattr(AsyncUserStateStore, name))
            == inspect.signature(getattr(UserStateStore, name))
        )
