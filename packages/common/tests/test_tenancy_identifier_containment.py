"""A ``tenant_id`` is an isolation segment, so it must not carry structure.

``state_key_prefix()`` interpolates ``tenant_id`` into a *structured key
namespace* — ``"tenants/{tenant_id}/_state/"`` — that three backends
consume in three different ways: as a filesystem path, as an S3 object-key
prefix, and as an in-memory dict key. A separator inside the identifier
therefore fails in two distinct ways, and only one of them is a path
problem:

* on the **filesystem** backend the OS resolves ``..``, so the state write
  lands outside the backend's base directory;
* on the **key-string** backends nothing resolves, but the prefix and the
  ``domain_id`` are concatenated — so a crafted ``tenant_id`` can produce
  the *same* state key as a different ``(tenant, domain)`` pair and read
  its state. No traversal is involved; the namespace simply merges.

That second failure is why the composing-site guard is not sufficient on
its own. ``safe_join`` at the file backend's ``_kb_path`` closes the first
and cannot see the second, because there is no path there to contain.
The identifier is checked where it is constructed.
"""

from __future__ import annotations

import pytest

from dataknobs_common.tenancy import (
    BoundTenantContext,
    PrefixedTenantContext,
    SharedCorpusTenantContext,
    create_tenant_context,
)

ESCAPING_IDS = [
    "../elsewhere",
    "a/../b",
    "team/alpha",
    "..",
    ".",
    "back\\slash",
    "with\x00nul",
    "/absolute",
]


@pytest.mark.parametrize("tenant_id", ESCAPING_IDS)
def test_bound_context_rejects_a_structured_tenant_id(tenant_id: str) -> None:
    with pytest.raises(ValueError):
        BoundTenantContext(tenant_id=tenant_id, domain_id="dom")


@pytest.mark.parametrize("tenant_id", ESCAPING_IDS)
def test_prefixed_context_rejects_a_structured_tenant_id(tenant_id: str) -> None:
    with pytest.raises(ValueError):
        PrefixedTenantContext(tenant_id=tenant_id, domain_id="dom", prefix_pattern="{tenant_id}/")


@pytest.mark.parametrize("tenant_id", ESCAPING_IDS)
def test_shared_corpus_context_rejects_a_structured_tenant_id(tenant_id: str) -> None:
    with pytest.raises(ValueError):
        SharedCorpusTenantContext(tenant_id=tenant_id, domain_id="dom", shared_corpus_id="corpus")


def test_the_factory_rejects_it_too() -> None:
    """Config-driven construction is the path a deployment actually takes."""
    with pytest.raises(ValueError):
        create_tenant_context({"domain_id": "dom", "tenant_id": "../elsewhere"})


def test_the_state_key_collision_is_no_longer_constructible() -> None:
    """The concrete cross-tenant merge, pinned at its root.

    ``tenants/{t}/_state/`` + ``domain_id`` gave the same key for
    ``(t="acme", d="proj/_state/secret")`` and
    ``(t="acme/_state/proj", d="secret")``, so the second tenant read the
    first's ingest state through a key-string backend. Rejecting the
    structured ``tenant_id`` removes the only crafted half — a
    ``domain_id`` may legitimately name a subdirectory, and does.
    """
    innocent = BoundTenantContext(tenant_id="acme", domain_id="proj/_state/secret")

    with pytest.raises(ValueError):
        BoundTenantContext(tenant_id="acme/_state/proj", domain_id="secret")

    assert innocent.state_key_prefix() == "tenants/acme/_state/"


def test_an_ordinary_tenant_id_is_untouched() -> None:
    """Every existing single- and multi-tenant deployment shape still builds."""
    for tenant_id in ("acme", "acme-corp", "acme_corp", "tenant.42", "ACME123"):
        ctx = BoundTenantContext(tenant_id=tenant_id, domain_id="dom")
        assert ctx.state_key_prefix() == f"tenants/{tenant_id}/_state/"


def test_a_domain_id_may_still_name_a_subdirectory() -> None:
    """The guard is on the isolation segment, not on every identifier."""
    ctx = BoundTenantContext(tenant_id="acme", domain_id="team/alpha")

    assert ctx.domain_id == "team/alpha"


def test_a_prefix_pattern_may_still_be_any_convention() -> None:
    """``prefix_pattern`` is a deployment's own convention, not user input.

    It is bounded at the composing site instead — the file backend's
    ``_kb_path`` calls ``safe_join`` on the prefix as well as the domain,
    precisely because the pattern's literal text is not a ``tenant_id``
    and is not checked here.
    """
    ctx = PrefixedTenantContext(
        tenant_id="acme", domain_id="dom", prefix_pattern="legacy/{tenant_id}-{domain_id}/"
    )

    assert ctx.state_key_prefix() == "legacy/acme-dom/"
