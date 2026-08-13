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

from dataclasses import dataclass

import pytest

from dataknobs_common import tenancy
from dataknobs_common.tenancy import (
    BoundTenantContext,
    PrefixedTenantContext,
    SharedCorpusTenantContext,
    create_tenant_context,
    validate_tenant_id,
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
    structured ``tenant_id`` removes the crafted half *at this layer*.

    This docstring used to add "a ``domain_id`` may legitimately name a
    subdirectory, and does", which was wrong: a nested domain addresses
    the literal segments a knowledge-base layout puts around it, and the
    backends now refuse it. What is still true is narrower and is the
    reason ``innocent`` below constructs — a ``TenantContext`` does not
    check ``domain_id``, because at *this* layer it is not one invariant
    (see :func:`test_a_domain_id_is_not_checked_at_this_layer`).
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


def test_a_domain_id_is_not_checked_at_this_layer() -> None:
    """``domain_id`` is not one invariant here, so there is no one check.

    This test used to assert the same thing for the opposite reason —
    that a ``domain_id`` "may legitimately name a subdirectory". It may
    not, where it reaches a knowledge-base layout: the backends refuse a
    structured ``domain_id``, because a nested one addresses the literal
    segments the layout puts around it.

    The check is not *also* here because a ``TenantContext`` does not
    know which of those a consumer will do with the value. It is a path
    segment when it reaches a KB layout; it is a hash input in
    ``dataknobs_data.user.store``, whose ``_document_id`` length-prefixes
    every component precisely so a separator inside one is structurally
    safe, and which builds a ``SingleTenantContext`` from that namespace.
    Rejecting a separator here would retract that documented tolerance to
    fail a case that is already refused where it actually composes.

    So the value passes through, and the composing site decides. The one
    thing this loses is fail-fast, which is the open question rather than
    the settled one.
    """
    ctx = BoundTenantContext(tenant_id="acme", domain_id="team/alpha")

    assert ctx.domain_id == "team/alpha"
    # And it never reaches this impl's own state prefix — only a
    # PrefixedTenantContext whose pattern interpolates {domain_id} does
    # that, and both knowledge backends bound the formatted result.
    assert ctx.state_key_prefix() == "tenants/acme/_state/"


def test_a_prefix_pattern_may_still_be_any_convention() -> None:
    """``prefix_pattern`` is a deployment's own convention, not user input.

    It is bounded at the composing site instead — the file backend's
    ``_kb_path`` calls ``safe_join`` on the prefix as well as the domain,
    precisely because the pattern's literal text is not a ``tenant_id``
    and is not checked here.
    """
    ctx = PrefixedTenantContext(
        tenant_id="acme", domain_id="dom", prefix_pattern="legacy/{tenant_id}/{domain_id}/"
    )

    assert ctx.state_key_prefix() == "legacy/acme/dom/"


def test_an_ambiguous_prefix_pattern_still_merges_two_tenants() -> None:
    """The ``tenant_id`` guard cannot see a delimiter the *pattern* chose.

    ``validate_tenant_id`` keeps path separators and NUL out of the id,
    but a pattern is free to join ``{tenant_id}`` and ``{domain_id}``
    with a character the ids may legally contain. Two different tenants
    then share one state namespace — the same failure the guard closes
    for separators, reached without one.

    **Pinned rather than fixed.** Deciding this automatically means
    parsing the pattern against the space of ids a deployment might use,
    which is a design question rather than a missing check. The class
    docstring tells a consumer to choose a delimiter their ids cannot
    contain; this test is the evidence for why that sentence exists, and
    it fails if a later change makes the collision impossible — at which
    point the docstring and this test both want revisiting rather than
    one of them silently going stale.
    """
    pattern = "legacy/{tenant_id}-{domain_id}/"
    a = PrefixedTenantContext(tenant_id="acme", domain_id="x-y", prefix_pattern=pattern)
    b = PrefixedTenantContext(tenant_id="acme-x", domain_id="y", prefix_pattern=pattern)

    assert a.state_key_prefix() == b.state_key_prefix() == "legacy/acme-x-y/"


def test_the_invariant_is_reachable_from_the_documented_extension_point() -> None:
    """A consumer impl must be able to enforce what the reference impls do.

    The module docstring invites writing a frozen class satisfying
    ``TenantContext``, and nothing calls ``__post_init__`` on a class
    this module never sees. If the check were private, a consumer
    following that invitation would accept a structured id and merge two
    tenants' namespaces — the failure the check exists to prevent, in the
    one construction path the reference impls do not cover.
    """
    assert "validate_tenant_id" in tenancy.__all__
    assert validate_tenant_id("acme") is None
    with pytest.raises(ValueError):
        validate_tenant_id("acme/eu-west")


def test_a_consumer_impl_enforcing_the_invariant_behaves_like_the_references() -> None:
    """The exported check, used the way a consumer would use it."""

    @dataclass(frozen=True)
    class ConsumerTenantContext:
        tenant_id: str
        domain_id: str

        def __post_init__(self) -> None:
            validate_tenant_id(self.tenant_id)

        def lock_key(self, operation: str) -> str:
            return f"{operation}:{self.tenant_id}:{self.domain_id}"

        def state_key_prefix(self) -> str:
            return f"custom/{self.tenant_id}/"

        def matches(self, other: object) -> bool:
            return self == other

    assert ConsumerTenantContext("acme", "dom").state_key_prefix() == "custom/acme/"
    with pytest.raises(ValueError, match="separator"):
        ConsumerTenantContext("acme/eu-west", "dom")
